#!/usr/bin/env python3
"""
Model Optimization Utilities for AIY Vision Kit BEV-OBB Detection

This module provides utilities for optimizing YOLOv5-based models for deployment
on resource-constrained hardware like the Raspberry Pi Zero W. It implements
various optimization techniques including quantization, pruning, and knowledge
distillation.

References:
    - Jacob, B., et al. (2018). Quantization and training of neural networks for 
      efficient integer-arithmetic-only inference. CVPR.
    - Han, S., et al. (2015). Learning both weights and connections for efficient 
      neural network. NeurIPS.
    - Hinton, G., et al. (2015). Distilling the knowledge in a neural network. 
      arXiv preprint arXiv:1503.02531.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.quantization as quantization
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import yaml
import time

from torch.utils.data import DataLoader
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.utils import prune


class QuantizedYOLOv5OBB(nn.Module):
    """
    Quantized version of YOLOv5 for Oriented Bounding Box detection.
    
    This class wraps the original YOLOv5-OBB model with quantization stubs
    for efficient int8 inference on CPU-only devices.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize quantized model wrapper.
        
        Args:
            model (nn.Module): Original YOLOv5-OBB model
        """
        super(QuantizedYOLOv5OBB, self).__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantization.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Model output
        """
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


class ModelOptimizer:
    """
    Comprehensive model optimization toolkit for Raspberry Pi Zero deployment.
    
    This class provides various optimization techniques to reduce model size,
    inference time, and memory consumption while maintaining detection accuracy.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the model optimizer.
        
        Args:
            config_path (str): Path to optimization configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cpu')  # Pi Zero only supports CPU
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.optimization_config = self.config.get('optimization', {})
        
    def quantize_model(self, model: nn.Module, 
                      calibration_loader: DataLoader,
                      quantization_type: str = 'dynamic') -> nn.Module:
        """
        Apply quantization to the model for efficient inference.
        
        Args:
            model (nn.Module): Original model
            calibration_loader (DataLoader): Data for calibration
            quantization_type (str): Type of quantization ('dynamic', 'static', 'qat')
            
        Returns:
            nn.Module: Quantized model
        """
        self.logger.info(f"Starting {quantization_type} quantization...")
        
        if quantization_type == 'dynamic':
            # Dynamic quantization - fastest to apply, good for CPU inference
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
            
        elif quantization_type == 'static':
            # Static quantization - requires calibration data
            model.eval()
            
            # Wrap model with quantization stubs
            quantized_model = QuantizedYOLOv5OBB(model)
            
            # Prepare for quantization
            quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(quantized_model, inplace=True)
            
            # Calibration
            self.logger.info("Calibrating model...")
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(calibration_loader):
                    if batch_idx >= 100:  # Limit calibration samples
                        break
                    quantized_model(data)
            
            # Convert to quantized model
            torch.quantization.convert(quantized_model, inplace=True)
            
        elif quantization_type == 'qat':
            # Quantization Aware Training - requires retraining
            model.train()
            quantized_model = QuantizedYOLOv5OBB(model)
            quantized_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(quantized_model, inplace=True)
            
            self.logger.warning("QAT requires additional training - returning prepared model")
            
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        self.logger.info("Quantization completed")
        return quantized_model
    
    def prune_model(self, model: nn.Module, 
                   pruning_ratio: float = 0.3,
                   pruning_type: str = 'unstructured') -> nn.Module:
        """
        Apply pruning to reduce model parameters.
        
        Args:
            model (nn.Module): Model to prune
            pruning_ratio (float): Fraction of parameters to prune
            pruning_type (str): Type of pruning ('unstructured', 'structured')
            
        Returns:
            nn.Module: Pruned model
        """
        self.logger.info(f"Starting {pruning_type} pruning with ratio {pruning_ratio}")
        
        if pruning_type == 'unstructured':
            # Magnitude-based unstructured pruning
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    parameters_to_prune.append((module, 'weight'))
            
            # Global magnitude pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
            
            # Remove pruning reparameterization
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
                
        elif pruning_type == 'structured':
            # Channel-wise structured pruning
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and module.out_channels > 1:
                    prune.ln_structured(
                        module, 
                        name='weight', 
                        amount=pruning_ratio, 
                        n=1, 
                        dim=0
                    )
                    prune.remove(module, 'weight')
        
        self.logger.info("Pruning completed")
        return model
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """
        Apply general inference optimizations.
        
        Args:
            model (nn.Module): Model to optimize
            
        Returns:
            nn.Module: Optimized model
        """
        self.logger.info("Optimizing model for inference...")
        
        # Set to evaluation mode
        model.eval()
        
        # Fuse operations where possible
        if hasattr(model, 'fuse'):
            model.fuse()
        
        # Apply torch.jit optimizations if supported
        try:
            # Create example input
            example_input = torch.randn(1, 3, 416, 416)
            traced_model = torch.jit.trace(model, example_input)
            traced_model = torch.jit.optimize_for_inference(traced_model)
            self.logger.info("Applied TorchScript optimizations")
            return traced_model
        except Exception as e:
            self.logger.warning(f"TorchScript optimization failed: {e}")
            return model
    
    def knowledge_distillation(self, teacher_model: nn.Module,
                             student_model: nn.Module,
                             train_loader: DataLoader,
                             num_epochs: int = 10,
                             temperature: float = 4.0,
                             alpha: float = 0.7) -> nn.Module:
        """
        Apply knowledge distillation to create a smaller student model.
        
        Args:
            teacher_model (nn.Module): Pre-trained teacher model
            student_model (nn.Module): Smaller student model
            train_loader (DataLoader): Training data
            num_epochs (int): Number of training epochs
            temperature (float): Distillation temperature
            alpha (float): Weight for distillation loss
            
        Returns:
            nn.Module: Trained student model
        """
        self.logger.info("Starting knowledge distillation...")
        
        teacher_model.eval()
        student_model.train()
        
        # Optimizers
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        
        # Loss functions
        kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
        task_loss_fn = nn.MSELoss()  # For bbox regression
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Teacher and student predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                
                student_outputs = student_model(data)
                
                # Knowledge distillation loss
                kd_loss = kd_loss_fn(
                    torch.log_softmax(student_outputs / temperature, dim=1),
                    torch.softmax(teacher_outputs / temperature, dim=1)
                ) * (temperature ** 2)
                
                # Task-specific loss
                task_loss = task_loss_fn(student_outputs, targets)
                
                # Combined loss
                loss = alpha * kd_loss + (1 - alpha) * task_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    self.logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")
        
        self.logger.info("Knowledge distillation completed")
        return student_model
    
    def benchmark_model(self, model: nn.Module, 
                       input_shape: Tuple[int, int, int, int] = (1, 3, 416, 416),
                       num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark model performance metrics.
        
        Args:
            model (nn.Module): Model to benchmark
            input_shape (Tuple): Input tensor shape
            num_runs (int): Number of inference runs
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        self.logger.info(f"Benchmarking model with {num_runs} runs...")
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Timing
        inference_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                inference_times.append(time.time() - start_time)
        
        # Calculate statistics
        inference_times = np.array(inference_times)
        
        # Model size
        param_size = sum(p.numel() for p in model.parameters())
        buffer_size = sum(b.numel() for b in model.buffers())
        model_size = param_size + buffer_size
        
        # Memory usage (approximate)
        memory_usage = sum(p.numel() * p.element_size() for p in model.parameters())
        memory_usage += sum(b.numel() * b.element_size() for b in model.buffers())
        
        metrics = {
            'avg_inference_time': float(np.mean(inference_times)),
            'std_inference_time': float(np.std(inference_times)),
            'min_inference_time': float(np.min(inference_times)),
            'max_inference_time': float(np.max(inference_times)),
            'fps': float(1.0 / np.mean(inference_times)),
            'parameters': int(param_size),
            'model_size_mb': float(memory_usage / (1024 * 1024)),
            'throughput': float(num_runs / np.sum(inference_times))
        }
        
        self.logger.info("Benchmark results:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
        
        return metrics
    
    def optimize_pipeline(self, model: nn.Module,
                         calibration_loader: Optional[DataLoader] = None,
                         train_loader: Optional[DataLoader] = None) -> nn.Module:
        """
        Apply complete optimization pipeline.
        
        Args:
            model (nn.Module): Original model
            calibration_loader (DataLoader): Calibration data for quantization
            train_loader (DataLoader): Training data for knowledge distillation
            
        Returns:
            nn.Module: Fully optimized model
        """
        self.logger.info("Starting complete optimization pipeline...")
        
        # Step 1: Initial benchmark
        initial_metrics = self.benchmark_model(model)
        self.logger.info(f"Initial model metrics: {initial_metrics}")
        
        optimized_model = model
        
        # Step 2: Pruning (if enabled)
        if self.optimization_config.get('enable_pruning', False):
            pruning_ratio = self.optimization_config.get('pruning_ratio', 0.3)
            optimized_model = self.prune_model(optimized_model, pruning_ratio)
        
        # Step 3: Knowledge Distillation (if enabled and data available)
        if (self.optimization_config.get('enable_distillation', False) and 
            train_loader is not None):
            # Create a smaller student model
            student_model = self._create_student_model(optimized_model)
            optimized_model = self.knowledge_distillation(
                optimized_model, student_model, train_loader
            )
        
        # Step 4: Quantization
        if calibration_loader is not None:
            quantization_type = self.optimization_config.get('quantization_type', 'dynamic')
            optimized_model = self.quantize_model(
                optimized_model, calibration_loader, quantization_type
            )
        else:
            # Apply dynamic quantization without calibration data
            optimized_model = self.quantize_model(optimized_model, None, 'dynamic')
        
        # Step 5: General inference optimizations
        optimized_model = self.optimize_for_inference(optimized_model)
        
        # Step 6: Final benchmark
        final_metrics = self.benchmark_model(optimized_model)
        self.logger.info(f"Final model metrics: {final_metrics}")
        
        # Calculate improvements
        speedup = initial_metrics['avg_inference_time'] / final_metrics['avg_inference_time']
        size_reduction = (initial_metrics['model_size_mb'] - final_metrics['model_size_mb']) / initial_metrics['model_size_mb']
        
        self.logger.info(f"Optimization results:")
        self.logger.info(f"  Speedup: {speedup:.2f}x")
        self.logger.info(f"  Size reduction: {size_reduction*100:.1f}%")
        
        return optimized_model
    
    def _create_student_model(self, teacher_model: nn.Module) -> nn.Module:
        """
        Create a smaller student model for knowledge distillation.
        
        Args:
            teacher_model (nn.Module): Teacher model
            
        Returns:
            nn.Module: Student model
        """
        # This is a simplified implementation
        # In practice, you would design a specific smaller architecture
        
        # For now, create a copy with reduced channels
        student_model = type(teacher_model)(teacher_model.config)
        
        # Reduce model capacity
        for name, module in student_model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Reduce output channels by half
                new_out_channels = max(1, module.out_channels // 2)
                new_module = nn.Conv2d(
                    module.in_channels, new_out_channels,
                    module.kernel_size, module.stride,
                    module.padding, module.dilation,
                    module.groups, module.bias is not None
                )
                # Copy weights for compatible channels
                with torch.no_grad():
                    new_module.weight[:, :, :, :] = module.weight[:new_out_channels, :, :, :]
                    if module.bias is not None:
                        new_module.bias[:] = module.bias[:new_out_channels]
                
                # Replace module
                parent = student_model
                components = name.split('.')
                for component in components[:-1]:
                    parent = getattr(parent, component)
                setattr(parent, components[-1], new_module)
        
        return student_model
    
    def save_optimized_model(self, model: nn.Module, 
                           save_path: str,
                           metadata: Optional[Dict] = None):
        """
        Save optimized model with metadata.
        
        Args:
            model (nn.Module): Optimized model
            save_path (str): Path to save model
            metadata (Dict): Additional metadata
        """
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimization_config': self.optimization_config,
            'timestamp': time.time()
        }
        
        if metadata:
            save_dict.update(metadata)
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(save_dict, save_path)
        self.logger.info(f"Optimized model saved to {save_path}")


def main():
    """Example usage of the model optimizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize YOLOv5-OBB model for Raspberry Pi')
    parser.add_argument('--config', type=str, default='config/optimization_config.yaml',
                       help='Path to optimization configuration')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model to optimize')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save optimized model')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = ModelOptimizer(args.config)
    
    # Load model
    model = torch.load(args.model, map_location='cpu')
    
    # Optimize model
    optimized_model = optimizer.optimize_pipeline(model)
    
    # Save optimized model
    optimizer.save_optimized_model(optimized_model, args.output)


if __name__ == "__main__":
    main()