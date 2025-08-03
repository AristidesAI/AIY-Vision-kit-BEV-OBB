# Installation Guide - AIY Vision Kit BEV-OBB Detection System

Complete installation guide for deploying the Birds-Eye View Oriented Bounding Box detection system on Google AIY Vision Kit and Raspberry Pi Zero W hardware.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Hardware Setup](#hardware-setup)
- [Automated Installation](#automated-installation)
- [Manual Installation](#manual-installation)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

- **Raspberry Pi Zero W** with AIY Vision Kit
- **MicroSD card** (16GB minimum, 32GB recommended, Class 10)
- **Raspberry Pi Camera Module v2** (8MP, IMX219 sensor)
- **Power Supply** (5V 2.1A micro-USB, stable power essential)
- **Internet connection** for initial setup and package downloads

### Recommended Hardware for Drone Integration

- **FPV Drone Frame** compatible with Raspberry Pi Zero W mounting
- **Camera Gimbal** for downward-facing orientation stabilization  
- **Extended Battery Pack** (recommended 2S-3S LiPo for extended flight time)
- **Vibration Dampeners** to reduce camera shake during flight
- **GPS Module** (optional, for mapping applications)

## Hardware Setup

### AIY Vision Kit Assembly

1. **Assemble Vision Kit** following [Google's official guide](https://aiyprojects.withgoogle.com/vision/)
2. **Camera Orientation**: Mount camera facing downward for BEV perspective
   - Adjust camera ribbon to allow 90° rotation
   - Secure mounting to prevent vibration during flight
3. **Vision Bonnet Connection**: Ensure proper GPIO connections
4. **microSD Card**: Flash with latest Raspberry Pi OS Lite

### Drone Integration Setup

For drone mounting applications:

```bash
# Camera mounting considerations for drone integration
# - Mount camera perpendicular to ground (90° downward)
# - Ensure ribbon cable has sufficient length and flexibility
# - Add vibration dampening between camera and drone frame
# - Consider gimbal integration for image stabilization
```

## Automated Installation

### Quick Installation Script

Download and run the automated installation script:

```bash
# Download installation script
curl -sSL https://raw.githubusercontent.com/yourusername/aiy-vision-kit-bev-obb/main/scripts/install_aiy_bev_obb.sh -o install_aiy_bev_obb.sh

# Make executable
chmod +x install_aiy_bev_obb.sh

# Run installation (requires sudo privileges)
./install_aiy_bev_obb.sh
```

The automated script performs:

1. **System Updates**: Updates Raspberry Pi OS and essential packages
2. **Hardware Configuration**: Enables camera, I2C, SPI interfaces
3. **Python Environment**: Creates isolated virtual environment
4. **PyTorch Installation**: Installs ARM-optimized PyTorch for Pi Zero W
5. **OpenCV Setup**: Installs optimized OpenCV for image processing
6. **AIY Integration**: Configures Vision Kit hardware components
7. **Model Optimization**: Downloads and optimizes pre-trained models
8. **Performance Tuning**: Applies Pi Zero W specific optimizations

### Installation Progress

The installation process takes approximately 30-60 minutes on Raspberry Pi Zero W:

```
[INFO] Checking if running on Raspberry Pi...
[SUCCESS] Raspberry Pi Zero detected
[INFO] Updating system packages...
[INFO] Setting up additional swap space for Pi Zero...
[INFO] Installing AIY Python packages...
[INFO] Installing PyTorch for ARMv6 (Pi Zero)...
[INFO] Installing OpenCV...
[SUCCESS] Installation completed successfully!
```

## Manual Installation

For users who prefer step-by-step control over the installation process:

### Step 1: System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install build dependencies
sudo apt install -y \
    build-essential cmake pkg-config git wget curl \
    libjpeg-dev libtiff5-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libfontconfig1-dev libcairo2-dev \
    libgdk-pixbuf2.0-dev libpango1.0-dev \
    libgtk2.0-dev libgtk-3-dev \
    libatlas-base-dev gfortran \
    libhdf5-dev libhdf5-serial-dev \
    python3-dev python3-pip python3-venv
```

### Step 2: Hardware Interface Configuration

```bash
# Enable camera interface
sudo raspi-config nonint do_camera 0

# Enable I2C for AIY Vision Bonnet
sudo raspi-config nonint do_i2c 0

# Enable SPI for AIY Vision Bonnet
sudo raspi-config nonint do_spi 0

# Configure GPU memory split (minimal for Pi Zero W)
sudo raspi-config nonint do_memory_split 16

# Verify camera detection
vcgencmd get_camera
# Expected output: supported=1 detected=1
```

### Step 3: Performance Optimization for Pi Zero W

```bash
# Configure CPU performance governor
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Add performance optimizations to boot config
sudo tee -a /boot/config.txt << EOF
# Performance optimizations for Pi Zero W
arm_freq=1000
core_freq=500
sdram_freq=500
over_voltage=2
max_usb_current=1

# Camera and I2C configuration
start_x=1
gpu_mem=128
dtparam=i2c_arm=on
dtparam=spi=on
dtparam=audio=off
EOF
```

### Step 4: Swap Space Configuration

Pi Zero W requires additional swap space for compilation:

```bash
# Configure swap for compilation phase
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Verify swap configuration
free -h
```

### Step 5: Python Environment Setup

```bash
# Create virtual environment
python3 -m venv ~/aiy_env
source ~/aiy_env/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install AIY-specific packages
pip install aiy-python-wheels
pip install aiy-vision-kit
pip install gpiozero picamera RPi.GPIO
```

### Step 6: PyTorch Installation (ARM-optimized)

For Raspberry Pi Zero W (ARMv6 architecture):

```bash
# Download pre-compiled PyTorch wheel for ARMv6
wget -O torch-1.9.0-cp37-cp37m-linux_armv6l.whl \
    https://github.com/KumaTea/pytorch-aarch64/releases/download/v1.9.0/torch-1.9.0-cp37-cp37m-linux_armv6l.whl

# Install PyTorch
pip install torch-1.9.0-cp37-cp37m-linux_armv6l.whl

# Install TorchVision (lightweight version)
pip install torchvision==0.10.0 --no-deps

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

### Step 7: OpenCV Installation

```bash
# Install optimized OpenCV for Pi Zero W
pip install opencv-python-headless==4.5.3.56

# Verify OpenCV installation
python -c "import cv2; print(f'OpenCV {cv2.__version__} installed successfully')"
```

### Step 8: Project Installation

```bash
# Clone project repository
git clone https://github.com/yourusername/aiy-vision-kit-bev-obb.git
cd aiy-vision-kit-bev-obb

# Install project dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .

# Create necessary directories
mkdir -p {models,data/{raw,processed,annotations},logs,output}
```

### Step 9: Model Setup and Optimization

```bash
# Download base YOLOv5 model
mkdir -p models
wget -O models/yolov5s.pt \
    https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt

# Optimize model for Pi Zero W deployment
python src/model_optimizer.py \
    --config config/optimization_config.yaml \
    --model models/yolov5s.pt \
    --output models/yolov5s_optimized.pt

# Verify optimized model
python -c "
import torch
model = torch.load('models/yolov5s_optimized.pt', map_location='cpu')
print(f'Optimized model loaded successfully')
print(f'Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters')
"
```

## Configuration

### System Configuration Files

The system uses YAML configuration files for flexible parameter management:

#### Main System Configuration (`config/bev_obb_config.yaml`)

```yaml
# Model configuration for drone BEV applications
model:
  name: "yolov5s_bev_obb"
  backbone: "mobilenet_v3_small"
  input_size: [416, 416]  # Optimized for Pi Zero W

# Detection classes for aerial/drone applications
classes:
  names: ['vehicle', 'person', 'bicycle', 'motorcycle', 'bus', 'truck', 'boat', 'airplane']
  nc: 8

# BEV-specific configuration for drone applications
bev:
  height_range: [2.0, 100.0]  # Drone altitude range (meters)
  width_range: [-50.0, 50.0]  # Field of view width (meters)
  resolution: 0.1  # Ground resolution (meters per pixel)
  
# Drone-specific camera configuration
aiy:
  camera_resolution: [1640, 1232]  # Full sensor resolution
  framerate: 15  # Optimized for processing capability
  rotation: 0  # Adjust based on camera mounting orientation
  
# Performance optimization for flight applications
hardware:
  device: "cpu"
  quantization: true
  threads: 1
  memory_limit_mb: 128
```

#### Optimization Configuration (`config/optimization_config.yaml`)

```yaml
optimization:
  # Aggressive optimization for drone applications
  optimization_level: "aggressive"
  
  # Quantization for real-time performance
  enable_quantization: true
  quantization_type: "dynamic"
  
  # Model pruning for memory efficiency
  enable_pruning: true
  pruning_ratio: 0.3
  
# Performance targets for drone deployment
performance_targets:
  max_inference_time_ms: 400  # 2.5 FPS minimum
  target_fps: 2.5
  max_memory_usage_mb: 128
  battery_efficiency_mode: true
```

### Environment Variables

Set environment variables for optimal performance:

```bash
# Add to ~/.bashrc for persistent configuration
echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc
echo 'export MKL_NUM_THREADS=1' >> ~/.bashrc
echo 'export OPENBLAS_NUM_THREADS=1' >> ~/.bashrc
echo 'export PYTORCH_CPU_ALLOCATOR_POLICY=expandable_segments:False' >> ~/.bashrc

# Reload configuration
source ~/.bashrc
```

## Verification

### System Verification Tests

Run comprehensive system tests to verify installation:

```bash
# Activate environment
source ~/aiy_env/bin/activate

# Run system verification
python -c "
import sys
import torch
import cv2
import numpy as np
from src.bev_obb_detector import BEVOBBDetector

print('=== System Verification ===')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')

# Test hardware components
try:
    detector = BEVOBBDetector('config/bev_obb_config.yaml')
    print('✓ Detection system initialized successfully')
except Exception as e:
    print(f'✗ Detection system error: {e}')

# Test camera
try:
    from picamera import PiCamera
    camera = PiCamera()
    camera.close()
    print('✓ Camera interface working')
except Exception as e:
    print(f'✗ Camera error: {e}')

# Test AIY hardware
try:
    from aiy.leds import Leds
    from gpiozero import Button
    print('✓ AIY hardware interfaces available')
except Exception as e:
    print(f'✗ AIY hardware error: {e}')

print('=== Verification Complete ===')
"
```

### Performance Benchmark

Run performance benchmark to verify optimization:

```bash
# Run system benchmark
python scripts/benchmark_system.py \
    --config config/bev_obb_config.yaml \
    --runs 25 \
    --output benchmark_results.json

# Expected performance targets:
# - Inference time: <400ms
# - Memory usage: <128MB  
# - FPS: >2.5
```

### Camera Test

Test camera functionality with BEV orientation:

```bash
# Test camera capture
python -c "
from picamera import PiCamera
from time import sleep
import numpy as np

camera = PiCamera()
camera.resolution = (1640, 1232)
camera.rotation = 0  # Adjust for downward-facing mount

print('Testing camera capture...')
camera.start_preview()
sleep(2)
camera.capture('test_bev_image.jpg')
camera.stop_preview()
camera.close()

print('✓ Test image captured: test_bev_image.jpg')
print('Verify image shows downward BEV perspective')
"
```

## Troubleshooting

### Common Installation Issues

#### Issue: PyTorch Installation Fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch
```

**Solution:**
```bash
# For Pi Zero W (ARMv6), use pre-compiled wheel
wget https://github.com/KumaTea/pytorch-aarch64/releases/download/v1.9.0/torch-1.9.0-cp37-cp37m-linux_armv6l.whl
pip install torch-1.9.0-cp37-cp37m-linux_armv6l.whl

# Verify architecture
uname -m  # Should show armv6l for Pi Zero W
```

#### Issue: Camera Not Detected

**Symptoms:**
```
mmal: mmal_vc_component_create: failed to create component 'vc.ril.camera' (1:ENOMEM)
```

**Solution:**
```bash
# Enable camera interface
sudo raspi-config nonint do_camera 0

# Increase GPU memory
sudo raspi-config nonint do_memory_split 128

# Check camera connection
vcgencmd get_camera

# Reboot system
sudo reboot
```

#### Issue: Memory Errors During Model Loading

**Symptoms:**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:75] posix_memalign
```

**Solution:**
```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=1024/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Reduce model input size in config
# Edit config/bev_obb_config.yaml:
# input_size: [320, 320]  # Reduced from [416, 416]
```

#### Issue: AIY Vision Bonnet Not Recognized

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/dev/vision_spicomm'
```

**Solution:**
```bash
# Enable SPI and I2C
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0

# Add device tree overlays
echo "dtoverlay=aiy-vision-bonnet" | sudo tee -a /boot/config.txt

# Check hardware connections and reboot
sudo reboot
```

### Performance Optimization Issues

#### Issue: Slow Inference Performance (>1000ms)

**Solution:**
```bash
# Enable performance CPU governor
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Apply CPU overclocking (Pi Zero W)
sudo tee -a /boot/config.txt << EOF
arm_freq=1000
core_freq=500
over_voltage=2
EOF

# Use quantized model
python src/model_optimizer.py --quantize-only

# Reduce input resolution
# Edit config: input_size: [320, 320]
```

#### Issue: High Memory Usage (>200MB)

**Solution:**
```bash
# Enable aggressive optimization
# Edit config/optimization_config.yaml:
# optimization_level: "aggressive"
# pruning_ratio: 0.5

# Apply memory optimizations
export PYTORCH_CPU_ALLOCATOR_POLICY=expandable_segments:False

# Use smaller model variant
# Download yolov5n.pt instead of yolov5s.pt
```

### Drone-Specific Installation Considerations

#### Vibration Dampening

```bash
# For drone applications, consider:
# 1. Anti-vibration mounts for camera
# 2. Soft mounting for Raspberry Pi
# 3. Cable strain relief
# 4. Secure connection verification before flight
```

#### Power Management

```bash
# Optimize for battery operation
# Edit /boot/config.txt:
echo "dtparam=audio=off" | sudo tee -a /boot/config.txt
echo "dtparam=bluetooth=off" | sudo tee -a /boot/config.txt
echo "disable_wifi_sleep=1" | sudo tee -a /boot/config.txt

# Monitor power consumption
python -c "
import psutil
import time

for i in range(10):
    battery = psutil.sensors_battery()
    if battery:
        print(f'Battery: {battery.percent}% - {battery.secsleft}s remaining')
    time.sleep(5)
"
```

### Getting Help

If you encounter issues not covered in this guide:

1. **Check System Logs**: `tail -f /var/log/syslog`
2. **Enable Debug Logging**: Set `log_level: DEBUG` in configuration
3. **Run Diagnostics**: `python scripts/system_diagnostics.py`
4. **Community Support**: Visit the project repository issues page
5. **Hardware Verification**: Use `gpio readall` to check pin configurations

### Post-Installation Setup

After successful installation:

1. **Reboot System**: `sudo reboot`
2. **Activate Environment**: `source ~/aiy_env/bin/activate`
3. **Navigate to Project**: `cd ~/AIY-Vision-kit-BEV-OBB`
4. **Run Detection**: `python run_detection.py`
5. **Configure Auto-start**: `sudo systemctl enable bev-obb-detector`

For drone integration, ensure proper camera orientation and secure mounting before flight operations.