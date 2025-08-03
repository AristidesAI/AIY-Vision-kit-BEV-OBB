#!/usr/bin/env python3
"""
AIY Vision Kit BEV-OBB Detection System Runner

This is the main entry point for running the Birds-Eye View Oriented 
Bounding Box detection system on Google AIY Vision Kit hardware.

Usage:
    python run_detection.py                          # Default config
    python run_detection.py config/custom.yaml       # Custom config
    python run_detection.py --help                   # Show help

Example:
    python run_detection.py \
        --config config/bev_obb_config.yaml \
        --output-dir /home/pi/detections \
        --log-level INFO \
        --auto-start
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from bev_obb_detector import BEVOBBDetector


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AIY Vision Kit BEV-OBB Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s config/custom_config.yaml
  %(prog)s --config config/bev_obb_config.yaml --auto-start
  %(prog)s --output-dir /home/pi/detections --log-level DEBUG
        """
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        default='config/bev_obb_config.yaml',
        help='Path to configuration YAML file (default: config/bev_obb_config.yaml)'
    )
    
    parser.add_argument(
        '--config',
        dest='config_alt',
        help='Alternative way to specify config file'
    )
    
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--auto-start',
        action='store_true',
        help='Automatically start detection without waiting for button press'
    )
    
    parser.add_argument(
        '--model-path',
        help='Override model path from config'
    )
    
    parser.add_argument(
        '--camera-res',
        nargs=2,
        type=int,
        metavar=('WIDTH', 'HEIGHT'),
        help='Override camera resolution (e.g., --camera-res 1640 1232)'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=float,
        help='Override confidence threshold from config'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='AIY Vision Kit BEV-OBB Detection System v1.0.0'
    )
    
    return parser.parse_args()


def setup_logging(level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('detection_system.log', mode='a')
        ]
    )


def validate_config_file(config_path: str) -> str:
    """Validate configuration file exists and return absolute path."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        config_file = script_dir / config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return str(config_file.absolute())


def apply_cli_overrides(config_path: str, args) -> str:
    """Apply command line overrides to configuration."""
    import yaml
    import tempfile
    
    # Load original config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.output_dir:
        config['logging']['output_dir'] = args.output_dir
    
    if args.model_path:
        config['model_path'] = args.model_path
    
    if args.camera_res:
        config['aiy']['camera_resolution'] = args.camera_res
    
    if args.conf_threshold:
        config['detection']['conf_threshold'] = args.conf_threshold
    
    if args.auto_start:
        config['auto_start'] = True
    
    # Save modified config to temporary file
    temp_config = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False
    )
    
    yaml.dump(config, temp_config, default_flow_style=False)
    temp_config.flush()
    
    return temp_config.name


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.log_level)
        logger = logging.getLogger(__name__)
        
        # Determine config file
        config_path = args.config_alt if args.config_alt else args.config
        
        # Validate config file exists
        config_path = validate_config_file(config_path)
        logger.info(f"Using configuration file: {config_path}")
        
        # Apply CLI overrides if any
        if any([args.output_dir, args.model_path, args.camera_res, 
                args.conf_threshold, args.auto_start]):
            config_path = apply_cli_overrides(config_path, args)
            logger.info("Applied command line configuration overrides")
        
        # Initialize and run detector
        logger.info("Initializing BEV-OBB Detection System...")
        detector = BEVOBBDetector(config_path)
        
        logger.info("Starting detection system...")
        logger.info("Press Ctrl+C to stop")
        
        # Run the detection system
        detector.run()
        
    except KeyboardInterrupt:
        logger.info("Detection system stopped by user")
        sys.exit(0)
        
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure all dependencies are installed:")
        logger.error("  source ~/aiy_env/bin/activate")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()