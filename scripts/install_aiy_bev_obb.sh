#!/bin/bash
# AIY Vision Kit BEV-OBB Detection System Installation Script
# Optimized for Raspberry Pi Zero W with AIY Vision Kit
#
# This script sets up the complete environment for running Birds-Eye View
# Oriented Bounding Box detection on the Google AIY Vision Kit.
#
# Usage: bash install_aiy_bev_obb.sh
#
# References:
#   - AIY Projects: https://aiyprojects.withgoogle.com/vision/
#   - YOLOv5: https://github.com/ultralytics/yolov5
#   - PyTorch ARM: https://pytorch.org/

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Raspberry Pi
check_raspberry_pi() {
    log_info "Checking if running on Raspberry Pi..."
    if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
        log_error "This script is designed for Raspberry Pi. Detected system:"
        cat /proc/cpuinfo | grep "model name" | head -1
        exit 1
    fi
    
    # Check for Pi Zero specifically
    if grep -q "Pi Zero" /proc/cpuinfo; then
        log_success "Raspberry Pi Zero detected"
        export PI_ZERO=true
    else
        log_warning "Non-Pi Zero detected. Some optimizations may not apply."
        export PI_ZERO=false
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: $python_version"
    
    if ! python3 -c "import sys; assert sys.version_info >= (3, 7)"; then
        log_error "Python 3.7 or higher required. Found: $python_version"
        exit 1
    fi
    
    # Check available memory
    memory_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    memory_mb=$((memory_kb / 1024))
    log_info "Available memory: ${memory_mb}MB"
    
    if [ $memory_mb -lt 400 ]; then
        log_warning "Low memory detected. Consider increasing swap space."
    fi
    
    # Check available disk space
    disk_space=$(df -h / | awk 'NR==2 {print $4}')
    log_info "Available disk space: $disk_space"
}

# Update system packages
update_system() {
    log_info "Updating system packages..."
    sudo apt-get update -y
    sudo apt-get upgrade -y
    
    # Install essential packages
    sudo apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        libjpeg-dev \
        libtiff5-dev \
        libpng-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libfontconfig1-dev \
        libcairo2-dev \
        libgdk-pixbuf2.0-dev \
        libpango1.0-dev \
        libgtk2.0-dev \
        libgtk-3-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-dev \
        libhdf5-serial-dev \
        libhdf5-103 \
        python3-pyqt5 \
        python3-dev \
        python3-pip \
        python3-venv \
        git \
        wget \
        curl \
        unzip
    
    log_success "System packages updated"
}

# Setup swap space for compilation
setup_swap() {
    if [ "$PI_ZERO" = true ]; then
        log_info "Setting up additional swap space for Pi Zero..."
        
        # Check current swap
        current_swap=$(free -m | grep Swap | awk '{print $2}')
        log_info "Current swap: ${current_swap}MB"
        
        if [ $current_swap -lt 1000 ]; then
            log_info "Creating additional swap file..."
            sudo dd if=/dev/zero of=/swapfile bs=1M count=1024
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
            sudo swapon /swapfile
            
            # Make permanent
            echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
            
            log_success "Swap space configured"
        fi
    fi
}

# Install AIY Python packages
install_aiy_packages() {
    log_info "Installing AIY Python packages..."
    
    # Create virtual environment
    python3 -m venv ~/aiy_env
    source ~/aiy_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install AIY packages
    pip install aiy-python-wheels
    pip install aiy-vision-kit
    
    # Install additional AIY dependencies
    pip install \
        gpiozero \
        picamera \
        RPi.GPIO
    
    log_success "AIY packages installed"
}

# Install PyTorch for ARM
install_pytorch_arm() {
    log_info "Installing PyTorch for ARM architecture..."
    
    source ~/aiy_env/bin/activate
    
    if [ "$PI_ZERO" = true ]; then
        # Use pre-compiled wheel for Pi Zero (ARMv6)
        log_info "Installing PyTorch for ARMv6 (Pi Zero)..."
        
        # Download pre-compiled PyTorch wheel
        wget -O torch-1.9.0-cp37-cp37m-linux_armv6l.whl \
            https://github.com/KumaTea/pytorch-aarch64/releases/download/v1.9.0/torch-1.9.0-cp37-cp37m-linux_armv6l.whl
        
        pip install torch-1.9.0-cp37-cp37m-linux_armv6l.whl
        rm torch-1.9.0-cp37-cp37m-linux_armv6l.whl
        
        # Install torchvision from source (lighter version)
        pip install torchvision==0.10.0 --no-deps
        
    else
        # Standard ARM installation for Pi 3/4
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    fi
    
    log_success "PyTorch installed"
}

# Install OpenCV optimized for Pi
install_opencv() {
    log_info "Installing OpenCV..."
    
    source ~/aiy_env/bin/activate
    
    if [ "$PI_ZERO" = true ]; then
        # Use pre-compiled OpenCV for Pi Zero
        pip install opencv-python-headless==4.5.3.56
    else
        # Full OpenCV installation
        pip install opencv-python
    fi
    
    log_success "OpenCV installed"
}

# Install project dependencies
install_dependencies() {
    log_info "Installing project dependencies..."
    
    source ~/aiy_env/bin/activate
    
    # Install from requirements.txt (excluding heavy packages already installed)
    pip install \
        numpy \
        scipy \
        matplotlib \
        seaborn \
        pandas \
        PyYAML \
        requests \
        tqdm \
        Pillow \
        shapely \
        psutil \
        sqlite3
    
    log_success "Project dependencies installed"
}

# Download and setup YOLOv5
setup_yolov5() {
    log_info "Setting up YOLOv5..."
    
    cd ~
    if [ ! -d "yolov5" ]; then
        git clone https://github.com/ultralytics/yolov5.git
    fi
    
    cd yolov5
    source ~/aiy_env/bin/activate
    
    # Install YOLOv5 requirements (minimal for Pi)
    pip install -r requirements.txt --no-deps || true
    
    # Download pre-trained model
    python detect.py --weights yolov5s.pt --source data/images --exist-ok
    
    log_success "YOLOv5 setup completed"
}

# Setup project structure
setup_project() {
    log_info "Setting up project structure..."
    
    # Create project directory
    PROJECT_DIR=~/AIY-Vision-kit-BEV-OBB
    mkdir -p $PROJECT_DIR
    cd $PROJECT_DIR
    
    # Create directory structure
    mkdir -p {src,config,models,data/{raw,processed,annotations},logs,scripts,tests,docs}
    
    # Copy configuration files
    if [ -f "config/bev_obb_config.yaml" ]; then
        log_info "Configuration files found"
    else
        log_warning "Configuration files not found. Please copy them manually."
    fi
    
    log_success "Project structure created"
}

# Configure system services
configure_services() {
    log_info "Configuring system services..."
    
    # Enable camera
    sudo raspi-config nonint do_camera 0
    
    # Enable I2C for AIY bonnet
    sudo raspi-config nonint do_i2c 0
    
    # Enable SPI for AIY bonnet
    sudo raspi-config nonint do_spi 0
    
    # Configure GPU memory split (minimal for Pi Zero)
    if [ "$PI_ZERO" = true ]; then
        sudo raspi-config nonint do_memory_split 16
    else
        sudo raspi-config nonint do_memory_split 64
    fi
    
    log_success "System services configured"
}

# Optimize system for performance
optimize_system() {
    log_info "Optimizing system performance..."
    
    # CPU governor for performance
    echo 'performance' | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
    
    # Increase USB current limit
    echo 'max_usb_current=1' | sudo tee -a /boot/config.txt
    
    # Optimize memory
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
    
    if [ "$PI_ZERO" = true ]; then
        # Pi Zero specific optimizations
        echo 'arm_freq=1000' | sudo tee -a /boot/config.txt
        echo 'core_freq=500' | sudo tee -a /boot/config.txt
        echo 'sdram_freq=500' | sudo tee -a /boot/config.txt
        echo 'over_voltage=2' | sudo tee -a /boot/config.txt
    fi
    
    log_success "System optimization completed"
}

# Setup startup script
setup_startup() {
    log_info "Setting up startup script..."
    
    # Create systemd service
    sudo tee /etc/systemd/system/bev-obb-detector.service > /dev/null << EOF
[Unit]
Description=BEV-OBB Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/AIY-Vision-kit-BEV-OBB
Environment=PATH=/home/pi/aiy_env/bin
ExecStart=/home/pi/aiy_env/bin/python src/bev_obb_detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    log_info "To enable auto-start, run: sudo systemctl enable bev-obb-detector"
    log_success "Startup script configured"
}

# Run tests
run_tests() {
    log_info "Running basic tests..."
    
    source ~/aiy_env/bin/activate
    
    # Test Python imports
    python3 -c "
import torch
import cv2
import numpy as np
import yaml
print('All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'OpenCV version: {cv2.__version__}')
print(f'NumPy version: {np.__version__}')
"
    
    # Test camera (if available)
    if [ -e "/dev/video0" ]; then
        log_info "Camera device detected"
    else
        log_warning "No camera device found. Please connect camera module."
    fi
    
    # Test AIY bonnet (if available)
    if [ -e "/dev/vision_spicomm" ]; then
        log_info "AIY Vision bonnet detected"
    else
        log_warning "AIY Vision bonnet not detected. Please check connection."
    fi
    
    log_success "Basic tests completed"
}

# Main installation function
main() {
    log_info "Starting AIY Vision Kit BEV-OBB Detection System installation..."
    log_info "This may take 30-60 minutes on Raspberry Pi Zero W"
    
    check_raspberry_pi
    check_requirements
    update_system
    setup_swap
    install_aiy_packages
    install_pytorch_arm
    install_opencv
    install_dependencies
    setup_yolov5
    setup_project
    configure_services
    optimize_system
    setup_startup
    run_tests
    
    log_success "Installation completed successfully!"
    echo
    log_info "Next steps:"
    echo "1. Reboot the system: sudo reboot"
    echo "2. Activate environment: source ~/aiy_env/bin/activate"
    echo "3. Navigate to project: cd ~/AIY-Vision-kit-BEV-OBB"
    echo "4. Run detection: python src/bev_obb_detector.py config/bev_obb_config.yaml"
    echo "5. Enable auto-start: sudo systemctl enable bev-obb-detector"
    echo
    log_info "For troubleshooting, check logs in ~/AIY-Vision-kit-BEV-OBB/logs/"
}

# Run main function
main "$@"