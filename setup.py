#!/usr/bin/env python3
"""
Setup script for AIY Vision Kit BEV-OBB Detection System

This script installs the Birds-Eye View Oriented Bounding Box detection
system optimized for Google AIY Vision Kit and Raspberry Pi Zero W.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="aiy-bev-obb-detector",
    version="1.0.0",
    author="AIY Vision Kit BEV-OBB Team",
    author_email="your.email@example.com",
    description="Birds-Eye View Oriented Bounding Box Detection for AIY Vision Kit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aiy-vision-kit-bev-obb",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "mypy>=0.800",
        ],
        "visualization": [
            "plotly>=5.0",
            "dash>=2.0",
            "streamlit>=1.0",
        ],
        "optimization": [
            "onnx>=1.8",
            "onnxruntime>=1.7",
            "tensorrt>=7.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "bev-obb-detect=src.bev_obb_detector:main",
            "bev-obb-optimize=src.model_optimizer:main",
            "bev-obb-train=src.train:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "models/*.pt", "data/samples/*"],
    },
    zip_safe=False,
    keywords=[
        "computer vision", "object detection", "oriented bounding box",
        "raspberry pi", "aiy vision kit", "birds eye view", "yolov5",
        "edge computing", "real-time detection"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/aiy-vision-kit-bev-obb/issues",
        "Source": "https://github.com/yourusername/aiy-vision-kit-bev-obb",
        "Documentation": "https://aiy-vision-kit-bev-obb.readthedocs.io/",
    },
)