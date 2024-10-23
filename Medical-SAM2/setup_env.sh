#!/bin/bash

cd /data3/zc/code/medical-sca/Medical-SAM2

eval "$(conda shell.bash hook)"

# Create the environment from the YAML file
conda env create -f environment.yml

# Activate the environment
conda activate medsam2

pip uninstall torch torchvision torchaudio -y

# Install specific PyTorch and related packages with CUDA support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install transformers library
pip install transformers

# Install flash-attn with no build isolation
pip install flash-attn --no-build-isolation
