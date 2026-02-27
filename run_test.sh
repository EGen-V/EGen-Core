#!/bin/bash
set -x

# Source conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
conda activate egen-core

# Install requirements
pip install torch transformers accelerate sentencepiece protobuf

# Run the test script
PYTHONUNBUFFERED=1 python test_inference_sa1q9.py
