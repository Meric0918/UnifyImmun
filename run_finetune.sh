#!/bin/bash
# Run fine-tuning script for liver data
# Usage: ./run_finetune.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unifyimmun

# Navigate to source directory
cd /home/mjp/Project/fine-tuning/UnifyImmun/source

# Run fine-tuning
echo "Starting fine-tuning process..."
python3 finetune_liver.py

echo "Fine-tuning completed!"