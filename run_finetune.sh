#!/bin/bash
# Run fine-tuning script for pTCR model
# Usage: ./run_finetune.sh <data_path>
# Example: ./run_finetune.sh ../data/data_liver/TCR_Peptide_balanced.csv
# Note: Path should be relative to source/ directory, or use absolute path

# Check if data_path is provided
if [ -z "$1" ]; then
    echo "Error: data_path is required"
    echo "Usage: ./run_finetune.sh <data_path>"
    echo "Example: ./run_finetune.sh ../data/data_liver/TCR_Peptide_balanced.csv"
    exit 1
fi

DATA_PATH="$1"

# If path doesn't start with / or ../, prepend ../
if [[ "$DATA_PATH" != /* ]] && [[ "$DATA_PATH" != ../* ]]; then
    DATA_PATH="../$DATA_PATH"
    echo "Adjusted path to: $DATA_PATH (relative to source directory)"
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unifyimmun

# Navigate to source directory
cd /home/mjp/Project/fine-tuning/UnifyImmun/source

# Run fine-tuning with specified data_path
echo "Starting fine-tuning process..."
echo "Data path: $DATA_PATH"
python3 -c "
import sys
sys.path.insert(0, '.')
from finetune_liver import run_fine_tuning
run_fine_tuning(data_path='$DATA_PATH')
"

echo "Fine-tuning completed!"