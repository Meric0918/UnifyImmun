# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

UnifyImmun is a unified cross-attention model for predicting antigen binding specificity to both HLA and TCR molecules. It uses a two-phase progressive training strategy with FGM adversarial training for robustness.

## Commands

### Training
```bash
cd source
# ESM2-based training (recommended)
python run_all_phases_esm2.py

# Original training (non-ESM2)
python run_all_phases.py
```

### Testing
```bash
cd source
# HLA binding prediction
python HLA_test.py

# TCR binding prediction
python TCR_test.py
```

### Output Scores
```bash
cd source
python HLA_output_score.py
python TCR_output_score.py
```

## Architecture

### Two-Phase Progressive Training Flow

**Phase 1:**
1. `HLA_ESM2.py` → trains HLA-peptide binding model, saves `encoder_P`
2. `TCR_ESM2.py` → loads `encoder_P` from HLA_ESM2, trains TCR-peptide model, saves `encoder_P`

**Phase 2:**
1. `HLA_ESM2_2.py` → loads `encoder_P` from TCR_ESM2, trains HLA model, saves `encoder_P`
2. `TCR_ESM2_2.py` → loads `encoder_P` from HLA_ESM2_2, trains TCR model, saves `encoder_P`

The peptide encoder (`encoder_P`) is transferred across phases, enabling knowledge sharing between HLA and TCR tasks.

### Model Components

Each model consists of:
- **ESM2 Embedding** (frozen): 650M parameter protein language model, outputs 1280-dim embeddings
- **Projection Layer** (trainable): 1280 → 64 dimensions
- **Encoder**: Transformer encoder with 1 layer, 1 head
- **Cross-Attention**: Peptide ↔ HLA/TCR interaction mechanism
- **Classification Head**: Linear layers (34*64 → 256 → 64 → 2)

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 64 |
| n_heads | 1 |
| n_layers | 1 |
| d_ff | 512 |
| batch_size | 64 (ESM2), 1024 (original) |
| epochs | 30 |
| learning_rate | 1e-4 |
| threshold | 0.5 |
| FGM epsilon | 1.0 |
| early_stopping patience | 5 |
| early_stopping min_delta | 0.001 |

### Sequence Lengths

| Sequence | Max Length |
|----------|------------|
| peptide | 15 |
| HLA | 34 |
| TCR | 34 |

## Data Format

Input CSV files require three columns:
- **HLA data**: `peptide`, `HLA`, `label`
- **TCR data**: `peptide`, `tcr`, `label`

Labels: 0 (non-binding), 1 (binding)

## Directory Structure

```
data/
  data_HLA/     # HLA-peptide binding datasets
  data_TCR/     # TCR-peptide binding datasets
source/
  models/       # Model definitions (HLA.py, TCR.py, esm2_embedding.py)
  HLA_ESM2.py   # Phase 1 HLA training
  TCR_ESM2.py   # Phase 1 TCR training
  HLA_ESM2_2.py # Phase 2 HLA training
  TCR_ESM2_2.py # Phase 2 TCR training
trained_model/  # Saved model weights
```

## External Dependencies

- ESM2 model: `/home/mclab/mjp/model/esm2_t33_650M_UR50D`
- Requires transformers library for ESM2

## FGM Adversarial Training

FGM (Fast Gradient Method) is applied during training:
- Only attacks trainable projection layer parameters (ESM2 is frozen)
- epsilon = 1.0
- BatchNorm `track_running_stats` is frozen during attack to prevent running stats corruption

## Early Stopping

Monitors validation performance average (roc_auc, accuracy, mcc, f1, aupr). Triggers when improvement < 0.001 for 5 consecutive epochs.