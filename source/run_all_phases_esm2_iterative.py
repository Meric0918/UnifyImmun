"""
Run all training phases iteratively with ESM2 embeddings.
Supports multiple training rounds with encoder_P transfer between rounds.
Includes early stopping based on test set performance (LiverCancer TCR_Peptide.csv).

Early stopping condition: Stop when test performance doesn't improve significantly
for 5 consecutive rounds (improvement < min_delta).

Usage:
    cd /home/mclab/mjp/unifyimmun/source

    # Train from Round 3 with early stopping (max 20 rounds):
    python run_all_phases_esm2_iterative.py --start_round 3 --max_rounds 20 --patience 5 --min_delta 0.001

    # Without early stopping (fixed rounds):
    python run_all_phases_esm2_iterative.py --start_round 3 --end_round 5 --no_early_stop
"""

import subprocess
import os
import argparse
import re
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, matthews_corrcoef, f1_score,
    precision_recall_curve, auc, confusion_matrix
)
from tqdm import tqdm


_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

# Test data path
TEST_DATA_PATH = os.path.join(_project_root, "data", "LiverCancer", "TCR_Peptide.csv")


def create_hla_script(round_num, load_from_dir, load_encoder_name, save_encoder_name, save_to_dir):
    """Create HLA training script for a specific round."""
    script_path = os.path.join(_current_dir, f"temp_HLA_ESM2_{round_num}.py")

    content = f'''"""
HLA_ESM2 Round {round_num}
Loads encoder_P from {load_from_dir}/{load_encoder_name}
Saves encoder_P to {save_to_dir}/{save_encoder_name}
"""
import re

original = open("HLA_ESM2_2.py").read()

# Replace encoder loading path
modified = re.sub(
    r'"TCR_ESM2", "encoder_P_ESM2\\.pth"',
    '"{load_from_dir}", "{load_encoder_name}"',
    original
)

# Replace encoder save filename
modified = re.sub(
    r'"encoder_P_ESM2_2\\.pth"',
    '"{save_encoder_name}"',
    modified
)

# Replace save directory
modified = re.sub(
    r'"HLA_ESM2_2"',
    '"{save_to_dir}"',
    modified
)

# Replace experiment name
modified = re.sub(
    r'experiment_name="HLA_ESM2[^"]*"',
    'experiment_name="HLA_ESM2_{round_num}"',
    modified
)

exec(modified)
'''
    with open(script_path, "w") as f:
        f.write(content)
    return script_path


def create_tcr_script(round_num, load_from_dir, load_encoder_name, save_encoder_name, save_to_dir):
    """Create TCR training script for a specific round."""
    script_path = os.path.join(_current_dir, f"temp_TCR_ESM2_{round_num}.py")

    content = f'''"""
TCR_ESM2 Round {round_num}
Loads encoder_P from {load_from_dir}/{load_encoder_name}
Saves encoder_P to {save_to_dir}/{save_encoder_name}
"""
import re

original = open("TCR_ESM2_2.py").read()

# Replace encoder loading path
modified = re.sub(
    r'"HLA_ESM2_2", "encoder_P_ESM2_2\\.pth"',
    '"{load_from_dir}", "{load_encoder_name}"',
    original
)

# Replace encoder save filename
modified = re.sub(
    r'"encoder_P_ESM2_2\\.pth"',
    '"{save_encoder_name}"',
    modified
)

# Replace save directory
modified = re.sub(
    r'"TCR_ESM2_2"',
    '"{save_to_dir}"',
    modified
)

# Replace experiment name
modified = re.sub(
    r'experiment_name="TCR_ESM2[^"]*"',
    'experiment_name="TCR_ESM2_{round_num}"',
    modified
)

exec(modified)
'''
    with open(script_path, "w") as f:
        f.write(content)
    return script_path


def run_training(script_path, phase_name):
    """Run a training script."""
    print("\n" + "=" * 60)
    print(f"Starting {phase_name}...")
    print("=" * 60)

    result = subprocess.run(
        ["python", script_path],
        cwd=_current_dir,
    )

    print(f"\n{phase_name} finished")
    print("-" * 60)

    return result.returncode == 0


def evaluate_on_testset(model_path, test_data_path, batch_size=64):
    """
    Evaluate TCR model on test dataset.

    Args:
        model_path: Path to the saved model (.pkl file)
        test_data_path: Path to test CSV file (peptide,tcr,label)
        batch_size: Batch size for evaluation

    Returns:
        dict with performance metrics
    """
    # Import model and data loading functions
    import sys
    sys.path.insert(0, _current_dir)

    from models.esm2_embedding import (
        Mymodel_TCR_ESM2,
        ESM2TokenizerWrapper,
        device,
        pep_max_len,
        tcr_max_len,
        esm2_max_len,
    )
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    # Load test data
    test_df = pd.read_csv(test_data_path).dropna()
    print(f"Test dataset: {len(test_df)} samples")

    # Tokenizer
    tokenizer = ESM2TokenizerWrapper()

    # Process data
    pep_sequences = []
    tcr_sequences = []
    labels = []

    for pep, tcr, label in zip(test_df.peptide, test_df.tcr, test_df.label):
        pep_truncated = pep[:pep_max_len] if len(pep) > pep_max_len else pep
        tcr_truncated = tcr[:esm2_max_len] if len(tcr) > esm2_max_len else tcr
        pep_sequences.append(pep_truncated)
        tcr_sequences.append(tcr_truncated)
        labels.append(label)

    # Encode sequences
    pep_encoded = tokenizer.encode_sequences(pep_sequences, max_length=pep_max_len)
    tcr_encoded = tokenizer.encode_sequences(tcr_sequences, max_length=esm2_max_len)

    # Create dataset
    class TestDataset(Dataset):
        def __init__(self, pep_ids, tcr_ids, labels, pep_masks, tcr_masks):
            self.pep_ids = pep_ids
            self.tcr_ids = tcr_ids
            self.labels = labels
            self.pep_masks = pep_masks
            self.tcr_masks = tcr_masks

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return (
                self.pep_ids[idx],
                self.tcr_ids[idx],
                self.labels[idx],
                self.pep_masks[idx],
                self.tcr_masks[idx],
            )

    test_dataset = TestDataset(
        pep_encoded["input_ids"],
        tcr_encoded["input_ids"],
        torch.LongTensor(labels),
        pep_encoded["attention_mask"],
        tcr_encoded["attention_mask"],
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = Mymodel_TCR_ESM2(freeze_esm2=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluate
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for pep_inputs, tcr_inputs, labels_batch, pep_masks, tcr_masks in tqdm(test_loader, desc="Evaluating"):
            pep_inputs = pep_inputs.to(device)
            tcr_inputs = tcr_inputs.to(device)
            labels_batch = labels_batch.to(device)
            pep_masks = pep_masks.to(device)
            tcr_masks = tcr_masks.to(device)

            outputs, _ = model(
                pep_inputs, tcr_inputs,
                pep_attention_mask=pep_masks,
                tcr_attention_mask=tcr_masks
            )

            y_true = labels_batch.cpu().numpy()
            y_pred = nn.Softmax(dim=1)(outputs)[:, 1].cpu().detach().numpy()
            y_true_list.extend(y_true)
            y_pred_list.extend(y_pred)

    # Calculate metrics
    y_pred_transfer = np.array([[0, 1][x > 0.5] for x in y_pred_list])

    roc_auc = roc_auc_score(y_true_list, y_pred_list)
    accuracy = accuracy_score(y_true_list, y_pred_transfer)
    mcc = matthews_corrcoef(y_true_list, y_pred_transfer)
    f1 = f1_score(y_true_list, y_pred_transfer)
    prec, reca, _ = precision_recall_curve(y_true_list, y_pred_list)
    aupr = auc(reca, prec)

    tn, fp, fn, tp = confusion_matrix(y_true_list, y_pred_transfer, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Performance average (same as training: first 5 metrics)
    performance_avg = (roc_auc + accuracy + mcc + f1 + aupr) / 5

    metrics = {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "mcc": mcc,
        "f1": f1,
        "aupr": aupr,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "performance_avg": performance_avg,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Iterative ESM2 Training with Early Stopping")
    parser.add_argument("--start_round", type=int, default=3,
                        help="Starting round number (default: 3)")
    parser.add_argument("--end_round", type=int, default=None,
                        help="Fixed end round (if specified, no early stopping)")
    parser.add_argument("--max_rounds", type=int, default=20,
                        help="Maximum rounds when using early stopping (default: 20)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="Minimum improvement threshold (default: 0.001)")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Disable early stopping (use fixed end_round)")
    parser.add_argument("--test_data", type=str, default=TEST_DATA_PATH,
                        help="Test dataset path for early stopping evaluation")
    args = parser.parse_args()

    # Determine training mode
    if args.no_early_stop or args.end_round is not None:
        early_stop_enabled = False
        end_round = args.end_round if args.end_round else args.start_round + 2
    else:
        early_stop_enabled = True
        end_round = args.start_round + args.max_rounds - 1

    print("\n" + "#" * 60)
    print("# Iterative ESM2 Training Pipeline with Early Stopping")
    print("#" * 60)
    print(f"Starting from Round {args.start_round}")
    print(f"Maximum rounds: {args.max_rounds if early_stop_enabled else 'fixed to ' + str(end_round)}")
    print(f"Early stopping: {'Enabled' if early_stop_enabled else 'Disabled'}")
    if early_stop_enabled:
        print(f"  Patience: {args.patience} rounds")
        print(f"  Min delta: {args.min_delta}")
        print(f"  Test dataset: {args.test_data}")

    # Check test data exists
    if not os.path.exists(args.test_data):
        print(f"Error: Test dataset not found: {args.test_data}")
        return

    # Create directories
    for r in range(args.start_round, end_round + 1):
        for dir_name in [f"HLA_ESM2_{r}", f"TCR_ESM2_{r}"]:
            dir_path = os.path.join(_project_root, "trained_model", dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created: {dir_path}")

    # encoder_P naming convention
    def get_encoder_name(round_num):
        if round_num == 1:
            return "encoder_P_ESM2.pth"
        else:
            return f"encoder_P_ESM2_{round_num}.pth"

    # Early stopping tracking
    performance_best = 0
    no_improve_count = 0
    performance_history = []

    # Run training for each round
    for r in range(args.start_round, end_round + 1):
        print(f"\n{'#' * 60}")
        print(f"# Round {r}")
        print(f"{'#' * 60}")

        # Determine paths
        prev_round = r - 1
        prev_tcr_dir = f"TCR_ESM2_{prev_round}" if prev_round >= 2 else "TCR_ESM2_2"
        prev_encoder_name = get_encoder_name(prev_round)
        current_encoder_name = get_encoder_name(r)
        hla_dir = f"HLA_ESM2_{r}"
        tcr_dir = f"TCR_ESM2_{r}"

        print(f"encoder_P transfer:")
        print(f"  HLA: {prev_tcr_dir}/{prev_encoder_name} → {hla_dir}/{current_encoder_name}")
        print(f"  TCR: {hla_dir}/{current_encoder_name} → {tcr_dir}/{current_encoder_name}")

        # HLA training
        hla_script = create_hla_script(
            r, prev_tcr_dir, prev_encoder_name, current_encoder_name, hla_dir
        )
        success = run_training(hla_script, f"HLA_ESM2_{r}")
        os.remove(hla_script)

        if not success:
            print(f"Error in HLA_ESM2_{r}, stopping pipeline")
            break

        # TCR training
        tcr_script = create_tcr_script(
            r, hla_dir, current_encoder_name, current_encoder_name, tcr_dir
        )
        success = run_training(tcr_script, f"TCR_ESM2_{r}")
        os.remove(tcr_script)

        if not success:
            print(f"Error in TCR_ESM2_{r}, stopping pipeline")
            break

        # Evaluate on test set (for early stopping decision)
        if early_stop_enabled:
            print(f"\n{'=' * 60}")
            print(f"Evaluating TCR_ESM2_{r} on test dataset...")
            print(f"{'=' * 60}")

            model_path = os.path.join(_project_root, "trained_model", tcr_dir, f"model_TCR_ESM2_{r}.pkl")

            if os.path.exists(model_path):
                metrics = evaluate_on_testset(model_path, args.test_data)

                performance_current = metrics["performance_avg"]
                performance_history.append({
                    "round": r,
                    "performance_avg": performance_current,
                    "roc_auc": metrics["roc_auc"],
                    "accuracy": metrics["accuracy"],
                    "mcc": metrics["mcc"],
                    "f1": metrics["f1"],
                    "aupr": metrics["aupr"],
                })

                print(f"\nTest set performance (Round {r}):")
                print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
                print(f"  Accuracy:   {metrics['accuracy']:.4f}")
                print(f"  MCC:        {metrics['mcc']:.4f}")
                print(f"  F1:         {metrics['f1']:.4f}")
                print(f"  AUPR:       {metrics['aupr']:.4f}")
                print(f"  Avg (5):    {performance_current:.4f}")
                print(f"  Best so far: {performance_best:.4f}")

                # Check improvement
                if performance_current > performance_best + args.min_delta:
                    performance_best = performance_current
                    no_improve_count = 0
                    print(f"  ✓ Improved! New best: {performance_best:.4f}")
                else:
                    no_improve_count += 1
                    improvement = performance_current - performance_best
                    print(f"  ✗ No significant improvement ({improvement:+.4f})")
                    print(f"  No improve count: {no_improve_count}/{args.patience}")

                # Early stopping check
                if no_improve_count >= args.patience:
                    print(f"\n{'!' * 60}")
                    print(f"! Early stopping triggered!")
                    print(f"! No improvement for {args.patience} consecutive rounds")
                    print(f"! Best performance: {performance_best:.4f}")
                    print(f"{'!' * 60}")
                    break
            else:
                print(f"Warning: Model not found at {model_path}, skipping evaluation")

    # Print summary
    print("\n" + "#" * 60)
    print("# Training Summary")
    print("#" * 60)

    if performance_history:
        print("\nPerformance history on test set:")
        print("-" * 50)
        for entry in performance_history:
            print(f"  Round {entry['round']}: Avg={entry['performance_avg']:.4f}, "
                  f"AUC={entry['roc_auc']:.4f}, Acc={entry['accuracy']:.4f}")

        best_entry = max(performance_history, key=lambda x: x['performance_avg'])
        print("-" * 50)
        print(f"Best round: {best_entry['round']} (performance_avg={best_entry['performance_avg']:.4f})")

    print("\nSaved models:")
    for r in range(args.start_round, r + 1):
        for model_type in ["HLA", "TCR"]:
            dir_path = os.path.join(_project_root, "trained_model", f"{model_type}_ESM2_{r}")
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                print(f"  {model_type}_ESM2_{r}: {files}")


if __name__ == "__main__":
    main()