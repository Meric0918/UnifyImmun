"""
Test script for fine-tuned pTCR model
Run after fine-tuning to evaluate model on test set
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_auc_score, auc, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

from finetune_liver import (
    FineTuningModel, device, data_process_liver, LiverDataSet,
    vocab, batch_size, threshold, performance, performance_pd, transfer
)
import torch.utils.data as Data
import os

def test_model(model_path, test_data_path='../data/data_liver/test.csv'):
    """
    Test a fine-tuned model on test set.

    Args:
        model_path: Path to the fine-tuned model (.pkl file)
        test_data_path: Path to test data CSV
    """
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    print(f"Loading test data from: {test_data_path}")

    # Load test data
    test_data = pd.read_csv(test_data_path).dropna()
    print(f"Test samples: {len(test_data)}")
    print(f"Label distribution: {Counter(test_data.label)}")

    # Process data
    pep_inputs, tcr_inputs, labels = data_process_liver(test_data)
    test_loader = Data.DataLoader(
        LiverDataSet(pep_inputs, tcr_inputs, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # Load model
    model = FineTuningModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")

    # Test
    print("\nTesting...")
    with torch.no_grad():
        y_true_list, y_pred_list = [], []
        for pep, tcr, label in tqdm(test_loader, colour='green'):
            pep, tcr, label = pep.to(device), tcr.to(device), label.to(device)
            outputs, _ = model(pep, tcr)
            pred = nn.Softmax(dim=1)(outputs)[:, 1].cpu().detach().numpy()
            y_true_list.extend(label.cpu().numpy())
            y_pred_list.extend(pred)

    # Evaluate
    y_pred_transfer = transfer(y_pred_list, threshold)
    test_performance = performance(y_true_list, y_pred_list, y_pred_transfer)

    return test_performance


def test_all_folds(model_dir='../trained_model/finetune_liver',
                   test_data_path='../data/data_liver/test.csv',
                   n_folds=5):
    """
    Test all fold models and average results.

    Args:
        model_dir: Directory containing fine-tuned models
        test_data_path: Path to test data CSV
        n_folds: Number of fold models to test
    """
    print(f"\n{'='*60}")
    print(f"Testing All Folds")
    print(f"{'='*60}")

    all_performances = []

    for fold in range(1, n_folds + 1):
        model_path = os.path.join(model_dir, f'model_finetune_fold{fold}.pkl')
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found at {model_path}")
            continue

        print(f"\n--- Testing Fold {fold} ---")
        performance = test_model(model_path, test_data_path)
        all_performances.append(performance)

    # Summary
    print(f"\n{'='*60}")
    print(f"All Folds Test Results Summary")
    print(f"{'='*60}")
    print(performance_pd(all_performances).to_string())

    return all_performances


def predict_single(model_path, peptide, tcr):
    """
    Predict binding probability for a single peptide-TCR pair.

    Args:
        model_path: Path to the fine-tuned model
        peptide: Peptide sequence string
        tcr: TCR sequence string

    Returns:
        probability: Binding probability (0-1)
    """
    model = FineTuningModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Process sequences
    tcr_max_len = 34
    peptide = peptide.ljust(tcr_max_len, '-')
    tcr = tcr.ljust(tcr_max_len, '-')

    pep_input = torch.LongTensor([[vocab[n] for n in peptide]]).to(device)
    tcr_input = torch.LongTensor([[vocab[n] for n in tcr]]).to(device)

    with torch.no_grad():
        output, _ = model(pep_input, tcr_input)
        prob = nn.Softmax(dim=1)(output)[0, 1].cpu().item()

    return prob


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # Test specific model
        model_path = sys.argv[1]
        test_performance = test_model(model_path)
    else:
        # Test all folds
        all_performances = test_all_folds(
            model_dir='../trained_model/finetune_liver',
            test_data_path='../data/data_liver/test.csv',
            n_folds=5
        )