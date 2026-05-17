"""
HLA binding model training Phase 2 with ESM2 embeddings.
Loads encoder_P from corresponding TCR_ESM2 fold and continues training.
ESM2 parameters are frozen, only projection layers are trained.
Uses swanlab for logging.
5-fold cross-validation.
"""

import time
import random
import warnings
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    auc,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import swanlab

from models.esm2_embedding import (
    Mymodel_HLA_ESM2,
    data_load_HLA_ESM2,
    ESM2TokenizerWrapper,
    transfer,
    batch_size,
    epochs,
    threshold,
    d_model,
    n_heads,
    n_layers,
    device,
    pep_max_len,
    hla_max_len,
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

warnings.filterwarnings("ignore")
seed = 66


def performance(y_true, y_pred, y_pred_transfer):
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred_transfer)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_transfer, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true=y_true, y_pred=y_pred_transfer, zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred_transfer, zero_division=0)
    f1 = f1_score(y_true=y_true, y_pred=y_pred_transfer, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred)
    prec, reca, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(reca, prec)
    mcc = matthews_corrcoef(y_true, y_pred_transfer)
    print("tn = {}, fp = {}, fn = {}, tp = {}".format(tn, fp, fn, tp))
    print(
        "y_pred: 0 = {} | 1 = {}".format(
            Counter(y_pred_transfer)[0], Counter(y_pred_transfer)[1]
        )
    )
    print("y_true: 0 = {} | 1 = {}".format(Counter(y_true)[0], Counter(y_true)[1]))
    print(
        "auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}".format(
            roc_auc, sensitivity, specificity, accuracy, mcc
        )
    )
    print(
        "precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}".format(
            precision, recall, f1, aupr
        )
    )
    return (
        roc_auc,
        accuracy,
        mcc,
        f1,
        aupr,
        sensitivity,
        specificity,
        precision,
        recall,
    )


f_mean = lambda l: sum(l) / len(l)


def performances_to_pd(performances_list):
    metrics_name = [
        "roc_auc",
        "accuracy",
        "mcc",
        "f1",
        "aupr",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
    ]
    performances_pd = pd.DataFrame(performances_list, columns=metrics_name)
    performances_pd.loc["mean"] = performances_pd.mean(axis=0)
    performances_pd.loc["std"] = performances_pd.std(axis=0)
    return performances_pd


class FGM_ESM2:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.bn_track_backup = {}

    def attack(self, epsilon=1.0):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                self.bn_track_backup[name] = module.track_running_stats
                module.track_running_stats = False

        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name:
                self.backup[name] = param.data.clone().detach()
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

        for name, module in self.model.named_modules():
            if name in self.bn_track_backup:
                module.track_running_stats = self.bn_track_backup[name]
        self.bn_track_backup = {}


def train_HLA(model, train_loader, fold, epoch, epochs, criterion, optimizer):
    train_time = 0
    model.train()
    model.encoder_H.src_emb.esm2.eval()
    model.encoder_P.src_emb.esm2.eval()

    y_true_list, y_pred_list, attention_list = [], [], []
    loss_list = []
    fgm = FGM_ESM2(model)

    for pep_inputs, hla_inputs, labels, pep_masks, hla_masks in tqdm(
        train_loader, colour="yellow", desc=f"Fold-{fold} Train Epoch-{epoch}"
    ):
        pep_inputs = pep_inputs.to(device)
        hla_inputs = hla_inputs.to(device)
        labels = labels.to(device)
        pep_masks = pep_masks.to(device)
        hla_masks = hla_masks.to(device)

        start = time.time()
        train_outputs, cross_attention = model(
            pep_inputs, hla_inputs,
            pep_attention_mask=pep_masks,
            hla_attention_mask=hla_masks
        )
        train_loss = criterion(train_outputs, labels)
        train_time += time.time() - start
        train_loss.backward()

        fgm.attack(epsilon=1.0)
        train_outputs2, _ = model(
            pep_inputs, hla_inputs,
            pep_attention_mask=pep_masks,
            hla_attention_mask=hla_masks
        )
        loss_sum = criterion(train_outputs2, labels)
        loss_sum.backward()
        fgm.restore()

        optimizer.step()
        optimizer.zero_grad()

        y_true = labels.cpu().numpy()
        y_pred = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()
        y_true_list.extend(y_true)
        y_pred_list.extend(y_pred)
        loss_list.append(train_loss)
        attention_list.append(cross_attention)

    y_pred_transfer_list = transfer(y_pred_list, threshold)
    result_train = (y_true_list, y_pred_list, y_pred_transfer_list)
    avg_loss = f_mean([l.item() for l in loss_list])
    print(
        "Fold-{} Train: Epoch:{}/{} Loss = {:.4f} Time = {:.4f} seconds".format(
            fold, epoch, epochs, avg_loss, train_time
        )
    )
    performance_train = performance(y_true_list, y_pred_list, y_pred_transfer_list)
    return result_train, performance_train, train_time, attention_list, avg_loss


def valid_HLA(model, val_loader, fold, epoch, epochs, criterion):
    model.eval()
    torch.manual_seed(66)
    torch.cuda.manual_seed(66)

    with torch.no_grad():
        y_true_val_list, y_pred_val_list, attention_val_list = [], [], []
        loss_val_list = []

        for pep_inputs, hla_inputs, labels, pep_masks, hla_masks in tqdm(
            val_loader, colour="blue", desc=f"Fold-{fold} Valid Epoch-{epoch}"
        ):
            pep_inputs = pep_inputs.to(device)
            hla_inputs = hla_inputs.to(device)
            labels = labels.to(device)
            pep_masks = pep_masks.to(device)
            hla_masks = hla_masks.to(device)

            val_outputs, cross_attention_val = model(
                pep_inputs, hla_inputs,
                pep_attention_mask=pep_masks,
                hla_attention_mask=hla_masks
            )
            val_loss = criterion(val_outputs, labels)

            y_true_val = labels.cpu().numpy()
            y_pred_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
            y_true_val_list.extend(y_true_val)
            y_pred_val_list.extend(y_pred_val)
            loss_val_list.append(val_loss)

        y_pred_transfer_val_list = transfer(y_pred_val_list, threshold)
        result_val = (y_true_val_list, y_pred_val_list, y_pred_transfer_val_list)
        avg_val_loss = f_mean([l.item() for l in loss_val_list])
        print(
            "Fold-{} Valid: Epoch:{}/{} Loss = {:.4f}".format(
                fold, epoch, epochs, avg_val_loss
            )
        )
        performance_val = performance(
            y_true_val_list, y_pred_val_list, y_pred_transfer_val_list
        )
    return result_val, performance_val, cross_attention_val, avg_val_loss


def train_single_fold(fold):
    """Train HLA model for a single fold, loading encoder_P from TCR_ESM2 fold."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Initialize fresh model for each fold
    model = Mymodel_HLA_ESM2(freeze_esm2=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )

    # Initialize swanlab for this fold
    swanlab.init(
        project="unifyimmun_5fold",
        experiment_name=f"HLA_ESM2_2_fold_{fold}",
        config={
            "model": "HLA_ESM2_2",
            "embedding": "ESM2_650M",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 1e-4,
            "threshold": threshold,
            "seed": seed,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "freeze_esm2": True,
            "fold": fold,
            "pretrained_encoder": f"TCR_ESM2_fold_{fold}",
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.001,
        },
    )

    # Load encoder_P from corresponding TCR_ESM2 fold
    encoder_path_load = os.path.join(
        _project_root, "trained_model", "TCR_ESM2", f"encoder_P_ESM2_fold_{fold}.pth"
    )
    if os.path.exists(encoder_path_load):
        model.encoder_P.load_state_dict(torch.load(encoder_path_load, map_location=device))
        print(f"Loaded encoder_P from TCR_ESM2 fold-{fold}: {encoder_path_load}")
    else:
        print(f"Warning: encoder_P not found at {encoder_path_load}, using fresh encoder")

    # Load test datasets
    independent_loader = data_load_HLA_ESM2(type_="independent", batch_size=batch_size)
    external_loader = data_load_HLA_ESM2(type_="external", batch_size=batch_size)

    print(f"\n{'='*60}")
    print(f"Fold-{fold} Training (Phase 2)")
    print(f"{'='*60}")
    print("Load HLA Data with ESM2 embedding:")

    train_loader = data_load_HLA_ESM2(type_="train", fold=fold, batch_size=batch_size)
    val_loader = data_load_HLA_ESM2(type_="val", fold=fold, batch_size=batch_size)

    train_data = pd.read_csv(
        os.path.join(_project_root, "data", "data_HLA", f"train_fold_{fold}.csv")
    )
    val_data = pd.read_csv(
        os.path.join(_project_root, "data", "data_HLA", f"val_fold_{fold}.csv")
    )
    print(
        "Label: Train = {} | Val = {}".format(
            Counter(train_data.label), Counter(val_data.label)
        )
    )

    # Paths for this fold
    path_all = os.path.join(_project_root, "trained_model", "HLA_ESM2_2")
    if not os.path.exists(path_all):
        os.makedirs(path_all)
    save_path = os.path.join(path_all, f"model_HLA_ESM2_2_fold_{fold}.pkl")
    encoder_save_path = os.path.join(path_all, f"encoder_P_ESM2_2_fold_{fold}.pth")

    epoch_best = -1
    time_train = 0

    # Early stopping parameters
    patience = 10
    min_delta = 0.001
    no_improve_count = 0
    val_performance_best = 0

    for epoch in range(1, epochs + 1):
        result_train, performance_train, train_time_ep, attention_score, train_loss = train_HLA(
            model, train_loader, fold, epoch, epochs, criterion, optimizer
        )
        result_val, performance_val, attention_score_val, val_loss = valid_HLA(
            model, val_loader, fold, epoch, epochs, criterion
        )
        train_performance_avg = sum(performance_train[:5]) / 5
        val_performance_avg = sum(performance_val[:5]) / 5

        swanlab.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_roc_auc": performance_train[0],
                "train_accuracy": performance_train[1],
                "train_mcc": performance_train[2],
                "train_f1": performance_train[3],
                "train_aupr": performance_train[4],
                "train_performance_avg": train_performance_avg,
                "val_loss": val_loss,
                "val_roc_auc": performance_val[0],
                "val_accuracy": performance_val[1],
                "val_mcc": performance_val[2],
                "val_f1": performance_val[3],
                "val_aupr": performance_val[4],
                "val_performance_avg": val_performance_avg,
                "no_improve_count": no_improve_count,
            }
        )

        val_improved = val_performance_avg > val_performance_best + min_delta

        if val_improved:
            val_performance_best, epoch_best = val_performance_avg, epoch
            print(
                "Save model: Best epoch = {} | Val_performance_avg = {:.4f}".format(
                    epoch_best, val_performance_best
                )
            )
            torch.save(model.eval().state_dict(), save_path)
            torch.save(model.eval().encoder_P.state_dict(), encoder_save_path)
            swanlab.log(
                {"best_epoch": epoch_best, "best_performance_avg": val_performance_best}
            )

        # Early stopping
        if val_improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs (patience={patience})")
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        scheduler.step(val_performance_avg)
        current_lr = optimizer.param_groups[0]['lr']
        swanlab.log({"learning_rate": current_lr})

        time_train += train_time_ep

    # Evaluation
    fold_results = {}
    if epoch_best >= 0:
        model.load_state_dict(torch.load(save_path, map_location=device))
        model_eval = model.eval()

        _, valid_performance, _, _ = valid_HLA(
            model_eval, val_loader, fold, epoch_best, epochs, criterion
        )
        _, independent_performance, _, _ = valid_HLA(
            model_eval, independent_loader, fold, epoch_best, epochs, criterion
        )
        _, external_performance, _, _ = valid_HLA(
            model_eval, external_loader, fold, epoch_best, epochs, criterion
        )

        fold_results = {
            "fold": fold,
            "epoch_best": epoch_best,
            "val": valid_performance[:5],
            "independent": independent_performance[:5],
            "external": external_performance[:5],
        }

        print(f"\nFold-{fold} Results:")
        print("  Val: roc_auc={:.4f}, acc={:.4f}, mcc={:.4f}, f1={:.4f}, aupr={:.4f}".format(*valid_performance[:5]))
        print("  Independent: roc_auc={:.4f}, acc={:.4f}, mcc={:.4f}, f1={:.4f}, aupr={:.4f}".format(*independent_performance[:5]))
        print("  External: roc_auc={:.4f}, acc={:.4f}, mcc={:.4f}, f1={:.4f}, aupr={:.4f}".format(*external_performance[:5]))

    swanlab.finish()
    return fold_results


def main():
    print("\n" + "#" * 60)
    print("# HLA_ESM2_2 5-Fold Cross-Validation Training (Phase 2)")
    print("#" * 60)
    print("# Loads encoder_P from corresponding TCR_ESM2 fold")

    # Create directory
    path_all = os.path.join(_project_root, "trained_model", "HLA_ESM2_2")
    if not os.path.exists(path_all):
        os.makedirs(path_all)

    all_fold_results = []
    total_start = time.time()

    for fold in range(1, 6):
        fold_result = train_single_fold(fold)
        all_fold_results.append(fold_result)

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "#" * 60)
    print("# 5-Fold Cross-Validation Summary")
    print("#" * 60)

    metrics_names = ["roc_auc", "accuracy", "mcc", "f1", "aupr"]

    for dataset_type in ["val", "independent", "external"]:
        print(f"\n{dataset_type.upper()} Set Results:")
        metrics_values = [r[dataset_type] for r in all_fold_results if r]
        metrics_pd = performances_to_pd(metrics_values)
        print(metrics_pd.to_string())

    print(f"\nTotal training time: {total_elapsed/3600:.2f} hours")
    print("\nSaved models:")
    for fold in range(1, 6):
        save_path = os.path.join(path_all, f"model_HLA_ESM2_2_fold_{fold}.pkl")
        encoder_path = os.path.join(path_all, f"encoder_P_ESM2_2_fold_{fold}.pth")
        if os.path.exists(save_path):
            print(f"  Fold-{fold}: model_HLA_ESM2_2_fold_{fold}.pkl, encoder_P_ESM2_2_fold_{fold}.pth")


if __name__ == "__main__":
    main()