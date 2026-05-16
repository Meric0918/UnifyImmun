"""
TCR binding model training with ESM2 embeddings.
ESM2 parameters are frozen, only projection layers are trained.
Uses swanlab for logging.
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
import swanlab

from models.esm2_embedding import (
    Mymodel_TCR_ESM2,
    data_load_TCR_ESM2,
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
    tcr_max_len,
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

warnings.filterwarnings("ignore")
seed = 66
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Initialize ESM2 TCR model
model_tcr = Mymodel_TCR_ESM2(freeze_esm2=True).to(device)
criterion_tcr = nn.CrossEntropyLoss()
optimizer_tcr = optim.Adam(
    filter(lambda p: p.requires_grad, model_tcr.parameters()), lr=1e-4
)
# Dynamic learning rate scheduler
scheduler_tcr = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_tcr, mode='max', factor=0.5, patience=3, min_lr=1e-6, verbose=True
)


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
    """
    FGM adversarial training for ESM2 model.
    Only attacks trainable parameters (projection layers), not frozen ESM2.

    Fixed version: Freezes BatchNorm track_running_stats during attack
    to prevent running_mean/running_var from being updated during adversarial forward.
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.bn_track_backup = {}  # Backup BatchNorm track_running_stats state

    def attack(self, epsilon=1.0):
        # Step 1: Freeze BatchNorm running stats updates during attack forward
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                self.bn_track_backup[name] = module.track_running_stats
                module.track_running_stats = False  # Critical: freeze BN running stats

        # Step 2: Attack projection layer parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name:
                self.backup[name] = param.data.clone().detach()
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self):
        # Step 1: Restore projection layer parameters
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

        # Step 2: Restore BatchNorm track_running_stats state
        for name, module in self.model.named_modules():
            if name in self.bn_track_backup:
                module.track_running_stats = self.bn_track_backup[name]
        self.bn_track_backup = {}


def train_tcr(model, train_loader, fold, epoch, epochs):
    train_time = 0
    model.train()
    model.encoder_T.src_emb.esm2.eval()
    model.encoder_P.src_emb.esm2.eval()

    y_true_list, y_pred_list, attention_list = [], [], []
    loss_list = []
    fgm = FGM_ESM2(model)

    for pep_inputs, tcr_inputs, labels, pep_masks, tcr_masks in tqdm(
        train_loader, colour="yellow"
    ):
        pep_inputs = pep_inputs.to(device)
        tcr_inputs = tcr_inputs.to(device)
        labels = labels.to(device)
        pep_masks = pep_masks.to(device)
        tcr_masks = tcr_masks.to(device)

        start = time.time()
        train_outputs, cross_attention = model(
            pep_inputs, tcr_inputs,
            pep_attention_mask=pep_masks,
            tcr_attention_mask=tcr_masks
        )
        train_loss = criterion_tcr(train_outputs, labels)
        train_time += time.time() - start
        train_loss.backward()

        fgm.attack(epsilon=1.0)
        train_outputs2, _ = model(
            pep_inputs, tcr_inputs,
            pep_attention_mask=pep_masks,
            tcr_attention_mask=tcr_masks
        )
        loss_sum = criterion_tcr(train_outputs2, labels)
        loss_sum.backward()
        fgm.restore()

        optimizer_tcr.step()
        optimizer_tcr.zero_grad()

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
        "Fold-{} Train: Epoch:{}/{} Loss = {:.4f} | Time = {:.4f} sec".format(
            fold, epoch, epochs, avg_loss, train_time
        )
    )
    performance_train = performance(y_true_list, y_pred_list, y_pred_transfer_list)
    return result_train, performance_train, train_time, attention_list, avg_loss


def valid_tcr(model, val_loader, fold, epoch, epochs):
    model.eval()
    torch.manual_seed(66)
    torch.cuda.manual_seed(66)
    cross_attention_val = None

    with torch.no_grad():
        y_true_val_list, y_pred_val_list, attention_val_list = [], [], []
        loss_val_list = []

        for pep_inputs, tcr_inputs, labels, pep_masks, tcr_masks in tqdm(
            val_loader, colour="blue"
        ):
            pep_inputs = pep_inputs.to(device)
            tcr_inputs = tcr_inputs.to(device)
            labels = labels.to(device)
            pep_masks = pep_masks.to(device)
            tcr_masks = tcr_masks.to(device)

            val_outputs, cross_attention_val = model(
                pep_inputs, tcr_inputs,
                pep_attention_mask=pep_masks,
                tcr_attention_mask=tcr_masks
            )
            val_loss = criterion_tcr(val_outputs, labels)

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


# Initialize swanlab
swanlab.init(
    project="unifyimmun",
    experiment_name="TCR_ESM2",
    config={
        "model": "TCR_ESM2",
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
        "pretrained_encoder": "HLA_ESM2",
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.001,
    },
)

# Load test datasets
independent_loader = data_load_TCR_ESM2(type_="independent", batch_size=batch_size)
covid_loader = data_load_TCR_ESM2(type_="covid", batch_size=batch_size)
triple_loader = data_load_TCR_ESM2(type_="triple", batch_size=batch_size)

attention_train_dict, attention_val_dict = {}, {}

start_time = time.time()

# Training (single split, no cross-validation)
fold = 1
print("Training with single split:")
print("Load Encoder from HLA_ESM2 Phase 1")

# Load encoder_P from HLA_ESM2 Phase 1
encoder_path_load = os.path.join(
    _project_root, "trained_model", "HLA_ESM2", "encoder_P_ESM2.pth"
)
if os.path.exists(encoder_path_load):
    model_tcr.encoder_P.load_state_dict(torch.load(encoder_path_load, map_location=device))
    print("Loaded encoder_P from: ", encoder_path_load)
else:
    print("Warning: encoder_P not found at {}, using fresh encoder".format(encoder_path_load))

print("Load TCR Data with ESM2 embedding:")
train_loader = data_load_TCR_ESM2(type_="train", fold=fold, batch_size=batch_size)
val_loader = data_load_TCR_ESM2(type_="val", fold=fold, batch_size=batch_size)

train_data = pd.read_csv(
    os.path.join(_project_root, "data", "data_TCR", "train_fold_{}.csv".format(fold))
).dropna()
val_data = pd.read_csv(
    os.path.join(_project_root, "data", "data_TCR", "val_fold_{}.csv".format(fold))
).dropna()
print(
    "Label: Train = {} | Val = {}".format(
        Counter(train_data.label), Counter(val_data.label)
    )
)

print("TCR Train with ESM2:")
path_all = os.path.join(_project_root, "trained_model", "TCR_ESM2")
save_path = os.path.join(path_all, "model_TCR_ESM2.pkl")
encoder_path = os.path.join(path_all, "encoder_P_ESM2.pth")
print("save path: ", save_path)

epoch_best = -1
time_train = 0

# Early stopping parameters
patience = 5  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum improvement threshold
no_improve_count = 0

# Track both train and validation performance for early stopping
train_performance_best = 0
val_performance_best = 0

for epoch in range(1, epochs + 1):
    result_train, performance_train, train_time_ep, attention_score, train_loss = train_tcr(
        model_tcr, train_loader, fold, epoch, epochs
    )
    result_val, performance_val, attention_score_val, val_loss = valid_tcr(
        model_tcr, val_loader, fold, epoch, epochs
    )
    train_performance_avg = sum(performance_train[:5]) / 5
    val_performance_avg = sum(performance_val[:5]) / 5

    # Log to swanlab
    swanlab.log(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_roc_auc": performance_train[0],
            "train_accuracy": performance_train[1],
            "train_mcc": performance_train[2],
            "train_f1": performance_train[3],
            "train_aupr": performance_train[4],
            "train_sensitivity": performance_train[5],
            "train_specificity": performance_train[6],
            "train_performance_avg": train_performance_avg,
            "val_loss": val_loss,
            "val_roc_auc": performance_val[0],
            "val_accuracy": performance_val[1],
            "val_mcc": performance_val[2],
            "val_f1": performance_val[3],
            "val_aupr": performance_val[4],
            "val_sensitivity": performance_val[5],
            "val_specificity": performance_val[6],
            "val_performance_avg": val_performance_avg,
            "no_improve_count": no_improve_count,
        }
    )

    # Check improvement on both train and validation
    train_improved = train_performance_avg > train_performance_best + min_delta
    val_improved = val_performance_avg > val_performance_best + min_delta

    if val_improved:
        val_performance_best, epoch_best = val_performance_avg, epoch
        if not os.path.exists(path_all):
            os.makedirs(path_all)
        print(
            "****Saving model: Best epoch = {} | Val_performance_avg = {:.4f}".format(
                epoch_best, val_performance_best
            )
        )
        print("*****Path saver: ", save_path)
        torch.save(model_tcr.eval().state_dict(), save_path)
        torch.save(model_tcr.eval().encoder_P.state_dict(), encoder_path)

        swanlab.log(
            {
                "best_epoch": epoch_best,
                "best_performance_avg": val_performance_best,
            }
        )

    if train_improved:
        train_performance_best = train_performance_avg
        print(f"Train performance improved: {train_performance_best:.4f}")

    # Early stopping: only when both train and val don't improve significantly
    if train_improved or val_improved:
        no_improve_count = 0  # Reset counter if either improves
    else:
        no_improve_count += 1
        print(f"No improvement for {no_improve_count} epochs (patience={patience})")
        print(f"  Train: best={train_performance_best:.4f}, current={train_performance_avg:.4f}")
        print(f"  Val:   best={val_performance_best:.4f}, current={val_performance_avg:.4f}")
        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Update learning rate based on validation performance
    scheduler_tcr.step(val_performance_avg)
    current_lr = optimizer_tcr.param_groups[0]['lr']
    swanlab.log({"learning_rate": current_lr})
    print(f"Current learning rate: {current_lr:.6f}")

    time_train += train_time_ep

print(f"TCR ESM2 Training Finished (best epoch: {epoch_best})")
print("-----Evaluate Results-----")

if epoch_best >= 0:
    print("*****Path saver: ", save_path)
    model_tcr.load_state_dict(torch.load(save_path, map_location=device))
    model_eval = model_tcr.eval()

    independent_result, independent_performance, independent_attention, _ = valid_tcr(
        model_eval, independent_loader, fold, epoch_best, epochs
    )
    covid_result, covid_performance, covid_attention, _ = valid_tcr(
        model_eval, covid_loader, fold, epoch_best, epochs
    )
    triple_result, triple_performance, triple_attention, _ = valid_tcr(
        model_eval, triple_loader, fold, epoch_best, epochs
    )

    # Log final evaluation results
    swanlab.log(
        {
            "independent_roc_auc": independent_performance[0],
            "independent_accuracy": independent_performance[1],
            "covid_roc_auc": covid_performance[0],
            "covid_accuracy": covid_performance[1],
            "triple_roc_auc": triple_performance[0],
            "triple_accuracy": triple_performance[1],
        }
    )

    print("****TCR ESM2 Independent set:")
    print("roc_auc={:.4f}, accuracy={:.4f}, mcc={:.4f}, f1={:.4f}".format(
        independent_performance[0], independent_performance[1], independent_performance[2], independent_performance[3]
    ))
    print("****TCR ESM2 Covid set:")
    print("roc_auc={:.4f}, accuracy={:.4f}, mcc={:.4f}, f1={:.4f}".format(
        covid_performance[0], covid_performance[1], covid_performance[2], covid_performance[3]
    ))
    print("****TCR ESM2 Triple set:")
    print("roc_auc={:.4f}, accuracy={:.4f}, mcc={:.4f}, f1={:.4f}".format(
        triple_performance[0], triple_performance[1], triple_performance[2], triple_performance[3]
    ))
    print("****Val set:")
    print("roc_auc={:.4f}, accuracy={:.4f}, mcc={:.4f}, f1={:.4f}".format(
        performance_val[0], performance_val[1], performance_val[2], performance_val[3]
    ))

print("Total training time: {:6.2f} sec".format(time_train))

end_time = time.time()
use_time = end_time - start_time
print("Use Time: {:6.2f} seconds".format(use_time))

swanlab.finish()