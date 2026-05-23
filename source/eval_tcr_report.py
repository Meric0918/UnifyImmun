import os
import sys
import json
import time
import random
import logging
import argparse
import warnings
import psutil
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    roc_curve,
)

from models.TCR import *
from models.TCR import data_process_tcr, MyDataSet_tcr, data_load_tcr
from models.esm2_embedding import (
    Mymodel_TCR_ESM2,
    data_process_TCR_ESM2,
    MyDataSet_TCR_ESM2,
    ESM2TokenizerWrapper,
    get_tokenizer_wrapper,
    data_load_TCR_ESM2,
)
import torch.utils.data as Data

warnings.filterwarnings("ignore")


# =========================
# 1. argparse
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained TCR model with multi-fold support and auto report generation"
    )

    parser.add_argument(
        "--model-path-template",
        type=str,
        default="../trained_model/TCR_{fold}/model_TCR.pkl",
        help="Checkpoint path template. Use {fold} placeholder if needed.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="./results/tcr_experiment",
        help="Root result directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="tcr_eval_report",
        help="Experiment name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for positive prediction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=66,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Only used for logging display",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[1],
        help="Fold list, e.g. --folds 1 2 3 4 5",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["independent", "covid", "triple"],
        help="Dataset list to evaluate, e.g. --datasets independent covid triple",
    )
    parser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save attention tensors",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to custom CSV data file (e.g., LungCancer_TCR.csv). If provided, use this instead of predefined datasets.",
    )
    parser.add_argument(
        "--use-esm2",
        action="store_true",
        help="Use ESM2-based model architecture (Mymodel_TCR_ESM2).",
    )

    return parser.parse_args()


# =========================
# 2. utils
# =========================
def set_seed(seed: int = 66):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


GLOBAL_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


def setup_logger(
    log_file: str, logger_name: str, global_log_name: str = "eval_tcr_report.log"
):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    ensure_dir(GLOBAL_LOG_DIR)
    global_log_path = os.path.join(GLOBAL_LOG_DIR, global_log_name)
    global_handler = logging.FileHandler(global_log_path, encoding="utf-8", mode="a")
    global_handler.setLevel(logging.INFO)
    global_handler.setFormatter(formatter)
    logger.addHandler(global_handler)

    return logger


def safe_mean(values):
    return float(sum(values) / len(values)) if len(values) > 0 else 0.0


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    return mem_mb


def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        return allocated, reserved
    return None, None


def log_memory_status(logger, prefix=""):
    mem_mb = get_memory_usage()
    gpu_alloc, gpu_reserved = get_gpu_memory()

    msg = f"{prefix}Memory: RAM={mem_mb:.1f}MB"
    if gpu_alloc is not None:
        msg += f" | GPU_Alloc={gpu_alloc:.1f}MB | GPU_Reserved={gpu_reserved:.1f}MB"
    logger.info(msg)


def resolve_model_path(model_path_template, fold):
    if "{fold}" in model_path_template:
        return model_path_template.format(fold=fold)
    return model_path_template


def dump_json(obj, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================
# 3. metrics
# =========================
def safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0


def safe_aupr(y_true, y_score):
    try:
        prec, reca, _ = precision_recall_curve(y_true, y_score)
        return float(auc(reca, prec))
    except Exception:
        return 0.0


def calculate_metrics(y_true, y_score, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    roc_auc = safe_auc(y_true, y_score)
    aupr = safe_aupr(y_true, y_score)

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "roc_auc": roc_auc,
        "accuracy": float(accuracy),
        "mcc": float(mcc),
        "f1": float(f1),
        "aupr": float(aupr),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(recall),
        "pred_0": int(Counter(y_pred)[0]),
        "pred_1": int(Counter(y_pred)[1]),
        "true_0": int(Counter(y_true)[0]),
        "true_1": int(Counter(y_true)[1]),
    }


def log_metrics(logger, prefix, metrics):
    logger.info(
        f"{prefix} | tn={metrics['tn']}, fp={metrics['fp']}, fn={metrics['fn']}, tp={metrics['tp']}"
    )
    logger.info(
        f"{prefix} | y_pred: 0={metrics['pred_0']} | 1={metrics['pred_1']} | "
        f"y_true: 0={metrics['true_0']} | 1={metrics['true_1']}"
    )
    logger.info(
        f"{prefix} | auc={metrics['roc_auc']:.4f} | sensitivity={metrics['sensitivity']:.4f} | "
        f"specificity={metrics['specificity']:.4f} | acc={metrics['accuracy']:.4f} | "
        f"mcc={metrics['mcc']:.4f}"
    )
    logger.info(
        f"{prefix} | precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f} | "
        f"f1={metrics['f1']:.4f} | aupr={metrics['aupr']:.4f}"
    )


# =========================
# 4. io helpers
# =========================
def save_sample_predictions(result_tuple, save_path):
    y_true_list, y_score_list, y_pred_list = result_tuple

    rows = []
    for i in range(len(y_true_list)):
        rows.append(
            {
                "sample_id": i,
                "y_true": int(y_true_list[i]),
                "y_pred": int(y_pred_list[i]),
                "y_prob": float(y_score_list[i]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df


def save_attention(attention_all, save_path):
    try:
        torch.save(attention_all, save_path)
    except Exception as e:
        print(f"Warning: failed to save attention to {save_path}: {e}")


def data_load_tcr_from_csv(csv_path, batch_size=64, use_esm2=False):
    data = pd.read_csv(csv_path)
    if use_esm2:
        tokenizer_wrapper = get_tokenizer_wrapper()
        pep_inputs, tcr_inputs, labels, pep_masks, tcr_masks = data_process_TCR_ESM2(
            data, tokenizer_wrapper
        )
        loader = Data.DataLoader(
            MyDataSet_TCR_ESM2(pep_inputs, tcr_inputs, labels, pep_masks, tcr_masks),
            batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
    else:
        pep_inputs, tcr_inputs, labels = data_process_tcr(data)
        loader = Data.DataLoader(
            MyDataSet_tcr(pep_inputs, tcr_inputs, labels),
            batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
    return loader, len(data)


def save_metrics_csv(df, save_path):
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def summarize_by_dataset(all_metrics_df):
    metric_cols = [
        "roc_auc",
        "accuracy",
        "mcc",
        "f1",
        "aupr",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "tn",
        "fp",
        "fn",
        "tp",
        "pred_0",
        "pred_1",
        "true_0",
        "true_1",
        "loss",
    ]

    summary_rows = []
    for dataset_name in sorted(all_metrics_df["dataset"].unique().tolist()):
        sub_df = all_metrics_df[all_metrics_df["dataset"] == dataset_name].copy()

        mean_row = {"dataset": dataset_name, "fold": "mean"}
        std_row = {"dataset": dataset_name, "fold": "std"}

        for col in metric_cols:
            mean_row[col] = sub_df[col].mean()
            std_row[col] = sub_df[col].std()

        summary_rows.append(mean_row)
        summary_rows.append(std_row)

    return pd.DataFrame(summary_rows)


# =========================
# 5. plotting
# =========================
def plot_roc_curve(y_true, y_score, save_path, title):
    plt.figure(figsize=(6, 5))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    except Exception:
        plt.plot([0, 1], [0, 1], label="ROC unavailable")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pr_curve(y_true, y_score, save_path, title):
    plt.figure(figsize=(6, 5))
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"AUPR = {pr_auc:.4f}")
    except Exception:
        plt.plot([0, 1], [0, 1], label="PR unavailable")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_metric_lines(all_metrics_df, save_dir):
    metric_names = [
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

    datasets = sorted(all_metrics_df["dataset"].unique().tolist())

    for metric in metric_names:
        plt.figure(figsize=(8, 5))

        for dataset_name in datasets:
            sub_df = all_metrics_df[all_metrics_df["dataset"] == dataset_name].copy()
            sub_df = sub_df.sort_values("fold")
            x = [str(v) for v in sub_df["fold"].tolist()]
            y = sub_df[metric].tolist()
            plt.plot(x, y, marker="o", label=dataset_name)

        plt.xlabel("Fold")
        plt.ylabel(metric)
        plt.title(f"{metric} across folds")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_across_folds.png"), dpi=200)
        plt.close()


def plot_mean_std_bars(summary_df, save_dir):
    metric_names = [
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

    datasets = sorted(summary_df["dataset"].unique().tolist())

    for metric in metric_names:
        means = []
        stds = []
        names = []

        for dataset_name in datasets:
            sub_df = summary_df[summary_df["dataset"] == dataset_name].copy()
            mean_row = sub_df[sub_df["fold"] == "mean"]
            std_row = sub_df[sub_df["fold"] == "std"]

            if len(mean_row) == 0:
                continue

            names.append(dataset_name)
            means.append(float(mean_row.iloc[0][metric]))
            stds.append(float(std_row.iloc[0][metric]) if len(std_row) > 0 else 0.0)

        plt.figure(figsize=(6, 5))
        plt.bar(names, means, yerr=stds, capsize=5)
        plt.ylabel(metric)
        plt.title(f"{metric} mean ± std")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_mean_std.png"), dpi=200)
        plt.close()


# =========================
# 6. evaluation
# =========================
def evaluate(
    model,
    data_loader,
    criterion,
    device,
    threshold,
    logger,
    fold,
    epoch,
    epochs,
    dataset_name,
    use_esm2=False,
):
    model.eval()

    y_true_all = []
    y_score_all = []
    losses = []
    attention_all = []

    num_batches = len(data_loader)
    num_samples = len(data_loader.dataset)
    logger.info(
        f"[START] fold-{fold} {dataset_name} | samples={num_samples} | batches={num_batches}"
    )
    log_memory_status(logger, f"[{dataset_name}] ")

    eval_start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(
            tqdm(
                data_loader,
                colour="blue",
                desc=f"fold-{fold} {dataset_name}",
            )
        ):
            if use_esm2:
                # ESM2: batch_data = (pep_inputs, tcr_inputs, labels, pep_masks, tcr_masks)
                anti_inputs, tcr_inputs, labels, pep_masks, tcr_masks = batch_data
                anti_inputs = anti_inputs.to(device)
                tcr_inputs = tcr_inputs.to(device)
                labels = labels.to(device)
                pep_masks = pep_masks.to(device)
                tcr_masks = tcr_masks.to(device)
                outputs, cross_attention = model(
                    anti_inputs, tcr_inputs, pep_attention_mask=pep_masks, tcr_attention_mask=tcr_masks
                )
            else:
                # Non-ESM2: batch_data = (pep_inputs, tcr_inputs, labels)
                anti_inputs, tcr_inputs, labels = batch_data
                anti_inputs = anti_inputs.to(device)
                tcr_inputs = tcr_inputs.to(device)
                labels = labels.to(device)
                outputs, cross_attention = model(anti_inputs, tcr_inputs)

            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            y_true_all.extend(labels.cpu().numpy().tolist())
            y_score_all.extend(probs.detach().cpu().numpy().tolist())
            losses.append(loss.item())

            if isinstance(cross_attention, torch.Tensor):
                attention_all.append(cross_attention.detach().cpu())
            else:
                attention_all.append(cross_attention)

    eval_elapsed = time.time() - eval_start_time

    y_pred_all = transfer(y_score_all, threshold)
    avg_loss = safe_mean(losses)

    logger.info(
        f"Fold-{fold} | {dataset_name} | Epoch:{epoch}/{epochs} | Loss={avg_loss:.4f}"
    )
    logger.info(
        f"[DONE] fold-{fold} {dataset_name} | elapsed={eval_elapsed:.2f}s | throughput={num_samples / eval_elapsed:.1f} samples/s"
    )
    log_memory_status(logger, f"[{dataset_name}] ")

    metrics = calculate_metrics(y_true_all, y_score_all, y_pred_all)
    metrics["dataset"] = dataset_name
    metrics["fold"] = fold
    metrics["loss"] = avg_loss

    log_metrics(logger, f"fold-{fold} {dataset_name}", metrics)

    result_tuple = (y_true_all, y_score_all, y_pred_all)
    return result_tuple, metrics, attention_all


# =========================
# 7. report generation
# =========================
def make_experiment_dirs(root_result_dir, experiment_name):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(root_result_dir, f"{experiment_name}_{timestamp}")

    dirs = {
        "exp_dir": exp_dir,
        "plots_dir": os.path.join(exp_dir, "plots"),
        "tables_dir": os.path.join(exp_dir, "tables"),
        "logs_dir": os.path.join(exp_dir, "logs"),
        "folds_dir": os.path.join(exp_dir, "folds"),
        "artifacts_dir": os.path.join(exp_dir, "artifacts"),
    }

    for p in dirs.values():
        ensure_dir(p)

    return dirs


def generate_report_md(exp_dirs, args, all_metrics_df, summary_df):
    report_path = os.path.join(exp_dirs["exp_dir"], "report.md")

    lines = []
    lines.append("# TCR Evaluation Report\n")
    lines.append("## Experiment Settings\n")
    lines.append(f"- experiment_name: {args.experiment_name}")
    lines.append(f"- model_path_template: {args.model_path_template}")
    lines.append(f"- folds: {args.folds}")
    lines.append(f"- datasets: {args.datasets}")
    lines.append(f"- batch_size: {args.batch_size}")
    lines.append(f"- threshold: {args.threshold}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- device: {args.device}")
    lines.append(f"- use_esm2: {args.use_esm2}")
    lines.append("")

    lines.append("## Summary Table")
    lines.append("")
    try:
        lines.append(summary_df.to_markdown(index=False))
    except Exception:
        lines.append("See tables/summary_by_dataset.csv")
    lines.append("")

    lines.append("## All Fold Metrics")
    lines.append("")
    try:
        lines.append(all_metrics_df.to_markdown(index=False))
    except Exception:
        lines.append("See tables/all_fold_metrics.csv")
    lines.append("")

    lines.append("## Generated Figures")
    lines.append("")
    lines.append("- plots/roc_*")
    lines.append("- plots/pr_*")
    lines.append("- plots/*_across_folds.png")
    lines.append("- plots/*_mean_std.png")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 8. main
# =========================
def main():
    args = parse_args()

    exp_dirs = make_experiment_dirs(args.result_dir, args.experiment_name)

    run_id = f"{args.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger = setup_logger(
        os.path.join(exp_dirs["logs_dir"], "run.log"),
        logger_name="tcr_eval_report",
        global_log_name="eval_tcr_report.log",
    )

    logger.info("=" * 80)
    logger.info("TCR Model Evaluation Report Generator")
    logger.info(f"RUN ID: {run_id}")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Experiment directory: {exp_dirs['exp_dir']}")
    logger.info("")
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("")

    dump_json(vars(args), os.path.join(exp_dirs["artifacts_dir"], "config.json"))

    set_seed(args.seed)
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info("")

    log_memory_status(logger, "[INIT] ")
    logger.info("")

    # vocab check only needed for non-ESM2 models
    if not args.use_esm2:
        if "vocab" not in globals():
            raise ValueError(
                "`vocab` is not defined. Please ensure it exists in models.TCR"
            )

    all_metrics = []
    total_start_time = time.time()

    for fold in args.folds:
        logger.info("-" * 60)
        logger.info(f"[FOLD {fold}] Processing...")
        fold_start_time = time.time()

        fold_dir = os.path.join(exp_dirs["folds_dir"], f"fold_{fold}")
        ensure_dir(fold_dir)

        model_path = resolve_model_path(args.model_path_template, fold)
        logger.info(f"Model path: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found for fold {fold}: {model_path}"
            )

        # Select model architecture based on use_esm2 flag
        if args.use_esm2:
            model = Mymodel_TCR_ESM2(freeze_esm2=True).to(device)
        else:
            model = Mymodel_tcr().to(device)
        criterion = nn.CrossEntropyLoss()

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Model loaded successfully (use_esm2={args.use_esm2})")

        model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_params:,}")
        log_memory_status(logger, "[MODEL] ")
        logger.info("")

        if args.data_path:
            logger.info(f"Loading custom data from: {args.data_path}")
            custom_loader, custom_samples = data_load_tcr_from_csv(
                args.data_path, batch_size=args.batch_size, use_esm2=args.use_esm2
            )
            logger.info(f"  Samples: {custom_samples}")
            logger.info(f"  Batches: {len(custom_loader)}")
            logger.info(f"  Batch size: {args.batch_size}")

            result_tuple, metrics, attention_all = evaluate(
                model=model,
                data_loader=custom_loader,
                criterion=criterion,
                device=device,
                threshold=args.threshold,
                logger=logger,
                fold=fold,
                epoch=-1,
                epochs=args.epochs,
                dataset_name="custom",
                use_esm2=args.use_esm2,
            )
            all_metrics.append(metrics)

            pred_df = save_sample_predictions(
                result_tuple,
                os.path.join(fold_dir, "custom_result_data.csv"),
            )

            plot_roc_curve(
                pred_df["y_true"].tolist(),
                pred_df["y_prob"].tolist(),
                os.path.join(exp_dirs["plots_dir"], f"roc_custom_fold_{fold}.png"),
                f"ROC - custom - fold {fold}",
            )
            plot_pr_curve(
                pred_df["y_true"].tolist(),
                pred_df["y_prob"].tolist(),
                os.path.join(exp_dirs["plots_dir"], f"pr_custom_fold_{fold}.png"),
                f"PR - custom - fold {fold}",
            )

            if args.save_attention:
                save_attention(
                    attention_all,
                    os.path.join(fold_dir, "custom_attention.pt"),
                )
                logger.info(f"Attention saved")
        else:
            for dataset_name in args.datasets:
                logger.info(f"Loading dataset: {dataset_name}")
                if args.use_esm2:
                    loader = data_load_TCR_ESM2(
                        type_=dataset_name,
                        fold=fold,
                        batch_size=args.batch_size,
                    )
                else:
                    loader = data_load_tcr(
                        type_=dataset_name,
                        fold=fold,
                        batch_size=args.batch_size,
                    )
                logger.info(f"  Samples: {len(loader.dataset)}")
                logger.info(f"  Batches: {len(loader)}")
                logger.info(f"  Batch size: {args.batch_size}")

                result_tuple, metrics, attention_all = evaluate(
                    model=model,
                    data_loader=loader,
                    criterion=criterion,
                    device=device,
                    threshold=args.threshold,
                    logger=logger,
                    fold=fold,
                    epoch=-1,
                    epochs=args.epochs,
                    dataset_name=dataset_name,
                    use_esm2=args.use_esm2,
                )
                all_metrics.append(metrics)

                pred_df = save_sample_predictions(
                    result_tuple,
                    os.path.join(fold_dir, f"{dataset_name}_result_data.csv"),
                )

                plot_roc_curve(
                    pred_df["y_true"].tolist(),
                    pred_df["y_prob"].tolist(),
                    os.path.join(
                        exp_dirs["plots_dir"], f"roc_{dataset_name}_fold_{fold}.png"
                    ),
                    f"ROC - {dataset_name} - fold {fold}",
                )
                plot_pr_curve(
                    pred_df["y_true"].tolist(),
                    pred_df["y_prob"].tolist(),
                    os.path.join(
                        exp_dirs["plots_dir"], f"pr_{dataset_name}_fold_{fold}.png"
                    ),
                    f"PR - {dataset_name} - fold {fold}",
                )

                if args.save_attention:
                    save_attention(
                        attention_all,
                        os.path.join(fold_dir, f"{dataset_name}_attention.pt"),
                    )
                    logger.info(f"Attention saved for {dataset_name}")

        fold_elapsed = time.time() - fold_start_time
        logger.info(f"[FOLD {fold}] Completed in {fold_elapsed:.2f}s")
        log_memory_status(logger, "[FOLD END] ")
        logger.info("")

    logger.info("=" * 80)
    logger.info("Generating summary reports...")

    all_metrics_df = pd.DataFrame(all_metrics)
    summary_df = summarize_by_dataset(all_metrics_df)

    save_metrics_csv(
        all_metrics_df, os.path.join(exp_dirs["tables_dir"], "all_fold_metrics.csv")
    )
    save_metrics_csv(
        summary_df, os.path.join(exp_dirs["tables_dir"], "summary_by_dataset.csv")
    )

    plot_metric_lines(all_metrics_df, exp_dirs["plots_dir"])
    plot_mean_std_bars(summary_df, exp_dirs["plots_dir"])

    generate_report_md(exp_dirs, args, all_metrics_df, summary_df)

    total_elapsed = time.time() - total_start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"Total elapsed time: {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)"
    )
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Experiment directory: {exp_dirs['exp_dir']}")
    logger.info(f"Global log file: {GLOBAL_LOG_DIR}/eval_tcr_report.log")
    log_memory_status(logger, "[FINAL] ")
    logger.info("=" * 80)
    logger.info(f"RUN END: {run_id}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
