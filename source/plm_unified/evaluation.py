"""Standalone checkpoint evaluation helpers."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from .config import ExperimentConfig
from .metrics import BinaryMetrics, compute_binary_metrics
from .model import Task
from .trainer import resolve_device


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader,
    task: Task,
    config: ExperimentConfig,
) -> BinaryMetrics:
    device = resolve_device(config.training.device)
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    labels = []
    probabilities = []
    loss_sum = 0.0
    samples = 0
    precision = config.training.mixed_precision
    use_autocast = device.type == "cuda" and precision != "none"
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    hidden_dtype = (
        autocast_dtype
        if use_autocast
        else torch.float32
    )

    for batch in loader:
        peptide_hidden = batch["peptide_hidden"].to(device, dtype=hidden_dtype)
        receptor_hidden = batch["receptor_hidden"].to(device, dtype=hidden_dtype)
        peptide_mask = batch["peptide_mask"].to(device, dtype=torch.bool)
        receptor_mask = batch["receptor_mask"].to(device, dtype=torch.bool)
        batch_labels = batch["labels"].to(device, dtype=torch.long)
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if use_autocast
            else nullcontext()
        )
        with context:
            output = model(
                task,
                peptide_hidden,
                peptide_mask,
                receptor_hidden,
                receptor_mask,
            )
            loss = criterion(output.logits, batch_labels)
        batch_samples = int(batch_labels.shape[0])
        samples += batch_samples
        loss_sum += float(loss.cpu()) * batch_samples
        probabilities.append(
            torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy()
        )
        labels.append(batch_labels.cpu().numpy())

    return compute_binary_metrics(
        np.concatenate(labels),
        np.concatenate(probabilities),
        loss=loss_sum / samples,
        threshold=config.training.threshold,
    )
