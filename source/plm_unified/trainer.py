"""Stage 1A warm-up and stage 1B alternating dual-task training."""

from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from .artifacts import (
    create_artifact_timestamp,
    timestamped_filename,
    validate_artifact_timestamp,
)
from .config import ExperimentConfig
from .encoders import OnlineEncoderBundle
from .fgm import FGMStats, ModuleFGM
from .metrics import (
    BinaryMetrics,
    compute_binary_metrics,
    prefixed_metrics,
    task_score,
)
from .model import Task, UnifiedBindingModel, extract_task_state_dict
from .tracking import SwanLabTracker

Stage = Literal["warmup", "stage1b", "complete"]
TASK_CHECKPOINT_FILENAMES: dict[Task, str] = {
    "phla": "best_phla.pt",
    "ptcr": "best_ptcr.pt",
}


def _metric_summary(metrics: BinaryMetrics) -> str:
    return (
        f"loss={metrics.loss:.4f} "
        f"AUROC={metrics.auroc:.4f} "
        f"AUPR={metrics.aupr:.4f} "
        f"ACC={metrics.accuracy:.4f} "
        f"MCC={metrics.mcc:.4f} "
        f"F1={metrics.f1:.4f}"
    )


@dataclass
class EpochResult:
    metrics: BinaryMetrics
    clean_loss: float
    adversarial_loss: float
    total_loss: float
    optimizer_steps: int
    samples: int
    attacked_parameter_count: float
    fgm_mean_gradient_norm: float
    fgm_max_gradient_norm: float
    model_gradient_norm: float

    def to_log_dict(self, prefix: str) -> dict[str, float | int]:
        values = prefixed_metrics(prefix, self.metrics)
        values.update(
            {
                f"{prefix}/clean_loss": self.clean_loss,
                f"{prefix}/adversarial_loss": self.adversarial_loss,
                f"{prefix}/total_loss": self.total_loss,
                f"{prefix}/optimizer_steps": self.optimizer_steps,
                f"{prefix}/samples": self.samples,
                f"{prefix}/fgm_attacked_parameters": self.attacked_parameter_count,
                f"{prefix}/fgm_mean_gradient_norm": self.fgm_mean_gradient_norm,
                f"{prefix}/fgm_max_gradient_norm": self.fgm_max_gradient_norm,
                f"{prefix}/model_gradient_norm": self.model_gradient_norm,
            }
        )
        return values


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_optimizer(
    model: UnifiedBindingModel,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    training = config.training
    parameter_groups = [
        {
            "name": "adapters",
            "params": list(model.peptide_adapter.parameters())
            + list(model.hla_adapter.parameters())
            + list(model.tcr_adapter.parameters()),
            "lr": training.adapter_lr,
        },
        {
            "name": "cross_attention",
            "params": list(model.phla_cross_attention.parameters())
            + list(model.ptcr_cross_attention.parameters()),
            "lr": training.cross_attention_lr,
        },
        {
            "name": "pooling",
            "params": list(model.phla_pooling.parameters())
            + list(model.ptcr_pooling.parameters()),
            "lr": training.pooling_lr,
        },
        {
            "name": "classifiers",
            "params": list(model.phla_classifier.parameters())
            + list(model.ptcr_classifier.parameters()),
            "lr": training.classifier_lr,
        },
    ]
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=training.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, round(total_steps * warmup_ratio))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = min(
            1.0,
            (step - warmup_steps) / max(1, total_steps - warmup_steps),
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class Stage1Trainer:
    def __init__(
        self,
        model: UnifiedBindingModel,
        config: ExperimentConfig,
        train_loaders: Mapping[Task, Any],
        val_loaders: Mapping[Task, Any],
        tracker: SwanLabTracker,
        cache_metadata: Optional[Mapping[str, Any]] = None,
        online_encoders: Optional[OnlineEncoderBundle] = None,
        input_mode: str = "cache",
        run_timestamp: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.train_loaders = dict(train_loaders)
        self.val_loaders = dict(val_loaders)
        self.data_filter_stats = {}
        for split_name, loaders in (
            ("train", self.train_loaders),
            ("val", self.val_loaders),
        ):
            for task, loader in loaders.items():
                dataset = getattr(loader, "dataset", None)
                stats = getattr(dataset, "filter_stats", None)
                if stats:
                    self.data_filter_stats[f"{task}/{split_name}"] = dict(stats)
        self.tracker = tracker
        self.cache_metadata = dict(cache_metadata or {})
        self.online_encoders = online_encoders
        self.input_mode = input_mode
        self.device = resolve_device(config.training.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = build_optimizer(model, config)
        steps_per_pair = sum(len(loader) for loader in self.train_loaders.values())
        total_steps = steps_per_pair * (
            config.training.warmup_epochs_per_task + config.training.max_rounds
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_ratio=config.training.scheduler_warmup_ratio,
        )
        scaler_enabled = (
            self.device.type == "cuda"
            and config.training.mixed_precision == "fp16"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        self.fgm = ModuleFGM(model)

        self.stage: Stage = "warmup"
        self.warmup_completed: dict[Task, int] = {"phla": 0, "ptcr": 0}
        self.current_round = 0
        self.best_task_scores: dict[Task, float] = {
            "phla": -float("inf"),
            "ptcr": -float("inf"),
        }
        self.best_val_metrics: dict[Task, dict[str, float | int]] = {}
        self.no_improve_rounds: dict[Task, int] = {"phla": 0, "ptcr": 0}
        self.stopped_tasks: dict[Task, bool] = {"phla": False, "ptcr": False}
        self.global_step = 0
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        self.last_checkpoint_path = self.run_dir / timestamped_filename(
            "last.pt", self.run_timestamp
        )
        self.warmup_checkpoint_path = self.run_dir / timestamped_filename(
            "warmup_last.pt", self.run_timestamp
        )
        self.task_checkpoint_paths: dict[Task, Path] = {
            task: self.run_dir / timestamped_filename(filename, self.run_timestamp)
            for task, filename in TASK_CHECKPOINT_FILENAMES.items()
        }

    def _autocast(self):
        precision = self.config.training.mixed_precision
        if self.device.type != "cuda" or precision == "none":
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _move_batch(
        self,
        task: Task,
        batch: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        non_blocking = self.device.type == "cuda"
        hidden_dtype = torch.float32
        if self.device.type == "cuda":
            if self.config.training.mixed_precision == "fp16":
                hidden_dtype = torch.float16
            elif self.config.training.mixed_precision == "bf16":
                hidden_dtype = torch.bfloat16
        if "peptide_sequences" in batch:
            if self.online_encoders is None:
                raise RuntimeError("Raw sequences require online_encoders")
            encoded = self.online_encoders.encode(
                task,
                batch["peptide_sequences"],
                batch["receptor_sequences"],
            )
        else:
            encoded = batch
        return {
            "peptide_hidden": encoded["peptide_hidden"].to(
                self.device, dtype=hidden_dtype, non_blocking=non_blocking
            ),
            "peptide_mask": encoded["peptide_mask"].to(
                self.device, dtype=torch.bool, non_blocking=non_blocking
            ),
            "receptor_hidden": encoded["receptor_hidden"].to(
                self.device, dtype=hidden_dtype, non_blocking=non_blocking
            ),
            "receptor_mask": encoded["receptor_mask"].to(
                self.device, dtype=torch.bool, non_blocking=non_blocking
            ),
            "labels": batch["labels"].to(
                self.device, dtype=torch.long, non_blocking=non_blocking
            ),
        }

    def _forward(self, task: Task, batch: Mapping[str, torch.Tensor]):
        return self.model(
            task=task,
            peptide_hidden=batch["peptide_hidden"],
            peptide_mask=batch["peptide_mask"],
            receptor_hidden=batch["receptor_hidden"],
            receptor_mask=batch["receptor_mask"],
        )

    def train_epoch(
        self,
        task: Task,
        loader,
        *,
        phase_name: str,
        use_fgm: bool,
    ) -> EpochResult:
        self.model.set_trainable_task(task)
        self.model.train()
        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        clean_loss_sum = 0.0
        adversarial_loss_sum = 0.0
        samples = 0
        optimizer_steps = 0
        attacked_counts: list[int] = []
        fgm_mean_norms: list[float] = []
        fgm_max_norms: list[float] = []
        model_gradient_norms: list[float] = []

        for batch_index, cpu_batch in enumerate(loader, start=1):
            batch = self._move_batch(task, cpu_batch)
            batch_samples = int(batch["labels"].shape[0])
            self.optimizer.zero_grad(set_to_none=True)
            gradient_scale = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0

            with self._autocast():
                clean_output = self._forward(task, batch)
                clean_loss = self.criterion(clean_output.logits, batch["labels"])
            self.scaler.scale(clean_loss).backward()

            adversarial_loss_value = 0.0
            fgm_stats: Optional[FGMStats] = None
            if use_fgm:
                fgm_stats = self.fgm.attack(
                    self.model.fgm_modules(task),
                    epsilon=self.config.training.fgm_epsilon,
                )
                try:
                    with self._autocast():
                        adversarial_output = self._forward(task, batch)
                        adversarial_loss = self.criterion(
                            adversarial_output.logits,
                            batch["labels"],
                        )
                    self.scaler.scale(adversarial_loss).backward()
                    adversarial_loss_value = float(adversarial_loss.detach().cpu())
                finally:
                    self.fgm.restore()

            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                self.config.training.gradient_clip_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            optimizer_steps += 1

            clean_loss_value = float(clean_loss.detach().cpu())
            clean_loss_sum += clean_loss_value * batch_samples
            adversarial_loss_sum += adversarial_loss_value * batch_samples
            samples += batch_samples
            probabilities.append(
                torch.softmax(clean_output.logits.detach(), dim=1)[:, 1]
                .float()
                .cpu()
                .numpy()
            )
            labels.append(batch["labels"].detach().cpu().numpy())
            model_gradient_norms.append(float(gradient_norm.detach().cpu()))
            if fgm_stats is not None:
                attacked_counts.append(fgm_stats.attacked_parameter_count)
                fgm_mean_norms.append(
                    fgm_stats.mean_gradient_norm / gradient_scale
                )
                fgm_max_norms.append(fgm_stats.max_gradient_norm / gradient_scale)

            if (
                batch_index % self.config.training.log_every_steps == 0
                or batch_index == len(loader)
            ):
                learning_rates = {
                    f"learning_rate/{group.get('name', index)}": group["lr"]
                    for index, group in enumerate(self.optimizer.param_groups)
                }
                self.tracker.log(
                    {
                        f"{phase_name}/{task}/batch_clean_loss": clean_loss_value,
                        f"{phase_name}/{task}/batch_adversarial_loss": (
                            adversarial_loss_value
                        ),
                        f"{phase_name}/{task}/batch_gradient_norm": float(
                            gradient_norm.detach().cpu()
                        ),
                        f"{phase_name}/{task}/fgm_epsilon": (
                            self.config.training.fgm_epsilon if use_fgm else 0.0
                        ),
                        **learning_rates,
                    },
                    step=self.global_step,
                )

        average_clean_loss = clean_loss_sum / samples
        average_adversarial_loss = adversarial_loss_sum / samples
        metrics = compute_binary_metrics(
            np.concatenate(labels),
            np.concatenate(probabilities),
            loss=average_clean_loss,
            threshold=self.config.training.threshold,
        )
        return EpochResult(
            metrics=metrics,
            clean_loss=average_clean_loss,
            adversarial_loss=average_adversarial_loss,
            total_loss=average_clean_loss + average_adversarial_loss,
            optimizer_steps=optimizer_steps,
            samples=samples,
            attacked_parameter_count=_mean(attacked_counts),
            fgm_mean_gradient_norm=_mean(fgm_mean_norms),
            fgm_max_gradient_norm=max(fgm_max_norms, default=0.0),
            model_gradient_norm=_mean(model_gradient_norms),
        )

    @torch.inference_mode()
    def evaluate(self, task: Task, loader) -> BinaryMetrics:
        self.model.eval()
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        loss_sum = 0.0
        samples = 0
        for cpu_batch in loader:
            batch = self._move_batch(task, cpu_batch)
            with self._autocast():
                output = self._forward(task, batch)
                loss = self.criterion(output.logits, batch["labels"])
            batch_samples = int(batch["labels"].shape[0])
            samples += batch_samples
            loss_sum += float(loss.detach().cpu()) * batch_samples
            probabilities.append(
                torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy()
            )
            labels.append(batch["labels"].cpu().numpy())
        return compute_binary_metrics(
            np.concatenate(labels),
            np.concatenate(probabilities),
            loss=loss_sum / samples,
            threshold=self.config.training.threshold,
        )

    def fit(self) -> dict[str, Any]:
        if self.stage == "complete":
            return self.summary()

        if self.stage == "warmup":
            for task in ("phla", "ptcr"):
                first_epoch = self.warmup_completed[task] + 1
                for epoch in range(
                    first_epoch,
                    self.config.training.warmup_epochs_per_task + 1,
                ):
                    result = self.train_epoch(
                        task,
                        self.train_loaders[task],
                        phase_name="warmup",
                        use_fgm=False,
                    )
                    self.warmup_completed[task] = epoch
                    self.tracker.log(
                        {
                            "stage": "warmup",
                            "task": task,
                            "warmup_epoch": epoch,
                            **result.to_log_dict(f"warmup/{task}/train"),
                        },
                        step=self.global_step,
                    )
                    print(
                        f"[Stage 1A][seed={self.config.training.seed}]"
                        f"[task={'pHLA' if task == 'phla' else 'pTCR'}]"
                        f"[epoch={epoch}/"
                        f"{self.config.training.warmup_epochs_per_task}]"
                        f"[step={self.global_step}] "
                        f"{_metric_summary(result.metrics)}",
                        flush=True,
                    )
                    self.save_checkpoint(self.last_checkpoint_path)

            self.stage = "stage1b"
            self.save_checkpoint(self.warmup_checkpoint_path)
            self.save_checkpoint(self.last_checkpoint_path)

        start_round = self.current_round + 1
        for round_id in range(start_round, self.config.training.max_rounds + 1):
            active_tasks: list[Task] = [
                task
                for task in ("phla", "ptcr")
                if not self.stopped_tasks[task]
            ]
            if not active_tasks:
                break

            train_results: dict[Task, EpochResult] = {}
            for task in active_tasks:
                train_results[task] = self.train_epoch(
                    task,
                    self.train_loaders[task],
                    phase_name=f"round_{round_id}",
                    use_fgm=True,
                )

            val_metrics = {
                task: self.evaluate(task, self.val_loaders[task])
                for task in active_tasks
            }
            scores: dict[Task, float] = {}
            improved: dict[Task, bool] = {}
            newly_stopped: list[Task] = []
            for task in active_tasks:
                scores[task] = task_score(val_metrics[task])
                improved[task] = (
                    scores[task]
                    > self.best_task_scores[task]
                    + self.config.training.early_stopping_min_delta
                )
                if improved[task]:
                    self.best_task_scores[task] = scores[task]
                    self.best_val_metrics[task] = val_metrics[task].to_dict()
                    self.no_improve_rounds[task] = 0
                else:
                    self.no_improve_rounds[task] += 1
                    if (
                        self.no_improve_rounds[task]
                        >= self.config.training.early_stopping_patience
                    ):
                        self.stopped_tasks[task] = True
                        newly_stopped.append(task)
            self.current_round = round_id

            log_values: dict[str, Any] = {
                "stage": "stage1b",
                "round": round_id,
                "round/phla_optimizer_steps": (
                    train_results["phla"].optimizer_steps
                    if "phla" in train_results
                    else 0
                ),
                "round/ptcr_optimizer_steps": (
                    train_results["ptcr"].optimizer_steps
                    if "ptcr" in train_results
                    else 0
                ),
                "round/phla_to_ptcr_step_ratio": (
                    (
                        train_results["phla"].optimizer_steps
                        if "phla" in train_results
                        else 0
                    )
                    / max(
                        1,
                        (
                            train_results["ptcr"].optimizer_steps
                            if "ptcr" in train_results
                            else 0
                        ),
                    )
                ),
            }
            for task in active_tasks:
                log_values.update(
                    train_results[task].to_log_dict(f"round/{task}/train")
                )
                log_values.update(prefixed_metrics(f"round/{task}/val", val_metrics[task]))
                log_values[f"early_stopping/{task}/score"] = scores[task]
                log_values[f"early_stopping/{task}/best_score"] = (
                    self.best_task_scores[task]
                )
                log_values[f"early_stopping/{task}/no_improve_rounds"] = (
                    self.no_improve_rounds[task]
                )
                log_values[f"early_stopping/{task}/stopped"] = int(
                    self.stopped_tasks[task]
                )
            self.tracker.log(log_values, step=self.global_step)
            print(
                f"[Stage 1B][seed={self.config.training.seed}]"
                f"[round={round_id}/{self.config.training.max_rounds}]"
                f"[step={self.global_step}] "
                f"active={','.join(active_tasks)}",
                flush=True,
            )
            for task in active_tasks:
                task_name = "pHLA" if task == "phla" else "pTCR"
                print(
                    f"  {task_name} train: "
                    f"{_metric_summary(train_results[task].metrics)} | "
                    f"val: {_metric_summary(val_metrics[task])} | "
                    f"score={scores[task]:.4f} "
                    f"best={self.best_task_scores[task]:.4f} "
                    f"improved={'yes' if improved[task] else 'no'} "
                    f"patience={self.no_improve_rounds[task]}/"
                    f"{self.config.training.early_stopping_patience} "
                    f"stopped={'yes' if self.stopped_tasks[task] else 'no'}",
                    flush=True,
                )

            self.save_checkpoint(self.last_checkpoint_path)
            for task in active_tasks:
                if improved[task]:
                    self.save_best_task_checkpoint(task)
            for task in newly_stopped:
                task_name = "pHLA" if task == "phla" else "pTCR"
                print(
                    f"[Early stopping][seed={self.config.training.seed}]"
                    f"[task={task_name}] round={round_id} "
                    f"best_score={self.best_task_scores[task]:.4f}",
                    flush=True,
                )
            if all(self.stopped_tasks.values()):
                break

        self.stage = "complete"
        self.save_checkpoint(self.last_checkpoint_path)
        print(
            f"[Complete][seed={self.config.training.seed}] "
            f"round={self.current_round} step={self.global_step} "
            f"best_pHLA={self.best_task_scores['phla']:.4f} "
            f"best_pTCR={self.best_task_scores['ptcr']:.4f}",
            flush=True,
        )
        return self.summary()

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "stage": self.stage,
            "warmup_completed": dict(self.warmup_completed),
            "current_round": self.current_round,
            "best_task_scores": dict(self.best_task_scores),
            "best_val_metrics": self.best_val_metrics,
            "no_improve_rounds": dict(self.no_improve_rounds),
            "stopped_tasks": dict(self.stopped_tasks),
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "data_filter_stats": self.data_filter_stats,
            "cache_metadata": self.cache_metadata,
            "input_mode": self.input_mode,
            "run_timestamp": self.run_timestamp,
            "swanlab_run_id": self.tracker.run_id,
            "dataloader_rng_state": {
                task: loader.generator.get_state()
                for task, loader in self.train_loaders.items()
                if getattr(loader, "generator", None) is not None
            },
            "rng_state": _capture_rng_state(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        _atomic_torch_save(self.checkpoint_state(), path)

    def save_best_task_checkpoint(self, task: Task) -> None:
        """Save only the task whose own validation score just improved."""

        training_state = self.checkpoint_state()
        task_state = build_task_checkpoint_state(
            training_state,
            task,
            source_checkpoint=self.last_checkpoint_path,
        )
        _atomic_torch_save(
            task_state,
            self.task_checkpoint_paths[task],
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = load_training_checkpoint(path, map_location="cpu")
        format_version = checkpoint.get("format_version")
        if format_version not in {1, 2}:
            raise ValueError("Unsupported checkpoint format")
        checkpoint_config = checkpoint.get("config", {})
        if checkpoint.get("input_mode", "cache") != self.input_mode:
            raise ValueError(
                f"Checkpoint input mode {checkpoint.get('input_mode')} does not match "
                f"current mode {self.input_mode}"
            )
        checkpoint_training = checkpoint_config.get("training", {})
        for key in ("fold", "seed"):
            current_value = getattr(self.config.training, key)
            if checkpoint_training.get(key) != current_value:
                raise ValueError(
                    f"Checkpoint {key}={checkpoint_training.get(key)} does not match "
                    f"current {key}={current_value}"
                )
        if checkpoint_config.get("model") != self.config.to_dict()["model"]:
            raise ValueError("Checkpoint model configuration does not match current config")
        checkpoint_cache = checkpoint.get("cache_metadata", {})
        for entity, current_manifest in self.cache_metadata.items():
            saved_manifest = checkpoint_cache.get(entity, {})
            current_fingerprint = current_manifest.get("model", {}).get(
                "checkpoint_fingerprint"
            )
            saved_fingerprint = saved_manifest.get("model", {}).get(
                "checkpoint_fingerprint"
            )
            if saved_fingerprint and saved_fingerprint != current_fingerprint:
                raise ValueError(
                    f"Checkpoint cache fingerprint mismatch for entity '{entity}'"
                )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        self.stage = checkpoint["stage"]
        self.warmup_completed = {
            "phla": int(checkpoint["warmup_completed"]["phla"]),
            "ptcr": int(checkpoint["warmup_completed"]["ptcr"]),
        }
        self.current_round = int(checkpoint["current_round"])
        self.best_val_metrics = dict(checkpoint.get("best_val_metrics", {}))
        if format_version == 2:
            self.best_task_scores = {
                task: float(checkpoint["best_task_scores"][task])
                for task in ("phla", "ptcr")
            }
            self.no_improve_rounds = {
                task: int(checkpoint["no_improve_rounds"][task])
                for task in ("phla", "ptcr")
            }
            self.stopped_tasks = {
                task: bool(checkpoint["stopped_tasks"][task])
                for task in ("phla", "ptcr")
            }
        else:
            self.best_task_scores = {
                task: (
                    task_score(self.best_val_metrics[task])
                    if task in self.best_val_metrics
                    else -float("inf")
                )
                for task in ("phla", "ptcr")
            }
            self.no_improve_rounds = {"phla": 0, "ptcr": 0}
            self.stopped_tasks = {"phla": False, "ptcr": False}
        self.global_step = int(checkpoint["global_step"])
        for task, generator_state in checkpoint.get(
            "dataloader_rng_state", {}
        ).items():
            loader = self.train_loaders.get(task)
            generator = getattr(loader, "generator", None)
            if generator is not None:
                generator.set_state(generator_state)
        _restore_rng_state(checkpoint.get("rng_state"))

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "fold": self.config.training.fold,
            "seed": self.config.training.seed,
            "round": self.current_round,
            "best_task_scores": dict(self.best_task_scores),
            "best_val_metrics": self.best_val_metrics,
            "no_improve_rounds": dict(self.no_improve_rounds),
            "stopped_tasks": dict(self.stopped_tasks),
            "data_filter_stats": self.data_filter_stats,
            "global_step": self.global_step,
            "run_timestamp": self.run_timestamp,
            "run_dir": str(self.run_dir),
            "last_checkpoint": str(self.last_checkpoint_path),
            "warmup_checkpoint": str(self.warmup_checkpoint_path),
            "task_checkpoints": {
                task: str(path)
                for task, path in self.task_checkpoint_paths.items()
            },
        }


def peek_swanlab_run_id(path: str | Path) -> Optional[str]:
    checkpoint = load_training_checkpoint(path, map_location="cpu")
    return checkpoint.get("swanlab_run_id")


def load_training_checkpoint(
    path: str | Path,
    *,
    map_location,
) -> dict[str, Any]:
    """Load trusted local training state across PyTorch 2.1+ defaults."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_task_checkpoint_state(
    training_checkpoint: Mapping[str, Any],
    task: Task,
    *,
    source_checkpoint: str | Path,
) -> dict[str, Any]:
    """Build a lean inference checkpoint containing exactly one task branch."""

    if training_checkpoint.get("checkpoint_type") == "task_model":
        raise ValueError("Expected a training checkpoint, got a task model")
    model_state = training_checkpoint.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError("Training checkpoint does not contain a model state")
    best_metrics = training_checkpoint.get("best_val_metrics", {})
    best_task_scores = training_checkpoint.get("best_task_scores", {})
    validation_metrics = best_metrics.get(task, {})
    best_score = best_task_scores.get(task)
    if best_score is None and validation_metrics:
        best_score = task_score(validation_metrics)
    no_improve = training_checkpoint.get("no_improve_rounds", {})
    stopped = training_checkpoint.get("stopped_tasks", {})
    return {
        "format_version": 2,
        "checkpoint_type": "task_model",
        "task": task,
        "model": extract_task_state_dict(model_state, task),
        "config": training_checkpoint.get("config", {}),
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_format_version": training_checkpoint.get("format_version"),
        "stage": training_checkpoint.get("stage"),
        "current_round": training_checkpoint.get("current_round"),
        "global_step": training_checkpoint.get("global_step"),
        "run_timestamp": training_checkpoint.get("run_timestamp"),
        "best_score": best_score,
        "validation_metrics": validation_metrics,
        "early_stopping": {
            "no_improve_rounds": (
                no_improve.get(task) if isinstance(no_improve, Mapping) else None
            ),
            "stopped": stopped.get(task) if isinstance(stopped, Mapping) else None,
        },
        "data_filter_stats": training_checkpoint.get("data_filter_stats", {}),
        "cache_metadata": training_checkpoint.get("cache_metadata", {}),
        "input_mode": training_checkpoint.get("input_mode", "cache"),
    }


def export_task_checkpoints(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    artifact_timestamp: str | None = None,
) -> dict[Task, Path]:
    """Split a trusted legacy full checkpoint into pHLA and pTCR model files."""

    source = Path(checkpoint_path).resolve()
    checkpoint = load_training_checkpoint(source, map_location="cpu")
    destination = Path(output_dir).resolve() if output_dir else source.parent
    timestamp = validate_artifact_timestamp(
        artifact_timestamp or create_artifact_timestamp()
    )
    targets = {
        task: destination / timestamped_filename(filename, timestamp)
        for task, filename in TASK_CHECKPOINT_FILENAMES.items()
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Task checkpoint already exists: {names}")
    for task, target in targets.items():
        state = build_task_checkpoint_state(
            checkpoint,
            task,
            source_checkpoint=source,
        )
        _atomic_torch_save(state, target)
    return targets


def _atomic_torch_save(state: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, target)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _mean(values) -> float:
    return float(sum(values) / len(values)) if values else 0.0
