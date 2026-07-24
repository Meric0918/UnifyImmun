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

from .config import ExperimentConfig
from .encoders import OnlineEncoderBundle
from .fgm import FGMStats, ModuleFGM
from .metrics import (
    BinaryMetrics,
    compute_binary_metrics,
    joint_score,
    prefixed_metrics,
)
from .model import Task, UnifiedBindingModel
from .tracking import SwanLabTracker

Stage = Literal["warmup", "stage1b", "complete"]


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
        self.best_joint_score = -float("inf")
        self.best_val_metrics: dict[str, dict[str, float | int]] = {}
        self.no_improve_rounds = 0
        self.global_step = 0
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)

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
                    self.save_checkpoint(self.run_dir / "last.pt")

            self.stage = "stage1b"
            self.save_checkpoint(self.run_dir / "warmup_last.pt")
            self.save_checkpoint(self.run_dir / "last.pt")

        start_round = self.current_round + 1
        for round_id in range(start_round, self.config.training.max_rounds + 1):
            train_results: dict[Task, EpochResult] = {}
            for task in ("phla", "ptcr"):
                train_results[task] = self.train_epoch(
                    task,
                    self.train_loaders[task],
                    phase_name=f"round_{round_id}",
                    use_fgm=True,
                )

            val_metrics = {
                task: self.evaluate(task, self.val_loaders[task])
                for task in ("phla", "ptcr")
            }
            score = joint_score(val_metrics["phla"], val_metrics["ptcr"])
            improved = (
                score
                > self.best_joint_score
                + self.config.training.early_stopping_min_delta
            )
            if improved:
                self.best_joint_score = score
                self.best_val_metrics = {
                    task: val_metrics[task].to_dict() for task in ("phla", "ptcr")
                }
                self.no_improve_rounds = 0
            else:
                self.no_improve_rounds += 1
            self.current_round = round_id

            log_values: dict[str, Any] = {
                "stage": "stage1b",
                "round": round_id,
                "joint_score": score,
                "best_joint_score": self.best_joint_score,
                "no_improve_rounds": self.no_improve_rounds,
                "round/phla_optimizer_steps": train_results["phla"].optimizer_steps,
                "round/ptcr_optimizer_steps": train_results["ptcr"].optimizer_steps,
                "round/phla_to_ptcr_step_ratio": (
                    train_results["phla"].optimizer_steps
                    / max(1, train_results["ptcr"].optimizer_steps)
                ),
            }
            for task in ("phla", "ptcr"):
                log_values.update(
                    train_results[task].to_log_dict(f"round/{task}/train")
                )
                log_values.update(prefixed_metrics(f"round/{task}/val", val_metrics[task]))
            self.tracker.log(log_values, step=self.global_step)
            print(
                f"[Stage 1B][seed={self.config.training.seed}]"
                f"[round={round_id}/{self.config.training.max_rounds}]"
                f"[step={self.global_step}] "
                f"joint={score:.4f} best={self.best_joint_score:.4f} "
                f"improved={'yes' if improved else 'no'} "
                f"patience={self.no_improve_rounds}/"
                f"{self.config.training.early_stopping_patience}",
                flush=True,
            )
            for task in ("phla", "ptcr"):
                task_name = "pHLA" if task == "phla" else "pTCR"
                print(
                    f"  {task_name} train: "
                    f"{_metric_summary(train_results[task].metrics)} | "
                    f"val: {_metric_summary(val_metrics[task])}",
                    flush=True,
                )

            self.save_checkpoint(self.run_dir / "last.pt")
            if improved:
                self.save_checkpoint(self.run_dir / "best_joint.pt")
            if (
                self.no_improve_rounds
                >= self.config.training.early_stopping_patience
            ):
                print(
                    f"[Early stopping][seed={self.config.training.seed}] "
                    f"round={round_id} best_joint={self.best_joint_score:.4f}",
                    flush=True,
                )
                break

        self.stage = "complete"
        self.save_checkpoint(self.run_dir / "last.pt")
        print(
            f"[Complete][seed={self.config.training.seed}] "
            f"round={self.current_round} step={self.global_step} "
            f"best_joint={self.best_joint_score:.4f}",
            flush=True,
        )
        return self.summary()

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "stage": self.stage,
            "warmup_completed": dict(self.warmup_completed),
            "current_round": self.current_round,
            "best_joint_score": self.best_joint_score,
            "best_val_metrics": self.best_val_metrics,
            "no_improve_rounds": self.no_improve_rounds,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "data_filter_stats": self.data_filter_stats,
            "cache_metadata": self.cache_metadata,
            "input_mode": self.input_mode,
            "swanlab_run_id": self.tracker.run_id,
            "dataloader_rng_state": {
                task: loader.generator.get_state()
                for task, loader in self.train_loaders.items()
                if getattr(loader, "generator", None) is not None
            },
            "rng_state": _capture_rng_state(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(self.checkpoint_state(), temporary)
        os.replace(temporary, target)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = load_training_checkpoint(path, map_location="cpu")
        if checkpoint.get("format_version") != 1:
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
        self.best_joint_score = float(checkpoint["best_joint_score"])
        self.best_val_metrics = dict(checkpoint.get("best_val_metrics", {}))
        self.no_improve_rounds = int(checkpoint["no_improve_rounds"])
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
            "best_joint_score": self.best_joint_score,
            "best_val_metrics": self.best_val_metrics,
            "data_filter_stats": self.data_filter_stats,
            "global_step": self.global_step,
            "run_dir": str(self.run_dir),
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
