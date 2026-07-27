"""Progressive stage-2 PLM fine-tuning from chapter 15."""

from __future__ import annotations

import gc
import math
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from .artifacts import (
    create_artifact_timestamp,
    timestamped_filename,
    validate_artifact_timestamp,
)
from .config import ExperimentConfig
from .finetuning import FineTuningStage, ProgressiveFineTuningModel
from .metrics import (
    BinaryMetrics,
    compute_binary_metrics,
    prefixed_metrics,
    task_score,
)
from .model import Task, extract_task_state_dict
from .tracking import SwanLabTracker
from .trainer import (
    EpochResult,
    _atomic_torch_save,
    _capture_rng_state,
    _restore_rng_state,
    build_scheduler,
    load_training_checkpoint,
    resolve_device,
)

STAGE2_TASK_CHECKPOINT_FILENAMES: dict[Task, str] = {
    "phla": "stage2_best_phla.pt",
    "ptcr": "stage2_best_ptcr.pt",
}


def _load_progressive_model_state(
    model: ProgressiveFineTuningModel,
    state_dict: Mapping[str, Any],
) -> None:
    """Strictly load learned tensors while tolerating TE runtime metadata."""

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in missing if not key.endswith("._extra_state")]
    unexpected = [key for key in unexpected if not key.endswith("._extra_state")]
    if missing or unexpected:
        raise RuntimeError(
            "Stage-2 model state_dict mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _console_log(message: str) -> None:
    print(
        f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] {message}",
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _format_parameter_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def _metric_summary(metrics: BinaryMetrics) -> str:
    return (
        f"loss={metrics.loss:.4f} "
        f"AUROC={metrics.auroc:.4f} "
        f"AUPR={metrics.aupr:.4f} "
        f"ACC={metrics.accuracy:.4f} "
        f"MCC={metrics.mcc:.4f} "
        f"F1={metrics.f1:.4f}"
    )


def build_stage2_optimizer(
    model: ProgressiveFineTuningModel,
    config: ExperimentConfig,
    stage: FineTuningStage,
) -> torch.optim.Optimizer:
    """Build discriminative parameter groups for one unfreezing stage."""

    values = config.stage2
    groups: list[dict[str, Any]] = []

    def add_group(name: str, parameters, lr: float) -> None:
        parameters = list(parameters)
        if parameters:
            groups.append({"name": name, "params": parameters, "lr": lr})

    peptide_blocks = model.peptide_encoder.transformer_blocks()
    add_group(
        "peptide_plm_last",
        peptide_blocks[-1].parameters(),
        values.peptide_top_lr,
    )
    if model.peptide_unfrozen_layers > 1:
        add_group(
            "peptide_plm_lower",
            (
                parameter
                for block in peptide_blocks[-model.peptide_unfrozen_layers : -1]
                for parameter in block.parameters()
            ),
            values.peptide_lower_lr,
        )

    if stage == "stage2b":
        hla_blocks = model.hla_encoder.transformer_blocks()
        add_group(
            "hla_plm_top",
            (
                parameter
                for block in hla_blocks[-model.hla_unfrozen_layers :]
                for parameter in block.parameters()
            ),
            values.hla_top_lr,
        )
        tcr_blocks = model.tcr_encoder.transformer_blocks()
        add_group(
            "tcr_plm_last",
            tcr_blocks[-1].parameters(),
            values.tcr_top_lr,
        )
        if model.tcr_unfrozen_layers > 1:
            add_group(
                "tcr_plm_lower",
                (
                    parameter
                    for block in tcr_blocks[-model.tcr_unfrozen_layers : -1]
                    for parameter in block.parameters()
                ),
                values.tcr_lower_lr,
            )

    binding = model.binding_model
    add_group(
        "adapters",
        (
            parameter
            for module in (
                binding.peptide_adapter,
                binding.hla_adapter,
                binding.tcr_adapter,
            )
            for parameter in module.parameters()
        ),
        values.adapter_lr,
    )
    add_group(
        "cross_attention",
        (
            parameter
            for module in (
                binding.phla_cross_attention,
                binding.ptcr_cross_attention,
            )
            for parameter in module.parameters()
        ),
        values.cross_attention_lr,
    )
    add_group(
        "pooling",
        (
            parameter
            for module in (binding.phla_pooling, binding.ptcr_pooling)
            for parameter in module.parameters()
        ),
        values.pooling_lr,
    )
    add_group(
        "classifiers",
        (
            parameter
            for module in (binding.phla_classifier, binding.ptcr_classifier)
            for parameter in module.parameters()
        ),
        values.classifier_lr,
    )
    return torch.optim.AdamW(groups, weight_decay=values.weight_decay)


class Stage2Trainer:
    """Run stage 2A and 2B as pHLA/pTCR alternating rounds."""

    def __init__(
        self,
        model: ProgressiveFineTuningModel,
        config: ExperimentConfig,
        train_loaders: Mapping[Task, Any],
        val_loaders: Mapping[Task, Any],
        tracker: SwanLabTracker,
        *,
        source_stage1_checkpoint: str | Path | None = None,
        run_timestamp: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.train_loaders = dict(train_loaders)
        self.val_loaders = dict(val_loaders)
        self.tracker = tracker
        self.device = resolve_device(config.training.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.stage: FineTuningStage | str = "stage2a"
        self.stage2a_round = 0
        self.stage2b_round = 0
        self.best_joint_scores = {
            "stage2a": -float("inf"),
            "stage2b": -float("inf"),
        }
        self.best_val_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
        self.best_task_scores: dict[Task, float] = {
            "phla": -float("inf"),
            "ptcr": -float("inf"),
        }
        self.best_task_val_metrics: dict[Task, dict[str, float | int]] = {}
        self.no_improve_rounds = 0
        self.global_step = 0
        self.source_stage1_checkpoint = (
            str(Path(source_stage1_checkpoint).resolve())
            if source_stage1_checkpoint
            else None
        )
        self.data_filter_stats = {}
        for split_name, loaders in (
            ("train", self.train_loaders),
            ("val", self.val_loaders),
        ):
            for task, loader in loaders.items():
                stats = getattr(getattr(loader, "dataset", None), "filter_stats", None)
                if stats:
                    self.data_filter_stats[f"{task}/{split_name}"] = dict(stats)

        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint_path = self.run_dir / timestamped_filename(
            "stage2_last.pt",
            self.run_timestamp,
        )
        self.stage2a_best_path = self.run_dir / timestamped_filename(
            "stage2a_best_joint.pt",
            self.run_timestamp,
        )
        self.stage2b_best_path = self.run_dir / timestamped_filename(
            "stage2b_best_joint.pt",
            self.run_timestamp,
        )
        self.task_checkpoint_paths: dict[Task, Path] = {
            task: self.run_dir / timestamped_filename(filename, self.run_timestamp)
            for task, filename in STAGE2_TASK_CHECKPOINT_FILENAMES.items()
        }
        self.optimizer_stage: FineTuningStage = "stage2a"
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LambdaLR
        self._build_optimization("stage2a")
        scaler_enabled = (
            self.device.type == "cuda"
            and config.stage2.mixed_precision == "fp16"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

        def loader_summary(task: str, loader: Any) -> str:
            dataset = getattr(loader, "dataset", None)
            sample_text = f"{len(dataset):,}" if dataset is not None else "unknown"
            return f"{task}={sample_text} samples/{len(loader):,} batches"

        train_sizes = ", ".join(
            loader_summary(task, loader)
            for task, loader in self.train_loaders.items()
        )
        val_sizes = ", ".join(
            loader_summary(task, loader)
            for task, loader in self.val_loaders.items()
        )
        _console_log(
            "[STAGE2 INIT] "
            f"fold={config.training.fold} seed={config.training.seed} "
            f"device={self.device} precision={config.stage2.mixed_precision} "
            f"micro_batch={config.stage2.micro_batch_size} "
            f"eval_batch={config.stage2.eval_batch_size} "
            f"accumulation={config.stage2.gradient_accumulation_steps} "
            f"effective_batch="
            f"{config.stage2.micro_batch_size * config.stage2.gradient_accumulation_steps}"
        )
        _console_log(f"[DATA][train] {train_sizes}")
        _console_log(f"[DATA][val] {val_sizes}")
        _console_log(
            "[CACHE][STAGE2A][HLA] enabled=true storage=device "
            "policy=one-residue-embedding-per-unique-HLA; "
            "automatically cleared before Stage2B"
        )

    def _build_optimization(self, stage: FineTuningStage) -> None:
        self.optimizer_stage = stage
        self.optimizer = build_stage2_optimizer(self.model, self.config, stage)
        rounds = (
            self.config.stage2.stage2a_rounds
            if stage == "stage2a"
            else self.config.stage2.stage2b_max_rounds
        )
        accumulation = self.config.stage2.gradient_accumulation_steps
        steps_per_round = sum(
            math.ceil(len(loader) / accumulation)
            for loader in self.train_loaders.values()
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=max(1, steps_per_round * rounds),
            warmup_ratio=self.config.stage2.scheduler_warmup_ratio,
        )
        group_summary = ", ".join(
            f"{group.get('name', index)}={group['lr']:.2e}"
            for index, group in enumerate(self.optimizer.param_groups)
        )
        total_steps = max(1, steps_per_round * rounds)
        warmup_steps = max(
            1,
            round(total_steps * self.config.stage2.scheduler_warmup_ratio),
        )
        _console_log(
            f"[OPTIMIZER][{stage.upper()}] "
            f"steps_per_round={steps_per_round:,} "
            f"planned_rounds={rounds} total_steps={total_steps:,} "
            f"warmup_steps={warmup_steps:,} lr_groups=({group_summary})"
        )

    def _gpu_memory(self) -> tuple[str, dict[str, float]]:
        if self.device.type != "cuda":
            return "gpu_memory=n/a", {}
        gib = 1024**3
        allocated = torch.cuda.memory_allocated(self.device) / gib
        reserved = torch.cuda.memory_reserved(self.device) / gib
        peak = torch.cuda.max_memory_allocated(self.device) / gib
        return (
            f"gpu={allocated:.2f}GiB/{reserved:.2f}GiB peak={peak:.2f}GiB",
            {
                "gpu/memory_allocated_gib": allocated,
                "gpu/memory_reserved_gib": reserved,
                "gpu/max_memory_allocated_gib": peak,
            },
        )

    def _autocast(self):
        precision = self.config.stage2.mixed_precision
        if self.device.type != "cuda" or precision == "none":
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def train_epoch(
        self,
        task: Task,
        loader,
        *,
        stage: FineTuningStage,
        round_id: int,
    ) -> EpochResult:
        self.model.set_trainable_task(task, stage)
        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        accumulation = self.config.stage2.gradient_accumulation_steps
        total_batches = len(loader)
        total_optimizer_steps = math.ceil(total_batches / accumulation)
        trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        loss_sum = 0.0
        samples = 0
        optimizer_steps = 0
        gradient_norms: list[float] = []
        started_at = time.perf_counter()
        _console_log(
            f"[TRAIN START][{stage.upper()}][{task.upper()}] "
            f"round={round_id} batches={total_batches:,} "
            f"optimizer_steps={total_optimizer_steps:,} "
            f"accumulation={accumulation} "
            f"trainable_params={_format_parameter_count(trainable_count)}"
        )

        for batch_index, batch in enumerate(loader, start=1):
            window_start = ((batch_index - 1) // accumulation) * accumulation + 1
            window_size = min(accumulation, total_batches - window_start + 1)
            if batch_index == window_start:
                self.optimizer.zero_grad(set_to_none=True)

            batch_labels = batch["labels"].to(
                self.device,
                dtype=torch.long,
                non_blocking=self.device.type == "cuda",
            )
            with self._autocast():
                output = self.model(
                    task,
                    batch["peptide_sequences"],
                    batch["receptor_sequences"],
                )
                loss = self.criterion(output.logits, batch_labels)
            self.scaler.scale(loss / window_size).backward()

            batch_samples = int(batch_labels.shape[0])
            batch_loss = float(loss.detach().cpu())
            samples += batch_samples
            loss_sum += batch_loss * batch_samples
            probabilities.append(
                torch.softmax(output.logits.detach(), dim=1)[:, 1]
                .float()
                .cpu()
                .numpy()
            )
            labels.append(batch_labels.detach().cpu().numpy())

            end_of_window = batch_index - window_start + 1 == window_size
            if end_of_window:
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    self.config.stage2.gradient_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1
                optimizer_steps += 1
                gradient_norms.append(float(gradient_norm.detach().cpu()))

                if (
                    optimizer_steps
                    % self.config.stage2.log_every_optimizer_steps
                    == 0
                    or batch_index == total_batches
                ):
                    learning_rates = {
                        f"learning_rate/{group.get('name', index)}": group["lr"]
                        for index, group in enumerate(self.optimizer.param_groups)
                    }
                    learning_rate_summary = ", ".join(
                        f"{group.get('name', index)}={group['lr']:.2e}"
                        for index, group in enumerate(self.optimizer.param_groups)
                    )
                    elapsed = time.perf_counter() - started_at
                    progress = batch_index / max(1, total_batches)
                    eta = (
                        elapsed * (total_batches - batch_index) / batch_index
                        if batch_index
                        else 0.0
                    )
                    samples_per_second = samples / max(elapsed, 1e-9)
                    gpu_summary, gpu_values = self._gpu_memory()
                    running_loss = loss_sum / samples
                    self.tracker.log(
                        {
                            "stage": stage,
                            "round": round_id,
                            "task": task,
                            f"{stage}/{task}/batch": batch_index,
                            f"{stage}/{task}/optimizer_step": optimizer_steps,
                            f"{stage}/{task}/progress_percent": progress * 100.0,
                            f"{stage}/{task}/batch_loss": batch_loss,
                            f"{stage}/{task}/running_loss": running_loss,
                            f"{stage}/{task}/gradient_norm": gradient_norms[-1],
                            f"{stage}/{task}/samples_per_second": (
                                samples_per_second
                            ),
                            f"{stage}/{task}/elapsed_seconds": elapsed,
                            f"{stage}/{task}/eta_seconds": eta,
                            f"{stage}/{task}/fgm_enabled": 0,
                            **learning_rates,
                            **gpu_values,
                        },
                        step=self.global_step,
                    )
                    _console_log(
                        f"[TRAIN PROGRESS][{stage.upper()}][{task.upper()}] "
                        f"round={round_id} "
                        f"opt={optimizer_steps:,}/{total_optimizer_steps:,} "
                        f"batch={batch_index:,}/{total_batches:,} "
                        f"progress={progress * 100:.1f}% "
                        f"samples={samples:,} "
                        f"loss={batch_loss:.4f} avg_loss={running_loss:.4f} "
                        f"grad_norm={gradient_norms[-1]:.4f} "
                        f"speed={samples_per_second:.2f} samples/s "
                        f"elapsed={_format_duration(elapsed)} "
                        f"eta={_format_duration(eta)} "
                        f"{gpu_summary} lr=({learning_rate_summary})"
                    )

        if samples == 0:
            raise RuntimeError(f"{task} training loader is empty")
        average_loss = loss_sum / samples
        metrics = compute_binary_metrics(
            np.concatenate(labels),
            np.concatenate(probabilities),
            loss=average_loss,
            threshold=self.config.training.threshold,
        )
        elapsed = time.perf_counter() - started_at
        _console_log(
            f"[TRAIN END][{stage.upper()}][{task.upper()}] "
            f"round={round_id} samples={samples:,} "
            f"optimizer_steps={optimizer_steps:,} "
            f"elapsed={_format_duration(elapsed)} {_metric_summary(metrics)}"
        )
        if stage == "stage2a" and task == "phla":
            cache = self.model.stage2a_hla_cache_stats()
            _console_log(
                "[CACHE][STAGE2A][HLA] "
                f"entries={cache['entries']:,} requests={cache['requests']:,} "
                f"hits={cache['hits']:,} misses={cache['misses']:,}"
            )
        return EpochResult(
            metrics=metrics,
            clean_loss=average_loss,
            adversarial_loss=0.0,
            total_loss=average_loss,
            optimizer_steps=optimizer_steps,
            samples=samples,
            attacked_parameter_count=0.0,
            fgm_mean_gradient_norm=0.0,
            fgm_max_gradient_norm=0.0,
            model_gradient_norm=(
                float(np.mean(gradient_norms)) if gradient_norms else 0.0
            ),
        )

    @torch.inference_mode()
    def evaluate(
        self,
        task: Task,
        loader,
        *,
        stage: FineTuningStage | None = None,
        round_id: int | None = None,
    ) -> BinaryMetrics:
        self.model.set_evaluation()
        stage_name = stage.upper() if stage else "STAGE2"
        round_text = str(round_id) if round_id is not None else "n/a"
        total_batches = len(loader)
        log_every_batches = max(
            1,
            self.config.stage2.gradient_accumulation_steps
            * self.config.stage2.log_every_optimizer_steps,
        )
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        loss_sum = 0.0
        samples = 0
        started_at = time.perf_counter()
        _console_log(
            f"[VAL START][{stage_name}][{task.upper()}] "
            f"round={round_text} batches={total_batches:,}"
        )
        for batch_index, batch in enumerate(loader, start=1):
            batch_labels = batch["labels"].to(
                self.device,
                dtype=torch.long,
                non_blocking=self.device.type == "cuda",
            )
            with self._autocast():
                output = self.model(
                    task,
                    batch["peptide_sequences"],
                    batch["receptor_sequences"],
                )
                loss = self.criterion(output.logits, batch_labels)
            batch_samples = int(batch_labels.shape[0])
            samples += batch_samples
            loss_sum += float(loss.detach().cpu()) * batch_samples
            probabilities.append(
                torch.softmax(output.logits, dim=1)[:, 1].float().cpu().numpy()
            )
            labels.append(batch_labels.cpu().numpy())
            if (
                batch_index % log_every_batches == 0
                or batch_index == total_batches
            ):
                elapsed = time.perf_counter() - started_at
                progress = batch_index / max(1, total_batches)
                eta = (
                    elapsed * (total_batches - batch_index) / batch_index
                    if batch_index
                    else 0.0
                )
                samples_per_second = samples / max(elapsed, 1e-9)
                running_loss = loss_sum / samples
                gpu_summary, gpu_values = self._gpu_memory()
                metric_prefix = f"{stage or 'stage2'}/{task}/val_progress"
                self.tracker.log(
                    {
                        "stage": stage or "stage2",
                        "round": round_id or 0,
                        "task": task,
                        f"{metric_prefix}/batch": batch_index,
                        f"{metric_prefix}/progress_percent": progress * 100.0,
                        f"{metric_prefix}/running_loss": running_loss,
                        f"{metric_prefix}/samples_per_second": samples_per_second,
                        f"{metric_prefix}/elapsed_seconds": elapsed,
                        f"{metric_prefix}/eta_seconds": eta,
                        **gpu_values,
                    },
                    step=self.global_step,
                )
                _console_log(
                    f"[VAL PROGRESS][{stage_name}][{task.upper()}] "
                    f"round={round_text} "
                    f"batch={batch_index:,}/{total_batches:,} "
                    f"progress={progress * 100:.1f}% samples={samples:,} "
                    f"avg_loss={running_loss:.4f} "
                    f"speed={samples_per_second:.2f} samples/s "
                    f"elapsed={_format_duration(elapsed)} "
                    f"eta={_format_duration(eta)} {gpu_summary}"
                )
        if samples == 0:
            raise RuntimeError(f"{task} validation loader is empty")
        metrics = compute_binary_metrics(
            np.concatenate(labels),
            np.concatenate(probabilities),
            loss=loss_sum / samples,
            threshold=self.config.training.threshold,
        )
        elapsed = time.perf_counter() - started_at
        _console_log(
            f"[VAL END][{stage_name}][{task.upper()}] "
            f"round={round_text} samples={samples:,} "
            f"elapsed={_format_duration(elapsed)} {_metric_summary(metrics)}"
        )
        return metrics

    def _run_round(
        self,
        stage: FineTuningStage,
        round_id: int,
    ) -> tuple[
        dict[Task, EpochResult],
        dict[Task, BinaryMetrics],
        float,
    ]:
        round_started_at = time.perf_counter()
        _console_log(
            f"[ROUND START][{stage.upper()}] round={round_id} "
            "order=train:pHLA->pTCR, val:pHLA->pTCR"
        )
        train_results: dict[Task, EpochResult] = {}
        for task in ("phla", "ptcr"):
            train_results[task] = self.train_epoch(
                task,
                self.train_loaders[task],
                stage=stage,
                round_id=round_id,
            )
        val_metrics: dict[Task, BinaryMetrics] = {}
        for task in ("phla", "ptcr"):
            val_metrics[task] = self.evaluate(
                task,
                self.val_loaders[task],
                stage=stage,
                round_id=round_id,
            )
        joint_score = sum(
            metric
            for task in ("phla", "ptcr")
            for metric in (val_metrics[task].auroc, val_metrics[task].aupr)
        ) / 4.0
        _console_log(
            f"[ROUND METRICS][{stage.upper()}] round={round_id} "
            f"joint={joint_score:.4f} "
            f"pHLA=({_metric_summary(val_metrics['phla'])}) "
            f"pTCR=({_metric_summary(val_metrics['ptcr'])}) "
            f"elapsed={_format_duration(time.perf_counter() - round_started_at)}"
        )
        return train_results, val_metrics, joint_score

    def _record_round(
        self,
        stage: FineTuningStage,
        round_id: int,
        train_results: Mapping[Task, EpochResult],
        val_metrics: Mapping[Task, BinaryMetrics],
        joint_score: float,
        improved: bool,
        task_improved: Mapping[Task, bool],
    ) -> None:
        values: dict[str, Any] = {
            "stage": stage,
            "round": round_id,
            f"{stage}/joint_score": joint_score,
            f"{stage}/best_joint_score": self.best_joint_scores[stage],
            f"{stage}/no_improve_rounds": self.no_improve_rounds,
        }
        for task in ("phla", "ptcr"):
            values[f"{stage}/{task}/score"] = task_score(val_metrics[task])
            values[f"{stage}/{task}/best_score"] = self.best_task_scores[task]
            values[f"{stage}/{task}/improved"] = int(task_improved[task])
            values.update(
                train_results[task].to_log_dict(f"{stage}/{task}/train")
            )
            values.update(
                prefixed_metrics(f"{stage}/{task}/val", val_metrics[task])
            )
        self.tracker.log(values, step=self.global_step)
        _console_log(
            f"[{stage.upper()}][seed={self.config.training.seed}]"
            f"[round={round_id}]"
            f"[step={self.global_step}] "
            f"joint={joint_score:.4f} "
            f"best={self.best_joint_scores[stage]:.4f} "
            f"improved={'yes' if improved else 'no'} "
            f"patience={self.no_improve_rounds} "
            f"pHLA_score={task_score(val_metrics['phla']):.4f}"
            f"{'*' if task_improved['phla'] else ''} "
            f"pTCR_score={task_score(val_metrics['ptcr']):.4f}"
            f"{'*' if task_improved['ptcr'] else ''}"
        )

    def _update_task_bests(
        self,
        val_metrics: Mapping[Task, BinaryMetrics],
    ) -> dict[Task, bool]:
        improved: dict[Task, bool] = {}
        for task in ("phla", "ptcr"):
            score = task_score(val_metrics[task])
            task_improved = (
                score
                > self.best_task_scores[task]
                + self.config.stage2.early_stopping_min_delta
            )
            improved[task] = task_improved
            if task_improved:
                self.best_task_scores[task] = score
                self.best_task_val_metrics[task] = val_metrics[task].to_dict()
        return improved

    def save_best_task_checkpoint(self, task: Task) -> None:
        """Save the task-specific PLMs and branch, like stage-1 task weights."""

        path = self.task_checkpoint_paths[task]
        started_at = time.perf_counter()
        _console_log(
            f"[CHECKPOINT SAVE START][{task.upper()}] "
            f"score={self.best_task_scores[task]:.4f} path={path}"
        )
        state = build_stage2_task_checkpoint_state(
            {
                "model": self.model.state_dict(),
                "checkpoint_type": "stage2_training",
                "config": self.config.to_dict(),
                "encoder_metadata": self.model.encoder_metadata(),
                "stage": self.stage,
                "stage2a_round": self.stage2a_round,
                "stage2b_round": self.stage2b_round,
                "global_step": self.global_step,
                "run_timestamp": self.run_timestamp,
                "best_task_scores": self.best_task_scores,
                "best_task_val_metrics": self.best_task_val_metrics,
                "source_stage1_checkpoint": self.source_stage1_checkpoint,
                "input_mode": "online_stage2",
                "fgm_enabled": False,
            },
            task,
            source_checkpoint=self.last_checkpoint_path,
        )
        _atomic_torch_save(state, path)
        size_gib = path.stat().st_size / 1024**3
        _console_log(
            f"[CHECKPOINT SAVE END][{task.upper()}] "
            f"size={size_gib:.2f}GiB "
            f"elapsed={_format_duration(time.perf_counter() - started_at)} "
            f"path={path}"
        )

    def fit(self) -> dict[str, Any]:
        if self.stage == "complete":
            for task in ("phla", "ptcr"):
                if not self.task_checkpoint_paths[task].is_file():
                    self.save_best_task_checkpoint(task)
            return self.summary()

        if self.stage == "stage2a":
            for round_id in range(
                self.stage2a_round + 1,
                self.config.stage2.stage2a_rounds + 1,
            ):
                train_results, val_metrics, joint_score = self._run_round(
                    "stage2a",
                    round_id,
                )
                improved = (
                    joint_score
                    > self.best_joint_scores["stage2a"]
                    + self.config.stage2.early_stopping_min_delta
                )
                if improved:
                    self.best_joint_scores["stage2a"] = joint_score
                    self.best_val_metrics["stage2a"] = {
                        task: val_metrics[task].to_dict()
                        for task in ("phla", "ptcr")
                    }
                task_improved = self._update_task_bests(val_metrics)
                self.stage2a_round = round_id
                self.no_improve_rounds = 0 if improved else self.no_improve_rounds + 1
                self._record_round(
                    "stage2a",
                    round_id,
                    train_results,
                    val_metrics,
                    joint_score,
                    improved,
                    task_improved,
                )
                for task in ("phla", "ptcr"):
                    if task_improved[task]:
                        self.save_best_task_checkpoint(task)
                if improved:
                    self.save_checkpoint(self.stage2a_best_path)
                self.save_checkpoint(self.last_checkpoint_path)

            _console_log(
                "[STAGE TRANSITION] Stage 2A finished; "
                f"loading best joint checkpoint {self.stage2a_best_path}"
            )
            transition_started_at = time.perf_counter()
            best_stage2a = load_stage2_checkpoint(
                self.stage2a_best_path,
                map_location="cpu",
            )
            _load_progressive_model_state(self.model, best_stage2a["model"])
            del best_stage2a
            gc.collect()
            _console_log(
                "[STAGE TRANSITION] Best Stage 2A weights loaded "
                f"in {_format_duration(time.perf_counter() - transition_started_at)}; "
                "entering Stage 2B"
            )
            self.stage = "stage2b"
            self.stage2b_round = 0
            self.no_improve_rounds = 0
            self._build_optimization("stage2b")
            self.save_checkpoint(self.last_checkpoint_path)

        for round_id in range(
            self.stage2b_round + 1,
            self.config.stage2.stage2b_max_rounds + 1,
        ):
            train_results, val_metrics, joint_score = self._run_round(
                "stage2b",
                round_id,
            )
            improved = (
                joint_score
                > self.best_joint_scores["stage2b"]
                + self.config.stage2.early_stopping_min_delta
            )
            if improved:
                self.best_joint_scores["stage2b"] = joint_score
                self.best_val_metrics["stage2b"] = {
                    task: val_metrics[task].to_dict()
                    for task in ("phla", "ptcr")
                }
                self.no_improve_rounds = 0
            else:
                self.no_improve_rounds += 1
            task_improved = self._update_task_bests(val_metrics)
            self.stage2b_round = round_id
            self._record_round(
                "stage2b",
                round_id,
                train_results,
                val_metrics,
                joint_score,
                improved,
                task_improved,
            )
            for task in ("phla", "ptcr"):
                if task_improved[task]:
                    self.save_best_task_checkpoint(task)
            if improved:
                self.save_checkpoint(self.stage2b_best_path)
            self.save_checkpoint(self.last_checkpoint_path)
            if (
                round_id >= self.config.stage2.stage2b_min_rounds
                and self.no_improve_rounds
                >= self.config.stage2.early_stopping_patience
            ):
                _console_log(
                    f"[Stage 2B early stopping]"
                    f"[seed={self.config.training.seed}] "
                    f"round={round_id} "
                    f"best_joint={self.best_joint_scores['stage2b']:.4f}"
                )
                break

        self.stage = "complete"
        self.save_checkpoint(self.last_checkpoint_path)
        _console_log(
            f"[Stage 2 complete][seed={self.config.training.seed}] "
            f"round={self.stage2b_round} "
            f"step={self.global_step} "
            f"best_joint={self.best_joint_scores['stage2b']:.4f}"
        )
        return self.summary()

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a full checkpoint, including all frozen PLM parameters."""

        return {
            "format_version": 1,
            "checkpoint_type": "stage2_training",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_stage": self.optimizer_stage,
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "stage": self.stage,
            "stage2a_round": self.stage2a_round,
            "stage2b_round": self.stage2b_round,
            "best_joint_scores": dict(self.best_joint_scores),
            "best_val_metrics": self.best_val_metrics,
            "best_task_scores": dict(self.best_task_scores),
            "best_task_val_metrics": self.best_task_val_metrics,
            "no_improve_rounds": self.no_improve_rounds,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "encoder_metadata": self.model.encoder_metadata(),
            "source_stage1_checkpoint": self.source_stage1_checkpoint,
            "data_filter_stats": self.data_filter_stats,
            "fgm_enabled": False,
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
        path = Path(path)
        started_at = time.perf_counter()
        _console_log(
            f"[CHECKPOINT SAVE START][JOINT] stage={self.stage} "
            f"round2a={self.stage2a_round} round2b={self.stage2b_round} "
            f"step={self.global_step} path={path}"
        )
        _atomic_torch_save(self.checkpoint_state(), path)
        size_gib = path.stat().st_size / 1024**3
        _console_log(
            f"[CHECKPOINT SAVE END][JOINT] size={size_gib:.2f}GiB "
            f"elapsed={_format_duration(time.perf_counter() - started_at)} "
            f"path={path}"
        )

    def load_checkpoint(self, path: str | Path) -> None:
        started_at = time.perf_counter()
        _console_log(f"[CHECKPOINT LOAD START] path={path}")
        checkpoint = load_stage2_checkpoint(path, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_training = checkpoint_config.get("training", {})
        for key in ("fold", "seed"):
            current_value = getattr(self.config.training, key)
            if checkpoint_training.get(key) != current_value:
                raise ValueError(
                    f"Checkpoint {key}={checkpoint_training.get(key)} does not "
                    f"match current {key}={current_value}"
                )
        current_config = self.config.to_dict()
        if checkpoint_config.get("model") != current_config["model"]:
            raise ValueError("Checkpoint model configuration does not match")
        saved_stage2_config = dict(checkpoint_config.get("stage2", {}))
        current_stage2_config = dict(current_config["stage2"])
        # Logging and evaluation batching do not affect optimizer/scheduler
        # state or training sample order, so they are safe runtime changes
        # when resuming an older checkpoint.
        for runtime_key in ("log_every_optimizer_steps", "eval_batch_size"):
            saved_stage2_config.pop(runtime_key, None)
            current_stage2_config.pop(runtime_key, None)
        if saved_stage2_config != current_stage2_config:
            raise ValueError("Checkpoint stage2 configuration does not match")
        saved_metadata = checkpoint.get("encoder_metadata", {})
        current_metadata = self.model.encoder_metadata()
        for entity in ("peptide", "hla", "tcr"):
            saved_fingerprint = saved_metadata.get(entity, {}).get(
                "checkpoint_fingerprint"
            )
            current_fingerprint = current_metadata[entity][
                "checkpoint_fingerprint"
            ]
            if saved_fingerprint and saved_fingerprint != current_fingerprint:
                raise ValueError(
                    f"Checkpoint base-PLM fingerprint mismatch for {entity}"
                )

        _load_progressive_model_state(self.model, checkpoint["model"])
        self.stage = checkpoint["stage"]
        self.stage2a_round = int(checkpoint["stage2a_round"])
        self.stage2b_round = int(checkpoint["stage2b_round"])
        self.best_joint_scores = {
            key: float(checkpoint["best_joint_scores"][key])
            for key in ("stage2a", "stage2b")
        }
        self.best_val_metrics = dict(checkpoint.get("best_val_metrics", {}))
        saved_task_scores = checkpoint.get("best_task_scores", {})
        saved_task_metrics = checkpoint.get("best_task_val_metrics", {})
        if not saved_task_metrics:
            stage_metrics = checkpoint.get("best_val_metrics", {})
            saved_task_metrics = stage_metrics.get("stage2b") or stage_metrics.get(
                "stage2a", {}
            )
        self.best_task_val_metrics = dict(saved_task_metrics)
        self.best_task_scores = {
            task: float(
                saved_task_scores.get(
                    task,
                    task_score(saved_task_metrics[task])
                    if task in saved_task_metrics
                    else -float("inf"),
                )
            )
            for task in ("phla", "ptcr")
        }
        self.no_improve_rounds = int(checkpoint["no_improve_rounds"])
        self.global_step = int(checkpoint["global_step"])
        self.source_stage1_checkpoint = checkpoint.get(
            "source_stage1_checkpoint"
        )
        optimizer_stage = checkpoint.get(
            "optimizer_stage",
            "stage2a" if self.stage == "stage2a" else "stage2b",
        )
        self._build_optimization(optimizer_stage)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        for task, generator_state in checkpoint.get(
            "dataloader_rng_state", {}
        ).items():
            loader = self.train_loaders.get(task)
            generator = getattr(loader, "generator", None)
            if generator is not None:
                generator.set_state(generator_state)
        _restore_rng_state(checkpoint.get("rng_state"))
        _console_log(
            f"[CHECKPOINT LOAD END] stage={self.stage} "
            f"round2a={self.stage2a_round} round2b={self.stage2b_round} "
            f"step={self.global_step} "
            f"elapsed={_format_duration(time.perf_counter() - started_at)}"
        )

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "fold": self.config.training.fold,
            "seed": self.config.training.seed,
            "stage2a_round": self.stage2a_round,
            "stage2b_round": self.stage2b_round,
            "best_joint_scores": dict(self.best_joint_scores),
            "best_val_metrics": self.best_val_metrics,
            "best_task_scores": dict(self.best_task_scores),
            "best_task_val_metrics": self.best_task_val_metrics,
            "no_improve_rounds": self.no_improve_rounds,
            "global_step": self.global_step,
            "run_timestamp": self.run_timestamp,
            "run_dir": str(self.run_dir),
            "last_checkpoint": str(self.last_checkpoint_path),
            "stage2a_best_checkpoint": str(self.stage2a_best_path),
            "stage2b_best_checkpoint": str(self.stage2b_best_path),
            "task_checkpoints": {
                task: str(path)
                for task, path in self.task_checkpoint_paths.items()
            },
            "source_stage1_checkpoint": self.source_stage1_checkpoint,
        }


def initialise_from_stage1(
    model: ProgressiveFineTuningModel,
    checkpoint_path: str | Path,
    config: ExperimentConfig,
) -> Path:
    """Load the complete unified downstream model from a stage-1 last file."""

    source = Path(checkpoint_path).resolve()
    checkpoint = load_training_checkpoint(source, map_location="cpu")
    if checkpoint.get("checkpoint_type") == "task_model":
        raise ValueError(
            "Stage 2 requires a full stage-1 last checkpoint, not a task model"
        )
    if checkpoint.get("stage") != "complete":
        raise ValueError(
            "Stage 2 requires a completed stage-1 last checkpoint"
        )
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config.get("model") != config.to_dict()["model"]:
        raise ValueError("Stage-1 checkpoint model configuration does not match")
    checkpoint_training = checkpoint_config.get("training", {})
    for key in ("fold", "seed"):
        expected = getattr(config.training, key)
        if checkpoint_training.get(key) != expected:
            raise ValueError(
                f"Stage-1 checkpoint {key}={checkpoint_training.get(key)} "
                f"does not match current {key}={expected}"
            )
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-1 checkpoint does not contain the unified model")
    model.binding_model.load_state_dict(state)
    return source


def load_stage2_checkpoint(
    path: str | Path,
    *,
    map_location,
) -> dict[str, Any]:
    checkpoint = load_training_checkpoint(path, map_location=map_location)
    if (
        checkpoint.get("checkpoint_type") != "stage2_training"
        or checkpoint.get("format_version") != 1
    ):
        raise ValueError("Unsupported stage-2 checkpoint format")
    return checkpoint


def build_stage2_task_checkpoint_state(
    training_checkpoint: Mapping[str, Any],
    task: Task,
    *,
    source_checkpoint: str | Path,
) -> dict[str, Any]:
    """Build a deployable task checkpoint with only the needed PLMs/branch."""

    if task not in STAGE2_TASK_CHECKPOINT_FILENAMES:
        raise ValueError(f"Unknown task: {task}")
    if training_checkpoint.get("checkpoint_type") == "stage2_task_model":
        raise ValueError("Expected a full stage-2 checkpoint, got a task model")
    model_state = training_checkpoint.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError("Stage-2 checkpoint does not contain a model state")

    receptor_entity = "hla" if task == "phla" else "tcr"

    def select_prefix(prefix: str) -> dict[str, torch.Tensor]:
        selected = {}
        for name, value in model_state.items():
            if name.startswith(prefix):
                if name.endswith("._extra_state"):
                    continue
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Checkpoint value for {name} is not a tensor")
                selected[name[len(prefix) :]] = value.detach().cpu()
        return selected

    peptide_state = select_prefix("peptide_encoder.")
    receptor_state = select_prefix(f"{receptor_entity}_encoder.")
    binding_state = select_prefix("binding_model.")
    if not peptide_state or not receptor_state or not binding_state:
        raise ValueError(
            "Stage-2 checkpoint is missing peptide, receptor, or binding state"
        )
    task_binding_state = extract_task_state_dict(binding_state, task)
    metadata = training_checkpoint.get("encoder_metadata", {})
    selected_metadata = {
        "peptide": metadata.get("peptide", {}),
        receptor_entity: metadata.get(receptor_entity, {}),
    }
    best_task_scores = training_checkpoint.get("best_task_scores", {})
    best_task_metrics = training_checkpoint.get("best_task_val_metrics", {})
    if not best_task_metrics:
        # Checkpoints written before task-specific outputs were introduced only
        # tracked the stage-level joint validation dictionary.
        stage_metrics = training_checkpoint.get("best_val_metrics", {})
        best_task_metrics = stage_metrics.get("stage2b") or stage_metrics.get(
            "stage2a", {}
        )
    validation_metrics = best_task_metrics.get(task, {})
    best_score = best_task_scores.get(task)
    if best_score is None and validation_metrics:
        best_score = task_score(validation_metrics)

    current_round = (
        training_checkpoint.get("stage2a_round", 0)
        if training_checkpoint.get("stage") == "stage2a"
        else training_checkpoint.get("stage2b_round", 0)
    )
    return {
        "format_version": 1,
        "checkpoint_type": "stage2_task_model",
        "task": task,
        "model": {
            "peptide_encoder": peptide_state,
            "receptor_encoder": receptor_state,
            "binding_model": task_binding_state,
        },
        "receptor_entity": receptor_entity,
        "plm_entities": ["peptide", receptor_entity],
        "config": training_checkpoint.get("config", {}),
        "encoder_metadata": selected_metadata,
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_checkpoint_type": training_checkpoint.get("checkpoint_type"),
        "stage": training_checkpoint.get("stage"),
        "current_round": current_round,
        "global_step": training_checkpoint.get("global_step"),
        "run_timestamp": training_checkpoint.get("run_timestamp"),
        "best_score": best_score,
        "validation_metrics": validation_metrics,
        "input_mode": training_checkpoint.get("input_mode", "online_stage2"),
        "fgm_enabled": False,
        "source_stage1_checkpoint": training_checkpoint.get(
            "source_stage1_checkpoint"
        ),
    }
