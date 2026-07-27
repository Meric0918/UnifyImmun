#!/usr/bin/env python3
"""Progressively unfreeze PLM top layers for unified stage-2 training."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Transformer Engine 1.x discovers NVRTC through CUDA_HOME.  The
# unifyimmun environment provides a CUDA 12.4 overlay assembled from the
# matching PyTorch CUDA wheels, so use it automatically for stage-2 launches.
_CONDA_CUDA_HOME = Path(sys.executable).resolve().parent.parent / "cuda"
if "CUDA_HOME" not in os.environ and _CONDA_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_CONDA_CUDA_HOME)

from plm_unified.artifacts import (
    create_artifact_timestamp,
    validate_artifact_timestamp,
)
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import make_online_dataloader, split_path
from plm_unified.finetuning import (
    ESMCFineTuningEncoder,
    ProgressiveFineTuningModel,
    TCRBertFineTuningEncoder,
)
from plm_unified.model import UnifiedBindingModel
from plm_unified.stage2_trainer import (
    Stage2Trainer,
    initialise_from_stage1,
    load_stage2_checkpoint,
)
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import resolve_device, set_global_seed


def log_status(message: str) -> None:
    print(
        f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] {message}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / "plm_stage2.yaml"
        ),
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        help="Completed stage-1 last_<timestamp>.pt; required for a new run.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume a full stage2_last_<timestamp>.pt checkpoint.",
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
    )
    parser.add_argument(
        "--run-timestamp",
        help="Optional shared artifact suffix in YYYYMMDD_HHMMSS_ffffff format.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug only: limit rows per train/validation split.",
    )
    return parser.parse_args()


def make_loaders(config, limit=None):
    common = {
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
        "peptide_min_length": config.model.peptide_min_length,
        "peptide_max_length": config.model.peptide_max_length,
        "limit": limit,
    }
    train_loaders = {}
    val_loaders = {}
    for task in ("phla", "ptcr"):
        train_loaders[task] = make_online_dataloader(
            split_path(
                config.paths.data_root,
                task,
                "train",
                config.training.fold,
            ),
            task,
            batch_size=config.stage2.micro_batch_size,
            shuffle=True,
            seed=config.training.seed,
            **common,
        )
        val_loaders[task] = make_online_dataloader(
            split_path(
                config.paths.data_root,
                task,
                "val",
                config.training.fold,
            ),
            task,
            batch_size=config.stage2.eval_batch_size,
            shuffle=False,
            seed=config.training.seed,
            **common,
        )
    return train_loaders, val_loaders


def build_model(config, device):
    precision_to_encoder_dtype = {
        "none": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    dtype = precision_to_encoder_dtype[config.stage2.mixed_precision]
    encoder_kwargs = {
        "device": device,
        "dtype": dtype,
        "gradient_checkpointing": config.stage2.gradient_checkpointing,
    }
    started_at = time.perf_counter()
    log_status(
        f"[MODEL LOAD START][Peptide ESM-C] path={config.paths.esmc_model} "
        f"device={device} dtype={dtype}"
    )
    peptide_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model,
        config.model.peptide_max_length,
        **encoder_kwargs,
    )
    log_status(
        "[MODEL LOAD END][Peptide ESM-C] "
        f"layers={len(peptide_encoder.transformer_blocks())} "
        f"hidden={peptide_encoder.hidden_size} "
        f"elapsed={time.perf_counter() - started_at:.1f}s"
    )
    started_at = time.perf_counter()
    log_status(
        f"[MODEL LOAD START][HLA ESM-C] path={config.paths.esmc_model} "
        f"device={device} dtype={dtype}"
    )
    hla_encoder = ESMCFineTuningEncoder(
        config.paths.esmc_model,
        config.model.hla_max_length,
        **encoder_kwargs,
    )
    log_status(
        "[MODEL LOAD END][HLA ESM-C] "
        f"layers={len(hla_encoder.transformer_blocks())} "
        f"hidden={hla_encoder.hidden_size} "
        f"elapsed={time.perf_counter() - started_at:.1f}s"
    )
    started_at = time.perf_counter()
    log_status(
        f"[MODEL LOAD START][TCR-BERT] path={config.paths.tcr_bert_model} "
        f"device={device} dtype={dtype}"
    )
    tcr_encoder = TCRBertFineTuningEncoder(
        config.paths.tcr_bert_model,
        config.model.tcr_max_length,
        **encoder_kwargs,
    )
    log_status(
        "[MODEL LOAD END][TCR-BERT] "
        f"layers={len(tcr_encoder.transformer_blocks())} "
        f"hidden={tcr_encoder.hidden_size} "
        f"elapsed={time.perf_counter() - started_at:.1f}s"
    )
    if peptide_encoder.hidden_size != config.model.peptide_input_dim:
        raise ValueError("Peptide ESM-C hidden size does not match model config")
    if hla_encoder.hidden_size != config.model.hla_input_dim:
        raise ValueError("HLA ESM-C hidden size does not match model config")
    if tcr_encoder.hidden_size != config.model.tcr_input_dim:
        raise ValueError("TCR-BERT hidden size does not match model config")
    return ProgressiveFineTuningModel(
        UnifiedBindingModel(config.model),
        peptide_encoder,
        hla_encoder,
        tcr_encoder,
        peptide_unfrozen_layers=config.stage2.peptide_unfrozen_layers,
        hla_unfrozen_layers=config.stage2.hla_unfrozen_layers,
        tcr_unfrozen_layers=config.stage2.tcr_unfrozen_layers,
    )


def main() -> None:
    args = parse_args()
    if args.stage1_checkpoint and args.resume:
        raise ValueError("Use either --stage1-checkpoint or --resume, not both")
    if not args.stage1_checkpoint and not args.resume:
        raise ValueError(
            "A new run requires --stage1-checkpoint; resuming requires --resume"
        )

    resume_state = (
        load_stage2_checkpoint(args.resume, map_location="cpu")
        if args.resume
        else None
    )
    resume_timestamp = resume_state.get("run_timestamp") if resume_state else None
    resume_id = resume_state.get("swanlab_run_id") if resume_state else None
    del resume_state
    gc.collect()
    run_timestamp = validate_artifact_timestamp(
        args.run_timestamp or resume_timestamp or create_artifact_timestamp()
    )
    config = apply_overrides(
        load_config(args.config),
        fold=args.fold,
        seed=args.seed,
        swanlab_mode=args.swanlab_mode,
        device=args.device,
        num_workers=args.num_workers,
    )
    config.validate(require_models=True)
    set_global_seed(config.training.seed)
    device = resolve_device(config.training.device)
    log_status(
        f"[RUN START] fold={config.training.fold} seed={config.training.seed} "
        f"device={device} swanlab={config.swanlab.mode} "
        f"run_timestamp={run_timestamp} limit={args.limit}"
    )
    log_status(
        f"[SCHEDULE] Stage2A={config.stage2.stage2a_rounds} alternating rounds; "
        f"Stage2B={config.stage2.stage2b_min_rounds}-"
        f"{config.stage2.stage2b_max_rounds} rounds; "
        f"patience={config.stage2.early_stopping_patience}; "
        "order=pHLA->pTCR"
    )
    model = build_model(config, device)
    source_stage1 = None
    if args.stage1_checkpoint:
        started_at = time.perf_counter()
        log_status(
            f"[STAGE1 LOAD START] checkpoint={args.stage1_checkpoint}"
        )
        source_stage1 = initialise_from_stage1(
            model,
            args.stage1_checkpoint,
            config,
        )
        log_status(
            f"[STAGE1 LOAD END] source={source_stage1} "
            f"elapsed={time.perf_counter() - started_at:.1f}s"
        )

    started_at = time.perf_counter()
    log_status("[DATA LOAD START] reading raw sequence CSV splits")
    train_loaders, val_loaders = make_loaders(config, args.limit)
    log_status(
        "[DATA LOAD END] "
        + ", ".join(
            f"{split}/{task}={len(loader.dataset):,}"
            for split, loaders in (("train", train_loaders), ("val", val_loaders))
            for task, loader in loaders.items()
        )
        + f" elapsed={time.perf_counter() - started_at:.1f}s"
    )
    log_status(
        f"[SWANLAB INIT START] mode={config.swanlab.mode} "
        f"project={config.swanlab.project}"
    )
    with SwanLabTracker(
        config,
        resume_id=resume_id,
        run_timestamp=run_timestamp,
    ) as tracker:
        log_status(f"[SWANLAB INIT END] run_id={tracker.run_id or 'disabled'}")
        data_log = {
            "input/mode": "online_stage2",
            "stage2/micro_batch_size": config.stage2.micro_batch_size,
            "stage2/eval_batch_size": config.stage2.eval_batch_size,
            "stage2/gradient_accumulation_steps": (
                config.stage2.gradient_accumulation_steps
            ),
            "stage2/effective_batch_size": (
                config.stage2.micro_batch_size
                * config.stage2.gradient_accumulation_steps
            ),
            "stage2/fgm_enabled": 0,
            "stage2/gradient_checkpointing": int(
                config.stage2.gradient_checkpointing
            ),
        }
        for entity, metadata in model.encoder_metadata().items():
            for key, value in metadata.items():
                data_log[f"encoder/{entity}/{key}"] = value
        for split_name, loaders in (
            ("train", train_loaders),
            ("val", val_loaders),
        ):
            for task, loader in loaders.items():
                for key, value in loader.dataset.filter_stats.items():
                    data_log[f"data/{task}/{split_name}/{key}"] = value
        tracker.log(data_log, step=0)

        trainer = Stage2Trainer(
            model,
            config,
            train_loaders,
            val_loaders,
            tracker,
            source_stage1_checkpoint=source_stage1,
            run_timestamp=run_timestamp,
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        summary = trainer.fit()
        tracker.log(
            {
                "training/complete": 1,
                "training/stage2b_round": summary["stage2b_round"],
                "training/best_joint_score": summary["best_joint_scores"][
                    "stage2b"
                ],
                "training/best_phla_score": summary["best_task_scores"][
                    "phla"
                ],
                "training/best_ptcr_score": summary["best_task_scores"][
                    "ptcr"
                ],
            },
            step=trainer.global_step,
        )
        log_status("[SWANLAB FINISH] flushing final metrics")
    log_status(
        f"[RUN END] fold={config.training.fold} seed={config.training.seed} "
        f"stage={summary['stage']} global_step={summary['global_step']}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
