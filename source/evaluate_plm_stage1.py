#!/usr/bin/env python3
"""Evaluate a unified PLM checkpoint on configured HLA/TCR splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.artifacts import (
    create_artifact_timestamp,
    find_latest_common_artifacts,
    timestamped_filename,
)
from plm_unified.cache import EmbeddingCache
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import make_cached_dataloader, make_online_dataloader, split_path
from plm_unified.evaluation import evaluate_model
from plm_unified.encoders import OnlineEncoderBundle, checkpoint_fingerprint
from plm_unified.model import TaskSpecificBindingModel, UnifiedBindingModel
from plm_unified.trainer import (
    TASK_CHECKPOINT_FILENAMES,
    load_training_checkpoint,
    resolve_device,
)


def _online_encoded_loader(loader, encoders, task):
    """Wrap an online string-batch loader into the hidden/mask dict evaluate_model expects."""
    for batch in loader:
        encoded = encoders.encode(
            task,
            batch["peptide_sequences"],
            batch["receptor_sequences"],
        )
        yield {
            "peptide_hidden": encoded["peptide_hidden"],
            "peptide_mask": encoded["peptide_mask"],
            "receptor_hidden": encoded["receptor_hidden"],
            "receptor_mask": encoded["receptor_mask"],
            "labels": batch["labels"],
        }


def _task_model(checkpoint, config, expected_task):
    if checkpoint.get("checkpoint_type") != "task_model":
        raise ValueError("Expected a task_model checkpoint")
    checkpoint_task = checkpoint.get("task")
    if checkpoint_task != expected_task:
        raise ValueError(
            f"Checkpoint task '{checkpoint_task}' does not match '{expected_task}'"
        )
    saved_model_config = checkpoint.get("config", {}).get("model")
    if (
        saved_model_config is not None
        and saved_model_config != config.to_dict()["model"]
    ):
        raise ValueError(
            "Task checkpoint model configuration does not match current config"
        )
    model = TaskSpecificBindingModel(config.model, expected_task)
    model.load_state_dict(checkpoint["model"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / "plm_stage1.yaml"
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--hla-split", default="val")
    parser.add_argument("--tcr-split", default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Encode frozen PLMs online instead of reading the HDF5 cache. "
        "Use for splits not present in the cache (e.g. independent/external). "
        "Loads ESM-C and TCR-BERT on the device and encodes on the fly.",
    )
    parser.add_argument(
        "--tasks",
        default="phla,ptcr",
        help="Comma-separated subset of phla,ptcr to evaluate (default: both).",
    )
    parser.add_argument(
        "--online-batch-size",
        type=int,
        help="Override training.online_batch_size for online encoding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(
        load_config(args.config),
        fold=args.fold,
        seed=args.seed,
        device=args.device,
    )
    splits = {"phla": args.hla_split, "ptcr": args.tcr_split}
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in splits]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Choose from: {sorted(splits)}")
    if not tasks:
        raise ValueError("At least one task must be selected")

    models = {}
    checkpoint_paths = {}
    checkpoint_types = {}
    if args.checkpoint:
        checkpoint_path = args.checkpoint.resolve()
        checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
        checkpoint_type = checkpoint.get("checkpoint_type", "training")
        checkpoint_task = checkpoint.get("task")
        if checkpoint_type == "task_model":
            if tasks != [checkpoint_task]:
                raise ValueError(
                    f"Task checkpoint '{checkpoint_task}' requires "
                    f"--tasks {checkpoint_task}"
                )
            models[checkpoint_task] = _task_model(
                checkpoint,
                config,
                checkpoint_task,
            )
        else:
            model = UnifiedBindingModel(config.model)
            model.load_state_dict(checkpoint["model"])
            models = {task: model for task in tasks}
        for task in tasks:
            checkpoint_paths[task] = str(checkpoint_path)
            checkpoint_types[task] = checkpoint_type
    else:
        resolved_checkpoints = find_latest_common_artifacts(
            config.run_dir(),
            {
                task: TASK_CHECKPOINT_FILENAMES[task]
                for task in tasks
            },
        )
        for task in tasks:
            checkpoint_path = resolved_checkpoints[task]
            checkpoint = load_training_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )
            models[task] = _task_model(checkpoint, config, task)
            checkpoint_paths[task] = str(checkpoint_path)
            checkpoint_types[task] = "task_model"

    online_encoders = None
    if args.no_cache:
        config.validate(require_models=True)
        online_encoders = OnlineEncoderBundle(
            esmc_path=config.paths.esmc_model,
            tcr_bert_path=config.paths.tcr_bert_model,
            peptide_max_length=config.model.peptide_max_length,
            hla_max_length=config.model.hla_max_length,
            tcr_max_length=config.model.tcr_max_length,
            device=resolve_device(config.training.device),
            dtype=config.cache.dtype,
        )
        if online_encoders.esmc.hidden_size != config.model.peptide_input_dim:
            raise ValueError("Online ESM-C hidden size does not match model config")
        if online_encoders.tcr_bert.hidden_size != config.model.tcr_input_dim:
            raise ValueError("Online TCR-BERT hidden size does not match model config")

    cache_dir = config.cache_dir()
    esmc_fingerprint = (
        checkpoint_fingerprint(config.paths.esmc_model)
        if config.paths.esmc_model.is_dir()
        else None
    )
    tcr_fingerprint = (
        checkpoint_fingerprint(config.paths.tcr_bert_model)
        if config.paths.tcr_bert_model.is_dir()
        else None
    )
    peptide_cache = None
    receptor_caches = {}
    if not args.no_cache:
        peptide_cache = EmbeddingCache(
            cache_dir / "peptide",
            expected_hidden_dim=config.model.peptide_input_dim,
            expected_fingerprint=esmc_fingerprint,
        )
        if "phla" in tasks:
            receptor_caches["phla"] = EmbeddingCache(
                cache_dir / "hla",
                expected_hidden_dim=config.model.hla_input_dim,
                expected_fingerprint=esmc_fingerprint,
            )
        if "ptcr" in tasks:
            receptor_caches["ptcr"] = EmbeddingCache(
                cache_dir / "tcr",
                expected_hidden_dim=config.model.tcr_input_dim,
                expected_fingerprint=tcr_fingerprint,
            )

    online_batch_size = args.online_batch_size or config.training.online_batch_size
    # make_online_dataloader requires an int limit; None means "no cap".
    online_limit = args.limit if args.limit is not None else (2**31 - 1)

    results = {}
    for task in tasks:
        csv_path = split_path(
            config.paths.data_root,
            task,
            splits[task],
            config.training.fold,
        )
        if args.no_cache:
            loader = make_online_dataloader(
                csv_path,
                task,
                batch_size=online_batch_size,
                shuffle=False,
                seed=config.training.seed,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory,
                peptide_min_length=config.model.peptide_min_length,
                peptide_max_length=config.model.peptide_max_length,
                limit=online_limit,
            )
            loader = _online_encoded_loader(loader, online_encoders, task)
        else:
            loader = make_cached_dataloader(
                csv_path,
                task,
                peptide_cache,
                receptor_caches[task],
                batch_size=config.training.batch_size,
                shuffle=False,
                seed=config.training.seed,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory,
                persistent_workers=config.training.persistent_workers,
                peptide_min_length=config.model.peptide_min_length,
                peptide_max_length=config.model.peptide_max_length,
                limit=args.limit,
            )
        results[task] = evaluate_model(
            models[task],
            loader,
            task,
            config,
        ).to_dict()

    report_timestamp = create_artifact_timestamp()
    report_filename = timestamped_filename(
        f"evaluation-hla_{args.hla_split}-tcr_{args.tcr_split}.json",
        report_timestamp,
    )
    report_path = config.run_dir() / report_filename
    report = {
        "artifact_timestamp": report_timestamp,
        "checkpoints": checkpoint_paths,
        "checkpoint_types": checkpoint_types,
        "fold": config.training.fold,
        "seed": config.training.seed,
        "splits": splits,
        "metrics": results,
        "report_path": str(report_path.resolve()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Report written to: {report_path}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
