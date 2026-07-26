#!/usr/bin/env python3
"""Evaluate a fold/seed's best_ptcr.pt on pTCR CSV test data.

Examples
--------
Evaluate all discovered test sets for fold 5, seed 2026::

    python source/test_ptcr_best.py --fold 5 --seed 2026 --device cuda

Evaluate only the independent and triple sets::

    python source/test_ptcr_best.py --fold 5 --sets independent triple

Evaluate multiple fold/seed combinations (Cartesian product)::

    python source/test_ptcr_best.py --folds 1 3 5 --seeds 42 2026

Evaluate an explicit CSV whose sequences are not in the embedding cache::

    python source/test_ptcr_best.py --fold 2 --seed 42 \
        --input-csv data/data_liver/TCR_Peptide_balanced.csv --no-cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.artifacts import (
    create_artifact_timestamp,
    find_latest_artifact,
    timestamped_path,
)
from plm_unified.cache import EmbeddingCache
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import (
    make_cached_dataloader,
    make_online_dataloader,
    split_path,
)
from plm_unified.evaluation import evaluate_model
from plm_unified.encoders import (
    OnlineEncoderBundle,
    checkpoint_fingerprint,
)
from plm_unified.model import TaskSpecificBindingModel
from plm_unified.trainer import load_training_checkpoint, resolve_device


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "plm_stage1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Training YAML used to reconstruct the model and cache paths.",
    )
    fold_group = parser.add_mutually_exclusive_group()
    fold_group.add_argument("--fold", type=int, help="One fold number, for example 1 or 5.")
    fold_group.add_argument(
        "--folds",
        type=int,
        nargs="+",
        help="Multiple fold numbers; evaluated with every selected seed.",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, help="One seed (default: 2026).")
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Multiple seeds; evaluated with every selected fold.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional explicit best_ptcr.pt path; overrides --fold/--seed lookup.",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        help="Test split names without _set, e.g. independent triple. "
        "Defaults to every *_set.csv found in data_TCR.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Explicit CSV containing peptide,tcr,label columns. Mutually exclusive "
        "with --sets.",
    )
    parser.add_argument(
        "--input-name",
        help="Report key for --input-csv (default: CSV filename stem).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Encode sequences online with the frozen ESM-C and TCR-BERT models. "
        "Required when the selected CSV is not fully represented in the cache.",
    )
    parser.add_argument(
        "--online-batch-size",
        type=int,
        help="Online encoder batch size (default: training.online_batch_size).",
    )
    parser.add_argument("--device", help="Device override, e.g. cuda or cpu.")
    parser.add_argument("--limit", type=int, help="Evaluate at most this many rows per set.")
    parser.add_argument(
        "--num-workers",
        type=int,
        help="DataLoader worker override (default: value from config).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Base JSON report path for a single fold/seed run; a timestamp is "
        "automatically inserted before the extension. In batch mode, each run "
        "is saved under its own checkpoint directory.",
    )
    return parser.parse_args()


def discover_test_sets(data_root: Path) -> list[str]:
    task_dir = data_root / "data_TCR"
    discovered = sorted(
        path.stem.removesuffix("_set")
        for path in task_dir.glob("*_set.csv")
        if path.is_file()
    )
    if not discovered:
        raise FileNotFoundError(f"No *_set.csv files found in {task_dir}")
    return discovered


def load_ptcr_model(checkpoint_path: Path, config):
    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_type") != "task_model":
        raise ValueError(
            f"Expected a task_model checkpoint, got {checkpoint.get('checkpoint_type')!r}: "
            f"{checkpoint_path}"
        )
    if checkpoint.get("task") != "ptcr":
        raise ValueError(
            f"Expected a pTCR checkpoint, got task={checkpoint.get('task')!r}: "
            f"{checkpoint_path}"
        )
    saved_model_config = checkpoint.get("config", {}).get("model")
    if saved_model_config is not None and saved_model_config != config.to_dict()["model"]:
        raise ValueError(
            "Checkpoint model configuration differs from the selected config. "
            "Use the YAML that was used to train this checkpoint."
        )
    model = TaskSpecificBindingModel(config.model, "ptcr")
    model.load_state_dict(checkpoint["model"])
    return model


def _online_encoded_loader(loader, encoders):
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        encoded = encoders.encode(
            "ptcr",
            batch["peptide_sequences"],
            batch["receptor_sequences"],
        )
        yield {
            **encoded,
            "labels": batch["labels"],
        }
        if batch_index % 50 == 0 or batch_index == total_batches:
            print(
                f"  online encoding: {batch_index}/{total_batches} batches",
                flush=True,
            )


def evaluate_run(
    args: argparse.Namespace,
    config,
    datasets: dict[str, Path],
    checkpoint_path: Path,
):
    model = load_ptcr_model(checkpoint_path, config)
    peptide_cache = None
    tcr_cache = None
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
    else:
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
        peptide_cache = EmbeddingCache(
            cache_dir / "peptide",
            expected_hidden_dim=config.model.peptide_input_dim,
            expected_fingerprint=esmc_fingerprint,
        )
        tcr_cache = EmbeddingCache(
            cache_dir / "tcr",
            expected_hidden_dim=config.model.tcr_input_dim,
            expected_fingerprint=tcr_fingerprint,
        )

    results: dict[str, dict] = {}
    try:
        for split, csv_path in datasets.items():
            if args.no_cache:
                online_limit = args.limit if args.limit is not None else 2**31 - 1
                source_loader = make_online_dataloader(
                    csv_path,
                    "ptcr",
                    batch_size=(
                        args.online_batch_size
                        or config.training.online_batch_size
                    ),
                    shuffle=False,
                    seed=config.training.seed,
                    num_workers=config.training.num_workers,
                    pin_memory=config.training.pin_memory,
                    peptide_min_length=config.model.peptide_min_length,
                    peptide_max_length=config.model.peptide_max_length,
                    limit=online_limit,
                )
                filter_stats = dict(source_loader.dataset.filter_stats)
                loader = _online_encoded_loader(source_loader, online_encoders)
            else:
                loader = make_cached_dataloader(
                    csv_path,
                    "ptcr",
                    peptide_cache,
                    tcr_cache,
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
                filter_stats = dict(loader.dataset.filter_stats)
            metrics = evaluate_model(model, loader, "ptcr", config).to_dict()
            results[split] = {
                "file": str(csv_path),
                "filter_stats": filter_stats,
                "metrics": metrics,
            }
            print(
                f"{split}: samples={metrics['samples']} loss={metrics['loss']:.6f} "
                f"AUROC={metrics['auroc']:.6f} AUPR={metrics['aupr']:.6f} "
                f"ACC={metrics['accuracy']:.6f} MCC={metrics['mcc']:.6f} "
                f"F1={metrics['f1']:.6f}",
                flush=True,
            )
    finally:
        if peptide_cache is not None:
            peptide_cache.close()
        if tcr_cache is not None:
            tcr_cache.close()

    return results


def main() -> None:
    args = parse_args()
    if args.input_csv and args.sets:
        raise SystemExit("--input-csv and --sets are mutually exclusive")
    if args.input_name and not args.input_csv:
        raise SystemExit("--input-name requires --input-csv")
    if args.online_batch_size is not None and args.online_batch_size < 1:
        raise SystemExit("--online-batch-size must be positive")
    if args.fold is None and not args.folds:
        raise SystemExit("one of --fold or --folds is required")
    folds = [args.fold] if args.fold is not None else args.folds
    seeds = [args.seed] if args.seed is not None else (args.seeds or [2026])
    if args.checkpoint and (len(folds) != 1 or len(seeds) != 1):
        raise SystemExit("--checkpoint can only be used with one fold and one seed")
    if args.output and (len(folds) != 1 or len(seeds) != 1):
        raise SystemExit(
            "--output with multiple fold/seed combinations is ambiguous; "
            "omit it to save each report in its own seed directory"
        )

    if args.input_csv:
        input_path = args.input_csv.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
        split_name = args.input_name or input_path.stem
        datasets = {split_name: input_path}
    else:
        base_config = load_config(args.config)
        discovered_sets = discover_test_sets(base_config.paths.data_root)
        split_names = args.sets or discovered_sets
        unknown = sorted(set(split_names) - set(discovered_sets))
        if unknown:
            raise ValueError(
                f"Unknown test set(s): {unknown}. Discovered: {discovered_sets}"
            )
        datasets = {
            split: split_path(
                base_config.paths.data_root,
                "ptcr",
                split,
                base_config.training.fold,
            )
            for split in split_names
        }

    for fold in folds:
        for seed in seeds:
            config = apply_overrides(
                load_config(args.config),
                fold=fold,
                seed=seed,
                device=args.device,
                num_workers=args.num_workers,
            )
            checkpoint_path = (
                args.checkpoint.expanduser().resolve()
                if args.checkpoint
                else find_latest_artifact(
                    config.paths.output_root
                    / f"fold_{fold}"
                    / f"seed_{seed}",
                    "best_ptcr.pt",
                )
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            run_key = f"fold_{fold}/seed_{seed}"
            print(f"\n[{run_key}] checkpoint={checkpoint_path}", flush=True)
            metrics = evaluate_run(
                args,
                config,
                datasets,
                checkpoint_path,
            )
            report_timestamp = create_artifact_timestamp()
            output_base = (
                args.output.expanduser().resolve()
                if args.output
                else checkpoint_path.parent
                / (
                    f"evaluation_ptcr_{next(iter(datasets))}.json"
                    if args.input_csv
                    else "evaluation_ptcr_sets.json"
                )
            )
            output_path = timestamped_path(output_base, report_timestamp)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "artifact_timestamp": report_timestamp,
                "checkpoint": str(checkpoint_path),
                "fold": fold,
                "seed": seed,
                "task": "ptcr",
                "input_mode": "online" if args.no_cache else "cache",
                "sets": list(datasets),
                "metrics": metrics,
                "report_path": str(output_path),
            }
            output_path.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
