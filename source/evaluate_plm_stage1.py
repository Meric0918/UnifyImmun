#!/usr/bin/env python3
"""Evaluate a unified PLM checkpoint on configured HLA/TCR splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.cache import EmbeddingCache
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import make_cached_dataloader, split_path
from plm_unified.evaluation import evaluate_model
from plm_unified.encoders import checkpoint_fingerprint
from plm_unified.model import UnifiedBindingModel
from plm_unified.trainer import load_training_checkpoint


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(
        load_config(args.config),
        fold=args.fold,
        seed=args.seed,
        device=args.device,
    )
    checkpoint_path = args.checkpoint or config.run_dir() / "best_joint.pt"
    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
    model = UnifiedBindingModel(config.model)
    model.load_state_dict(checkpoint["model"])

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
    receptor_caches = {
        "phla": EmbeddingCache(
            cache_dir / "hla",
            expected_hidden_dim=config.model.hla_input_dim,
            expected_fingerprint=esmc_fingerprint,
        ),
        "ptcr": EmbeddingCache(
            cache_dir / "tcr",
            expected_hidden_dim=config.model.tcr_input_dim,
            expected_fingerprint=tcr_fingerprint,
        ),
    }
    splits = {"phla": args.hla_split, "ptcr": args.tcr_split}
    results = {}
    for task in ("phla", "ptcr"):
        loader = make_cached_dataloader(
            split_path(
                config.paths.data_root,
                task,
                splits[task],
                config.training.fold,
            ),
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
        results[task] = evaluate_model(model, loader, task, config).to_dict()

    report = {
        "checkpoint": str(checkpoint_path),
        "fold": config.training.fold,
        "seed": config.training.seed,
        "splits": splits,
        "metrics": results,
    }
    report_path = config.run_dir() / (
        f"evaluation-hla_{args.hla_split}-tcr_{args.tcr_split}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
