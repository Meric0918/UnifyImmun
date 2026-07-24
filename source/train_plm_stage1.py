#!/usr/bin/env python3
"""Train stage 1A/1B of the unified ESM-C/TCR-BERT model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.cache import EmbeddingCache
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import (
    make_cached_dataloader,
    make_online_dataloader,
    split_path,
)
from plm_unified.encoders import OnlineEncoderBundle, checkpoint_fingerprint
from plm_unified.model import UnifiedBindingModel
from plm_unified.tracking import SwanLabTracker
from plm_unified.trainer import (
    Stage1Trainer,
    peek_swanlab_run_id,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / "plm_stage1.yaml"
        ),
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override DataLoader workers; use 0 to debug cache reads in-process.",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Debug only: encode frozen PLMs online; requires --limit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug only: limit rows per train/validation split.",
    )
    return parser.parse_args()


def open_caches(config):
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
    return {
        "peptide": EmbeddingCache(
            cache_dir / "peptide",
            expected_hidden_dim=config.model.peptide_input_dim,
            expected_fingerprint=esmc_fingerprint,
        ),
        "hla": EmbeddingCache(
            cache_dir / "hla",
            expected_hidden_dim=config.model.hla_input_dim,
            expected_fingerprint=esmc_fingerprint,
        ),
        "tcr": EmbeddingCache(
            cache_dir / "tcr",
            expected_hidden_dim=config.model.tcr_input_dim,
            expected_fingerprint=tcr_fingerprint,
        ),
    }


def make_loaders(config, caches, limit, *, online: bool):
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
        "persistent_workers": config.training.persistent_workers,
        "peptide_min_length": config.model.peptide_min_length,
        "peptide_max_length": config.model.peptide_max_length,
        "limit": limit,
    }
    train_loaders = {}
    val_loaders = {}
    for task, receptor_name in (("phla", "hla"), ("ptcr", "tcr")):
        paths = {
            split: split_path(
                config.paths.data_root,
                task,
                split,
                config.training.fold,
            )
            for split in ("train", "val")
        }
        if online:
            online_common = {
                "batch_size": config.training.online_batch_size,
                "num_workers": config.training.num_workers,
                "pin_memory": config.training.pin_memory,
                "peptide_min_length": config.model.peptide_min_length,
                "peptide_max_length": config.model.peptide_max_length,
                "limit": limit,
            }
            train_loaders[task] = make_online_dataloader(
                paths["train"],
                task,
                shuffle=True,
                seed=config.training.seed,
                **online_common,
            )
            val_loaders[task] = make_online_dataloader(
                paths["val"],
                task,
                shuffle=False,
                seed=config.training.seed,
                **online_common,
            )
        else:
            train_loaders[task] = make_cached_dataloader(
                paths["train"],
                task,
                caches["peptide"],
                caches[receptor_name],
                shuffle=True,
                seed=config.training.seed,
                **common,
            )
            val_loaders[task] = make_cached_dataloader(
                paths["val"],
                task,
                caches["peptide"],
                caches[receptor_name],
                shuffle=False,
                seed=config.training.seed,
                **common,
            )
    return train_loaders, val_loaders


def main() -> None:
    args = parse_args()
    config = apply_overrides(
        load_config(args.config),
        fold=args.fold,
        seed=args.seed,
        swanlab_mode=args.swanlab_mode,
        device=args.device,
        num_workers=args.num_workers,
    )
    if args.no_cache and args.limit is None:
        raise ValueError("--no-cache is a debug mode and requires an explicit --limit")
    set_global_seed(config.training.seed)
    online_encoders = None
    if args.no_cache:
        from plm_unified.trainer import resolve_device

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
        caches = {}
    else:
        caches = open_caches(config)
    train_loaders, val_loaders = make_loaders(
        config,
        caches,
        args.limit,
        online=args.no_cache,
    )
    model = UnifiedBindingModel(config.model)

    resume_id = peek_swanlab_run_id(args.resume) if args.resume else None
    with SwanLabTracker(config, resume_id=resume_id) as tracker:
        data_log = {}
        for split_name, loaders in (
            ("train", train_loaders),
            ("val", val_loaders),
        ):
            for task, loader in loaders.items():
                for key, value in loader.dataset.filter_stats.items():
                    data_log[f"data/{task}/{split_name}/{key}"] = value
        for entity, cache in caches.items():
            data_log[f"cache/{entity}/count"] = cache.manifest["count"]
            data_log[f"cache/{entity}/hidden_dim"] = cache.manifest["hidden_dim"]
            data_log[f"cache/{entity}/max_length"] = cache.manifest["max_length"]
            data_log[f"cache/{entity}/checkpoint_fingerprint"] = cache.manifest[
                "model"
            ]["checkpoint_fingerprint"]
        if online_encoders is not None:
            data_log["input/mode"] = "online_debug"
            data_log["input/esmc_checkpoint_fingerprint"] = (
                online_encoders.esmc.checkpoint_fingerprint
            )
            data_log["input/tcr_bert_checkpoint_fingerprint"] = (
                online_encoders.tcr_bert.checkpoint_fingerprint
            )
        tracker.log(data_log, step=0)
        trainer = Stage1Trainer(
            model,
            config,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            tracker=tracker,
            cache_metadata={
                entity: cache.manifest for entity, cache in caches.items()
            }
            if caches
            else {"online": online_encoders.metadata()},
            online_encoders=online_encoders,
            input_mode="online_debug" if args.no_cache else "cache",
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        summary = trainer.fit()
        tracker.log(
            {
                "training/complete": 1,
                "training/final_round": summary["round"],
                "training/best_joint_score": summary["best_joint_score"],
            },
            step=trainer.global_step,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
