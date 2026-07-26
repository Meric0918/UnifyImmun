#!/usr/bin/env python3
"""Precompute frozen ESM-C/TCR-BERT residue representations."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from plm_unified.artifacts import create_artifact_timestamp, timestamped_filename
from plm_unified.cache import build_embedding_cache
from plm_unified.config import apply_overrides, load_config
from plm_unified.data import collect_cache_sequences
from plm_unified.encoders import ESMCResidueEncoder, TCRBertResidueEncoder
from plm_unified.trainer import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / "plm_stage1.yaml"
        ),
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument(
        "--entities",
        nargs="+",
        choices=("peptide", "hla", "tcr"),
        default=("peptide", "hla", "tcr"),
    )
    parser.add_argument("--device", help="Examples: cuda, cuda:1, cpu")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the selected entity caches if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(
        load_config(args.config),
        fold=args.fold,
        device=args.device,
    )
    config.validate(require_models=True)
    device = resolve_device(config.training.device)
    artifact_timestamp = create_artifact_timestamp()
    cache_dir = config.cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    sequences, filter_stats = collect_cache_sequences(
        config.paths.data_root,
        config.training.fold,
        config.cache.hla_splits,
        config.cache.tcr_splits,
        config.model.peptide_min_length,
        config.model.peptide_max_length,
    )
    for split_name, values in filter_stats.items():
        dropped = values["peptide_too_short"] + values["peptide_too_long"]
        print(
            f"{split_name}: kept={values['kept']:,}, "
            f"dropped_out_of_range={dropped:,}"
        )
    for entity in ("peptide", "hla", "tcr"):
        print(f"{entity}: {len(sequences[entity]):,} unique sequences")
    filter_stats_path = cache_dir / timestamped_filename(
        "filter_stats.json", artifact_timestamp
    )
    filter_stats_path.write_text(
        json.dumps(filter_stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Filter statistics written to: {filter_stats_path}")

    selected = set(args.entities)
    if selected.intersection({"peptide", "hla"}):
        esmc_max_length = (
            config.model.hla_max_length
            if "hla" in selected
            else config.model.peptide_max_length
        )
        esmc = ESMCResidueEncoder(
            config.paths.esmc_model,
            max_length=esmc_max_length,
            device=device,
            dtype=config.cache.dtype,
        )
        if esmc.hidden_size != config.model.peptide_input_dim:
            raise ValueError(
                f"ESM-C hidden size {esmc.hidden_size} does not match "
                f"configured input dim {config.model.peptide_input_dim}"
            )
        if "peptide" in selected:
            esmc.max_length = config.model.peptide_max_length
            build_embedding_cache(
                esmc,
                sequences["peptide"],
                cache_root=cache_dir,
                entity="peptide",
                batch_size=config.cache.precompute_batch_size,
                shard_size=config.cache.shard_size,
                compression=config.cache.compression,
                overwrite=args.overwrite,
            )
        if "hla" in selected:
            esmc.max_length = config.model.hla_max_length
            build_embedding_cache(
                esmc,
                sequences["hla"],
                cache_root=cache_dir,
                entity="hla",
                batch_size=config.cache.precompute_batch_size,
                shard_size=config.cache.shard_size,
                compression=config.cache.compression,
                overwrite=args.overwrite,
            )
        del esmc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "tcr" in selected:
        tcr_bert = TCRBertResidueEncoder(
            config.paths.tcr_bert_model,
            max_length=config.model.tcr_max_length,
            device=device,
            dtype=config.cache.dtype,
        )
        if tcr_bert.hidden_size != config.model.tcr_input_dim:
            raise ValueError(
                f"TCR-BERT hidden size {tcr_bert.hidden_size} does not match "
                f"configured input dim {config.model.tcr_input_dim}"
            )
        build_embedding_cache(
            tcr_bert,
            sequences["tcr"],
            cache_root=cache_dir,
            entity="tcr",
            batch_size=config.cache.precompute_batch_size,
            shard_size=config.cache.shard_size,
            compression=config.cache.compression,
            overwrite=args.overwrite,
        )
    print(f"Embedding cache ready: {cache_dir}")


if __name__ == "__main__":
    main()
