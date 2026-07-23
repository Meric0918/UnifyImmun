#!/usr/bin/env python3
"""Server-side smoke test for local ESM-C and TCR-BERT checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm_unified.config import apply_overrides, load_config
from plm_unified.encoders import (
    ESMCResidueEncoder,
    TCRBertResidueEncoder,
    encoder_metadata,
)
from plm_unified.trainer import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / "plm_stage1.yaml"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), device=args.device)
    config.validate(require_models=True)
    device = resolve_device(config.training.device)
    esmc = ESMCResidueEncoder(
        config.paths.esmc_model,
        max_length=config.model.hla_max_length,
        device=device,
        dtype=config.cache.dtype,
    )
    esmc_output = esmc.encode(["LLFGYPVYV", "YDSGYREKYRQADVSNLYFRYDFYTLAADAYTWY"])
    tcr_bert = TCRBertResidueEncoder(
        config.paths.tcr_bert_model,
        max_length=config.model.tcr_max_length,
        device=device,
        dtype=config.cache.dtype,
    )
    tcr_output = tcr_bert.encode(["CASSLGQAYEQYF", "CASSPVTGGIYGYTF"])
    report = {
        "esmc": {
            **encoder_metadata(esmc),
            "shape": list(esmc_output.hidden.shape),
            "lengths": esmc_output.lengths.cpu().tolist(),
        },
        "tcr_bert": {
            **encoder_metadata(tcr_bert),
            "shape": list(tcr_output.hidden.shape),
            "lengths": tcr_output.lengths.cpu().tolist(),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
