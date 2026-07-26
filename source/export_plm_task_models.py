#!/usr/bin/env python3
"""Split unified Stage-1 checkpoints into deployable pHLA and pTCR models."""

from __future__ import annotations

import argparse
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp
from plm_unified.trainer import export_task_checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="One or more trusted legacy full checkpoints, such as best_joint.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; valid only when exporting one checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs only if their timestamped paths already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir and len(args.checkpoint) != 1:
        raise ValueError("--output-dir requires exactly one --checkpoint")
    artifact_timestamp = create_artifact_timestamp()
    for checkpoint in args.checkpoint:
        outputs = export_task_checkpoints(
            checkpoint,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            artifact_timestamp=artifact_timestamp,
        )
        print(f"Source: {checkpoint.resolve()}")
        print(f"  pHLA: {outputs['phla']}")
        print(f"  pTCR: {outputs['ptcr']}")


if __name__ == "__main__":
    main()
