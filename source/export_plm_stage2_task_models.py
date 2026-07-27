#!/usr/bin/env python3
"""Export separate pHLA and pTCR stage-2 task checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from plm_unified.artifacts import (
    create_artifact_timestamp,
    timestamped_filename,
    validate_artifact_timestamp,
)
from plm_unified.stage2_trainer import (
    STAGE2_TASK_CHECKPOINT_FILENAMES,
    build_stage2_task_checkpoint_state,
    load_stage2_checkpoint,
)
from plm_unified.trainer import _atomic_torch_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Full stage2_last or stage2b_best_joint checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory; defaults to the checkpoint directory.",
    )
    parser.add_argument("--run-timestamp")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing timestamp-matched task checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.checkpoint.resolve()
    checkpoint = load_stage2_checkpoint(source, map_location="cpu")
    destination = (args.output_dir or source.parent).resolve()
    timestamp = validate_artifact_timestamp(
        args.run_timestamp
        or checkpoint.get("run_timestamp")
        or create_artifact_timestamp()
    )
    targets = {
        task: destination / timestamped_filename(filename, timestamp)
        for task, filename in STAGE2_TASK_CHECKPOINT_FILENAMES.items()
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Task checkpoint already exists: "
            + ", ".join(str(path) for path in existing)
        )
    for task, target in targets.items():
        state = build_stage2_task_checkpoint_state(
            checkpoint,
            task,
            source_checkpoint=source,
        )
        _atomic_torch_save(state, target)
        print(f"[{task}] {target}")


if __name__ == "__main__":
    main()
