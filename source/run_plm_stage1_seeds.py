#!/usr/bin/env python3
"""Run the confirmed fold across the configured three random seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from plm_unified.config import apply_overrides, load_config
from plm_unified.trainer import load_training_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / "plm_stage1.yaml"
        ),
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device")
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), fold=args.fold)
    seeds = args.seeds or config.training.seeds
    training_script = Path(__file__).with_name("train_plm_stage1.py")
    checkpoints = []
    for seed in seeds:
        command = [
            sys.executable,
            str(training_script),
            "--config",
            str(Path(args.config).resolve()),
            "--fold",
            str(config.training.fold),
            "--seed",
            str(seed),
        ]
        if args.device:
            command.extend(["--device", args.device])
        if args.swanlab_mode:
            command.extend(["--swanlab-mode", args.swanlab_mode])
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        checkpoints.append(
            config.paths.output_root
            / f"fold_{config.training.fold}"
            / f"seed_{seed}"
            / "best_joint.pt"
        )
    summary = aggregate_checkpoints(checkpoints)
    summary_path = (
        config.paths.output_root
        / f"fold_{config.training.fold}"
        / "seed_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Seed summary: {summary_path}")


def aggregate_checkpoints(checkpoints: list[Path]) -> dict:
    runs = []
    for checkpoint_path in checkpoints:
        checkpoint = load_training_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
        runs.append(
            {
                "path": str(checkpoint_path),
                "seed": checkpoint["config"]["training"]["seed"],
                "joint_score": checkpoint["best_joint_score"],
                "metrics": checkpoint["best_val_metrics"],
            }
        )

    aggregate = {}
    metric_names = ("auroc", "aupr", "accuracy", "mcc", "f1")
    for task in ("phla", "ptcr"):
        aggregate[task] = {}
        for metric in metric_names:
            values = [float(run["metrics"][task][metric]) for run in runs]
            aggregate[task][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    joint_values = [float(run["joint_score"]) for run in runs]
    aggregate["joint_score"] = {
        "mean": statistics.fmean(joint_values),
        "std": statistics.stdev(joint_values) if len(joint_values) > 1 else 0.0,
    }
    return {"runs": runs, "aggregate": aggregate}


if __name__ == "__main__":
    main()
