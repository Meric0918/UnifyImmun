#!/usr/bin/env python3
"""Run the confirmed fold across the configured three random seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from plm_unified.artifacts import create_artifact_timestamp, timestamped_filename
from plm_unified.config import apply_overrides, load_config
from plm_unified.metrics import task_score
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
    batch_timestamp = create_artifact_timestamp()
    checkpoints = []
    for seed in seeds:
        run_timestamp = create_artifact_timestamp()
        command = [
            sys.executable,
            str(training_script),
            "--config",
            str(Path(args.config).resolve()),
            "--fold",
            str(config.training.fold),
            "--seed",
            str(seed),
            "--run-timestamp",
            run_timestamp,
        ]
        if args.device:
            command.extend(["--device", args.device])
        if args.swanlab_mode:
            command.extend(["--swanlab-mode", args.swanlab_mode])
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        run_dir = (
            config.paths.output_root
            / f"fold_{config.training.fold}"
            / f"seed_{seed}"
        )
        checkpoints.append(
            {
                "phla": run_dir
                / timestamped_filename("best_phla.pt", run_timestamp),
                "ptcr": run_dir
                / timestamped_filename("best_ptcr.pt", run_timestamp),
            }
        )
    summary = aggregate_checkpoints(checkpoints)
    summary_path = (
        config.paths.output_root
        / f"fold_{config.training.fold}"
        / timestamped_filename("seed_summary.json", batch_timestamp)
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Seed summary: {summary_path}")


def aggregate_checkpoints(checkpoints: list[dict[str, Path]]) -> dict:
    runs = []
    for task_paths in checkpoints:
        task_checkpoints = {
            task: load_training_checkpoint(path, map_location="cpu")
            for task, path in task_paths.items()
        }
        seeds = {
            checkpoint["config"]["training"]["seed"]
            for checkpoint in task_checkpoints.values()
        }
        if len(seeds) != 1:
            raise ValueError(f"Task checkpoints have different seeds: {seeds}")
        runs.append(
            {
                "paths": {
                    task: str(path) for task, path in task_paths.items()
                },
                "seed": seeds.pop(),
                "best_scores": {
                    task: float(
                        checkpoint.get(
                            "best_score",
                            task_score(checkpoint["validation_metrics"]),
                        )
                    )
                    for task, checkpoint in task_checkpoints.items()
                },
                "metrics": {
                    task: checkpoint["validation_metrics"]
                    for task, checkpoint in task_checkpoints.items()
                },
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
        score_values = [float(run["best_scores"][task]) for run in runs]
        aggregate[task]["best_score"] = {
            "mean": statistics.fmean(score_values),
            "std": statistics.stdev(score_values) if len(score_values) > 1 else 0.0,
        }
    return {"runs": runs, "aggregate": aggregate}


if __name__ == "__main__":
    main()
