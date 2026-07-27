"""Small SwanLab adapter that keeps tests and disabled runs dependency-free."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .artifacts import create_artifact_timestamp, validate_artifact_timestamp
from .config import ExperimentConfig


class SwanLabTracker:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        resume_id: Optional[str] = None,
        run_timestamp: Optional[str] = None,
    ):
        self.mode = config.swanlab.mode
        self.run = None
        self.run_id: Optional[str] = None
        self.text_factory = None
        self.run_timestamp = validate_artifact_timestamp(
            run_timestamp or create_artifact_timestamp()
        )
        if self.mode == "disabled":
            return
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab logging is required for this run. "
                "Install requirements-plm.txt or pass --swanlab-mode disabled."
            ) from exc
        self.text_factory = swanlab.Text

        logdir = config.paths.swanlog_root
        logdir.mkdir(parents=True, exist_ok=True)
        group = config.swanlab.group.format(fold=config.training.fold)
        experiment_name = (
            f"{config.swanlab.experiment_prefix}"
            f"-fold{config.training.fold}-seed{config.training.seed}"
            f"-{self.run_timestamp}"
        )
        if "stage2" in config.swanlab.experiment_prefix.lower():
            description = (
                "Unified ESM-C/TCR-BERT progressive stage-2 fine-tuning with "
                "alternating pHLA/pTCR training and task-specific checkpoints."
            )
        else:
            description = (
                "Unified ESM-C/TCR-BERT stage-1 training with shared Peptide "
                "Adapter and module-targeted FGM."
            )
        init_kwargs: dict[str, Any] = {
            "project": config.swanlab.project,
            "workspace": config.swanlab.workspace,
            "experiment_name": experiment_name,
            "description": description,
            "job_type": "train",
            "group": group,
            "tags": config.swanlab.tags,
            "config": config.to_dict(),
            "logdir": str(logdir),
            "mode": self.mode,
        }
        if resume_id:
            init_kwargs.update({"id": resume_id, "resume": "must"})
        else:
            init_kwargs["resume"] = "never"
        self.run = swanlab.init(**init_kwargs)
        self.run_id = getattr(self.run, "id", None)

    def log(self, values: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        if self.run is None:
            return
        payload = {
            key: self.text_factory(value)
            if isinstance(value, str) and self.text_factory is not None
            else value
            for key, value in values.items()
        }
        if step is not None:
            payload.setdefault("global_step", step)
        try:
            self.run.log(payload, step=step)
        except TypeError:
            self.run.log(payload)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None

    def __enter__(self) -> "SwanLabTracker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.finish()
