"""Typed configuration for the unified PLM stage-1 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass
class PathsConfig:
    project_root: Path
    data_root: Path
    cache_root: Path
    output_root: Path
    swanlog_root: Path
    esmc_model: Path = Path("/home/mjp/model/ESMC-300M")
    tcr_bert_model: Path = Path("/home/mjp/model/TCR-Bert")


@dataclass
class ModelConfig:
    peptide_input_dim: int = 960
    hla_input_dim: int = 960
    tcr_input_dim: int = 768
    adapter_hidden_dim: int = 256
    model_dim: int = 128
    num_attention_heads: int = 4
    feedforward_dim: int = 512
    adapter_dropout: float = 0.1
    attention_dropout: float = 0.1
    classifier_dropout: float = 0.2
    peptide_min_length: int = 8
    peptide_max_length: int = 15
    hla_max_length: int = 34
    tcr_max_length: int = 34


@dataclass
class CacheConfig:
    dtype: str = "float16"
    shard_size: int = 16384
    precompute_batch_size: int = 32
    compression: Optional[str] = None
    hla_splits: list[str] = field(
        default_factory=lambda: ["train", "val"]
    )
    tcr_splits: list[str] = field(
        default_factory=lambda: ["train", "val"]
    )


@dataclass
class TrainingConfig:
    fold: int = 1
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42, 3407, 2026])
    batch_size: int = 1024
    online_batch_size: int = 32
    num_workers: int = 2
    warmup_epochs_per_task: int = 2
    max_rounds: int = 10
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    adapter_lr: float = 3e-4
    cross_attention_lr: float = 2e-4
    pooling_lr: float = 3e-4
    classifier_lr: float = 3e-4
    weight_decay: float = 1e-2
    scheduler_warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    fgm_epsilon: float = 1.0
    threshold: float = 0.5
    mixed_precision: str = "bf16"
    log_every_steps: int = 50
    pin_memory: bool = True
    persistent_workers: bool = False
    device: str = "auto"


@dataclass
class SwanLabConfig:
    project: str = "unifyimmun"
    workspace: Optional[str] = None
    mode: str = "online"
    group: str = "plm-stage1-fold1"
    experiment_prefix: str = "PLM-Stage1"
    tags: list[str] = field(
        default_factory=lambda: ["ESMC-300M", "TCR-BERT", "FGM", "stage1"]
    )


@dataclass
class ExperimentConfig:
    paths: PathsConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    source_path: Optional[Path] = None

    def validate(self, require_models: bool = False) -> None:
        if self.model.model_dim % self.model.num_attention_heads != 0:
            raise ValueError("model_dim must be divisible by num_attention_heads")
        if self.model.peptide_min_length > self.model.peptide_max_length:
            raise ValueError("peptide_min_length cannot exceed peptide_max_length")
        for name in ("peptide_max_length", "hla_max_length", "tcr_max_length"):
            if getattr(self.model, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.cache.dtype != "float16":
            raise ValueError("cache.dtype must be float16 for the HDF5 cache")
        if self.cache.shard_size < 1:
            raise ValueError("cache.shard_size must be positive")
        if self.training.fold < 1:
            raise ValueError("fold must be a positive integer")
        if (
            self.training.batch_size < 1
            or self.training.online_batch_size < 1
            or self.cache.precompute_batch_size < 1
        ):
            raise ValueError("batch sizes must be positive")
        if self.training.warmup_epochs_per_task < 0:
            raise ValueError("warmup_epochs_per_task cannot be negative")
        if self.training.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if self.training.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.training.log_every_steps < 1:
            raise ValueError("log_every_steps must be positive")
        if not 0 <= self.training.scheduler_warmup_ratio < 1:
            raise ValueError("scheduler_warmup_ratio must be in [0, 1)")
        if self.training.fgm_epsilon <= 0:
            raise ValueError("fgm_epsilon must be positive")
        if self.training.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: none, fp16, bf16")
        if self.swanlab.mode not in {"online", "local", "offline", "disabled"}:
            raise ValueError(
                "swanlab.mode must be one of: online, local, offline, disabled"
            )
        if require_models:
            for name, path in (
                ("ESM-C", self.paths.esmc_model),
                ("TCR-BERT", self.paths.tcr_bert_model),
            ):
                if not path.is_dir():
                    raise FileNotFoundError(f"{name} checkpoint directory not found: {path}")

    def cache_dir(self) -> Path:
        return self.paths.cache_root / f"fold_{self.training.fold}"

    def run_dir(self) -> Path:
        return (
            self.paths.output_root
            / f"fold_{self.training.fold}"
            / f"seed_{self.training.seed}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return _serialise(asdict(self))


def _serialise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    return value


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _section(
    raw: Mapping[str, Any], name: str, allowed: Iterable[str]
) -> Dict[str, Any]:
    values = dict(raw.get(name, {}) or {})
    unexpected = sorted(set(values) - set(allowed))
    if unexpected:
        raise ValueError(f"Unknown keys in '{name}': {', '.join(unexpected)}")
    return values


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config and resolve every relative path against that file."""

    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load the experiment config. "
            "Install requirements-plm.txt."
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The root YAML value must be a mapping")

    base_dir = config_path.parent
    path_values = _section(raw, "paths", PathsConfig.__dataclass_fields__)
    required_paths = {"project_root", "data_root", "cache_root", "output_root", "swanlog_root"}
    missing_paths = sorted(required_paths - set(path_values))
    if missing_paths:
        raise ValueError(f"Missing required paths: {', '.join(missing_paths)}")
    path_values = {
        key: _resolve_path(value, base_dir) for key, value in path_values.items()
    }

    model_values = _section(raw, "model", ModelConfig.__dataclass_fields__)
    cache_values = _section(raw, "cache", CacheConfig.__dataclass_fields__)
    training_values = _section(raw, "training", TrainingConfig.__dataclass_fields__)
    swanlab_values = _section(raw, "swanlab", SwanLabConfig.__dataclass_fields__)

    config = ExperimentConfig(
        paths=PathsConfig(**path_values),
        model=ModelConfig(**model_values),
        cache=CacheConfig(**cache_values),
        training=TrainingConfig(**training_values),
        swanlab=SwanLabConfig(**swanlab_values),
        source_path=config_path,
    )
    config.validate()
    return config


def apply_overrides(
    config: ExperimentConfig,
    *,
    fold: Optional[int] = None,
    seed: Optional[int] = None,
    swanlab_mode: Optional[str] = None,
    device: Optional[str] = None,
) -> ExperimentConfig:
    """Apply common CLI overrides in place and validate the result."""

    if fold is not None:
        config.training.fold = fold
    if seed is not None:
        config.training.seed = seed
    if swanlab_mode is not None:
        config.swanlab.mode = swanlab_mode
    if device is not None:
        config.training.device = device
    config.validate()
    return config
