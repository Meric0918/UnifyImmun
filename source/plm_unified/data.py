"""CSV discovery and cache-backed pair datasets."""

from __future__ import annotations

import csv
import random
from array import array
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cache import CacheLocator, EmbeddingCache

Task = Literal["phla", "ptcr"]


def normalise_sequence(sequence: str) -> str:
    value = "".join(str(sequence).split()).replace("-", "").upper()
    if not value:
        raise ValueError("Encountered an empty sequence")
    return value


def split_path(
    data_root: str | Path,
    task: Task,
    split: str,
    fold: int,
) -> Path:
    root = Path(data_root)
    task_dir = root / ("data_HLA" if task == "phla" else "data_TCR")
    filename = (
        f"{split}_fold_{fold}.csv"
        if split in {"train", "val"}
        else f"{split}_set.csv"
    )
    path = task_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    return path


def task_columns(task: Task) -> tuple[str, str, str]:
    if task == "phla":
        return "peptide", "HLA", "label"
    if task == "ptcr":
        return "peptide", "tcr", "label"
    raise ValueError(f"Unknown task: {task}")


def iter_column(path: str | Path, column: str) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' not found in {path}")
        for row_number, row in enumerate(reader, start=2):
            raw = row.get(column)
            if raw is None or not str(raw).strip():
                raise ValueError(f"Missing {column} at {path}:{row_number}")
            yield normalise_sequence(raw)


def collect_cache_sequences(
    data_root: str | Path,
    fold: int,
    hla_splits: Sequence[str],
    tcr_splits: Sequence[str],
    peptide_min_length: int,
    peptide_max_length: int,
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    """Collect unique peptide/HLA/TCR strings required by configured splits."""

    entities: dict[str, set[str]] = {
        "peptide": set(),
        "hla": set(),
        "tcr": set(),
    }
    stats: dict[str, dict[str, int]] = {}
    for task, splits in (("phla", hla_splits), ("ptcr", tcr_splits)):
        peptide_col, receptor_col, _ = task_columns(task)
        for split in splits:
            path = split_path(data_root, task, split, fold)
            receptor_entity = "hla" if task == "phla" else "tcr"
            split_stats = {
                "total": 0,
                "kept": 0,
                "peptide_too_short": 0,
                "peptide_too_long": 0,
            }
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {peptide_col, receptor_col}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(f"{path} must contain columns: {sorted(required)}")
                for row_number, row in enumerate(reader, start=2):
                    split_stats["total"] += 1
                    peptide = normalise_sequence(row[peptide_col])
                    if len(peptide) < peptide_min_length:
                        split_stats["peptide_too_short"] += 1
                        continue
                    if len(peptide) > peptide_max_length:
                        split_stats["peptide_too_long"] += 1
                        continue
                    receptor = normalise_sequence(row[receptor_col])
                    entities["peptide"].add(peptide)
                    entities[receptor_entity].add(receptor)
                    split_stats["kept"] += 1
            stats[f"{task}/{split}"] = split_stats
    return entities, stats


class CachedPairDataset(Dataset):
    """Compact pair locations resolved once from a CSV file."""

    def __init__(
        self,
        csv_path: str | Path,
        task: Task,
        peptide_cache: EmbeddingCache,
        receptor_cache: EmbeddingCache,
        *,
        peptide_min_length: int = 8,
        peptide_max_length: int = 15,
        limit: int | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.task = task
        peptide_col, receptor_col, label_col = task_columns(task)
        values = array("q")
        self.filter_stats = {
            "total": 0,
            "kept": 0,
            "peptide_too_short": 0,
            "peptide_too_long": 0,
        }

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {peptide_col, receptor_col, label_col}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"{self.csv_path} must contain columns: {sorted(required)}"
                )
            for row_number, row in enumerate(reader, start=2):
                if limit is not None and len(values) // 5 >= limit:
                    break
                self.filter_stats["total"] += 1
                peptide = normalise_sequence(row[peptide_col])
                if len(peptide) < peptide_min_length:
                    self.filter_stats["peptide_too_short"] += 1
                    continue
                if len(peptide) > peptide_max_length:
                    self.filter_stats["peptide_too_long"] += 1
                    continue
                receptor = normalise_sequence(row[receptor_col])
                try:
                    peptide_location = peptide_cache.resolve(peptide)
                    receptor_location = receptor_cache.resolve(receptor)
                except KeyError as exc:
                    raise KeyError(
                        f"Cache miss while loading {self.csv_path}:{row_number}: {exc}"
                    ) from exc
                try:
                    label = int(row[label_col])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid label at {self.csv_path}:{row_number}"
                    ) from exc
                if label not in {0, 1}:
                    raise ValueError(
                        f"Label must be 0 or 1 at {self.csv_path}:{row_number}"
                    )
                values.extend(
                    (
                        peptide_location.shard,
                        peptide_location.row,
                        receptor_location.shard,
                        receptor_location.row,
                        label,
                    )
                )
                self.filter_stats["kept"] += 1
        self.locations = np.frombuffer(values, dtype=np.int64).reshape(-1, 5).copy()
        if not len(self.locations):
            raise ValueError(f"No usable rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.locations[index]


class CachedPairCollator:
    """Load one batch with grouped HDF5 reads from both entity caches."""

    def __init__(
        self,
        peptide_cache: EmbeddingCache,
        receptor_cache: EmbeddingCache,
    ):
        self.peptide_cache = peptide_cache
        self.receptor_cache = receptor_cache

    def __call__(self, rows: Sequence[np.ndarray]) -> dict[str, torch.Tensor]:
        matrix = np.asarray(rows, dtype=np.int64)
        peptide_locations = [
            CacheLocator(int(row[0]), int(row[1]), 0) for row in matrix
        ]
        receptor_locations = [
            CacheLocator(int(row[2]), int(row[3]), 0) for row in matrix
        ]
        peptide_hidden, peptide_mask = self.peptide_cache.get_many(peptide_locations)
        receptor_hidden, receptor_mask = self.receptor_cache.get_many(
            receptor_locations
        )
        return {
            "peptide_hidden": torch.from_numpy(peptide_hidden),
            "peptide_mask": torch.from_numpy(peptide_mask),
            "receptor_hidden": torch.from_numpy(receptor_hidden),
            "receptor_mask": torch.from_numpy(receptor_mask),
            "labels": torch.from_numpy(matrix[:, 4].copy()).long(),
        }


class OnlinePairDataset(Dataset):
    """Small debug dataset that keeps raw sequences for online frozen-PLM encoding."""

    def __init__(
        self,
        csv_path: str | Path,
        task: Task,
        *,
        peptide_min_length: int = 8,
        peptide_max_length: int = 15,
        limit: int,
    ):
        self.csv_path = Path(csv_path)
        self.task = task
        peptide_col, receptor_col, label_col = task_columns(task)
        self.rows: list[tuple[str, str, int]] = []
        self.filter_stats = {
            "total": 0,
            "kept": 0,
            "peptide_too_short": 0,
            "peptide_too_long": 0,
        }
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {peptide_col, receptor_col, label_col}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"{self.csv_path} must contain columns: {sorted(required)}"
                )
            for row_number, row in enumerate(reader, start=2):
                if len(self.rows) >= limit:
                    break
                self.filter_stats["total"] += 1
                peptide = normalise_sequence(row[peptide_col])
                if len(peptide) < peptide_min_length:
                    self.filter_stats["peptide_too_short"] += 1
                    continue
                if len(peptide) > peptide_max_length:
                    self.filter_stats["peptide_too_long"] += 1
                    continue
                receptor = normalise_sequence(row[receptor_col])
                try:
                    label = int(row[label_col])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid label at {self.csv_path}:{row_number}"
                    ) from exc
                if label not in {0, 1}:
                    raise ValueError(
                        f"Label must be 0 or 1 at {self.csv_path}:{row_number}"
                    )
                self.rows.append((peptide, receptor, label))
                self.filter_stats["kept"] += 1
        if not self.rows:
            raise ValueError(f"No usable rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, str, int]:
        return self.rows[index]


def _collate_online_pairs(
    rows: Sequence[tuple[str, str, int]],
) -> dict[str, list[str] | torch.Tensor]:
    return {
        "peptide_sequences": [row[0] for row in rows],
        "receptor_sequences": [row[1] for row in rows],
        "labels": torch.tensor([row[2] for row in rows], dtype=torch.long),
    }


def make_cached_dataloader(
    csv_path: str | Path,
    task: Task,
    peptide_cache: EmbeddingCache,
    receptor_cache: EmbeddingCache,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    peptide_min_length: int = 8,
    peptide_max_length: int = 15,
    limit: int | None = None,
) -> DataLoader:
    dataset = CachedPairDataset(
        csv_path,
        task,
        peptide_cache,
        receptor_cache,
        peptide_min_length=peptide_min_length,
        peptide_max_length=peptide_max_length,
        limit=limit,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    collator = CachedPairCollator(
        peptide_cache.worker_reader(),
        receptor_cache.worker_reader(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
        collate_fn=collator,
        generator=generator,
        worker_init_fn=_seed_worker,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


def make_online_dataloader(
    csv_path: str | Path,
    task: Task,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    peptide_min_length: int,
    peptide_max_length: int,
    limit: int,
) -> DataLoader:
    dataset = OnlinePairDataset(
        csv_path,
        task,
        peptide_min_length=peptide_min_length,
        peptide_max_length=peptide_max_length,
        limit=limit,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=False,
        collate_fn=_collate_online_pairs,
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
