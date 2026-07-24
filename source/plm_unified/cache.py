"""Sharded HDF5 residue cache with a SQLite sequence index."""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .encoders import FrozenResidueEncoder, encoder_metadata


MANIFEST_VERSION = 1


@dataclass(frozen=True)
class CacheLocator:
    shard: int
    row: int
    length: int


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required for the embedding cache. "
            "Install requirements-plm.txt."
        ) from exc
    return h5py


class EmbeddingCacheWriter:
    """Incrementally write fixed-width, residue-level HDF5 shards."""

    def __init__(
        self,
        root: str | Path,
        entity: str,
        *,
        hidden_dim: int,
        max_length: int,
        model_metadata: Mapping[str, Any],
        dtype: str = "float16",
        shard_size: int = 4096,
        compression: str | None = None,
        overwrite: bool = False,
    ):
        if dtype != "float16":
            raise ValueError("The HDF5 cache currently supports dtype='float16' only")
        if shard_size < 1:
            raise ValueError("shard_size must be positive")
        self.h5py = _require_h5py()
        self.entity = entity
        self.entity_dir = Path(root).resolve() / entity
        if self.entity_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Cache already exists: {self.entity_dir}. Use --overwrite explicitly."
                )
            shutil.rmtree(self.entity_dir)
        self.shards_dir = self.entity_dir / "shards"
        self.shards_dir.mkdir(parents=True)
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.dtype = dtype
        self.shard_size = shard_size
        self.compression = compression
        self.model_metadata = dict(model_metadata)

        self.connection = sqlite3.connect(self.entity_dir / "index.sqlite")
        self.connection.execute(
            """
            CREATE TABLE sequence_index (
                sequence TEXT PRIMARY KEY,
                shard INTEGER NOT NULL,
                row_idx INTEGER NOT NULL,
                length INTEGER NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX shard_row_index ON sequence_index(shard, row_idx)"
        )

        self.shard_id = -1
        self.shard_count = 0
        self.total_count = 0
        self.shard_summaries: list[dict[str, Any]] = []
        self.handle = None
        self.datasets: dict[str, Any] = {}

    def __enter__(self) -> "EmbeddingCacheWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._close_shard()
            self.connection.close()

    def _open_shard(self) -> None:
        self._close_shard()
        self.shard_id += 1
        self.shard_count = 0
        shard_path = self.shards_dir / f"part-{self.shard_id:05d}.h5"
        self.handle = self.h5py.File(shard_path, "w")
        chunk_rows = min(64, self.shard_size)
        string_type = self.h5py.string_dtype(encoding="utf-8")
        self.datasets = {
            "sequences": self.handle.create_dataset(
                "sequences",
                shape=(0,),
                maxshape=(self.shard_size,),
                dtype=string_type,
                chunks=(chunk_rows,),
            ),
            "hidden": self.handle.create_dataset(
                "hidden",
                shape=(0, self.max_length, self.hidden_dim),
                maxshape=(self.shard_size, self.max_length, self.hidden_dim),
                dtype=np.float16,
                chunks=(chunk_rows, self.max_length, self.hidden_dim),
                compression=self.compression,
            ),
            "mask": self.handle.create_dataset(
                "mask",
                shape=(0, self.max_length),
                maxshape=(self.shard_size, self.max_length),
                dtype=np.uint8,
                chunks=(chunk_rows, self.max_length),
                compression=self.compression,
            ),
            "lengths": self.handle.create_dataset(
                "lengths",
                shape=(0,),
                maxshape=(self.shard_size,),
                dtype=np.int16,
                chunks=(chunk_rows,),
            ),
        }

    def _close_shard(self) -> None:
        if self.handle is None:
            return
        self.handle.flush()
        shard_path = Path(self.handle.filename)
        self.handle.close()
        self.shard_summaries.append(
            {
                "id": self.shard_id,
                "file": shard_path.relative_to(self.entity_dir).as_posix(),
                "count": self.shard_count,
                "bytes": shard_path.stat().st_size,
            }
        )
        self.handle = None
        self.datasets = {}

    def add_batch(
        self,
        sequences: Sequence[str],
        hidden: np.ndarray,
        mask: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        if not sequences:
            return
        hidden = np.asarray(hidden, dtype=np.float16)
        mask = np.asarray(mask, dtype=np.uint8)
        lengths = np.asarray(lengths, dtype=np.int16)
        expected_hidden_shape = (len(sequences), self.max_length, self.hidden_dim)
        if hidden.shape != expected_hidden_shape:
            raise ValueError(
                f"Expected hidden shape {expected_hidden_shape}, got {hidden.shape}"
            )
        if mask.shape != (len(sequences), self.max_length):
            raise ValueError("Mask shape does not match cache configuration")
        if lengths.shape != (len(sequences),):
            raise ValueError("Lengths shape does not match sequence batch")

        offset = 0
        while offset < len(sequences):
            if self.handle is None or self.shard_count >= self.shard_size:
                self._open_shard()
            available = self.shard_size - self.shard_count
            take = min(available, len(sequences) - offset)
            start = self.shard_count
            end = start + take
            for dataset in self.datasets.values():
                dataset.resize((end,) + dataset.shape[1:])

            sequence_slice = list(sequences[offset : offset + take])
            self.datasets["sequences"][start:end] = sequence_slice
            self.datasets["hidden"][start:end] = hidden[offset : offset + take]
            self.datasets["mask"][start:end] = mask[offset : offset + take]
            self.datasets["lengths"][start:end] = lengths[offset : offset + take]

            rows = [
                (
                    sequence,
                    self.shard_id,
                    start + local_row,
                    int(lengths[offset + local_row]),
                )
                for local_row, sequence in enumerate(sequence_slice)
            ]
            try:
                self.connection.executemany(
                    "INSERT INTO sequence_index VALUES (?, ?, ?, ?)", rows
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Duplicate sequence encountered while writing cache") from exc

            self.shard_count = end
            self.total_count += take
            offset += take

    def close(self) -> None:
        self._close_shard()
        self.connection.commit()
        self.connection.close()
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "entity": self.entity,
            "count": self.total_count,
            "hidden_dim": self.hidden_dim,
            "max_length": self.max_length,
            "dtype": self.dtype,
            "shard_size": self.shard_size,
            "compression": self.compression,
            "model": self.model_metadata,
            "shards": self.shard_summaries,
        }
        (self.entity_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class EmbeddingCache:
    """Read cached representations and batch HDF5 access by shard."""

    def __init__(
        self,
        entity_dir: str | Path,
        *,
        expected_hidden_dim: int | None = None,
        expected_fingerprint: str | None = None,
        load_index: bool = True,
        max_open_shards: int = 64,
    ):
        self.h5py = _require_h5py()
        self.entity_dir = Path(entity_dir).resolve()
        manifest_path = self.entity_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError("Unsupported cache manifest version")
        self.hidden_dim = int(self.manifest["hidden_dim"])
        self.max_length = int(self.manifest["max_length"])
        if expected_hidden_dim is not None and self.hidden_dim != expected_hidden_dim:
            raise ValueError(
                f"Cache hidden dim {self.hidden_dim} != expected {expected_hidden_dim}"
            )
        fingerprint = self.manifest.get("model", {}).get("checkpoint_fingerprint")
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError("Cache checkpoint fingerprint does not match configured model")

        self.max_open_shards = max_open_shards
        self._handles: OrderedDict[int, Any] = OrderedDict()
        self._index: dict[str, CacheLocator] | None = None
        if load_index:
            self._index = self._load_index()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["h5py"] = None
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        self.h5py = _require_h5py()

    def __del__(self):
        self.close()

    def close(self) -> None:
        handles = getattr(self, "_handles", {})
        for handle in handles.values():
            handle.close()
        handles.clear()

    def _load_index(self) -> dict[str, CacheLocator]:
        connection = sqlite3.connect(
            f"file:{self.entity_dir / 'index.sqlite'}?mode=ro", uri=True
        )
        try:
            return {
                sequence: CacheLocator(int(shard), int(row), int(length))
                for sequence, shard, row, length in connection.execute(
                    "SELECT sequence, shard, row_idx, length FROM sequence_index"
                )
            }
        finally:
            connection.close()

    def resolve(self, sequence: str) -> CacheLocator:
        if self._index is None:
            connection = sqlite3.connect(
                f"file:{self.entity_dir / 'index.sqlite'}?mode=ro", uri=True
            )
            try:
                row = connection.execute(
                    "SELECT shard, row_idx, length FROM sequence_index WHERE sequence = ?",
                    (sequence,),
                ).fetchone()
            finally:
                connection.close()
            if row is None:
                raise KeyError(sequence)
            return CacheLocator(int(row[0]), int(row[1]), int(row[2]))
        try:
            return self._index[sequence]
        except KeyError as exc:
            raise KeyError(f"Sequence missing from {self.entity_dir.name} cache: {sequence}") from exc

    def worker_reader(self) -> "EmbeddingCache":
        """Return a lightweight reader for a spawned DataLoader worker."""

        return EmbeddingCache(
            self.entity_dir,
            load_index=False,
            max_open_shards=self.max_open_shards,
        )

    def _handle(self, shard_id: int):
        if shard_id in self._handles:
            handle = self._handles.pop(shard_id)
            self._handles[shard_id] = handle
            return handle
        shard = self.manifest["shards"][shard_id]
        handle = self.h5py.File(self.entity_dir / shard["file"], "r")
        self._handles[shard_id] = handle
        while len(self._handles) > self.max_open_shards:
            _, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        return handle

    def get_many(self, locators: Sequence[CacheLocator]) -> tuple[np.ndarray, np.ndarray]:
        """Return hidden states and masks in caller order using grouped shard reads."""

        count = len(locators)
        hidden = np.empty(
            (count, self.max_length, self.hidden_dim), dtype=np.float16
        )
        mask = np.empty((count, self.max_length), dtype=np.bool_)
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for output_row, locator in enumerate(locators):
            by_shard.setdefault(locator.shard, []).append((output_row, locator.row))

        for shard_id, requests in by_shard.items():
            handle = self._handle(shard_id)
            requested_rows = np.asarray([row for _, row in requests], dtype=np.int64)
            unique_rows, inverse = np.unique(requested_rows, return_inverse=True)
            shard_hidden = handle["hidden"][unique_rows]
            shard_mask = handle["mask"][unique_rows].astype(np.bool_, copy=False)
            for request_index, (output_row, _) in enumerate(requests):
                source_row = inverse[request_index]
                hidden[output_row] = shard_hidden[source_row]
                mask[output_row] = shard_mask[source_row]
        return hidden, mask


def batched(values: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def build_embedding_cache(
    encoder: FrozenResidueEncoder,
    sequences: Iterable[str],
    *,
    cache_root: str | Path,
    entity: str,
    batch_size: int,
    shard_size: int,
    compression: str | None,
    overwrite: bool,
) -> Path:
    """Encode deterministic unique sequences and write one entity cache."""

    unique_sequences = sorted(set(sequences))
    metadata = encoder_metadata(encoder)
    with EmbeddingCacheWriter(
        cache_root,
        entity,
        hidden_dim=encoder.hidden_size,
        max_length=encoder.max_length,
        model_metadata=metadata,
        shard_size=shard_size,
        compression=compression,
        overwrite=overwrite,
    ) as writer:
        for sequence_batch in _progress(
            batched(unique_sequences, batch_size),
            total=(len(unique_sequences) + batch_size - 1) // batch_size,
            description=f"cache:{entity}",
        ):
            encoded = encoder.encode(sequence_batch)
            writer.add_batch(
                sequence_batch,
                encoded.hidden.detach().cpu().to(dtype=_torch_float16()).numpy(),
                encoded.mask.detach().cpu().numpy(),
                encoded.lengths.detach().cpu().numpy(),
            )
    return Path(cache_root).resolve() / entity


def _torch_float16():
    import torch

    return torch.float16


def _progress(values, *, total: int, description: str):
    try:
        from tqdm import tqdm
    except ImportError:
        return values
    return tqdm(values, total=total, desc=description)
