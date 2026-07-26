"""Timestamped artifact naming helpers."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Mapping


ARTIFACT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
ARTIFACT_TIMESTAMP_PATTERN = r"\d{8}_\d{6}_\d{6}"


def create_artifact_timestamp(now: datetime | None = None) -> str:
    """Return a sortable local timestamp suitable for artifact filenames."""

    return (now or datetime.now().astimezone()).strftime(ARTIFACT_TIMESTAMP_FORMAT)


def validate_artifact_timestamp(timestamp: str) -> str:
    """Validate and return an artifact timestamp."""

    if re.fullmatch(ARTIFACT_TIMESTAMP_PATTERN, timestamp) is None:
        raise ValueError(
            "Artifact timestamp must use YYYYMMDD_HHMMSS_ffffff format, "
            f"got {timestamp!r}"
        )
    try:
        datetime.strptime(timestamp, ARTIFACT_TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise ValueError(f"Invalid artifact timestamp: {timestamp!r}") from exc
    return timestamp


def timestamped_filename(filename: str, timestamp: str) -> str:
    """Insert a validated timestamp immediately before the file extension."""

    timestamp = validate_artifact_timestamp(timestamp)
    path = Path(filename)
    if path.name != filename:
        raise ValueError(f"Expected a filename without directories, got {filename!r}")
    suffix = "".join(path.suffixes)
    stem = filename[: -len(suffix)] if suffix else filename
    return f"{stem}_{timestamp}{suffix}"


def timestamped_path(path: str | Path, timestamp: str) -> Path:
    target = Path(path)
    return target.with_name(timestamped_filename(target.name, timestamp))


def artifact_timestamp(path: str | Path, base_filename: str) -> str | None:
    """Extract the timestamp when ``path`` is a timestamped base artifact."""

    base = Path(base_filename)
    suffix = "".join(base.suffixes)
    stem = base.name[: -len(suffix)] if suffix else base.name
    match = re.fullmatch(
        rf"{re.escape(stem)}_({ARTIFACT_TIMESTAMP_PATTERN}){re.escape(suffix)}",
        Path(path).name,
    )
    return match.group(1) if match else None


def find_latest_artifact(
    directory: str | Path,
    base_filename: str,
    *,
    include_legacy: bool = True,
) -> Path:
    """Find the newest timestamped artifact, with optional legacy fallback."""

    root = Path(directory)
    candidates = [
        (timestamp, path)
        for path in root.glob(f"{Path(base_filename).stem}_*{Path(base_filename).suffix}")
        if (timestamp := artifact_timestamp(path, base_filename)) is not None
    ]
    if candidates:
        return max(candidates, key=lambda item: item[0])[1].resolve()
    legacy = root / base_filename
    if include_legacy and legacy.is_file():
        return legacy.resolve()
    raise FileNotFoundError(
        f"No timestamped artifact matching {base_filename!r} found in {root}"
    )


def find_latest_common_artifacts(
    directory: str | Path,
    base_filenames: Mapping[str, str],
    *,
    include_legacy: bool = True,
) -> dict[str, Path]:
    """Find artifacts that share the latest timestamp across all requested keys."""

    root = Path(directory)
    if not base_filenames:
        raise ValueError("At least one artifact filename is required")
    by_key: dict[str, dict[str, Path]] = {}
    for key, base_filename in base_filenames.items():
        paths: dict[str, Path] = {}
        base = Path(base_filename)
        for path in root.glob(f"{base.stem}_*{base.suffix}"):
            timestamp = artifact_timestamp(path, base_filename)
            if timestamp is not None:
                paths[timestamp] = path
        by_key[key] = paths

    common_timestamps = set.intersection(
        *(set(paths) for paths in by_key.values())
    )
    if common_timestamps:
        latest = max(common_timestamps)
        return {
            key: paths[latest].resolve()
            for key, paths in by_key.items()
        }

    legacy = {
        key: (root / base_filename)
        for key, base_filename in base_filenames.items()
    }
    if include_legacy and all(path.is_file() for path in legacy.values()):
        return {key: path.resolve() for key, path in legacy.items()}
    requested = ", ".join(base_filenames.values())
    raise FileNotFoundError(
        f"No complete timestamp-matched artifact set ({requested}) found in {root}"
    )
