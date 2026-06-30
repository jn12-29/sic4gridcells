from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sic4gridcells.logging_utils import to_jsonable, utc_timestamp

TRAIN_OUTPUT_MARKERS = (
    "config.yaml",
    "metrics.jsonl",
    "run.log",
    "train_events.jsonl",
    "tensorboard",
    "checkpoints",
)
EVAL_OUTPUT_MARKERS = (
    "config.yaml",
    "summary.json",
    "run.log",
    "eval_events.jsonl",
)
ABLATION_OUTPUT_MARKERS = (
    "run.log",
    "ablation_events.jsonl",
    "summary.json",
    "summary.csv",
)

_STEP_CHECKPOINT_RE = re.compile(r"step_(\d+)\.pt$")


class OutputDirectoryConflictError(RuntimeError):
    """Raised when a fresh run would overwrite an existing run output."""


@dataclass(frozen=True)
class RuntimeSnapshot:
    output_free_gb: float | None
    cuda_available: bool
    cuda_device_index: int | None
    cuda_device_name: str | None
    cuda_total_memory_mb: float | None
    cuda_memory_allocated_mb: float | None
    cuda_memory_reserved_mb: float | None
    cuda_max_memory_allocated_mb: float | None
    cuda_max_memory_reserved_mb: float | None

    def as_event_fields(self) -> dict[str, object]:
        return {
            "disk/output_free_gb": self.output_free_gb,
            "cuda/available": self.cuda_available,
            "cuda/device_index": self.cuda_device_index,
            "cuda/device_name": self.cuda_device_name,
            "cuda/total_memory_mb": self.cuda_total_memory_mb,
            "cuda/memory_allocated_mb": self.cuda_memory_allocated_mb,
            "cuda/memory_reserved_mb": self.cuda_memory_reserved_mb,
            "cuda/max_memory_allocated_mb": self.cuda_max_memory_allocated_mb,
            "cuda/max_memory_reserved_mb": self.cuda_max_memory_reserved_mb,
        }

    def as_metric_fields(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in self.as_event_fields().items():
            if isinstance(value, int | float):
                metrics[key] = float(value)
        return metrics


@dataclass(frozen=True)
class LatestCheckpoint:
    path: Path
    step: int


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.latest_path = self.checkpoint_dir / "latest.pt"
        self.manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"

    def step_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"step_{step}.pt"

    def save(self, payload: dict[str, Any], step: int) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.step_path(step)
        atomic_torch_save(payload, checkpoint_path)
        atomic_torch_save(payload, self.latest_path)
        self._write_manifest(checkpoint_path, step)
        return checkpoint_path

    def discover_latest(self) -> LatestCheckpoint | None:
        return discover_latest_checkpoint(self.checkpoint_dir)

    def _write_manifest(self, checkpoint_path: Path, step: int) -> None:
        previous = _read_checkpoint_manifest(self.manifest_path)
        rows = list(previous.get("checkpoints", []))
        checkpoint_row = {
            "step": step,
            "path": str(checkpoint_path),
            "saved_at": utc_timestamp(),
        }
        rows = [row for row in rows if row.get("step") != step]
        rows.append(checkpoint_row)
        rows.sort(key=lambda row: int(row["step"]))
        manifest = {
            "latest_step": step,
            "latest_checkpoint": str(checkpoint_path),
            "latest_alias": str(self.latest_path),
            "checkpoints": rows,
        }
        atomic_json_dump(manifest, self.manifest_path)


class StepTimer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start


def prepare_output_dir(
    output_dir: str | Path,
    *,
    resume: bool = False,
    overwrite: bool = False,
    markers: tuple[str, ...],
) -> Path:
    path = Path(output_dir)
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot both be true")
    if path.exists() and not path.is_dir():
        raise OutputDirectoryConflictError(f"Output path exists and is not a directory: {path}")
    if path.exists() and not resume and not overwrite:
        contents = sorted(item.name for item in path.iterdir())
        if contents:
            if markers:
                existing = sorted(marker for marker in markers if (path / marker).exists())
            else:
                existing = []
            details = ", ".join(existing) if existing else ", ".join(contents[:8])
            raise OutputDirectoryConflictError(
                f"Refusing to overwrite existing output directory {path}; "
                f"found contents: {details}. Use resume or an explicit overwrite flag."
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temporary_sibling(output_path)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_json_dump(payload: dict[str, Any] | list[Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temporary_sibling(output_path)
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def collect_runtime_snapshot(device: torch.device, output_dir: str | Path) -> RuntimeSnapshot:
    output_path = Path(output_dir)
    disk_usage = shutil.disk_usage(output_path if output_path.exists() else output_path.parent)
    output_free_gb = disk_usage.free / (1024 ** 3)
    cuda_available = torch.cuda.is_available()
    cuda_device_index: int | None = None
    cuda_device_name: str | None = None
    cuda_total_memory_mb: float | None = None
    cuda_memory_allocated_mb: float | None = None
    cuda_memory_reserved_mb: float | None = None
    cuda_max_memory_allocated_mb: float | None = None
    cuda_max_memory_reserved_mb: float | None = None
    if device.type == "cuda":
        cuda_device_index = device.index
        if cuda_device_index is None:
            cuda_device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(cuda_device_index)
        cuda_device_name = props.name
        cuda_total_memory_mb = props.total_memory / (1024 ** 2)
        cuda_memory_allocated_mb = torch.cuda.memory_allocated(cuda_device_index) / (1024 ** 2)
        cuda_memory_reserved_mb = torch.cuda.memory_reserved(cuda_device_index) / (1024 ** 2)
        cuda_max_memory_allocated_mb = torch.cuda.max_memory_allocated(cuda_device_index) / (1024 ** 2)
        cuda_max_memory_reserved_mb = torch.cuda.max_memory_reserved(cuda_device_index) / (1024 ** 2)
    return RuntimeSnapshot(
        output_free_gb=output_free_gb,
        cuda_available=cuda_available,
        cuda_device_index=cuda_device_index,
        cuda_device_name=cuda_device_name,
        cuda_total_memory_mb=cuda_total_memory_mb,
        cuda_memory_allocated_mb=cuda_memory_allocated_mb,
        cuda_memory_reserved_mb=cuda_memory_reserved_mb,
        cuda_max_memory_allocated_mb=cuda_max_memory_allocated_mb,
        cuda_max_memory_reserved_mb=cuda_max_memory_reserved_mb,
    )


def reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def discover_latest_checkpoint(output_dir_or_checkpoint_dir: str | Path) -> LatestCheckpoint | None:
    root = Path(output_dir_or_checkpoint_dir)
    checkpoint_dir = root if root.name == "checkpoints" else root / "checkpoints"
    candidates = _checkpoint_candidates(checkpoint_dir)
    manifest_latest = _latest_from_manifest(checkpoint_dir / "checkpoint_manifest.json")
    if manifest_latest is not None:
        candidates.append(manifest_latest)
    return max(candidates, key=lambda item: item.step) if candidates else None


def is_output_completed(output_dir: str | Path, max_optimizer_steps: int) -> bool:
    latest = discover_latest_checkpoint(output_dir)
    return latest is not None and latest.step >= max_optimizer_steps


def _read_checkpoint_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _latest_from_manifest(path: Path) -> LatestCheckpoint | None:
    manifest = _read_checkpoint_manifest(path)
    latest_path_value = manifest.get("latest_checkpoint") or manifest.get("latest_alias")
    latest_step = manifest.get("latest_step")
    if latest_path_value is None or latest_step is None:
        return None
    latest_path = Path(str(latest_path_value))
    if not latest_path.is_absolute():
        latest_path = path.parent / latest_path
    if latest_path.exists():
        return LatestCheckpoint(path=latest_path, step=int(latest_step))
    alias_value = manifest.get("latest_alias")
    if alias_value is not None:
        alias_path = Path(str(alias_value))
        if not alias_path.is_absolute():
            alias_path = path.parent / alias_path
        if alias_path.exists():
            return LatestCheckpoint(path=alias_path, step=int(latest_step))
    return None


def _checkpoint_candidates(checkpoint_dir: Path) -> list[LatestCheckpoint]:
    if not checkpoint_dir.exists():
        return []
    candidates: list[LatestCheckpoint] = []
    for path in checkpoint_dir.glob("step_*.pt"):
        match = _STEP_CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        candidates.append(LatestCheckpoint(path=path, step=int(match.group(1))))
    return candidates


def _temporary_sibling(path: Path) -> Path:
    stamp = f"{os.getpid()}-{time.time_ns()}"
    return path.with_name(f".{path.name}.{stamp}.tmp")
