from __future__ import annotations

import json
import logging
import math
import numbers
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def parse_log_level(value: str | int) -> int:
    if isinstance(value, int):
        return value
    normalized = value.upper()
    if normalized not in VALID_LOG_LEVELS:
        raise ValueError(
            "log level must be one of: " + ", ".join(VALID_LOG_LEVELS)
        )
    return int(getattr(logging, normalized))


@contextmanager
def cli_logging_context(level: str | int = "INFO") -> Iterator[None]:
    requested_level = parse_log_level(level)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        stream = getattr(handler, "stream", None)
        if stream is not None and getattr(stream, "closed", False):
            root_logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setLevel(requested_level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    old_level = root_logger.level
    root_logger.setLevel(requested_level)
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        handler.close()
        root_logger.setLevel(old_level)


@contextmanager
def log_file_context(
    path: str | Path,
    *,
    logger_names: Iterable[str],
    level: str | int = "INFO",
    mode: str = "a",
) -> Iterator[None]:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    requested_level = parse_log_level(level)
    handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    handler.setLevel(requested_level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    loggers: list[tuple[logging.Logger, int]] = []
    try:
        for name in logger_names:
            logger = logging.getLogger(name)
            loggers.append((logger, logger.level))
            logger.setLevel(min(logger.getEffectiveLevel(), requested_level))
            logger.addHandler(handler)
        yield
    finally:
        for logger, old_level in loggers:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
        handler.close()


class JsonlEventLogger:
    def __init__(self, path: str | Path, *, mode: str = "a") -> None:
        self.path = Path(path)
        self.mode = mode
        self._handle = None

    def __enter__(self) -> "JsonlEventLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open(self.mode, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def emit(self, event: str, **fields: Any) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlEventLogger must be used as a context manager")
        row = {
            "timestamp": utc_timestamp(),
            "event": event,
            **fields,
        }
        self._handle.write(
            json.dumps(to_jsonable(row), sort_keys=True, allow_nan=False) + "\n"
        )
        self._handle.flush()


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    return str(value)


def elapsed_seconds(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 6)


def trim_jsonl_events_to_step(path: str | Path, step: int) -> None:
    event_path = Path(path)
    if not event_path.exists():
        return
    kept_lines: list[str] = []
    pending_lines: list[str] = []
    with event_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not _event_row_is_at_or_before_step(row, step):
                break
            if _event_row_has_step_marker(row):
                kept_lines.extend(pending_lines)
                pending_lines.clear()
                kept_lines.append(line)
            elif pending_lines or row.get("event") == "train_start":
                pending_lines.append(line)
            else:
                kept_lines.append(line)
    with event_path.open("w", encoding="utf-8") as handle:
        for line in kept_lines:
            handle.write(line + "\n")


def _event_row_is_at_or_before_step(row: dict[str, Any], step: int) -> bool:
    for key in ("step", "final_step"):
        value = row.get(key)
        if isinstance(value, int | float) and value > step:
            return False
    return True


def _event_row_has_step_marker(row: dict[str, Any]) -> bool:
    return any(key in row for key in ("step", "final_step"))
