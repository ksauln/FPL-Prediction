from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple

from .config import LOGS_DIR

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
CONSOLE_FORMAT = "%(levelname)s - %(message)s"


def configure_run_logger(run_id: str | None = None) -> Tuple[logging.Logger, logging.FileHandler, Path]:
    """
    Configure a file-backed logger for the current pipeline run.

    Returns the logger instance, the file handler (for later updates), and the active log path.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_id or timestamp
    log_path = LOGS_DIR / f"run_{suffix}.log"

    logger = logging.getLogger("fplmodel")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers so repeated runs don't duplicate output
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    logger.addHandler(console_handler)

    logger.info("Logging initialised. Writing to %s", log_path)
    return logger, file_handler, log_path


@contextmanager
def log_timed_step(
    logger: logging.Logger,
    description: str,
    level: int = logging.INFO,
) -> Iterator[None]:
    """
    Context manager that logs the start and completion time for an expensive step.
    """
    clean_description = description.rstrip(".")
    logger.log(level, "%s...", clean_description)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        duration = time.perf_counter() - start
        logger.log(level, "%s failed after %.2fs", clean_description, duration)
        raise
    else:
        duration = time.perf_counter() - start
        logger.log(level, "%s completed in %.2fs", clean_description, duration)


def update_log_filename_for_gameweek(
    logger: logging.Logger,
    file_handler: logging.FileHandler,
    current_path: Path,
    next_gw: int,
) -> Tuple[logging.FileHandler, Path]:
    """
    Rename the run log to include the resolved next gameweek once known.

    Returns an updated file handler and log path. If the rename fails the original handler is reused.
    """
    suffix = current_path.stem[len("run_") :] if current_path.stem.startswith("run_") else current_path.stem
    new_path = current_path.with_name(f"run_gw{next_gw}_{suffix}.log")

    if new_path == current_path:
        return file_handler, current_path

    logger.info("Renaming log file to highlight gameweek %s", next_gw)
    try:
        formatter = logging.Formatter(LOG_FORMAT)
        logger.removeHandler(file_handler)
        file_handler.flush()
        file_handler.close()
        current_path.rename(new_path)

        new_handler = logging.FileHandler(new_path, encoding="utf-8")
        new_handler.setFormatter(formatter)
        logger.addHandler(new_handler)
        logger.info("Log file renamed to %s", new_path)
        return new_handler, new_path
    except OSError as exc:
        logger.warning("Could not rename log file to include gameweek: %s", exc)
        # Reattach old handler so logging continues even after failure.
        file_handler = logging.FileHandler(current_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
        return file_handler, current_path
