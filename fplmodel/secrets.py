"""Helpers for loading project secrets without committing them to version control."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from .config import PROJECT_ROOT

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore[import-not-found]


_SECRET_FILENAMES = ("secrets.toml",)


def _candidate_paths() -> list[Path]:
    """Return potential locations for the secrets file."""

    return [PROJECT_ROOT / name for name in _SECRET_FILENAMES]


@lru_cache(maxsize=1)
def _load_secret_mapping() -> Mapping[str, Any]:
    """Load secrets from the first existing secrets file."""

    for path in _candidate_paths():
        if not path.exists():
            continue
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
            if isinstance(data, MutableMapping):
                return dict(data)
        except Exception:
            # If parsing fails, ignore the file so that environment vars can be used instead.
            return {}
    return {}


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve a secret value, preferring environment variables."""

    value = os.getenv(key.upper())
    if value:
        return value

    # Normalise to lower-case keys for TOML lookups.
    secrets = _load_secret_mapping()
    lowered_key = key.lower()
    for candidate_key, candidate_value in secrets.items():
        if candidate_key.lower() == lowered_key:
            if isinstance(candidate_value, str):
                return candidate_value
            return str(candidate_value)
    return default


def secrets_available() -> bool:
    """Return whether any secrets file was found."""

    return bool(_load_secret_mapping())


__all__ = ["get_secret", "secrets_available"]
