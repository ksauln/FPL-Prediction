from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import requests

from .config import FPL_BOOTSTRAP, FPL_FIXTURES_ALL, FPL_ELEMENT_SUMMARY, RAW_DIR, CACHE_TTL_DAYS
from .utils import save_json, load_json, unix_now

def _safe_get_json(url: str) -> Any:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_bootstrap_static(force: bool = False) -> Dict[str, Any]:
    path = RAW_DIR / "bootstrap-static.json"
    if path.exists() and not force:
        return load_json(path)
    data = _safe_get_json(FPL_BOOTSTRAP)
    save_json(path, data)
    return data

def fetch_fixtures_all(force: bool = False) -> Any:
    path = RAW_DIR / "fixtures-all.json"
    if path.exists() and not force:
        return load_json(path)
    data = _safe_get_json(FPL_FIXTURES_ALL)
    save_json(path, data)
    return data

def _player_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        ts = meta.get("_fetched_ts", 0)
        age_days = (unix_now() - ts) / 86400.0
        return age_days <= CACHE_TTL_DAYS
    except Exception:
        return False

def fetch_player_history(player_id: int, force: bool = False) -> Dict[str, Any]:
    """
    Player element-summary includes current season 'history' and upcoming fixtures.
    """
    path = RAW_DIR / f"player_{player_id}.json"
    if (not force) and _player_cache_fresh(path):
        return load_json(path)
    url = FPL_ELEMENT_SUMMARY.format(player_id=player_id)
    data = _safe_get_json(url)
    data["_fetched_ts"] = unix_now()
    save_json(path, data)
    return data

def bulk_fetch_player_histories(player_ids: List[int], force: bool = False, sleep_s: float = 0.0) -> None:
    for i, pid in enumerate(player_ids, start=1):
        try:
            fetch_player_history(pid, force=force)
        except Exception as e:
            # Log but continue
            print(f"Failed fetch for player {pid}: {e}")
        if sleep_s:
            time.sleep(sleep_s)
