from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import requests

from .config import (
    FPL_BOOTSTRAP,
    FPL_FIXTURES_ALL,
    FPL_ELEMENT_SUMMARY,
    RAW_DIR,
    CACHE_TTL_DAYS,
    PLAYER_HISTORY_SEASONS_BACK,
)
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

def _player_cache_fresh(path: Path, min_seasons: int) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        ts = meta.get("_fetched_ts", 0)
        seasons_available = int(meta.get("_history_seasons", 1))
        if seasons_available < min_seasons:
            return False
        age_days = (unix_now() - ts) / 86400.0
        return age_days <= CACHE_TTL_DAYS
    except Exception:
        return False

def _current_season_start_year(now: datetime | None = None) -> int:
    now = now or datetime.utcnow()
    return now.year if now.month >= 7 else now.year - 1

def _format_season_code(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"

def _season_codes_to_fetch(seasons_back: int, now: datetime | None = None) -> List[str]:
    if seasons_back <= 0:
        return []
    start_year = _current_season_start_year(now)
    return [_format_season_code(start_year - offset) for offset in range(1, seasons_back + 1)]

def _annotate_history(entries: List[Dict[str, Any]] | None, season_code: str) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    if not entries:
        return annotated
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        item.setdefault("season_name", season_code)
        annotated.append(item)
    return annotated

def fetch_player_history(
    player_id: int,
    force: bool = False,
    seasons_back: int = PLAYER_HISTORY_SEASONS_BACK,
) -> Dict[str, Any]:
    """
    Player element-summary includes current season 'history' and upcoming fixtures.
    """
    path = RAW_DIR / f"player_{player_id}.json"
    required_seasons = max(seasons_back, 0) + 1  # always include current season
    if (not force) and _player_cache_fresh(path, required_seasons):
        return load_json(path)
    url = FPL_ELEMENT_SUMMARY.format(player_id=player_id)
    now = datetime.utcnow()
    current_season_code = _format_season_code(_current_season_start_year(now))
    data = _safe_get_json(url)
    history_all = _annotate_history(data.get("history", []), current_season_code)

    included_seasons = [current_season_code]
    past_seasons = _season_codes_to_fetch(seasons_back, now=now)
    for season_code in past_seasons:
        season_url = f"{url}?season={season_code}"
        try:
            season_payload = _safe_get_json(season_url)
        except requests.HTTPError as exc:
            if getattr(exc, "response", None) is not None and exc.response.status_code == 404:
                included_seasons.append(season_code)
                continue
            raise
        season_history = _annotate_history(season_payload.get("history", []), season_code)
        history_all.extend(season_history)
        included_seasons.append(season_code)

    included_seasons = list(dict.fromkeys(included_seasons))

    history_all.sort(
        key=lambda row: (
            row.get("season_name", ""),
            row.get("kickoff_time") or "",
            row.get("round") or 0,
            row.get("fixture") or 0,
        )
    )

    data["history"] = history_all
    data["_fetched_ts"] = unix_now()
    data["_history_season_codes"] = included_seasons
    data["_history_seasons"] = len(included_seasons)
    save_json(path, data)
    return data

def bulk_fetch_player_histories(
    player_ids: List[int],
    force: bool = False,
    sleep_s: float = 0.0,
    seasons_back: int = PLAYER_HISTORY_SEASONS_BACK,
) -> None:
    for i, pid in enumerate(player_ids, start=1):
        try:
            fetch_player_history(pid, force=force, seasons_back=seasons_back)
        except Exception as e:
            # Log but continue
            print(f"Failed fetch for player {pid}: {e}")
        if sleep_s:
            time.sleep(sleep_s)
