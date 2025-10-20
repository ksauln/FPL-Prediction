from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_S = 30
_REQUEST_ATTEMPTS = 4
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRYABLE_ERRORS = (requests.exceptions.Timeout, requests.exceptions.ConnectionError)


def _safe_get_json(url: str) -> Any:
    """
    Fetch JSON content from a URL with basic retry/backoff to soften transient API issues.
    """
    for attempt in range(1, _REQUEST_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=_REQUEST_TIMEOUT_S)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code not in _RETRYABLE_STATUS or attempt == _REQUEST_ATTEMPTS:
                logger.error(
                    "HTTP error fetching %s on attempt %s/%s: %s",
                    url,
                    attempt,
                    _REQUEST_ATTEMPTS,
                    exc,
                )
                raise
            logger.warning(
                "HTTP %s fetching %s (attempt %s/%s); retrying...",
                status_code,
                url,
                attempt,
                _REQUEST_ATTEMPTS,
            )
        except _RETRYABLE_ERRORS as exc:
            if attempt == _REQUEST_ATTEMPTS:
                logger.error(
                    "Network error fetching %s on attempt %s/%s: %s",
                    url,
                    attempt,
                    _REQUEST_ATTEMPTS,
                    exc,
                )
                raise
            logger.warning(
                "Network error fetching %s (%s) attempt %s/%s; retrying...",
                url,
                exc,
                attempt,
                _REQUEST_ATTEMPTS,
            )
        except Exception as exc:
            raise

        # Exponential backoff with jitter-free deterministic wait suits CLI runs
        sleep_for = min(5.0, 0.5 * (2 ** (attempt - 1)))
        time.sleep(sleep_for)

    raise RuntimeError(f"Failed to fetch JSON from {url} for unknown reasons")

def fetch_bootstrap_static(force: bool = False) -> Dict[str, Any]:
    path = RAW_DIR / "bootstrap-static.json"
    if path.exists() and not force:
        logger.info("Using cached bootstrap static from %s", path)
        return load_json(path)
    logger.info(
        "Fetching bootstrap static data from %s (force=%s)",
        FPL_BOOTSTRAP,
        force,
    )
    data = _safe_get_json(FPL_BOOTSTRAP)
    save_json(path, data)
    logger.info("Saved bootstrap static payload to %s", path)
    return data

def fetch_fixtures_all(force: bool = False) -> Any:
    path = RAW_DIR / "fixtures-all.json"
    if path.exists() and not force:
        logger.info("Using cached fixtures list from %s", path)
        return load_json(path)
    logger.info(
        "Fetching fixtures data from %s (force=%s)",
        FPL_FIXTURES_ALL,
        force,
    )
    data = _safe_get_json(FPL_FIXTURES_ALL)
    save_json(path, data)
    logger.info("Saved fixtures payload to %s", path)
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
        logger.debug(
            "Player %s cache is fresh (>= %s seasons) at %s; skipping fetch",
            player_id,
            required_seasons,
            path,
        )
        return load_json(path)
    existing_data: Dict[str, Any] | None = None
    existing_history_by_season: Dict[str, List[Dict[str, Any]]] = {}
    if path.exists():
        try:
            existing_data = load_json(path)
        except Exception:
            existing_data = None
        if isinstance(existing_data, dict):
            for entry in existing_data.get("history", []) or []:
                if not isinstance(entry, dict):
                    continue
                season_name = entry.get("season_name")
                if not season_name:
                    continue
                existing_history_by_season.setdefault(str(season_name), []).append(entry)
    url = FPL_ELEMENT_SUMMARY.format(player_id=player_id)
    now = datetime.utcnow()
    current_season_code = _format_season_code(_current_season_start_year(now))
    data = _safe_get_json(url)
    history_all = _annotate_history(data.get("history", []), current_season_code)

    included_seasons = [current_season_code]
    past_seasons = _season_codes_to_fetch(seasons_back, now=now)
    for season_code in past_seasons:
        cached_entries = existing_history_by_season.get(season_code)
        if cached_entries:
            logger.debug("Player %s season %s already cached; reusing local copy", player_id, season_code)
            for entry in cached_entries:
                copied = dict(entry)
                copied.setdefault("season_name", season_code)
                history_all.append(copied)
            included_seasons.append(season_code)
            continue
        logger.debug("Fetching player %s season %s history from API", player_id, season_code)
        season_url = f"{url}?season={season_code}"
        try:
            season_payload = _safe_get_json(season_url)
        except requests.HTTPError as exc:
            if getattr(exc, "response", None) is not None and exc.response.status_code == 404:
                included_seasons.append(season_code)
                logger.debug(
                    "Player %s season %s not available (404); marking season as included without data",
                    player_id,
                    season_code,
                )
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
    logger.debug(
        "Stored player %s history with %d entries across %d seasons at %s",
        player_id,
        len(history_all),
        len(included_seasons),
        path,
    )
    return data

def bulk_fetch_player_histories(
    player_ids: List[int],
    force: bool = False,
    sleep_s: float = 0.0,
    seasons_back: int = PLAYER_HISTORY_SEASONS_BACK,
) -> None:
    total = len(player_ids)
    logger.info(
        "Starting player history fetch for %d players (force=%s, seasons_back=%s)",
        total,
        force,
        seasons_back,
    )
    for i, pid in enumerate(player_ids, start=1):
        try:
            fetch_player_history(pid, force=force, seasons_back=seasons_back)
        except Exception as e:
            logger.warning("Failed fetch for player %s: %s", pid, e)
        else:
            if total >= 10 and (i == total or i % max(total // 10, 1) == 0):
                logger.info("Fetched player history %d/%d (player_id=%s)", i, total, pid)
        if sleep_s:
            logger.debug("Sleeping for %.2fs between player fetches", sleep_s)
            time.sleep(sleep_s)
    logger.info("Completed player history fetch cycle for %d players", total)
