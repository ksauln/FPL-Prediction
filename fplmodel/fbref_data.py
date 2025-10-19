"""Utilities for augmenting the FPL dataset with FBRef statistics."""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd
import requests

from .config import FBREF_API_BASE_URL, FBREF_DATA_DIR
from .secrets import get_secret
from .utils import load_json, save_json


LOGGER = logging.getLogger(__name__)


DEFAULT_COMPETITION_ID = 9  # Premier League (FBRef internal competition id)
DEFAULT_SEASON = "2023-2024"
EARLIEST_SUPPORTED_SEASON = "2017-2018"
PLAYER_MATCH_DEFAULT_STAT_TYPES: Sequence[str] = (
    "summary",
    "passing",
    "passing_types",
    "gca",
    "defense",
    "possession",
    "misc",
    "keeper",
)


class FBRefAPIError(RuntimeError):
    """Raised when the FBRef API returns a non-successful response."""


def normalise_season_code(season: str) -> str:
    """Ensure a season string uses FBRef's canonical format (YYYY-YYYY)."""

    season = season.strip()
    if "-" in season:
        start, end = season.split("-", 1)
        if len(end) == 2:
            # Expand shorthand like 2023-24 -> 2023-2024
            return f"{start}-{start[:2]}{end}"
        return season

    if "/" in season:
        # Accept alternative notation 2023/24
        start, end = season.split("/", 1)
        if len(end) == 2:
            return f"{start}-{start[:2]}{end}"
        if len(end) == 4:
            return f"{start}-{end}"

    # Fall back to returning the input to avoid surprising failures
    return season


def _season_to_ints(season: str) -> tuple[int, int]:
    """Return the starting and ending years for an FBRef season string."""

    normalised = normalise_season_code(season)
    if "-" not in normalised:
        raise ValueError(f"Season '{season}' is not in 'YYYY-YYYY' format")
    start, end = normalised.split("-", 1)
    return int(start), int(end)


def iter_seasons(
    start: str = EARLIEST_SUPPORTED_SEASON,
    end: Optional[str] = None,
) -> List[str]:
    """Generate inclusive FBRef season codes between ``start`` and ``end``."""

    start_year, _ = _season_to_ints(start)
    end_season = end or DEFAULT_SEASON
    end_year, _ = _season_to_ints(end_season)

    if end_year < start_year:
        raise ValueError(
            f"End season {end_season!r} occurs before start season {start!r}"
        )

    seasons: List[str] = []
    for year in range(start_year, end_year + 1):
        next_year = year + 1
        seasons.append(f"{year}-{next_year}")
    return seasons


def canonicalise_player_name(name: Any) -> str:
    """Return a normalised representation of a player's name for matching."""

    if not isinstance(name, str):
        return ""

    # Remove accents and punctuation so that "João" and "Joao" align.
    normalised = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in normalised if not unicodedata.combining(ch))
    cleaned = "".join(ch.lower() for ch in stripped if ch.isalnum())
    return cleaned


def compute_canonical_names(
    df: pd.DataFrame, name_columns: Sequence[str], *, target: Optional[str] = None
) -> pd.Series:
    """Return a Series of canonical names derived from ``name_columns``."""

    if not name_columns:
        result = pd.Series([""] * len(df), index=df.index, dtype="object")
        if target:
            df[target] = result
        return result

    canonical = pd.Series("", index=df.index, dtype="object")
    for column in name_columns:
        if column not in df.columns:
            continue
        values = df[column].fillna("").astype(str)
        # Skip already populated entries.
        mask = canonical == ""
        if not mask.any():
            break
        canonical.loc[mask] = values.loc[mask].map(
            lambda value: canonicalise_player_name(value) if value.strip() else ""
        )

    if target is not None:
        df[target] = canonical
    return canonical


def fbref_season_to_fpl_code(season: str) -> str:
    """Convert an FBRef season code (YYYY-YYYY) to FPL shorthand (YYYY-YY)."""

    canonical = normalise_season_code(season)
    if "-" not in canonical:
        return canonical
    start, end = canonical.split("-", 1)
    return f"{start}-{end[-2:]}"


@dataclass
class FBRefClient:
    """Lightweight client for the FBRef API."""

    api_key: Optional[str] = None
    api_key_header: str = "x-api-key"
    base_url: str = FBREF_API_BASE_URL
    timeout: int = 30
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = get_secret("fbref_api_key")
        if not self.api_key_header:
            self.api_key_header = (
                get_secret("fbref_api_key_header", default="x-api-key") or "x-api-key"
            )

        headers = {
            "Accept": "application/json",
            "User-Agent": "FPL-Prediction/FBRefClient",
        }

        # Only attach an auth header if we actually have a key.
        if self.api_key:
            headers[self.api_key_header] = self.api_key

        self.session.headers.update(headers)

    def _build_url(self, endpoint: str) -> str:
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url.rstrip('/')}/{endpoint}"

    def request(
        self,
        endpoint: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform a GET request against the FBRef API."""

        url = self._build_url(endpoint)
        response = self.session.get(url, params=params, timeout=self.timeout)

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network error path
            raise FBRefAPIError(str(exc)) from exc

        payload = response.json()
        if isinstance(payload, Mapping) and payload.get("success") is False:
            message = payload.get("message") or "Unknown FBRef API error"
            raise FBRefAPIError(message)
        return payload


def _ensure_cache_dir() -> Path:
    FBREF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return FBREF_DATA_DIR


def _cache_path(filename: str) -> Path:
    return _ensure_cache_dir() / filename


def _load_cached(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            return load_json(path)
        except Exception:  # pragma: no cover - corrupted cache path
            return None
    return None


def _dump_cache(path: Path, payload: Dict[str, Any]) -> None:
    save_json(path, payload)


def _merge_player_match_payload(
    accumulator: Dict[str, Dict[str, Any]],
    payload: Mapping[str, Any],
    *,
    stat_type: Optional[str] = None,
) -> None:
    """Merge a player-match payload into an accumulator keyed by match id."""

    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, Iterable):
        return

    for entry in data:
        if not isinstance(entry, Mapping):
            continue

        meta = dict(entry.get("meta_data") or {})
        match_id = meta.get("match_id") or ""
        if not match_id:
            match_id = "|".join(
                str(meta.get(field, "")) for field in ("date", "team_name", "opponent")
            )
        combined = accumulator.setdefault(match_id, {"meta_data": {}, "stats": {}})
        combined_meta = combined.setdefault("meta_data", {})
        combined_meta.update({k: v for k, v in meta.items() if v is not None})

        stats = entry.get("stats") or {}
        if not isinstance(stats, Mapping):
            continue

        nested_items = [
            (category, values)
            for category, values in stats.items()
            if isinstance(values, Mapping)
        ]

        if nested_items:
            for category, values in nested_items:
                target = combined.setdefault("stats", {}).setdefault(category, {})
                target.update(values)
            continue

        target_category = stat_type or "summary"
        target = combined.setdefault("stats", {}).setdefault(target_category, {})
        for key, value in stats.items():
            target[key] = value


def fetch_player_match_stats(
    fbref_player_id: str,
    *,
    league_id: int = DEFAULT_COMPETITION_ID,
    season: Optional[str] = None,
    stat_types: Optional[Sequence[str]] = None,
    force: bool = False,
    client: Optional[FBRefClient] = None,
) -> Dict[str, Any]:
    """Fetch per-match statistics for ``fbref_player_id`` across ``season``."""

    if not fbref_player_id:
        raise ValueError("fbref_player_id must be provided")

    season_code = normalise_season_code(season) if season else None
    cache_key = season_code or "latest"
    cache = _cache_path(
        f"player_match_stats_{fbref_player_id}_{league_id}_{cache_key}.json"
    )
    if not force:
        cached = _load_cached(cache)
        if cached is not None:
            return cached

    client = client or FBRefClient()
    base_params: Dict[str, Any] = {
        "player_id": fbref_player_id,
        "league_id": league_id,
    }
    if season_code:
        base_params["season_id"] = season_code

    accumulator: Dict[str, Dict[str, Any]] = {}

    base_payload = client.request("player-match-stats", params=base_params)
    _merge_player_match_payload(accumulator, base_payload)

    fetched_categories: set[str] = set()
    for combined in accumulator.values():
        stats = combined.get("stats") or {}
        if isinstance(stats, Mapping):
            fetched_categories.update(stats.keys())

    requested_stat_types = list(dict.fromkeys(stat_types or PLAYER_MATCH_DEFAULT_STAT_TYPES))
    remaining = [
        category
        for category in requested_stat_types
        if category not in fetched_categories
    ]

    for category in remaining:
        params = dict(base_params)
        params["stat_type"] = category
        try:
            payload = client.request("player-match-stats", params=params)
        except FBRefAPIError as exc:
            LOGGER.debug(
                "FBRef player match stats request failed for %s (%s): %s",
                fbref_player_id,
                category,
                exc,
            )
            continue
        _merge_player_match_payload(accumulator, payload, stat_type=category)

    for combined in accumulator.values():
        meta = combined.setdefault("meta_data", {})
        meta.setdefault("player_id", fbref_player_id)

    payload = {
        "data": list(accumulator.values()),
        "metadata": {
            "player_id": fbref_player_id,
            "league_id": league_id,
            "season": season_code,
            "stat_types": requested_stat_types,
        },
    }
    _dump_cache(cache, payload)
    return payload


def fetch_player_table(
    table: str,
    *,
    season: str = DEFAULT_SEASON,
    competition_id: int = DEFAULT_COMPETITION_ID,
    force: bool = False,
    client: Optional[FBRefClient] = None,
    extra_params: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fetch a player statistics table from FBRef and cache the raw payload."""

    season = normalise_season_code(season)
    params: Dict[str, Any] = {
        "type": table,
        "comp": competition_id,
        "season": season,
    }
    if extra_params:
        params.update(extra_params)

    filename = f"player_table_{table}_{competition_id}_{season}.json"
    cache = _cache_path(filename)
    if not force:
        cached = _load_cached(cache)
        if cached is not None:
            return cached

    client = client or FBRefClient()
    payload = client.request("player-table", params=params)
    if isinstance(payload, Mapping):
        payload = dict(payload)
    else:
        payload = {"data": payload}

    payload.setdefault("metadata", {})
    payload["metadata"].update({
        "table": table,
        "competition_id": competition_id,
        "season": season,
    })
    _dump_cache(cache, payload)
    return payload


def fetch_player_table_for_seasons(
    table: str,
    seasons: Sequence[str],
    *,
    competition_id: int = DEFAULT_COMPETITION_ID,
    force: bool = False,
    client: Optional[FBRefClient] = None,
    extra_params: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fetch ``table`` data for multiple seasons returning a season keyed mapping."""

    results: Dict[str, Dict[str, Any]] = {}
    for season in seasons:
        payload = fetch_player_table(
            table,
            season=season,
            competition_id=competition_id,
            force=force,
            client=client,
            extra_params=extra_params,
        )
        results[normalise_season_code(season)] = payload
    return results


def fetch_team_table(
    table: str,
    *,
    season: str = DEFAULT_SEASON,
    competition_id: int = DEFAULT_COMPETITION_ID,
    force: bool = False,
    client: Optional[FBRefClient] = None,
    extra_params: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fetch a team level statistics table from FBRef."""

    season = normalise_season_code(season)
    params: Dict[str, Any] = {
        "type": table,
        "comp": competition_id,
        "season": season,
    }
    if extra_params:
        params.update(extra_params)

    filename = f"team_table_{table}_{competition_id}_{season}.json"
    cache = _cache_path(filename)
    if not force:
        cached = _load_cached(cache)
        if cached is not None:
            return cached

    client = client or FBRefClient()
    payload = client.request("team-table", params=params)
    if isinstance(payload, Mapping):
        payload = dict(payload)
    else:
        payload = {"data": payload}

    payload.setdefault("metadata", {})
    payload["metadata"].update({
        "table": table,
        "competition_id": competition_id,
        "season": season,
    })
    _dump_cache(cache, payload)
    return payload


def _extract_records(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if "data" in payload and isinstance(payload["data"], Iterable):
        records = list(payload["data"])
        return [dict(row) for row in records if isinstance(row, Mapping)]

    # Some endpoints return a mapping keyed by player id.
    if isinstance(payload, Mapping):
        return [dict(v) for v in payload.values() if isinstance(v, Mapping)]

    return []


def payload_to_dataframe(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Convert an FBRef API payload into a :class:`pandas.DataFrame`."""

    records = _extract_records(payload)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    # Drop redundant rank columns if present
    drop_cols = [col for col in df.columns if str(col).lower().startswith("rank_")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def player_match_payload_to_dataframe(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Convert a player-match payload into a flattened :class:`pandas.DataFrame`."""

    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, Iterable):
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, Mapping):
            continue

        meta = dict(entry.get("meta_data") or {})
        stats = entry.get("stats") or {}

        record: Dict[str, Any] = {
            "fbref_match_id": meta.get("match_id"),
            "fbref_match_date": meta.get("date"),
            "fbref_match_round": meta.get("round"),
            "fbref_home_away": (meta.get("home_away") or "").lower() or None,
            "fbref_team_name": meta.get("team_name"),
            "fbref_opponent_name": meta.get("opponent"),
            "fbref_player_id": meta.get("player_id"),
        }

        if isinstance(stats, Mapping):
            for category, values in stats.items():
                if isinstance(values, Mapping):
                    for stat_name, stat_value in values.items():
                        column = f"fbref_match_{category}_{stat_name}"
                        record[column] = stat_value
                else:
                    column = f"fbref_match_{category}"
                    record[column] = values

        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    if "fbref_match_date" in df.columns:
        df["fbref_match_date"] = pd.to_datetime(df["fbref_match_date"], errors="coerce")
        df["match_date"] = df["fbref_match_date"].dt.strftime("%Y-%m-%d")
    else:
        df["match_date"] = pd.NA

    df["fbref_team_canonical"] = df["fbref_team_name"].map(canonicalise_player_name)
    df["fbref_opponent_canonical"] = df["fbref_opponent_name"].map(canonicalise_player_name)

    metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
    fbref_season = metadata.get("season")
    if fbref_season:
        fbref_season = normalise_season_code(fbref_season)
        df["fbref_season"] = fbref_season
        df["season_name"] = df["fbref_season"].map(fbref_season_to_fpl_code)
    else:
        df["fbref_season"] = pd.NA

    stat_columns = [
        column
        for column in df.columns
        if column.startswith("fbref_match_")
        and column
        not in {
            "fbref_match_id",
            "fbref_match_date",
            "fbref_match_round",
        }
    ]
    for column in stat_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.drop_duplicates(
        subset=["fbref_player_id", "fbref_match_id", "match_date", "fbref_match_round"],
        keep="first",
    )
    return df.reset_index(drop=True)


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert ``columns`` to numeric types when present."""

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a ratio while safely handling divide-by-zero cases."""

    ratio = numerator.divide(denominator)
    return ratio.replace([pd.NA, pd.NaT, float("inf"), -float("inf")], 0.0).fillna(0.0)


def rename_fbref_columns(
    df: pd.DataFrame,
    rename_map: Optional[Mapping[str, str]] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Return a DataFrame with FBRef column names transformed for modelling."""

    if df.empty:
        return df

    if rename_map:
        df = df.rename(columns=rename_map)

    if prefix:
        df = df.add_prefix(prefix)

    return df


def build_defensive_feature_table(
    *,
    season: Optional[str] = None,
    seasons: Optional[Sequence[str]] = None,
    competition_id: int = DEFAULT_COMPETITION_ID,
    force: bool = False,
    client: Optional[FBRefClient] = None,
) -> pd.DataFrame:
    """Fetch defensive actions and prepare a modelling-friendly table."""

    if seasons is None:
        target_season = season or DEFAULT_SEASON
        seasons = iter_seasons(EARLIEST_SUPPORTED_SEASON, target_season)

    frames: List[pd.DataFrame] = []
    for season_code in seasons:
        payload = fetch_player_table(
            "defense",
            season=season_code,
            competition_id=competition_id,
            force=force,
            client=client,
        )
        df = payload_to_dataframe(payload)
        if df.empty:
            continue
        df["season"] = normalise_season_code(season_code)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    rename_map = {
        "player": "player_name",
        "player_id": "player_id",
        "tackles": "tackles_total",
        "tkl": "tackles_total",
        "blocks": "blocks_total",
        "clr": "clearances",
        "int": "interceptions",
        "tklw": "tackles_won",
        "pressures": "pressures",
        "succ": "successful_pressures",
        "tackles_def_3rd": "def_3rd_tackles",
        "tackles_mid_3rd": "mid_3rd_tackles",
        "tackles_att_3rd": "att_3rd_tackles",
        "minutes": "minutes_played",
        "min": "minutes_played",
        "minutes_90s": "minutes_played_90s",
    }
    df = rename_fbref_columns(df, rename_map=rename_map, prefix="fbref_")

    numeric_columns = [
        "fbref_tackles_total",
        "fbref_blocks_total",
        "fbref_clearances",
        "fbref_interceptions",
        "fbref_tackles_won",
        "fbref_pressures",
        "fbref_successful_pressures",
        "fbref_def_3rd_tackles",
        "fbref_mid_3rd_tackles",
        "fbref_att_3rd_tackles",
        "fbref_minutes_played",
        "fbref_minutes_played_90s",
    ]
    df = _coerce_numeric(df, numeric_columns)

    minutes_90s_col = "fbref_minutes_played_90s"
    if minutes_90s_col not in df.columns and "fbref_minutes_played" in df.columns:
        df[minutes_90s_col] = df["fbref_minutes_played"] / 90.0
    if minutes_90s_col not in df.columns:
        df[minutes_90s_col] = pd.Series([pd.NA] * len(df), dtype="float64")

    minutes_90s = df[minutes_90s_col].replace({0: pd.NA})
    per90_columns = {
        "fbref_tackles_total": "fbref_tackles_per90",
        "fbref_blocks_total": "fbref_blocks_per90",
        "fbref_clearances": "fbref_clearances_per90",
        "fbref_interceptions": "fbref_interceptions_per90",
        "fbref_pressures": "fbref_pressures_per90",
        "fbref_successful_pressures": "fbref_successful_pressures_per90",
    }
    for source_col, feature_col in per90_columns.items():
        if source_col in df.columns:
            df[feature_col] = df[source_col].divide(minutes_90s).fillna(0.0)

    if "fbref_tackles_won" in df.columns and "fbref_tackles_total" in df.columns:
        df["fbref_tackles_won_pct"] = _safe_ratio(
            df["fbref_tackles_won"], df["fbref_tackles_total"]
        )

    if "fbref_successful_pressures" in df.columns and "fbref_pressures" in df.columns:
        df["fbref_successful_pressures_pct"] = _safe_ratio(
            df["fbref_successful_pressures"], df["fbref_pressures"]
        )

    if "fbref_tackles_total" in df.columns and "fbref_interceptions" in df.columns:
        df["fbref_tackles_plus_interceptions_per90"] = (
            df["fbref_tackles_per90"].fillna(0.0)
            + df["fbref_interceptions_per90"].fillna(0.0)
        )

    return df


def build_player_advanced_stats(
    *,
    stat_type: str,
    season: Optional[str] = None,
    seasons: Optional[Sequence[str]] = None,
    competition_id: int = DEFAULT_COMPETITION_ID,
    force: bool = False,
    client: Optional[FBRefClient] = None,
) -> pd.DataFrame:
    """Fetch an arbitrary player table and prepare columns for modelling."""

    if seasons is None:
        target_season = season or DEFAULT_SEASON
        seasons = iter_seasons(EARLIEST_SUPPORTED_SEASON, target_season)

    frames: List[pd.DataFrame] = []
    for season_code in seasons:
        payload = fetch_player_table(
            stat_type,
            season=season_code,
            competition_id=competition_id,
            force=force,
            client=client,
        )
        df = payload_to_dataframe(payload)
        if df.empty:
            continue
        df["season"] = normalise_season_code(season_code)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    prefix = f"fbref_{stat_type}_"
    df = rename_fbref_columns(df, prefix=prefix)

    minutes_col = None
    for candidate in (f"{prefix}minutes_played_90s", f"{prefix}minutes_played"):
        if candidate in df.columns:
            minutes_col = candidate
            break

    if minutes_col is not None:
        df = _coerce_numeric(df, [minutes_col])
        numeric_cols = [
            col
            for col in df.columns
            if col.startswith(prefix)
            and col not in {minutes_col, f"{prefix}player", f"{prefix}player_name"}
        ]
        df = _coerce_numeric(df, numeric_cols)
        minutes_series = df[minutes_col].replace({0: pd.NA})
        for col in numeric_cols:
            feature_name = f"{col}_per90"
            df[feature_name] = df[col].divide(minutes_series).fillna(0.0)

    return df


def collect_player_match_stats(
    player_id_map: Mapping[int, str],
    player_seasons: Mapping[int, Sequence[str]],
    *,
    competition_id: int = DEFAULT_COMPETITION_ID,
    stat_types: Optional[Sequence[str]] = None,
    force: bool = False,
    client: Optional[FBRefClient] = None,
    canonical_name_map: Optional[Mapping[int, str]] = None,
) -> pd.DataFrame:
    """Fetch player match stats for multiple players and return a combined frame."""

    if not player_id_map:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    stat_types = list(dict.fromkeys(stat_types or PLAYER_MATCH_DEFAULT_STAT_TYPES))

    for player_id, fbref_player_id in player_id_map.items():
        if not fbref_player_id:
            continue

        seasons = player_seasons.get(player_id) or []
        if not seasons:
            seasons = [DEFAULT_SEASON]

        for season_code in seasons:
            try:
                payload = fetch_player_match_stats(
                    fbref_player_id,
                    league_id=competition_id,
                    season=season_code,
                    stat_types=stat_types,
                    force=force,
                    client=client,
                )
            except FBRefAPIError as exc:
                LOGGER.debug(
                    "Skipping match stats for player %s season %s: %s",
                    fbref_player_id,
                    season_code,
                    exc,
                )
                continue

            df = player_match_payload_to_dataframe(payload)
            if df.empty:
                continue

            df["player_id"] = player_id
            if canonical_name_map:
                df["canonical_name"] = canonical_name_map.get(player_id, "")
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["player_id", "season_name", "match_date", "fbref_match_id"]
    )
    return combined.reset_index(drop=True)


def build_fbref_player_feature_matrix(
    elements_df: pd.DataFrame,
    *,
    seasons: Optional[Sequence[str]] = None,
    competition_id: int = DEFAULT_COMPETITION_ID,
    force: bool = False,
    client: Optional[FBRefClient] = None,
) -> pd.DataFrame:
    """Return FBRef defensive metrics aligned to ``elements_df`` players per season."""

    name_cols = [
        col
        for col in ("full_name", "web_name", "first_name", "second_name")
        if col in elements_df.columns
    ]
    if "player_id" not in elements_df.columns or not name_cols:
        raise KeyError(
            "elements_df must contain 'player_id' and at least one player name column"
        )

    if seasons is None:
        seasons = iter_seasons()
    seasons = [normalise_season_code(str(season)) for season in seasons]
    seasons = sorted(dict.fromkeys(seasons))

    defensive_df = build_defensive_feature_table(
        seasons=seasons,
        competition_id=competition_id,
        force=force,
        client=client,
    )
    if defensive_df.empty or "fbref_player_name" not in defensive_df.columns:
        return pd.DataFrame()

    working = defensive_df.copy()
    working["canonical_name"] = working["fbref_player_name"].map(canonicalise_player_name)
    working["season_name"] = working["fbref_season"].map(fbref_season_to_fpl_code)
    working = working[working["canonical_name"] != ""].copy()

    player_names = (
        elements_df[["player_id", *name_cols]]
        .melt(id_vars="player_id", value_vars=name_cols, value_name="player_name")
        .dropna(subset=["player_name"])
    )
    player_names["canonical_name"] = player_names["player_name"].map(
        canonicalise_player_name
    )
    player_names = player_names[player_names["canonical_name"] != ""]
    player_names = player_names.drop_duplicates(subset=["player_id", "canonical_name"])

    merged = working.merge(player_names[["player_id", "canonical_name"]], on="canonical_name", how="left")
    merged = merged.dropna(subset=["player_id"])

    feature_cols = [
        col
        for col in merged.columns
        if col.startswith("fbref_")
        and col
        not in {
            "fbref_player_name",
            "fbref_season",
        }
    ]
    if not feature_cols:
        return pd.DataFrame()

    numeric_cols = merged[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()

    agg_spec: Dict[str, Any] = {col: "mean" for col in numeric_cols}
    if "fbref_player_id" in merged.columns:
        agg_spec["fbref_player_id"] = "first"
    if "canonical_name" in merged.columns:
        agg_spec["canonical_name"] = "first"

    aggregated = (
        merged.groupby(["player_id", "season_name"], as_index=False)
        .agg(agg_spec)
        .sort_values(["player_id", "season_name"])
    )
    return aggregated


def merge_with_player_index(
    fbref_df: pd.DataFrame,
    player_index: pd.DataFrame,
    *,
    fbref_name_col: str = "fbref_player_name",
    player_index_name_col: str = "player_name",
) -> pd.DataFrame:
    """Merge FBRef data onto an existing player index DataFrame."""

    if fbref_df.empty:
        return player_index.copy()

    if player_index_name_col not in player_index.columns:
        raise KeyError(
            f"Column '{player_index_name_col}' not found in player_index dataframe"
        )

    working_fbref = fbref_df.copy()
    if fbref_name_col not in working_fbref.columns:
        # attempt to detect un-prefixed column name
        if "player_name" in working_fbref.columns:
            fbref_name_col = "player_name"
        elif "player" in working_fbref.columns:
            fbref_name_col = "player"
        else:
            raise KeyError("Player name column not found in FBRef dataframe")

    working_fbref = working_fbref.rename(columns={fbref_name_col: player_index_name_col})
    merged = player_index.merge(working_fbref, how="left", on=player_index_name_col)
    return merged


__all__ = [
    "FBRefClient",
    "FBRefAPIError",
    "build_fbref_player_feature_matrix",
    "canonicalise_player_name",
    "compute_canonical_names",
    "build_defensive_feature_table",
    "fetch_player_table_for_seasons",
    "iter_seasons",
    "build_player_advanced_stats",
    "collect_player_match_stats",
    "fetch_player_match_stats",
    "fetch_player_table",
    "fetch_team_table",
    "fbref_season_to_fpl_code",
    "player_match_payload_to_dataframe",
    "merge_with_player_index",
    "normalise_season_code",
    "payload_to_dataframe",
    "rename_fbref_columns",
]

