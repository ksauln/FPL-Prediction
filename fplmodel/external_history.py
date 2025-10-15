from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .config import EXTERNAL_HISTORY_DIR

logger = logging.getLogger(__name__)


HISTORY_COLS_DEFAULTS = {
    "clearances_blocks_interceptions": 0.0,
    "recoveries": 0.0,
    "tackles": 0.0,
    "defensive_contribution": 0.0,
    "starts": 0.0,
}


def _season_dir(season: str) -> Path:
    return EXTERNAL_HISTORY_DIR / season


def _season_gw_dir(season: str) -> Path:
    return _season_dir(season) / "gws"


def _load_team_map(season: str) -> dict[str, int]:
    teams_path = _season_dir(season) / "teams.csv"
    if not teams_path.exists():
        logger.warning("Missing teams.csv for season %s at %s", season, teams_path)
        return {}
    teams = pd.read_csv(teams_path)
    if "name" not in teams.columns or "id" not in teams.columns:
        logger.warning("teams.csv for season %s lacks expected columns", season)
        return {}
    return dict(zip(teams["name"].astype(str), teams["id"].astype(int)))


def _normalise_frame(df: pd.DataFrame, season: str, team_map: dict[str, int]) -> pd.DataFrame:
    rename_map = {
        "element": "player_id",
    }
    df = df.rename(columns=rename_map)

    df["season_name"] = season

    if "player_id" not in df.columns:
        logger.debug("Season %s frame missing player_id column after rename", season)
        return pd.DataFrame()

    # Coerce expected numeric columns
    numeric_cols = [
        "player_id",
        "fixture",
        "opponent_team",
        "total_points",
        "round",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "saves",
        "bonus",
        "bps",
        "team_h_score",
        "team_a_score",
        "value",
        "transfers_balance",
        "selected",
        "transfers_in",
        "transfers_out",
        "starts",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "round" in df.columns:
        df["round"] = df["round"].fillna(0).astype(int)

    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].astype(bool)

    if "modified" in df.columns:
        df["modified"] = df["modified"].astype(bool)

    # Map team names to ids where available
    if "team" in df.columns:
        mapped = df["team"].astype(str).map(team_map)
        df["team"] = pd.to_numeric(mapped, errors="coerce")

    # Ensure defaults for missing stats
    for col, default in HISTORY_COLS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    expected_columns = set(numeric_cols).union(
        {
            "kickoff_time",
            "opponent_team",
            "was_home",
            "modified",
            "expected_goals",
            "expected_assists",
            "expected_goal_involvements",
            "expected_goals_conceded",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "season_name",
            "team",
        }
    )
    existing_cols = set(df.columns)
    missing = expected_columns - existing_cols
    if missing:
        logger.debug("Season %s history missing columns: %s", season, ", ".join(sorted(missing)))
    return df


def _iter_gw_files(season: str) -> Iterable[Path]:
    gw_dir = _season_gw_dir(season)
    if not gw_dir.exists():
        logger.warning("Season %s gw directory not found at %s", season, gw_dir)
        return []
    return sorted(gw_dir.glob("gw*.csv"))


def load_external_histories(
    seasons: Sequence[str],
) -> pd.DataFrame:
    """
    Load per-gameweek histories from the vaastav/Fantasy-Premier-League dataset.

    Returns a DataFrame whose schema aligns with the official FPL `history` entries,
    suitable for concatenation with the API-derived history frame.
    """
    frames: list[pd.DataFrame] = []
    for season in seasons:
        gw_files = list(_iter_gw_files(season))
        if not gw_files:
            logger.info("No gameweek CSVs found for season %s", season)
            continue
        team_map = _load_team_map(season)
        season_frames = []
        for path in gw_files:
            try:
                gw_df = pd.read_csv(path)
            except Exception as exc:  # pragma: no cover - defensive read
                logger.warning("Failed to load %s: %s", path, exc)
                continue
            gw_df = _normalise_frame(gw_df, season, team_map)
            if gw_df.empty:
                continue
            season_frames.append(gw_df)
        if not season_frames:
            continue
        season_df = pd.concat(season_frames, ignore_index=True, sort=False)
        frames.append(season_df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    # Align column order with API histories
    if "player_id" in combined.columns:
        combined["player_id"] = combined["player_id"].astype(int)
    return combined
