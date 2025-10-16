"""Utilities for analysing user teams against the model optimal XI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .team_picker import pick_best_xi


@dataclass
class TeamSummary:
    """Summary statistics for a FPL squad."""

    squad: List[Dict[str, object]]
    bench: List[Dict[str, object]]
    starting_cost: float
    bench_cost: float
    total_cost: float
    expected_points_without_captain: float
    total_expected_points_with_captain: float
    bench_expected_points: float
    captain: Optional[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "squad": self.squad,
            "bench": self.bench,
            "starting_cost": self.starting_cost,
            "bench_cost": self.bench_cost,
            "total_cost": self.total_cost,
            "expected_points_without_captain": self.expected_points_without_captain,
            "total_expected_points_with_captain": self.total_expected_points_with_captain,
            "bench_expected_points": self.bench_expected_points,
            "captain": self.captain,
        }


def _get_players(df: pd.DataFrame, column: str, expected_value: int) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df[column] == expected_value].copy()


def summarise_team(team_df: pd.DataFrame, captain_id: Optional[int] = None) -> TeamSummary:
    """Create a :class:`TeamSummary` for a user provided squad.

    Parameters
    ----------
    team_df:
        DataFrame containing at least the columns ``player_id``, ``full_name``,
        ``expected_points`` and ``now_cost_millions``. Optional columns include
        ``starting`` (1 for starter, 0 otherwise), ``bench`` (1 if on bench) and
        ``captain`` (1 for the chosen captain).
    captain_id:
        Player identifier to be treated as captain. Overrides any ``captain``
        column present in ``team_df``.
    """

    required_cols = {"player_id", "full_name", "expected_points", "now_cost_millions"}
    missing = required_cols - set(team_df.columns)
    if missing:
        raise ValueError(f"team_df missing required columns: {sorted(missing)}")

    starters = _get_players(team_df, "starting", 1)
    bench = _get_players(team_df, "bench", 1)

    if starters.empty:
        starters = team_df.copy()
    if bench.empty:
        bench = team_df[~team_df.index.isin(starters.index)].copy()

    captain_col = "captain"
    if captain_id is not None:
        captain_pid = captain_id
    elif captain_col in team_df.columns:
        captain_series = team_df.loc[team_df[captain_col] == 1, "player_id"]
        captain_pid = int(captain_series.iloc[0]) if not captain_series.empty else None
    else:
        captain_pid = None

    starters_points = float(starters["expected_points"].sum())
    captain_bonus = 0.0
    captain_name: Optional[str] = None
    if captain_pid is not None:
        if captain_pid not in starters["player_id"].values:
            raise ValueError("Captain must be one of the starters.")
        captain_row = starters.loc[starters["player_id"] == captain_pid].iloc[0]
        captain_bonus = float(captain_row["expected_points"])
        captain_name = str(captain_row["full_name"])

    bench_points = float(bench["expected_points"].sum()) if not bench.empty else 0.0

    starting_cost = float(starters["now_cost_millions"].sum())
    bench_cost = float(bench["now_cost_millions"].sum()) if not bench.empty else 0.0
    total_cost = float(team_df["now_cost_millions"].sum())

    squad_records = starters.to_dict(orient="records")
    bench_records = bench.to_dict(orient="records") if not bench.empty else []

    return TeamSummary(
        squad=squad_records,
        bench=bench_records,
        starting_cost=starting_cost,
        bench_cost=bench_cost,
        total_cost=total_cost,
        expected_points_without_captain=starters_points,
        total_expected_points_with_captain=starters_points + captain_bonus,
        bench_expected_points=bench_points,
        captain=captain_name,
    )


def compare_team_to_optimal(
    predictions: pd.DataFrame,
    user_team: pd.DataFrame,
    captain_id: Optional[int] = None,
    budget_m: Optional[float] = None,
    formation: Optional[Dict[str, int]] = None,
    formations: Optional[Iterable[Dict[str, int]]] = None,
) -> Dict[str, object]:
    """Compare a user squad against the model optimal team for a gameweek."""

    comparison_team = user_team.copy()
    if "expected_points" not in comparison_team.columns:
        user_ep = predictions[["player_id", "expected_points"]]
        comparison_team = comparison_team.merge(user_ep, on="player_id", how="left")
    if comparison_team["expected_points"].isna().any():
        missing = comparison_team.loc[comparison_team["expected_points"].isna(), "player_id"].tolist()
        raise ValueError(f"Missing expected points for players: {missing}")

    user_summary = summarise_team(comparison_team, captain_id=captain_id)

    optimal_budget = budget_m if budget_m is not None else user_summary.total_cost
    optimal_team = pick_best_xi(
        predictions,
        budget_m=optimal_budget,
        formation=formation,
        formations=formations,
    )

    user_points = user_summary.total_expected_points_with_captain
    optimal_points = float(optimal_team["total_expected_points_with_captain"])
    points_gap = optimal_points - user_points
    rating = 0.0 if optimal_points <= 0 else (user_points / optimal_points) * 100.0

    return {
        "user_team": user_summary.as_dict(),
        "optimal_team": optimal_team,
        "comparison": {
            "user_expected_points": user_points,
            "optimal_expected_points": optimal_points,
            "points_gap": points_gap,
            "rating": rating,
        },
    }


__all__ = ["TeamSummary", "summarise_team", "compare_team_to_optimal"]

