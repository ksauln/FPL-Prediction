from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from .config import ROLLING_WINDOWS, MIN_MATCHES_FOR_FEATURES
from .state import ModelState

def _rolling_feats(hist: pd.DataFrame, windows=(3,5)) -> pd.DataFrame:
    """Create rolling means for key stats, grouped by player, ordered by round."""
    hist = hist.sort_values(["player_id", "round"]).copy()
    group = hist.groupby("player_id", group_keys=False)
    stats = ["total_points", "minutes", "goals_scored", "assists", "clean_sheets",
             "influence", "creativity", "threat", "ict_index"]
    for w in windows:
        for s in stats:
            col = f"{s}_ma{w}"
            hist[col] = group[s].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
    # last match lag features
    for s in stats:
        hist[f"{s}_lag1"] = group[s].shift(1)
    # Was home as int
    if "was_home" in hist.columns:
        hist["was_home"] = hist["was_home"].astype(int)
    # Mark usable rows (enough previous matches)
    hist["prev_matches"] = group["round"].transform(lambda x: x.rank(method="first") - 1)
    hist["enough_prev"] = hist["prev_matches"] >= MIN_MATCHES_FOR_FEATURES
    return hist

def _merge_team_strength(hist: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add team and opponent base strength features.
    """
    teams = teams_df[["team_id", "strength", "strength_attack_home", "strength_attack_away",
                      "strength_defence_home", "strength_defence_away"]].copy()
    teams = teams.rename(columns={
        "strength": "team_strength_overall",
        "strength_attack_home": "team_attack_home",
        "strength_attack_away": "team_attack_away",
        "strength_defence_home": "team_def_home",
        "strength_defence_away": "team_def_away",
    })
    hist = hist.merge(teams, left_on="team", right_on="team_id", how="left", suffixes=("",""))
    # opponent strength
    opp = teams_df[["team_id", "strength"]].rename(columns={"strength":"opp_strength_overall"})
    hist = hist.merge(opp, left_on="opponent_team", right_on="team_id", how="left", suffixes=("","_opp"))
    hist = hist.drop(columns=["team_id_opp"], errors="ignore")
    return hist

def build_training_and_pred_frames(
    elements_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    histories_df: pd.DataFrame,
    next_gw: int,
    last_finished_gw: int,
    state: ModelState,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Returns:
      X_train (DataFrame), y_train (Series of total_points), X_pred (DataFrame for next_gw)
    """
    # Merge element team ids into histories (histories has 'team')
    base = histories_df.merge(
        elements_df[["player_id", "team_id", "element_type"]].rename(columns={"team_id":"team"}),
        on="player_id", how="left"
    )
    base = _rolling_feats(base, windows=tuple(ROLLING_WINDOWS))
    base = _merge_team_strength(base, teams_df)

    # Include bias features
    base["player_bias"] = base["player_id"].astype(str).map(state.player_bias).fillna(0.0)
    base["pos_bias"] = base["element_type"].astype(str).map(state.position_bias).fillna(0.0)

    feature_cols = [c for c in base.columns if any(
        c.endswith(f"_ma{w}") for w in ROLLING_WINDOWS
    ) or c.endswith("_lag1")] + [
        "was_home", "team_strength_overall", "team_attack_home", "team_attack_away",
        "team_def_home", "team_def_away", "opp_strength_overall", "player_bias", "pos_bias"
    ]

    # TRAIN: rows with enough history and gw <= last_finished_gw
    train_rows = base[(base["enough_prev"]) & (base["round"] <= last_finished_gw)].copy()
    X_train = train_rows[feature_cols].fillna(0.0)
    y_train = train_rows["total_points"].astype(float)

    # PRED: need last_finished features to forecast next_gw per player (use most recent row per player <= last_finished_gw)
    last_rows = base[base["round"] <= last_finished_gw].sort_values(["player_id","round"]).groupby("player_id").tail(1)
    # but we must attach players' meta for identification (name, cost, team, element_type)
    last_rows = last_rows.merge(
        elements_df[["player_id", "full_name", "now_cost_millions", "team_id", "element_type"]],
        on="player_id", how="left"
    )
    X_pred = last_rows[["player_id", "full_name", "now_cost_millions", "team_id", "element_type"]].copy()
    X_pred_features = last_rows[feature_cols].fillna(0.0)
    # Return both meta and features separately for convenience
    X_pred = X_pred.join(X_pred_features.reset_index(drop=True))
    return X_train, y_train, X_pred

def expand_for_double_gw(pred_df: pd.DataFrame, fixtures_df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
    """
    If a player has multiple fixtures in next_gw, scale EP by number of fixtures.
    fixtures_df: all fixtures
    """
    # Count fixtures per team in next_gw
    gw_fx = fixtures_df[fixtures_df["event"] == next_gw]
    if gw_fx.empty:
        pred_df["fixture_multiplier"] = 1.0
        return pred_df
    team_counts = {}
    for _, row in gw_fx.iterrows():
        team_counts[row["team_h"]] = team_counts.get(row["team_h"], 0) + 1
        team_counts[row["team_a"]] = team_counts.get(row["team_a"], 0) + 1
    pred_df["fixture_multiplier"] = pred_df["team_id"].map(team_counts).fillna(1).astype(float)
    return pred_df
