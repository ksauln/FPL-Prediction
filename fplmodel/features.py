from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Sequence, Iterable

from .config import ROLLING_WINDOWS, MIN_MATCHES_FOR_FEATURES
from .fbref_data import (
    build_fbref_player_feature_matrix,
    canonicalise_player_name,
    collect_player_match_stats,
    compute_canonical_names,
    normalise_season_code,
)
from .state import ModelState


SET_PIECE_ORDER_MAP = {
    "corners_and_indirect_freekicks_order": "corners",
    "direct_freekicks_order": "direct_fk",
    "penalties_order": "penalty",
}

FBREF_DEFENSIVE_COMPONENTS = (
    "fbref_match_defense_clearances",
    "fbref_match_summary_blocks",
    "fbref_match_summary_int",
    "fbref_match_summary_tkl",
    "fbref_match_misc_ball_recov",
)


def _prepare_player_static_features(elements_df: pd.DataFrame) -> pd.DataFrame:
    """Enhance elements dataframe with numeric form metrics and set-piece flags."""
    df = elements_df.copy()

    name_cols = [
        col
        for col in ("full_name", "web_name", "first_name", "second_name")
        if col in df.columns
    ]
    if name_cols:
        compute_canonical_names(df, name_cols, target="canonical_name")
        df["canonical_name"] = df["canonical_name"].replace("", pd.NA)

    numeric_cols = [
        "form",
        "points_per_game",
        "selected_by_percent",
        "value_form",
        "value_season",
        "expected_goals_per_90",
        "expected_assists_per_90",
        "expected_goal_involvements_per_90",
        "expected_goals_conceded_per_90",
        "goals_conceded_per_90",
        "saves_per_90",
        "starts_per_90",
        "clean_sheets_per_90",
        "defensive_contribution_per_90",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "minutes" in df.columns:
        df["season_minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    else:
        df["season_minutes"] = 0.0

    score_components = []
    for order_col, prefix in SET_PIECE_ORDER_MAP.items():
        if order_col in df.columns:
            df[order_col] = pd.to_numeric(df[order_col], errors="coerce")
            order = df[order_col]
        else:
            order = pd.Series(np.nan, index=df.index, dtype=float)
            df[order_col] = order

        df[f"has_{prefix}_duty"] = (order.fillna(0) > 0).astype(int)
        df[f"primary_{prefix}_taker"] = (order == 1).astype(int)
        inv = 1.0 / order.replace({0: np.nan})
        inv = inv.where(order > 0).fillna(0)
        score_components.append(inv)

    if score_components:
        df["set_piece_duty_score"] = sum(score_components)
    else:
        df["set_piece_duty_score"] = 0.0

    return df

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator.replace({0: np.nan})
    return ratio.replace([np.inf, -np.inf], np.nan)


def _season_start_year(series: pd.Series) -> pd.Series:
    """Return the starting year (YYYY) extracted from FPL season codes."""

    if series.empty:
        return pd.Series(dtype=float)

    normalised = series.fillna("").astype(str).map(normalise_season_code)
    start_year = normalised.str.split("-", n=1).str[0]
    return pd.to_numeric(start_year, errors="coerce")


def _numeric_series(df: pd.DataFrame, column: str) -> Tuple[pd.Series, bool]:
    """Return a numeric series for ``column`` and flag whether it existed."""

    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float), False

    series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return series, True


def _compute_fbref_defensive_contribution(df: pd.DataFrame) -> pd.Series:
    """Sum FBRef defensive components into a defensive contribution count."""

    clearances, _ = _numeric_series(df, "fbref_match_defense_clearances")
    blocks, _ = _numeric_series(df, "fbref_match_summary_blocks")
    interceptions, _ = _numeric_series(df, "fbref_match_summary_int")
    tackles, _ = _numeric_series(df, "fbref_match_summary_tkl")
    recoveries, _ = _numeric_series(df, "fbref_match_misc_ball_recov")

    cbi = clearances + blocks + interceptions
    return (cbi + tackles + recoveries).fillna(0.0)


def _apply_historic_defensive_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """Populate defensive contribution for seasons prior to 2025-26."""

    if df.empty or "season_name" not in df.columns:
        return df

    if not any(col in df.columns for col in FBREF_DEFENSIVE_COMPONENTS):
        return df

    start_year = _season_start_year(df["season_name"])
    historic_mask = start_year < 2025
    if not historic_mask.any():
        return df

    fbref_contribution = _compute_fbref_defensive_contribution(df)

    if "defensive_contribution" not in df.columns:
        df["defensive_contribution"] = 0.0

    df.loc[historic_mask, "defensive_contribution"] = fbref_contribution.loc[
        historic_mask
    ]
    df["defensive_contribution"] = pd.to_numeric(
        df["defensive_contribution"], errors="coerce"
    ).fillna(0.0)

    return df


def _rolling_mean(group, window: int) -> pd.Series:
    return group.transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())


def _add_team_context_features(hist: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    """Create rolling team form features and merge opponent versions."""
    required_cols = {"team_h_score", "team_a_score", "was_home", "team", "fixture"}
    if not required_cols.issubset(hist.columns):
        return hist

    hist = hist.copy()
    was_home = hist["was_home"].astype(bool)
    hist["team_goals_for"] = np.where(was_home, hist["team_h_score"], hist["team_a_score"])
    hist["team_goals_against"] = np.where(was_home, hist["team_a_score"], hist["team_h_score"])
    hist["team_goal_diff"] = hist["team_goals_for"] - hist["team_goals_against"]
    hist["team_clean_sheet"] = (hist["team_goals_against"] == 0).astype(int)
    hist["team_conceded_two_plus"] = (hist["team_goals_against"] >= 2).astype(int)
    hist["team_match_points"] = np.select(
        [hist["team_goals_for"] > hist["team_goals_against"], hist["team_goals_for"] == hist["team_goals_against"]],
        [3, 1],
        default=0,
    )

    team_group = hist.groupby("team", group_keys=False)
    team_stats = [
        "team_goals_for",
        "team_goals_against",
        "team_goal_diff",
        "team_clean_sheet",
        "team_conceded_two_plus",
        "team_match_points",
    ]

    for w in windows:
        for stat in team_stats:
            ma_col = f"{stat}_ma{w}"
            hist[ma_col] = _rolling_mean(team_group[stat], w)

    team_feature_cols = [f"{stat}_ma{w}" for stat in team_stats for w in windows]
    team_feature_cols = [c for c in team_feature_cols if c in hist.columns]
    if team_feature_cols:
        team_frame = hist[["fixture", "team"] + team_feature_cols].drop_duplicates(subset=["fixture", "team"])
        rename_map = {
            col: col.replace("team_", "opp_team_", 1) if col.startswith("team_") else f"opp_{col}"
            for col in team_feature_cols
        }
        opponent_frame = team_frame.rename(columns={"team": "opponent_team", **rename_map})
        hist = hist.merge(opponent_frame, on=["fixture", "opponent_team"], how="left")

    return hist


def _rolling_feats(hist: pd.DataFrame, windows=(3, 5)) -> pd.DataFrame:
    """Create rolling means for key stats, grouped by player, ordered by round."""
    hist = hist.sort_values(["player_id", "round"]).copy()
    group = hist.groupby("player_id", group_keys=False)

    base_stats = [
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "goals_conceded",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "saves",
        "bonus",
        "bps",
        "clearances_blocks_interceptions",
        "recoveries",
        "tackles",
        "defensive_contribution",
        "starts",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "value",
        "transfers_balance",
        "selected",
        "transfers_in",
        "transfers_out",
    ]

    available_cols = set(hist.columns)
    for col in base_stats:
        if col in available_cols:
            hist[col] = _safe_numeric(hist[col])

    derived_features = []
    if {"goals_scored", "assists"}.issubset(available_cols):
        hist["attacking_returns"] = hist["goals_scored"].fillna(0) + hist["assists"].fillna(0)
        derived_features.append("attacking_returns")
    if {"goals_scored", "expected_goals"}.issubset(available_cols):
        hist["finishing_plus_minus"] = hist["goals_scored"].fillna(0) - hist["expected_goals"].fillna(0)
        hist["finishing_ratio"] = _safe_ratio(hist["goals_scored"].fillna(0), hist["expected_goals"].fillna(0))
        derived_features.extend(["finishing_plus_minus", "finishing_ratio"])
    if {"assists", "expected_assists"}.issubset(available_cols):
        hist["creation_plus_minus"] = hist["assists"].fillna(0) - hist["expected_assists"].fillna(0)
        hist["creation_ratio"] = _safe_ratio(hist["assists"].fillna(0), hist["expected_assists"].fillna(0))
        derived_features.extend(["creation_plus_minus", "creation_ratio"])
    if {"goals_scored", "assists", "expected_goal_involvements"}.issubset(available_cols):
        hist["xgi_plus_minus"] = (
            hist["goals_scored"].fillna(0) + hist["assists"].fillna(0) - hist["expected_goal_involvements"].fillna(0)
        )
        derived_features.append("xgi_plus_minus")
    if {"minutes"}.issubset(available_cols):
        hist["minutes_share"] = hist["minutes"].fillna(0) / 90.0
        derived_features.append("minutes_share")
    if {"minutes", "total_points"}.issubset(available_cols):
        minutes = hist["minutes"].replace({0: np.nan})
        hist["points_per_90"] = (hist["total_points"].fillna(0) * 90.0) / minutes
        hist["points_per_90"] = hist["points_per_90"].replace([np.inf, -np.inf], np.nan)
        derived_features.append("points_per_90")
    if {"tackles", "clearances_blocks_interceptions"}.issubset(available_cols):
        hist["tackles_plus_interceptions"] = hist["tackles"].fillna(0) + hist["clearances_blocks_interceptions"].fillna(0)
        derived_features.append("tackles_plus_interceptions")
    if {"tackles", "recoveries", "clearances_blocks_interceptions"}.issubset(available_cols):
        hist["defensive_actions"] = (
            hist["tackles"].fillna(0)
            + hist["recoveries"].fillna(0)
            + hist["clearances_blocks_interceptions"].fillna(0)
        )
        derived_features.append("defensive_actions")

    stats = [col for col in base_stats + derived_features if col in hist.columns]

    new_columns: Dict[str, pd.Series] = {}
    for w in windows:
        for s in stats:
            col = f"{s}_ma{w}"
            new_columns[col] = _rolling_mean(group[s], w)

    for s in stats:
        new_columns[f"{s}_lag1"] = group[s].shift(1)

    if new_columns:
        hist = pd.concat([hist, pd.DataFrame(new_columns, index=hist.index)], axis=1)

    if "was_home" in hist.columns:
        hist["was_home"] = hist["was_home"].astype(int)

    hist = _add_team_context_features(hist, windows)

    hist["prev_matches"] = group["round"].transform(lambda x: x.rank(method="first") - 1)
    hist["enough_prev"] = hist["prev_matches"] >= MIN_MATCHES_FOR_FEATURES
    return hist

def _merge_team_strength(hist: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Add team and opponent base strength and recent form features."""
    base_cols = [
        "team_id",
        "strength",
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
        "form",
        "points",
        "position",
        "played",
        "win",
        "draw",
        "loss",
    ]
    available_cols = [c for c in base_cols if c in teams_df.columns]
    teams = teams_df[available_cols].copy()

    rename_map = {
        "strength": "team_strength_overall",
        "strength_overall_home": "team_strength_home",
        "strength_overall_away": "team_strength_away",
        "strength_attack_home": "team_attack_home",
        "strength_attack_away": "team_attack_away",
        "strength_defence_home": "team_def_home",
        "strength_defence_away": "team_def_away",
        "form": "team_form_rating",
        "points": "team_points_table",
        "position": "team_league_position",
        "played": "team_matches_played",
        "win": "team_wins",
        "draw": "team_draws",
        "loss": "team_losses",
    }
    teams = teams.rename(columns=rename_map)
    hist = hist.merge(teams, left_on="team", right_on="team_id", how="left")

    opp_cols = {
        "team_strength_overall": "opp_strength_overall",
        "team_strength_home": "opp_strength_home",
        "team_strength_away": "opp_strength_away",
        "team_attack_home": "opp_attack_home",
        "team_attack_away": "opp_attack_away",
        "team_def_home": "opp_def_home",
        "team_def_away": "opp_def_away",
        "team_form_rating": "opp_form_rating",
        "team_points_table": "opp_points_table",
        "team_league_position": "opp_league_position",
        "team_matches_played": "opp_matches_played",
        "team_wins": "opp_wins",
        "team_draws": "opp_draws",
        "team_losses": "opp_losses",
    }
    opponent = teams.rename(columns=opp_cols)
    hist = hist.merge(opponent, left_on="opponent_team", right_on="team_id", how="left", suffixes=("", "_opp"))
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
    histories_df = histories_df.copy()
    if "kickoff_time" in histories_df.columns:
        kickoff_times = pd.to_datetime(histories_df["kickoff_time"], errors="coerce")
        histories_df["match_date"] = kickoff_times.dt.strftime("%Y-%m-%d")
    else:
        histories_df["match_date"] = pd.NA

    if "was_home" in histories_df.columns:
        histories_df["home_away"] = histories_df["was_home"].astype(bool).map(
            {True: "home", False: "away"}
        )
    else:
        histories_df["home_away"] = pd.NA

    elements_enhanced = _prepare_player_static_features(elements_df)

    fbref_feature_frame = pd.DataFrame()
    fbref_match_frame = pd.DataFrame()
    if "season_name" in histories_df.columns:
        history_seasons = [
            str(season)
            for season in histories_df["season_name"].dropna().unique()
            if str(season).strip() and str(season).strip().lower() not in {"nan", "none"}
        ]
        if history_seasons:
            try:
                fbref_feature_frame = build_fbref_player_feature_matrix(
                    elements_df, seasons=history_seasons
                )
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "FBRef feature build failed: %s", exc
                )
                fbref_feature_frame = pd.DataFrame()

            if not fbref_feature_frame.empty and "fbref_player_id" in fbref_feature_frame.columns:
                fbref_id_map = (
                    fbref_feature_frame.dropna(subset=["fbref_player_id"])  # type: ignore[arg-type]
                    .sort_values(["player_id", "season_name"])
                    .groupby("player_id")["fbref_player_id"]
                    .first()
                    .to_dict()
                )

                canonical_map = {}
                if "canonical_name" in elements_enhanced.columns:
                    canonical_map = (
                        elements_enhanced.dropna(subset=["canonical_name"])  # type: ignore[arg-type]
                        .drop_duplicates(subset=["player_id"])
                        .set_index("player_id")["canonical_name"]
                        .to_dict()
                    )

                season_lookup = histories_df.dropna(subset=["season_name"])
                if not season_lookup.empty:
                    season_lookup = season_lookup.assign(
                        season_name=season_lookup["season_name"].astype(str)
                    )
                    season_lookup = season_lookup[
                        season_lookup["season_name"].str.strip().ne("")
                    ]
                    player_seasons = (
                        season_lookup.groupby("player_id")["season_name"]
                        .apply(
                            lambda values: sorted(
                                dict.fromkeys(
                                    normalise_season_code(value)
                                    for value in values
                                    if isinstance(value, str)
                                    and value.strip()
                                    and value.strip().lower() not in {"nan", "none"}
                                )
                            )
                        )
                        .to_dict()
                    )
                else:
                    player_seasons = {}

                try:
                    fbref_match_frame = collect_player_match_stats(
                        player_id_map=fbref_id_map,
                        player_seasons=player_seasons,
                        canonical_name_map=canonical_map,
                    )
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "FBRef match stats build failed: %s", exc
                    )
                    fbref_match_frame = pd.DataFrame()

    if not fbref_match_frame.empty:
        team_name_map = {}
        if {"id", "name"}.issubset(teams_df.columns):
            team_name_map = {
                canonicalise_player_name(name): team_id
                for team_id, name in teams_df[["id", "name"]].dropna().itertuples(index=False)
            }
        if team_name_map:
            fbref_match_frame["fbref_team_id"] = fbref_match_frame["fbref_team_canonical"].map(
                team_name_map
            )
            fbref_match_frame["fbref_opponent_team_id"] = fbref_match_frame[
                "fbref_opponent_canonical"
            ].map(team_name_map)
        else:
            fbref_match_frame["fbref_team_id"] = pd.NA
            fbref_match_frame["fbref_opponent_team_id"] = pd.NA

        fbref_match_frame["home_away"] = fbref_match_frame["fbref_home_away"]
        fbref_match_frame = fbref_match_frame.drop(columns=["canonical_name"], errors="ignore")
        fbref_match_frame = fbref_match_frame.dropna(subset=["match_date"])
        fbref_match_frame = fbref_match_frame.drop_duplicates(
            subset=["player_id", "season_name", "match_date", "home_away"]
        )

    team_labels = teams_df[["team_id", "name"]].rename(columns={"name": "team_name"})
    elements_with_team = elements_enhanced.merge(team_labels, on="team_id", how="left")

    # Merge element team ids into histories (histories has 'team')
    static_feature_cols = [
        "player_id",
        "team_id",
        "element_type",
        "canonical_name",
        "form",
        "points_per_game",
        "value_form",
        "value_season",
        "selected_by_percent",
        "expected_goals_per_90",
        "expected_assists_per_90",
        "expected_goal_involvements_per_90",
        "expected_goals_conceded_per_90",
        "goals_conceded_per_90",
        "saves_per_90",
        "starts_per_90",
        "clean_sheets_per_90",
        "defensive_contribution_per_90",
        "season_minutes",
        "corners_and_indirect_freekicks_order",
        "direct_freekicks_order",
        "penalties_order",
        "has_corners_duty",
        "primary_corners_taker",
        "has_direct_fk_duty",
        "primary_direct_fk_taker",
        "has_penalty_duty",
        "primary_penalty_taker",
        "set_piece_duty_score",
    ]
    merge_cols = [c for c in static_feature_cols if c in elements_enhanced.columns]
    base = histories_df.merge(
        elements_enhanced[merge_cols].rename(columns={"team_id": "team"}),
        on="player_id", how="left",
        suffixes=("", "_current"),
    )

    if not fbref_match_frame.empty:
        base = base.merge(
            fbref_match_frame,
            on=["player_id", "season_name", "match_date", "home_away"],
            how="left",
        )

    base = _apply_historic_defensive_contribution(base)

    # Prefer historical team ids when available, but fall back to current squad assignment.
    if "team" not in base.columns and "team_current" in base.columns:
        base = base.rename(columns={"team_current": "team"})
    elif "team_current" in base.columns:
        base["team"] = base["team"].fillna(base["team_current"])
        base = base.drop(columns=["team_current"])
    base = _rolling_feats(base, windows=tuple(ROLLING_WINDOWS))
    if not fbref_feature_frame.empty:
        base = base.merge(
            fbref_feature_frame,
            on=["player_id", "season_name"],
            how="left",
        )
    base = _merge_team_strength(base, teams_df)

    # Include bias features
    base["player_bias"] = base["player_id"].astype(str).map(state.player_bias).fillna(0.0)
    base["pos_bias"] = base["element_type"].astype(str).map(state.position_bias).fillna(0.0)

    rolling_feature_cols = [
        c
        for c in base.columns
        if any(c.endswith(f"_ma{w}") for w in ROLLING_WINDOWS) or c.endswith("_lag1")
    ]
    manual_features = [
        "was_home",
        "team_strength_overall",
        "team_strength_home",
        "team_strength_away",
        "team_attack_home",
        "team_attack_away",
        "team_def_home",
        "team_def_away",
        "team_form_rating",
        "team_points_table",
        "team_league_position",
        "team_matches_played",
        "team_wins",
        "team_draws",
        "team_losses",
        "opp_strength_overall",
        "opp_strength_home",
        "opp_strength_away",
        "opp_attack_home",
        "opp_attack_away",
        "opp_def_home",
        "opp_def_away",
        "opp_form_rating",
        "opp_points_table",
        "opp_league_position",
        "opp_matches_played",
        "opp_wins",
        "opp_draws",
        "opp_losses",
        "form",
        "points_per_game",
        "value_form",
        "value_season",
        "selected_by_percent",
        "expected_goals_per_90",
        "expected_assists_per_90",
        "expected_goal_involvements_per_90",
        "expected_goals_conceded_per_90",
        "goals_conceded_per_90",
        "saves_per_90",
        "starts_per_90",
        "clean_sheets_per_90",
        "defensive_contribution_per_90",
        "corners_and_indirect_freekicks_order",
        "direct_freekicks_order",
        "penalties_order",
        "has_corners_duty",
        "primary_corners_taker",
        "has_direct_fk_duty",
        "primary_direct_fk_taker",
        "has_penalty_duty",
        "primary_penalty_taker",
        "set_piece_duty_score",
        "player_bias",
        "pos_bias",
    ]
    manual_feature_cols = [c for c in manual_features if c in base.columns]
    excluded_fbref_feature_cols = {"fbref_team_id", "fbref_opponent_team_id"}
    fbref_feature_cols = sorted(
        c
        for c in base.columns
        if c.startswith("fbref_")
        and pd.api.types.is_numeric_dtype(base[c])
        and c not in excluded_fbref_feature_cols
    )
    feature_cols = rolling_feature_cols + manual_feature_cols
    for col in fbref_feature_cols:
        if col not in feature_cols:
            feature_cols.append(col)

    # TRAIN: rows with enough history and gw <= last_finished_gw
    train_rows = base[(base["enough_prev"]) & (base["round"] <= last_finished_gw)].copy()
    X_train = train_rows[feature_cols].fillna(0.0)
    y_train = train_rows["total_points"].astype(float)

    # PRED: need last_finished features to forecast next_gw per player (use most recent row per player <= last_finished_gw)
    last_rows = base[base["round"] <= last_finished_gw].sort_values(["player_id","round"]).groupby("player_id").tail(1)
    # but we must attach players' meta for identification (name, cost, team, element_type)
    last_rows = last_rows.drop(columns=["team_id", "element_type"], errors="ignore").merge(
        elements_with_team[
            [
                "player_id",
                "full_name",
                "now_cost_millions",
                "team_id",
                "element_type",
                "team_name",
                "season_minutes",
            ]
        ],
        on="player_id",
        how="left",
        suffixes=("", "_meta"),
    )
    if "season_minutes_meta" in last_rows.columns:
        if "season_minutes" in last_rows.columns:
            last_rows["season_minutes"] = last_rows["season_minutes"].fillna(last_rows["season_minutes_meta"])
        else:
            last_rows = last_rows.rename(columns={"season_minutes_meta": "season_minutes"})
        last_rows = last_rows.drop(columns=["season_minutes_meta"])
    X_pred = last_rows[
        [
            "player_id",
            "full_name",
            "team_name",
            "now_cost_millions",
            "team_id",
            "element_type",
            "season_minutes",
        ]
    ].copy()
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


def log_model_feature_weights(
    logger: logging.Logger,
    feature_names: Sequence[str],
    model: object,
    model_label: str = "model",
    top_n: int = 15,
) -> None:
    """
    Log the most influential features (by absolute weight) for a fitted model.
    """
    if feature_names is None:
        logger.info("No feature names available to log for %s", model_label)
        return

    feature_names = list(feature_names)
    if not feature_names:
        logger.info("No feature names available to log for %s", model_label)
        return

    selector = None
    estimator = None
    if hasattr(model, "named_steps"):
        selector = model.named_steps.get("feature_selector")
        estimator = model.named_steps.get("est")
    if selector is not None and hasattr(selector, "features_to_keep_") and selector.features_to_keep_:
        feature_names = list(selector.features_to_keep_)
    if estimator is None and hasattr(model, "feature_importances_"):
        estimator = model
    if estimator is None:
        logger.info("Skipping feature weight logging for %s; estimator unavailable.", model_label)
        return

    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        logger.info(
            "Estimator %s for %s does not expose feature_importances_; skipping logging.",
            type(estimator).__name__,
            model_label,
        )
        return

    if len(importances) != len(feature_names):
        logger.warning(
            "Feature name count (%d) does not match weights (%d) for %s; logging skipped.",
            len(feature_names),
            len(importances),
            model_label,
        )
        return

    pairs = sorted(
        zip(feature_names, importances),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top_pairs = [(name, weight) for name, weight in pairs if abs(weight) > 0][:top_n]

    if not top_pairs:
        logger.info("All feature importances are zero for %s.", model_label)
        return

    formatted = ", ".join(f"{name}: {weight:.4f}" for name, weight in top_pairs)
    logger.info("Top feature weights for %s: %s", model_label, formatted)
