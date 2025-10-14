from __future__ import annotations
import pandas as pd
from typing import Tuple

from .state import ModelState

def evaluate_last_finished_gw_and_update_state(
    clf, reg, X_train_like: pd.DataFrame, histories_df: pd.DataFrame, last_finished_gw: int, state: ModelState
) -> pd.DataFrame:
    """
    Build a held-out style prediction for last_finished_gw using features from gw-1,
    compare to actual totals at last_finished_gw, update EMA biases in state.
    Returns residuals df used for update.
    """
    # We need rows where 'round' == last_finished_gw; features must be from previous match (already lagged in build).
    eval_rows = histories_df[histories_df["round"] == last_finished_gw].copy()
    if eval_rows.empty:
        return pd.DataFrame()

    # Ensure we have element_type metadata; if histories lack it, merge from provided features frame.
    if "element_type" not in eval_rows.columns:
        meta_map = X_train_like[["player_id", "element_type"]].drop_duplicates()
        eval_rows = eval_rows.merge(meta_map, on="player_id", how="left")

    # X_train_like contains engineered features up to last_finished_gw. We need to match eval_rows players to their latest feature row (gw-1).
    # This is already provided by X_train_like: it's the same schema as training features. We'll just pick by player_id.
    meta_cols = ["player_id", "element_type"]
    # Build a meta+feature frame for eval prediction
    feats_cols = [c for c in X_train_like.columns if c not in meta_cols]
    # For each player in eval_rows, find their last available feature row in X_train_like
    merged = eval_rows[["player_id", "element_type", "total_points"]].merge(
        X_train_like, on=["player_id", "element_type"], how="left", suffixes=("","")
    ).dropna(subset=feats_cols, how="all")

    if merged.empty:
        return pd.DataFrame()

    # Predict
    feats = merged[feats_cols].fillna(0.0)
    p_start = clf.predict_proba(feats)[:, 1]
    pts_hat = reg.predict(feats)
    ep = p_start * pts_hat

    merged["predicted_points"] = ep
    merged["residual"] = merged["total_points"].astype(float) - merged["predicted_points"].astype(float)
    merged["gw"] = int(last_finished_gw)

    # Update state EMA
    res_short = merged[["player_id", "element_type", "residual", "gw"]].copy()
    state.update_from_residuals(res_short)
    return res_short
