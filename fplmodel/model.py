from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import MODELS_DIR, REG_PARAMS, CLF_PARAMS, RANDOM_SEED
from .state import ModelState

REG_PATH = MODELS_DIR / "regressor.joblib"
CLF_PATH = MODELS_DIR / "classifier.joblib"

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, Pipeline]:
    """
    Train classifier (starts >=60) and regressor (points).
    For classifier target we derive from y_train's proxy minutes (not present here) ->
    we will approximate starts using 'minutes_ma3' threshold. If absent, fall back to points threshold.
    """
    # Heuristic "start" target from rolling minutes if available, else from last match minutes or points
    def derive_start_target(df: pd.DataFrame) -> np.ndarray:
        if "minutes_ma3" in df.columns:
            arr = (df["minutes_ma3"].values >= 60).astype(int)
            if arr.sum() > 10:  # valid
                return arr
        if "minutes_lag1" in df.columns:
            arr = (df["minutes_lag1"].values >= 60).astype(int)
            if arr.sum() > 10:
                return arr
        # last resort: proxy from points
        return (df.get("total_points_ma3", pd.Series(0, index=df.index)).values >= 2).astype(int)

    y_start = derive_start_target(X_train)
    # Pipelines
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("est", HistGradientBoostingClassifier(**CLF_PARAMS)),
    ])
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("est", HistGradientBoostingRegressor(**REG_PARAMS)),
    ])
    clf.fit(X_train, y_start)
    reg.fit(X_train, y_train)

    joblib.dump(clf, CLF_PATH)
    joblib.dump(reg, REG_PATH)
    return clf, reg

def load_models() -> Tuple[Pipeline, Pipeline]:
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    return clf, reg

def predict_expected_points(
    X_meta_and_feats: pd.DataFrame,
    clf: Pipeline,
    reg: Pipeline,
    state: ModelState,
) -> pd.DataFrame:
    """
    Input contains meta columns: player_id, full_name, now_cost_millions, team_id, element_type
    plus same feature columns used during training.
    Returns a DataFrame with expected_points and bias-corrected EP.
    """
    meta_cols = ["player_id", "full_name", "now_cost_millions", "team_id", "element_type"]
    meta = X_meta_and_feats[meta_cols].copy()
    feats = X_meta_and_feats.drop(columns=meta_cols)

    p_start = clf.predict_proba(feats)[:, 1]
    pts_hat = reg.predict(feats)
    ep = p_start * pts_hat

    # Apply bias corrections
    player_bias = np.array([state.get_player_bias(pid) for pid in meta["player_id"].values])
    pos_bias = np.array([state.get_position_bias(pos) for pos in meta["element_type"].values])
    ep_corrected = ep + player_bias + pos_bias

    out = meta.copy()
    out["p_start"] = p_start
    out["points_hat"] = pts_hat
    out["expected_points_raw"] = ep
    out["expected_points"] = ep_corrected.clip(min=0.0)  # no negatives
    return out
