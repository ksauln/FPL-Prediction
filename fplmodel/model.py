from __future__ import annotations
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Sequence

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .config import (
    MODELS_DIR,
    REG_PARAMS,
    CLF_PARAMS,
    RANDOM_SEED,
    FEATURE_CORRELATION_THRESHOLD,
    FEATURE_MIN_VARIANCE,
    ENABLE_HYPERPARAM_TUNING,
    HYPERPARAM_TUNING_MIN_SAMPLES,
    HYPERPARAM_TUNING_ITER,
    HYPERPARAM_TUNING_CV,
    REG_PARAM_DISTRIBUTIONS,
    CLF_PARAM_DISTRIBUTIONS,
)
from .state import ModelState

logger = logging.getLogger(__name__)

REG_PATH = MODELS_DIR / "regressor.joblib"
CLF_PATH = MODELS_DIR / "classifier.joblib"

class CorrelatedFeatureDropper(BaseEstimator, TransformerMixin):
    """
    Drop features with low variance or high pairwise correlation.
    """

    def __init__(self, correlation_threshold: float = 0.95, min_variance: float = 0.0):
        self.correlation_threshold = correlation_threshold
        self.min_variance = min_variance
        self.features_to_drop_: list[str] = []
        self.features_to_keep_: list[str] = []
        self.low_variance_features_: list[str] = []
        self.high_correlation_pairs_: list[tuple[str, str]] = []

    @staticmethod
    def _ensure_dataframe(X, copy: bool = True) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy() if copy else X
        return pd.DataFrame(X)

    def fit(self, X, y=None):  # noqa: D401 - standard sklearn signature
        df = self._ensure_dataframe(X)
        numeric_df = df.select_dtypes(include=[np.number])

        low_variance: set[str] = set()
        high_corr_drop: set[str] = set()
        high_corr_pairs: list[tuple[str, str]] = []

        if not numeric_df.empty:
            na_cols = numeric_df.columns[numeric_df.notna().sum() == 0]
            var_series = numeric_df.var(ddof=0).fillna(0.0)
            low_variance.update(na_cols.tolist())
            low_variance.update(var_series[var_series <= self.min_variance].index.tolist())

            corr_candidates = numeric_df.drop(columns=list(low_variance), errors="ignore")
            if not corr_candidates.empty:
                corr_matrix = corr_candidates.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                for col in upper.columns:
                    correlated = upper.index[upper[col] > self.correlation_threshold].tolist()
                    if correlated:
                        high_corr_drop.add(col)
                        high_corr_pairs.extend((row, col) for row in correlated)

        self.low_variance_features_ = sorted(low_variance)
        self.high_correlation_pairs_ = high_corr_pairs
        self.features_to_drop_ = sorted(low_variance.union(high_corr_drop))
        self.features_to_keep_ = [col for col in df.columns if col not in self.features_to_drop_]
        return self

    def transform(self, X):
        df = self._ensure_dataframe(X, copy=False)
        return df.drop(columns=self.features_to_drop_, errors="ignore")

def _param_space_size(param_grid: dict[str, Sequence]) -> int:
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total

def _should_tune(y: Sequence, require_two_classes: bool = False) -> bool:
    if not ENABLE_HYPERPARAM_TUNING:
        return False
    if y is None:
        return False
    if len(y) < max(HYPERPARAM_TUNING_MIN_SAMPLES, HYPERPARAM_TUNING_CV):
        return False
    unique_count = np.unique(np.asarray(y)).size
    if require_two_classes and unique_count < 2:
        logger.info("Skipping tuning: classifier target has a single class.")
        return False
    if unique_count <= 1 and not require_two_classes:
        logger.info("Skipping tuning: insufficient variation in target.")
        return False
    return True

def _fit_with_optional_tuning(
    pipeline: Pipeline,
    param_distributions: dict[str, Sequence] | None,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    label: str,
    scoring: str | None = None,
    require_two_classes: bool = False,
) -> Pipeline:
    if param_distributions and _should_tune(y, require_two_classes=require_two_classes):
        n_iter = min(HYPERPARAM_TUNING_ITER, _param_space_size(param_distributions))
        if n_iter <= 0:
            pipeline.fit(X, y)
            return pipeline
        try:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=HYPERPARAM_TUNING_CV,
                scoring=scoring,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(X, y)
            logger.info(
                "Best %s params from tuning: %s (score=%.4f)",
                label,
                search.best_params_,
                search.best_score_,
            )
            return search.best_estimator_
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Hyperparameter tuning for %s failed (%s). Using baseline parameters.",
                label,
                exc,
            )
    else:
        if ENABLE_HYPERPARAM_TUNING:
            logger.info("Skipping hyperparameter tuning for %s due to data or configuration.", label)
    pipeline.fit(X, y)
    return pipeline

def _build_pipeline(estimator: BaseEstimator) -> Pipeline:
    return Pipeline([
        ("feature_selector", CorrelatedFeatureDropper(
            correlation_threshold=FEATURE_CORRELATION_THRESHOLD,
            min_variance=FEATURE_MIN_VARIANCE,
        )),
        ("imputer", SimpleImputer(strategy="median")),
        ("est", estimator),
    ])

def _log_feature_selection(selector: CorrelatedFeatureDropper | None, label: str) -> None:
    if selector is None:
        return
    dropped = selector.features_to_drop_
    if dropped:
        preview = ", ".join(dropped[:10])
        more = "" if len(dropped) <= 10 else f", ... (+{len(dropped) - 10} more)"
        logger.info(
            "%s feature selector dropped %d column(s): %s%s",
            label.capitalize(),
            len(dropped),
            preview,
            more,
        )
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
    # Pipelines with feature selection + optional tuning
    clf_pipeline = _build_pipeline(HistGradientBoostingClassifier(**CLF_PARAMS))
    reg_pipeline = _build_pipeline(HistGradientBoostingRegressor(**REG_PARAMS))

    clf = _fit_with_optional_tuning(
        clf_pipeline,
        CLF_PARAM_DISTRIBUTIONS,
        X_train,
        y_start,
        label="classifier",
        scoring="balanced_accuracy",
        require_two_classes=True,
    )
    reg = _fit_with_optional_tuning(
        reg_pipeline,
        REG_PARAM_DISTRIBUTIONS,
        X_train,
        y_train,
        label="regressor",
        scoring="neg_mean_squared_error",
    )

    _log_feature_selection(clf.named_steps.get("feature_selector"), "classifier")
    _log_feature_selection(reg.named_steps.get("feature_selector"), "regressor")

    # Ensure model directory exists before persisting artifacts
    CLF_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    Input contains meta columns: player_id, full_name, team_name, now_cost_millions, team_id, element_type
    plus same feature columns used during training.
    Returns a DataFrame with expected_points and bias-corrected EP.
    """
    meta_cols = ["player_id", "full_name", "team_name", "now_cost_millions", "team_id", "element_type"]
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
