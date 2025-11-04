from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from typing import Any, Callable, Tuple, Sequence

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    BaseCrossValidator,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    SEASON_WEIGHT_DECAY,
    SEASON_WEIGHT_MIN,
    REG_PARAM_DISTRIBUTIONS,
    CLF_PARAM_DISTRIBUTIONS,
    RF_REG_PARAM_DISTRIBUTIONS,
    RF_CLF_PARAM_DISTRIBUTIONS,
    MLP_REG_PARAM_DISTRIBUTIONS,
    MLP_CLF_PARAM_DISTRIBUTIONS,
    XGB_REG_PARAM_DISTRIBUTIONS,
    XGB_CLF_PARAM_DISTRIBUTIONS,
    ENABLE_GPU_TRAINING,
    MIN_MATCHES_FOR_FEATURES,
)
from .state import ModelState

try:  # Optional dependency; only required if XGBoost models are used
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore
    XGBRegressor = None  # type: ignore

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


_SEASON_SORT_COL = "__season_sort__"
_ROUND_SORT_COL = "__round_sort__"
_KICKOFF_SORT_COL = "__kickoff_sort__"


def _season_start_year(label: Any) -> int | None:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, str):
        match = re.search(r"(\d{4})", label)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def _season_sort_key(label: Any) -> tuple[int, str]:
    year = _season_start_year(label)
    sentinel = 10**6
    if year is None:
        return (sentinel, "" if label is None else str(label))
    return (year, "" if label is None else str(label))


def _season_rank(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if series.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    valid_str = valid.astype(str)
    unique_sorted = sorted(valid_str.unique(), key=_season_sort_key)
    mapping = {season: idx for idx, season in enumerate(unique_sorted)}

    def mapper(val: Any) -> float:
        if pd.isna(val):
            return np.nan
        return float(mapping.get(str(val), np.nan))

    return series.astype(object).map(mapper)


def _compute_season_sample_weights(season_series: pd.Series) -> pd.Series:
    if season_series.empty:
        return pd.Series(np.ones(len(season_series)), index=season_series.index, dtype=float)
    valid = season_series.dropna()
    if valid.empty:
        return pd.Series(np.ones(len(season_series)), index=season_series.index, dtype=float)
    valid_str = valid.astype(str)
    unique_sorted = sorted(valid_str.unique(), key=_season_sort_key)
    n = len(unique_sorted)
    if n == 0:
        return pd.Series(np.ones(len(season_series)), index=season_series.index, dtype=float)

    weights: dict[str, float] = {}
    for rank, season in enumerate(unique_sorted):
        exponent = (n - 1) - rank
        weight = max(SEASON_WEIGHT_MIN, SEASON_WEIGHT_DECAY ** exponent)
        weights[season] = float(weight)

    def weight_lookup(val: Any) -> float:
        if pd.isna(val):
            return 1.0
        return float(weights.get(str(val), 1.0))

    return season_series.astype(object).map(weight_lookup).astype(float)


def _build_time_series_cv(
    metadata: pd.DataFrame,
    max_splits: int,
    n_samples: int,
) -> BaseCrossValidator | None:
    if metadata is None or metadata.empty:
        return None
    if n_samples < 3:
        return None
    if _SEASON_SORT_COL not in metadata.columns or _ROUND_SORT_COL not in metadata.columns:
        return None

    time_keys = (
        metadata[[_SEASON_SORT_COL, _ROUND_SORT_COL]]
        .dropna()
        .drop_duplicates()
        .sort_values(by=[_SEASON_SORT_COL, _ROUND_SORT_COL])
    )
    if time_keys.empty:
        return None

    possible_splits = min(len(time_keys) - 1, n_samples - 1)
    if possible_splits < 2:
        return None

    n_splits = min(max_splits, possible_splits)
    if n_splits < 2:
        return None

    return TimeSeriesSplit(n_splits=n_splits)


def _prepare_training_temporal_order(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    metadata: pd.DataFrame | None,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    np.ndarray | None,
    BaseCrossValidator | None,
    pd.Series | None,
]:
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)
    else:
        y = y.reindex(X.index)

    if metadata is None:
        metadata = pd.DataFrame(index=X.index)
    else:
        metadata = metadata.reindex(X.index).copy()

    season_series = metadata.get("season_name")
    season_rank = _season_rank(season_series) if season_series is not None else pd.Series(index=X.index, dtype=float)
    if season_rank.empty:
        season_rank = pd.Series(np.arange(len(metadata), dtype=float), index=metadata.index)
    if season_rank.notna().any():
        fill_value = float(season_rank.max(skipna=True)) + 1.0
        season_sort = season_rank.fillna(fill_value)
    else:
        season_sort = pd.Series(np.arange(len(metadata), dtype=float), index=metadata.index)
    metadata[_SEASON_SORT_COL] = season_sort

    if "round" in metadata.columns:
        round_numeric = pd.to_numeric(metadata["round"], errors="coerce")
        if round_numeric.notna().any():
            fallback = float(round_numeric.max(skipna=True)) + 1.0
            round_sort = round_numeric.fillna(fallback)
        else:
            round_sort = pd.Series(np.arange(len(metadata), dtype=float), index=metadata.index)
    else:
        round_sort = pd.Series(np.arange(len(metadata), dtype=float), index=metadata.index)
    metadata[_ROUND_SORT_COL] = round_sort

    if "kickoff_time" in metadata.columns:
        kickoff_sort = pd.to_datetime(metadata["kickoff_time"], errors="coerce")
        kickoff_sort = kickoff_sort.fillna(pd.Timestamp("1900-01-01"))
    else:
        kickoff_sort = pd.Series([pd.Timestamp("1900-01-01")] * len(metadata), index=metadata.index)
    metadata[_KICKOFF_SORT_COL] = kickoff_sort

    metadata_sorted = metadata.sort_values(
        by=[_SEASON_SORT_COL, _ROUND_SORT_COL, _KICKOFF_SORT_COL],
        kind="mergesort",
    )
    order_idx = metadata_sorted.index
    X_sorted = X.loc[order_idx].reset_index(drop=True)
    y_sorted = y.loc[order_idx].reset_index(drop=True)
    metadata_sorted = metadata_sorted.reset_index(drop=True)

    sample_weight_series: pd.Series | None = None
    sample_weight: np.ndarray | None = None
    if "season_name" in metadata_sorted.columns and metadata_sorted["season_name"].notna().any():
        sample_weight_series = _compute_season_sample_weights(metadata_sorted["season_name"])
        sample_weight = sample_weight_series.to_numpy()

    cv_strategy = _build_time_series_cv(metadata_sorted, HYPERPARAM_TUNING_CV, len(X_sorted))

    metadata_clean = metadata_sorted.drop(
        columns=[_SEASON_SORT_COL, _ROUND_SORT_COL, _KICKOFF_SORT_COL],
        errors="ignore",
    )

    return X_sorted, y_sorted, metadata_clean, sample_weight, cv_strategy, sample_weight_series


def _cv_split_count(
    cv: BaseCrossValidator | int,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> int:
    if isinstance(cv, int):
        return int(cv)
    try:
        return int(cv.get_n_splits(X, y))
    except TypeError:
        return int(cv.get_n_splits())
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
    override_params: dict[str, Any] | None = None,
    cv: BaseCrossValidator | int | None = None,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    fit_params: dict[str, Any] = {}
    if sample_weight is not None:
        fit_params["est__sample_weight"] = sample_weight
    if override_params:
        pipeline.set_params(**override_params)
        pipeline.fit(X, y, **fit_params)
        return pipeline
    if param_distributions and _should_tune(y, require_two_classes=require_two_classes):
        n_iter = min(HYPERPARAM_TUNING_ITER, _param_space_size(param_distributions))
        if n_iter <= 0:
            pipeline.fit(X, y, **fit_params)
            return pipeline
        try:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv if cv is not None else min(HYPERPARAM_TUNING_CV, len(y)),
                scoring=scoring,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(X, y, **fit_params)
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
    pipeline.fit(X, y, **fit_params)
    return pipeline


def _tune_and_score(
    pipeline_builder: Callable[[], Pipeline],
    param_distributions: dict[str, Sequence] | None,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    label: str,
    scoring: str,
    require_two_classes: bool = False,
    cv: BaseCrossValidator | int | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float, dict[str, Any] | None]:
    metric_mean = float("nan")
    metric_std = float("nan")
    best_params: dict[str, Any] | None = None
    fit_params: dict[str, Any] = {}
    if sample_weight is not None:
        fit_params["est__sample_weight"] = sample_weight

    if param_distributions and _should_tune(y, require_two_classes=require_two_classes):
        cv_splits = cv if cv is not None else min(HYPERPARAM_TUNING_CV, len(y))
        if cv_splits is None:
            cv_splits = min(HYPERPARAM_TUNING_CV, len(y))
        split_count = _cv_split_count(cv_splits, X, y) if cv_splits is not None else 0
        n_iter = min(HYPERPARAM_TUNING_ITER, _param_space_size(param_distributions))
        if n_iter > 0 and split_count >= 2:
            try:
                search = RandomizedSearchCV(
                    pipeline_builder(),
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    cv=cv_splits,
                    scoring=scoring,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    refit=True,
                    error_score="raise",
                    return_train_score=False,
                )
                search.fit(X, y, **fit_params)
                idx = search.best_index_
                score_mean = float(search.cv_results_["mean_test_score"][idx])
                score_std = float(search.cv_results_["std_test_score"][idx])
                negative_metric = scoring.startswith("neg_")
                if negative_metric:
                    metric_mean = -score_mean
                    metric_std = score_std
                else:
                    metric_mean = score_mean
                    metric_std = score_std
                best_params = search.best_params_
                logger.info(
                    "Best %s params from tuning: %s (score=%.4f)",
                    label,
                    best_params,
                    search.best_score_,
                )
                return metric_mean, metric_std, best_params
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Hyperparameter tuning for %s failed (%s). Falling back to baseline CV.",
                    label,
                    exc,
                )
        elif n_iter > 0:
            logger.info(
                "Skipping hyperparameter tuning for %s: insufficient CV splits (got %d).",
                label,
                split_count,
            )
    elif ENABLE_HYPERPARAM_TUNING:
        logger.info("Skipping hyperparameter tuning for %s due to data or configuration.", label)

    try:
        cv_splits = cv if cv is not None else min(HYPERPARAM_TUNING_CV, len(y))
        if cv_splits is None:
            return metric_mean, metric_std, best_params
        split_count = _cv_split_count(cv_splits, X, y)
        if split_count < 2:
            return metric_mean, metric_std, best_params
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            scores = cross_val_score(
                pipeline_builder(),
                X,
                y,
                scoring=scoring,
                cv=cv_splits,
                n_jobs=1,
                error_score=np.nan,
                fit_params=fit_params,
            )
        if scoring.startswith("neg_"):
            scores = -scores
        if not np.isnan(scores).all():
            metric_mean = float(np.nanmean(scores))
            metric_std = float(np.nanstd(scores))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Cross-validation for %s failed during fallback scoring: %s", label, exc)

    return metric_mean, metric_std, best_params

def _build_pipeline(estimator: BaseEstimator, use_scaler: bool = False) -> Pipeline:
    steps: list[tuple[str, TransformerMixin | BaseEstimator]] = [
        (
            "feature_selector",
            CorrelatedFeatureDropper(
                correlation_threshold=FEATURE_CORRELATION_THRESHOLD,
                min_variance=FEATURE_MIN_VARIANCE,
            ),
        ),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("est", estimator))
    return Pipeline(steps)

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


@dataclass(frozen=True)
class ModelCandidate:
    """Factory container for a classifier/regressor pair."""

    name: str
    display_name: str
    build_classifier: Callable[[], Pipeline]
    build_regressor: Callable[[], Pipeline]
    clf_param_distributions: dict[str, Sequence] | None = None
    reg_param_distributions: dict[str, Sequence] | None = None


@dataclass
class FittedModelBundle:
    """Concrete fitted pipelines for a model candidate."""

    name: str
    display_name: str
    classifier: Pipeline
    regressor: Pipeline


def _xgboost_available() -> bool:
    return XGBClassifier is not None and XGBRegressor is not None


@lru_cache(maxsize=1)
def _xgboost_gpu_enabled() -> bool:
    if not ENABLE_GPU_TRAINING:
        return False
    if not _xgboost_available():
        return False
    try:
        import os
        import xgboost  # type: ignore

        if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
            return False
        has_cuda = getattr(xgboost.core, "_has_cuda_support", None)
        if callable(has_cuda):
            return bool(has_cuda())
    except Exception:
        return False
    return True


def _build_model_candidates() -> list[ModelCandidate]:
    candidates: list[ModelCandidate] = []

    def default_classifier() -> Pipeline:
        return _build_pipeline(HistGradientBoostingClassifier(**CLF_PARAMS))

    def default_regressor() -> Pipeline:
        return _build_pipeline(HistGradientBoostingRegressor(**REG_PARAMS))

    candidates.append(
        ModelCandidate(
            name="hist_gbdt",
            display_name="Histogram Gradient Boosting",
            build_classifier=default_classifier,
            build_regressor=default_regressor,
            clf_param_distributions=CLF_PARAM_DISTRIBUTIONS,
            reg_param_distributions=REG_PARAM_DISTRIBUTIONS,
        )
    )

    def rf_classifier() -> Pipeline:
        return _build_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=RANDOM_SEED,
                class_weight="balanced_subsample",
            )
        )

    def rf_regressor() -> Pipeline:
        return _build_pipeline(
            RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )
        )

    candidates.append(
        ModelCandidate(
            name="random_forest",
            display_name="Random Forest",
            build_classifier=rf_classifier,
            build_regressor=rf_regressor,
            clf_param_distributions=RF_CLF_PARAM_DISTRIBUTIONS,
            reg_param_distributions=RF_REG_PARAM_DISTRIBUTIONS,
        )
    )

    def mlp_classifier() -> Pipeline:
        return _build_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=5e-4,
                max_iter=400,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=RANDOM_SEED,
            ),
            use_scaler=True,
        )

    def mlp_regressor() -> Pipeline:
        return _build_pipeline(
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=5e-4,
                max_iter=400,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=RANDOM_SEED,
            ),
            use_scaler=True,
        )

    candidates.append(
        ModelCandidate(
            name="neural_network",
            display_name="Neural Network (MLP)",
            build_classifier=mlp_classifier,
            build_regressor=mlp_regressor,
            clf_param_distributions=MLP_CLF_PARAM_DISTRIBUTIONS,
            reg_param_distributions=MLP_REG_PARAM_DISTRIBUTIONS,
        )
    )

    if _xgboost_available():
        gpu_enabled = _xgboost_gpu_enabled()
        if gpu_enabled:
            logger.info("GPU detected; configuring XGBoost candidates for gpu_hist training.")

        def xgb_classifier() -> Pipeline:
            base_params = dict(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=350,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                gamma=0.0,
                random_state=RANDOM_SEED,
                n_jobs=1,
                verbosity=0,
                tree_method="hist",
            )
            if gpu_enabled:
                base_params.update(
                    {
                        "tree_method": "gpu_hist",
                        "predictor": "gpu_predictor",
                    }
                )
                params_with_device = dict(base_params)
                params_with_device["device"] = "cuda"
                try:
                    estimator = XGBClassifier(**params_with_device)
                except TypeError:
                    estimator = XGBClassifier(**base_params)
            else:
                estimator = XGBClassifier(**base_params)
            return _build_pipeline(
                estimator
            )

        def xgb_regressor() -> Pipeline:
            base_params = dict(
                objective="reg:squarederror",
                n_estimators=400,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                gamma=0.0,
                random_state=RANDOM_SEED,
                n_jobs=1,
                verbosity=0,
                tree_method="hist",
            )
            if gpu_enabled:
                base_params.update(
                    {
                        "tree_method": "gpu_hist",
                        "predictor": "gpu_predictor",
                    }
                )
                params_with_device = dict(base_params)
                params_with_device["device"] = "cuda"
                try:
                    estimator = XGBRegressor(**params_with_device)
                except TypeError:
                    estimator = XGBRegressor(**base_params)
            else:
                estimator = XGBRegressor(**base_params)
            return _build_pipeline(
                estimator
            )

        candidates.append(
            ModelCandidate(
                name="xgboost",
                display_name="XGBoost",
                build_classifier=xgb_classifier,
                build_regressor=xgb_regressor,
                clf_param_distributions=XGB_CLF_PARAM_DISTRIBUTIONS,
                reg_param_distributions=XGB_REG_PARAM_DISTRIBUTIONS,
            )
        )

    return candidates


def _format_metric(mean: float | None, std: float | None, lower_is_better: bool = False) -> str:
    if mean is None or np.isnan(mean):
        return "n/a"
    if std is None or np.isnan(std):
        return f"{mean:.4f}"
    direction = "↓" if lower_is_better else "↑"
    return f"{mean:.4f} ± {std:.4f} {direction}"


def _evaluate_model_candidates(
    candidates: Sequence[ModelCandidate],
    X_train: pd.DataFrame,
    y_start: np.ndarray,
    y_points: pd.Series,
    sample_weight: np.ndarray | None,
    cv_strategy: BaseCrossValidator | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def _cv_splits(n_samples: int) -> BaseCrossValidator | int | None:
        if cv_strategy is not None:
            return cv_strategy
        if n_samples < 2:
            return None
        return min(HYPERPARAM_TUNING_CV, n_samples)

    clf_cv = _cv_splits(len(y_start))
    reg_cv = _cv_splits(len(y_points))

    for candidate in candidates:
        result: dict[str, Any] = {
            "candidate": candidate,
            "name": candidate.display_name,
            "clf_balanced_accuracy": np.nan,
            "clf_balanced_accuracy_std": np.nan,
            "reg_mae": np.nan,
            "reg_mae_std": np.nan,
            "clf_best_params": None,
            "reg_best_params": None,
        }

        if clf_cv is None or np.unique(y_start).size < 2:
            logger.info(
                "Skipping classifier CV for %s: insufficient target variation.",
                candidate.display_name,
            )
        else:
            mean, std, best_params = _tune_and_score(
                candidate.build_classifier,
                candidate.clf_param_distributions,
                X_train,
                y_start,
                label=f"classifier[{candidate.name}]",
                scoring="balanced_accuracy",
                require_two_classes=True,
                cv=clf_cv,
                sample_weight=sample_weight,
            )
            result["clf_balanced_accuracy"] = mean
            result["clf_balanced_accuracy_std"] = std
            result["clf_best_params"] = best_params

        if reg_cv is None:
            logger.info(
                "Skipping regressor CV for %s: insufficient samples.",
                candidate.display_name,
            )
        else:
            mean, std, best_params = _tune_and_score(
                candidate.build_regressor,
                candidate.reg_param_distributions,
                X_train,
                y_points,
                label=f"regressor[{candidate.name}]",
                scoring="neg_mean_absolute_error",
                cv=reg_cv,
                sample_weight=sample_weight,
            )
            result["reg_mae"] = mean
            result["reg_mae_std"] = std
            result["reg_best_params"] = best_params

        results.append(result)

        logger.info(
            "Candidate %s | classifier=%s | regressor=%s",
            candidate.display_name,
            _format_metric(
                result["clf_balanced_accuracy"],
                result["clf_balanced_accuracy_std"],
                lower_is_better=False,
            ),
            _format_metric(
                result["reg_mae"],
                result["reg_mae_std"],
                lower_is_better=True,
            ),
        )

    summary_records = []
    for res in results:
        summary_records.append(
            {
                "model": res["name"],
                "balanced_accuracy_mean": res["clf_balanced_accuracy"],
                "balanced_accuracy_std": res["clf_balanced_accuracy_std"],
                "mae_mean": res["reg_mae"],
                "mae_std": res["reg_mae_std"],
            }
        )

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_path = MODELS_DIR / "model_selection_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved model selection summary to %s", summary_path)

    return results


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metadata: pd.DataFrame | None = None,
) -> Tuple[Pipeline, Pipeline, list[FittedModelBundle]]:
    """
    Train classifier (starts >=60) and regressor (points) by comparing multiple model families.

    metadata provides optional columns (e.g., season_name, round, kickoff_time) used to
    construct chronological CV folds and per-season sample weights.

    Returns the selected classifier/regressor pipelines plus every fitted candidate pair
    for downstream ensembling.
    """

    def derive_start_target(df: pd.DataFrame) -> np.ndarray:
        if "minutes_ma3" in df.columns:
            arr = (df["minutes_ma3"].values >= 60).astype(int)
            if arr.sum() > 10:
                return arr
        if "minutes_lag1" in df.columns:
            arr = (df["minutes_lag1"].values >= 60).astype(int)
            if arr.sum() > 10:
                return arr
        return (df.get("total_points_ma3", pd.Series(0, index=df.index)).values >= 2).astype(int)

    X_train, y_train, metadata_sorted, sample_weight, cv_strategy, season_weights = _prepare_training_temporal_order(
        X_train, y_train, metadata
    )

    if season_weights is not None and metadata_sorted is not None and "season_name" in metadata_sorted.columns:
        weight_summary = (
            pd.DataFrame({"season_name": metadata_sorted["season_name"], "weight": season_weights})
            .dropna(subset=["season_name"])
        )
        if not weight_summary.empty:
            season_items = sorted(
                weight_summary.drop_duplicates("season_name").itertuples(index=False),
                key=lambda row: _season_sort_key(row.season_name),
            )
            summary_str = ", ".join(f"{row.season_name}: {row.weight:.2f}" for row in season_items)
            logger.info("Season sample weights applied: %s", summary_str)

    if cv_strategy is not None:
        try:
            n_splits = cv_strategy.get_n_splits()
        except TypeError:
            n_splits = cv_strategy.get_n_splits(X_train, y_train)
        logger.info("Using TimeSeriesSplit with %d folds for hyperparameter tuning.", n_splits)
    else:
        logger.info("Using default CV folds for hyperparameter tuning (time splits unavailable).")

    y_start = derive_start_target(X_train)

    candidates = _build_model_candidates()
    evaluation_results = _evaluate_model_candidates(
        candidates, X_train, y_start, y_train, sample_weight, cv_strategy
    )

    # Default selections fall back to the first candidate (histogram gradient boosting)
    best_clf_candidate = candidates[0]
    best_reg_candidate = candidates[0]
    best_clf_metrics: dict[str, Any] | None = None
    best_reg_metrics: dict[str, Any] | None = None

    for res in evaluation_results:
        if not np.isnan(res.get("clf_balanced_accuracy", np.nan)):
            if (
                best_clf_metrics is None
                or res["clf_balanced_accuracy"] > best_clf_metrics["clf_balanced_accuracy"]
            ):
                best_clf_metrics = res
                best_clf_candidate = res["candidate"]
        if not np.isnan(res.get("reg_mae", np.nan)):
            if best_reg_metrics is None or res["reg_mae"] < best_reg_metrics["reg_mae"]:
                best_reg_metrics = res
                best_reg_candidate = res["candidate"]

    if best_clf_metrics:
        logger.info(
            "Selected classifier model: %s (balanced_accuracy=%s)",
            best_clf_candidate.display_name,
            _format_metric(
                best_clf_metrics["clf_balanced_accuracy"],
                best_clf_metrics["clf_balanced_accuracy_std"],
                lower_is_better=False,
            ),
        )
    else:
        logger.info(
            "Falling back to default classifier model: %s",
            best_clf_candidate.display_name,
        )

    if best_reg_metrics:
        logger.info(
            "Selected regressor model: %s (MAE=%s)",
            best_reg_candidate.display_name,
            _format_metric(
                best_reg_metrics["reg_mae"],
                best_reg_metrics["reg_mae_std"],
                lower_is_better=True,
            ),
        )
    else:
        logger.info(
            "Falling back to default regressor model: %s",
            best_reg_candidate.display_name,
        )

    fitted_candidates: list[FittedModelBundle] = []
    fitted_map: dict[str, FittedModelBundle] = {}

    for res in evaluation_results:
        candidate = res["candidate"]
        clf_override_params = res.get("clf_best_params") if res else None
        reg_override_params = res.get("reg_best_params") if res else None

        try:
            candidate_clf = _fit_with_optional_tuning(
                candidate.build_classifier(),
                candidate.clf_param_distributions,
                X_train,
                y_start,
                label=f"classifier[{candidate.name}]",
                scoring="balanced_accuracy",
                require_two_classes=True,
                override_params=clf_override_params,
                cv=cv_strategy,
                sample_weight=sample_weight,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Training classifier %s failed (%s); skipping candidate for ensemble.",
                candidate.display_name,
                exc,
            )
            continue

        try:
            candidate_reg = _fit_with_optional_tuning(
                candidate.build_regressor(),
                candidate.reg_param_distributions,
                X_train,
                y_train,
                label=f"regressor[{candidate.name}]",
                scoring="neg_mean_absolute_error",
                override_params=reg_override_params,
                cv=cv_strategy,
                sample_weight=sample_weight,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Training regressor %s failed (%s); skipping candidate for ensemble.",
                candidate.display_name,
                exc,
            )
            continue

        bundle = FittedModelBundle(
            name=candidate.name,
            display_name=candidate.display_name,
            classifier=candidate_clf,
            regressor=candidate_reg,
        )
        fitted_candidates.append(bundle)
        fitted_map[candidate.name] = bundle

    if not fitted_candidates:
        fallback = candidates[0]
        logger.warning(
            "All candidate trainings failed; fitting fallback model %s.",
            fallback.display_name,
        )
        fallback_clf = _fit_with_optional_tuning(
            fallback.build_classifier(),
            fallback.clf_param_distributions,
            X_train,
            y_start,
            label=f"classifier[{fallback.name}]",
            scoring="balanced_accuracy",
            require_two_classes=True,
        )
        fallback_reg = _fit_with_optional_tuning(
            fallback.build_regressor(),
            fallback.reg_param_distributions,
            X_train,
            y_train,
            label=f"regressor[{fallback.name}]",
            scoring="neg_mean_absolute_error",
        )
        fallback_bundle = FittedModelBundle(
            name=fallback.name,
            display_name=fallback.display_name,
            classifier=fallback_clf,
            regressor=fallback_reg,
        )
        fitted_candidates.append(fallback_bundle)
        fitted_map[fallback.name] = fallback_bundle

    fallback_bundle = fitted_candidates[0]
    if best_clf_candidate.name not in fitted_map:
        logger.warning(
            "Selected classifier %s unavailable after fitting; reverting to %s.",
            best_clf_candidate.display_name,
            fallback_bundle.display_name,
        )
        best_clf_candidate = candidates[0]
    if best_reg_candidate.name not in fitted_map:
        logger.warning(
            "Selected regressor %s unavailable after fitting; reverting to %s.",
            best_reg_candidate.display_name,
            fallback_bundle.display_name,
        )
        best_reg_candidate = candidates[0]

    clf_bundle = fitted_map.get(best_clf_candidate.name, fallback_bundle)
    reg_bundle = fitted_map.get(best_reg_candidate.name, fallback_bundle)
    clf = clf_bundle.classifier
    reg = reg_bundle.regressor

    _log_feature_selection(
        clf.named_steps.get("feature_selector"),
        f"classifier ({best_clf_candidate.display_name})",
    )
    _log_feature_selection(
        reg.named_steps.get("feature_selector"),
        f"regressor ({best_reg_candidate.display_name})",
    )

    logger.info(
        "Final model selection: classifier=%s | regressor=%s",
        best_clf_candidate.display_name,
        best_reg_candidate.display_name,
    )

    # Ensure model directory exists before persisting artifacts
    CLF_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, CLF_PATH)
    joblib.dump(reg, REG_PATH)
    return clf, reg, fitted_candidates

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
    base_meta_cols = ["player_id", "full_name", "team_name", "now_cost_millions", "team_id", "element_type"]
    optional_meta_cols = [col for col in ("season_minutes",) if col in X_meta_and_feats.columns]
    meta_cols = base_meta_cols + optional_meta_cols
    meta = X_meta_and_feats[meta_cols].copy()
    feats = X_meta_and_feats.drop(columns=meta_cols)

    p_start = clf.predict_proba(feats)[:, 1]
    pts_hat = reg.predict(feats)
    ep = p_start * pts_hat

    if "season_minutes" in meta.columns:
        season_minutes = pd.to_numeric(meta["season_minutes"], errors="coerce").fillna(0.0)
        minutes_threshold = float(MIN_MATCHES_FOR_FEATURES * 90.0)
        if minutes_threshold > 0:
            reliability = np.clip(season_minutes / minutes_threshold, 0.0, 1.0)
        else:  # pragma: no cover - defensive
            reliability = pd.Series(1.0, index=meta.index)
    else:
        reliability = pd.Series(1.0, index=meta.index)
    reliability_np = reliability.to_numpy(dtype=float)

    # Apply bias corrections
    player_bias = np.array([state.get_player_bias(pid) for pid in meta["player_id"].values])
    pos_bias = np.array([state.get_position_bias(pos) for pos in meta["element_type"].values])
    ep_raw = ep * reliability_np
    ep_corrected = (ep + player_bias + pos_bias) * reliability_np

    out = meta.copy()
    extra_meta_cols = [
        "availability_this_round",
        "availability_next_round",
        "status_availability",
        "status_injury_flag",
        "injury_risk_flag",
    ]
    for col in extra_meta_cols:
        if col in X_meta_and_feats.columns:
            out[col] = X_meta_and_feats[col].values
    out["p_start"] = p_start
    out["points_hat"] = pts_hat
    out["reliability_weight"] = reliability_np
    out["expected_points_raw"] = ep_raw
    out["expected_points"] = ep_corrected.clip(min=0.0)  # no negatives
    return out
