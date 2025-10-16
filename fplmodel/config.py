"""
Config for FPL Expected Points (EP) model pipeline.
"""
from pathlib import Path

# --- Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
EXTERNAL_HISTORY_REPO = EXTERNAL_DATA_DIR / "Fantasy-Premier-League"
EXTERNAL_HISTORY_DIR = EXTERNAL_HISTORY_REPO / "data"

# --- API Endpoints (official FPL)
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_ALL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_ELEMENT_SUMMARY = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"

# --- General
RANDOM_SEED = 42

# Cache: refetch player histories if older than N days
CACHE_TTL_DAYS = 2

# Historical data
PLAYER_HISTORY_SEASONS_BACK = 5  # append this many prior seasons to player match history
USE_EXTERNAL_HISTORY = True
EXTERNAL_HISTORY_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]  # extend training with these completed seasons

# GPU acceleration
ENABLE_GPU_TRAINING = True  # attempt to use GPU-accelerated models when available

# Feature engineering
ROLLING_WINDOWS = [4]
MIN_MATCHES_FOR_FEATURES = 3  # min previous matches required to generate a training row

# Bias-correction (EMA) applied after each finished GW
EMA_ALPHA = 0.25  # weight of the most recent residual

# Feature selection
FEATURE_CORRELATION_THRESHOLD = 0.90
FEATURE_MIN_VARIANCE = 1e-6  # drop features with (near) zero variance

# Hyperparameter tuning
ENABLE_HYPERPARAM_TUNING = True
HYPERPARAM_TUNING_MIN_SAMPLES = 300
HYPERPARAM_TUNING_ITER = 12
HYPERPARAM_TUNING_CV = 3

# Model hyperparams (baseline defaults; tuning will explore around these)
REG_PARAMS = dict(  # HistGradientBoostingRegressor
    max_depth=6,
    max_iter=300,
    learning_rate=0.08,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=RANDOM_SEED,
)
CLF_PARAMS = dict(  # HistGradientBoostingClassifier
    max_depth=6,
    max_iter=250,
    learning_rate=0.08,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=RANDOM_SEED,
)

REG_PARAM_DISTRIBUTIONS = {
    "est__max_depth": [4, 6, 8, None],
    "est__learning_rate": [0.04, 0.06, 0.08, 0.12],
    "est__max_iter": [250, 300, 350, 400],
    "est__min_samples_leaf": [10, 20, 30, 40],
    "est__l2_regularization": [0.0, 0.05, 0.1, 0.3],
}
CLF_PARAM_DISTRIBUTIONS = {
    "est__max_depth": [4, 6, 8, None],
    "est__learning_rate": [0.04, 0.06, 0.08, 0.12],
    "est__max_iter": [200, 250, 300, 350],
    "est__min_samples_leaf": [10, 20, 30, 40],
    "est__l2_regularization": [0.0, 0.05, 0.1, 0.3],
}

RF_REG_PARAM_DISTRIBUTIONS = {
    "est__n_estimators": [200, 400, 600],
    "est__max_depth": [None, 10, 20],
    "est__min_samples_split": [2, 4, 6],
    "est__min_samples_leaf": [1, 2, 4],
    "est__max_features": [1.0, 0.7, "sqrt"],
}
RF_CLF_PARAM_DISTRIBUTIONS = {
    "est__n_estimators": [200, 400, 600],
    "est__max_depth": [None, 10, 20],
    "est__min_samples_split": [2, 4, 6],
    "est__min_samples_leaf": [1, 2, 4],
    "est__max_features": ["sqrt", "log2", 0.6],
}

MLP_REG_PARAM_DISTRIBUTIONS = {
    "est__hidden_layer_sizes": [(128, 64), (256, 128), (128, 128, 64)],
    "est__alpha": [1e-4, 5e-4, 1e-3],
    "est__learning_rate_init": [0.001, 0.003, 0.01],
    "est__beta_1": [0.85, 0.9, 0.95],
}
MLP_CLF_PARAM_DISTRIBUTIONS = {
    "est__hidden_layer_sizes": [(128, 64), (256, 128), (128, 128, 64)],
    "est__alpha": [1e-4, 5e-4, 1e-3],
    "est__learning_rate_init": [0.001, 0.003, 0.01],
    "est__beta_1": [0.85, 0.9, 0.95],
}

XGB_REG_PARAM_DISTRIBUTIONS = {
    "est__n_estimators": [250, 400, 550],
    "est__learning_rate": [0.05, 0.08, 0.12],
    "est__max_depth": [4, 6, 8],
    "est__subsample": [0.8, 0.9, 1.0],
    "est__colsample_bytree": [0.7, 0.85, 1.0],
    "est__reg_lambda": [0.5, 1.0, 1.5],
    "est__gamma": [0.0, 0.1, 0.3],
}
XGB_CLF_PARAM_DISTRIBUTIONS = {
    "est__n_estimators": [250, 400, 550],
    "est__learning_rate": [0.05, 0.08, 0.12],
    "est__max_depth": [4, 6, 8],
    "est__subsample": [0.8, 0.9, 1.0],
    "est__colsample_bytree": [0.7, 0.85, 1.0],
    "est__reg_lambda": [0.5, 1.0, 1.5],
    "est__gamma": [0.0, 0.1, 0.3],
}

# Team selection
BUDGET_MILLIONS = 100.0  # total budget for the full 15-player squad
FORMATION = {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3}  # starting XI shape
FORMATION_OPTIONS = [
    FORMATION,
    {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},  # 3-5-2
    {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},  # 4-4-2
    {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},  # 4-5-1
    {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},  # 4-3-3
    {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},  # 5-3-2
]
SQUAD_POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BENCH_SIZE = 4
BENCH_GK_COUNT = 1
MAX_PER_TEAM = 3
BENCH_EP_WEIGHT = 1.0  # weight for bench expected points in optimisation objective

# Training window: set None to use all finished GWs
MAX_TRAIN_GW = None
