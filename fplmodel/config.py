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

# --- API Endpoints (official FPL)
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_ALL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_ELEMENT_SUMMARY = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"

# --- General
RANDOM_SEED = 42

# Cache: refetch player histories if older than N days
CACHE_TTL_DAYS = 2

# Feature engineering
ROLLING_WINDOWS = [3, 5]
MIN_MATCHES_FOR_FEATURES = 2  # min previous matches required to generate a training row

# Bias-correction (EMA) applied after each finished GW
EMA_ALPHA = 0.35  # weight of the most recent residual

# Feature selection
FEATURE_CORRELATION_THRESHOLD = 0.95
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

# Training window: set None to use all finished GWs
MAX_TRAIN_GW = None
