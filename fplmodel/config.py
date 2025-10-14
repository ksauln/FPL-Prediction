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

# Model hyperparams (kept modest to run quickly on laptops)
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

# Team selection
BUDGET_MILLIONS = 100.0  # total budget for XI
FORMATION = {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3}  # standard 3-4-3
MAX_PER_TEAM = 3

# Training window: set None to use all finished GWs
MAX_TRAIN_GW = None
