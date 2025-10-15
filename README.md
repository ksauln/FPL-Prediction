# FPL Expected Points Pipeline

Automated Fantasy Premier League analytics pipeline that ingests raw data, engineers features, trains ML models, updates bias adjustments, and outputs next gameweek projections alongside an optimized squad selection.

---

## 1. System Overview

The pipeline is structured so each stage feeds the next while persisting intermediate artefacts for later inspection or incremental reruns:

1. **Logging/bootstrap** – capture run metadata and initialise logging.
2. **External data ingestion**
   - Official FPL API (bootstrap, fixtures, per-player histories).
   - Optional historical CSVs from the `vaastav/Fantasy-Premier-League` GitHub repo.
3. **Feature engineering** – derive rolling windows, lags, opponent strength context, set-piece indicators, and bias features.
4. **Model selection and training** – compare multiple estimator families (histogram gradient boosting, random forest, MLP, XGBoost with optional GPU support), fit each candidate, and retain both the selected pair and the full set for ensembling.
5. **Prediction** – score every fitted model, average their expected-point outputs into an ensemble forecast, and apply fixture multipliers.
6. **Evaluation & bias update** – score the most recently completed gameweek, update EMA-based player and positional residual corrections.
7. **Squad optimisation** – construct the best XI plus bench via mixed-integer optimisation subject to FPL constraints.
8. **Outputs** – write predictions, squads, residuals, and logs to disk.

---

## 2. Repository Layout

```
FPL-Prediction/
├── data/
│   ├── raw/          # API caches (bootstrap, fixtures, per-player histories)
│   ├── processed/    # intermediate cleaned data
│   └── external/     # cloned vaastav repo
├── fplmodel/
│   ├── config.py               # configurable constants (paths, model params, EMA alpha, etc.)
│   ├── data_pull.py            # HTTP pulls + caching logic
│   ├── data_cleaning.py        # normalise bootstrap and stack player histories
│   ├── external_history.py     # loader for historic CSV seasons
│   ├── features.py             # feature engineering functions
│   ├── model.py                # candidate builders, tuning, training, prediction helpers
│   ├── evaluation.py           # residual computation and EMA bias updates
│   ├── team_picker.py          # ILP squad optimisation
│   ├── state.py                # persistent bias storage (player & position)
│   ├── utils.py / logging_utils.py
│   └── display.py              # squad visualisation
├── models/                     # persisted models and state.json
├── outputs/                    # csv/json/png artefacts per run
├── logs/                       # detailed run logs
├── requirements.txt
└── main.py                     # orchestrates the full pipeline
```

---

## 3. Installation & Execution

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py              # runs the full pipeline
```

Environment variables can override logging paths or toggle features; consult `config.py` for available knobs.

---

## 4. Detailed Pipeline Walkthrough

### 4.1 Logging and Preparation
- `main.py` initialises a run-specific logger via `fplmodel.logging_utils.configure_run_logger`.
- `get_current_and_last_finished_gw` determines the next/open gameweek and the most recently completed one (using `events` from bootstrap).
- If `MAX_TRAIN_GW` is set, it caps the training horizon.

### 4.2 Data Ingestion
1. **Bootstrap static** (`fetch_bootstrap_static`)  
   Yields player metadata, team information, chip stats, and events.
2. **Fixtures** (`fetch_fixtures_all`)  
   Full schedule including future double/blank GWs.
3. **Player histories** (`bulk_fetch_player_histories`)  
   Pulls per-match history arrays for each player.  
   - Configurable `PLAYER_HISTORY_SEASONS_BACK` determines how many prior seasons to request via the API.
   - Files cached under `data/raw/player_<id>.json`.

4. **External history ingestion** (`external_history.load_external_histories`)  
   - Enabled when `USE_EXTERNAL_HISTORY = True`.  
   - Reads historical CSV snapshots (`gws/gw*.csv`) from the vaastav repo.
   - Normalises schema to match FPL API histories, ensures team IDs map correctly, and concatenates with API data.

### 4.3 Feature Engineering (`features.build_training_and_pred_frames`)
- Input: merged player metadata, team metrics, combined match history.
- Steps:
  - Convert static numeric columns to floats, build set-piece indicators.
  - Generate rolling averages (`ROLLING_WINDOWS`) and lag features for key stats (points, minutes, xG, xA, etc.).
  - Add team and opponent strength context (table position, strength ratings).
  - Merge persistent EMA bias features (`player_bias`, `pos_bias`) from `ModelState`.
  - Enforce `MIN_MATCHES_FOR_FEATURES` history before contributing to training.
  - Produce three DataFrames:
    - `X_train`: engineered features up to `last_finished_gw`.
    - `y_train`: actual total points.
    - `X_pred`: features + identifiers for each player to score for the next GW.

### 4.4 Model Training & Selection (`model.train_models`)
- Derives a binary `start` target (`P(minutes ≥ 60)`) from the engineered features.
- Builds candidates:
  - Histogram Gradient Boosting (default)
  - Random Forest
  - MLP (scaled features)
  - XGBoost (CPU or GPU depending on availability; `ENABLE_GPU_TRAINING`)
- For each candidate:
  - Runs cross-validation metrics for both classifier and regressor pipelines, with optional hyperparameter tuning (`RandomizedSearchCV`).
  - Records metrics and writes a summary to `models/model_selection_summary.csv`.
- Chooses the best-performing classifier/regressor pair based on balanced accuracy / MAE, while still fitting and keeping every candidate.
- Logs the final selection explicitly for traceability.
- Fits each candidate pipeline on the full training data, persists the selected pair under `models/`, and returns the full fitted bundle for downstream ensembling.

### 4.5 Prediction & Ensembling (`model.predict_expected_points`)
- Applies every fitted classifier/regressor pair to `X_pred` features.
- Computes raw expected points per model (`p_start * points_hat`) and stores both the raw and bias-corrected values with candidate-specific suffixes.
- Averages the raw outputs across models, re-applies EMA bias corrections (`player_bias`, `position_bias`), and clips at zero to produce the ensemble column used throughout the rest of the pipeline.
- `expand_for_double_gw` multiplies expected points by fixture counts for upcoming doubles/blanks (the per-model corrected columns are scaled too, enabling diagnostics on double/blank adjustments).

### 4.6 Evaluation & Bias Update (`evaluation.evaluate_last_finished_gw_and_update_state`)
- Reconstructs “gw-1” feature rows for players who played in `last_finished_gw`.
- Predicts that finished GW, compares to actual total points.
- Computes residuals and updates `state.json` via exponential moving average (`EMA_ALPHA`).
  - Separate EMA per player and per position keeps future predictions calibrated.

### 4.7 Squad Optimisation (`team_picker.pick_best_xi`)
- Formulates an integer linear program using PuLP:
  - Decision variables for each player (start, bench, captain, vice).
  - Constraints: budget, formation options (`FORMATION_OPTIONS`), positional minimums, max three per club, total squad size (15) and bench order.
- Produces:
  - Starting XI with captaincy applied.
  - Bench ordering.
  - Expected-point totals with and without captain.
- Optionally annotates fixtures on the squad objects.

### 4.8 Artefact Generation
- Saves:
  - Predictions: `outputs/predictions_gw<N>.csv` (includes ensemble columns plus per-model raw/corrected fields).
  - Starting XI / Bench CSVs.
  - Full squad JSON.
  - Squad image (if `display.create_best_xi_graphic` succeeds).
  - Residuals CSV for `last_finished_gw`.
- Maintains run-specific logs under `logs/`.

---

## 5. Configurability Highlights

- `config.py` contains most knobs:
  - **Paths** (`RAW_DIR`, `OUTPUTS_DIR`, etc.)
  - **Historical depth** (`PLAYER_HISTORY_SEASONS_BACK`, `EXTERNAL_HISTORY_SEASONS`)
  - **Model tuning** (`ENABLE_HYPERPARAM_TUNING`, distributions)
  - **Bias behaviour** (`EMA_ALPHA`, ability to clamp in `state.py`)
  - **Optimisation constraints** (budget, formations)
  - **GPU usage** (`ENABLE_GPU_TRAINING`)
- You can override constants via environment variables or secondary config layers if desired.
- Removing `models/state.json` resets all stored biases; otherwise the EMA continues from previous runs.

---

## 6. Development Tips

- Use `main.py` as the orchestrator; modules are decoupled enough to run individually for testing (e.g., call `features.build_training_and_pred_frames` in notebooks).
- Add unit tests for feature calculations or optimisation constraints as the codebase evolves.
- Watch the logs:
  - Selected models and hyperparameters.
  - Bias updates (number of residuals applied).
  - Squad selection metrics and constraints.
- If projections look unrealistic, inspect:
  - `models/state.json` for runaway biases.
  - `outputs/residuals_gw*.csv` for large residuals that feed the EMA.
  - Feature importances via the log entries from `log_model_feature_weights`.

---

## 7. Future Enhancements

- Fixture-level modelling (predict individual matches and aggregate).
- Injury/news sentiment ingestion to adjust playing time priors.
- Simulation-based points distributions rather than single EP values.
- Automated alerting when EMA biases exceed thresholds (and auto-resets or caps).
- Enhanced visualisation dashboards (Streamlit or similar) using the CSV outputs.

---

## 8. License & Acknowledgements

- Uses the official Fantasy Premier League API and historical data snapshots courtesy of [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).
- Respect FPL’s terms of service when sharing or automating.
- See `LICENSE` (if provided) for redistribution details.

Happy predicting!
