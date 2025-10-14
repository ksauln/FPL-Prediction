# FPL Expected Points (EP) Pipeline — Modular, Local-Friendly

This repo is a **from-scratch**, modular Fantasy Premier League prediction system that:
- pulls data from the **official FPL API**,
- cleans and engineers features,
- trains **tabular ML models** (gradient-boosted trees) to predict player **expected points**,
- **evaluates the last finished gameweek** and **updates bias-corrections** via EMA of residuals,
- predicts the next GW and **picks the best XI** via ILP (budget + formation + max 3 per club),
- outputs CSV/JSON artifacts.

> Designed to run on a laptop (no GPU required), easy to debug/edit per module.

---

## Quickstart

```bash
# In a fresh Python 3.10–3.12 environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Run
python main.py
```

Artifacts (written to `outputs/`):
- `predictions_gw<N>.csv` – expected points for each player for the **next** GW.
- `best_xi_gw<N>.json` – selected **starting XI** and captain with total expected points and cost.
- `residuals_gw<M>.csv` – residuals (actual - predicted) for the **last finished** GW, used to update EMA biases.

---

## Project Layout

```
fpl_ep_model/
  fplmodel/
    config.py             # all knobs: paths, URLs, rolling windows, EMA alpha, model params, budget, formation
    data_pull.py          # official FPL API pulls + basic on-disk caching
    data_cleaning.py      # normalize bootstrap + tidy player histories
    features.py           # feature engineering (rolling means, lags, team/opponent strength), double-GW scaling
    model.py              # train/load/predict with HGB (sklearn) for classifier + regressor
    evaluation.py         # evaluate last finished GW, compute residuals, update EMA biases
    team_picker.py        # ILP (PuLP) to pick best XI and captain under constraints
    state.py              # persistent EMA biases (player- and position-level)
    utils.py              # helpers
  data/                   # raw + processed caches
  models/                 # trained models + state.json
  outputs/                # predictions, best XI json, residuals
  main.py                 # orchestrates the whole workflow
  requirements.txt
  README.md
```

---

## How it works

### 1) Data
- **bootstrap-static** → players (`elements`), teams, events.
- **fixtures** → all fixtures with home/away + difficulty.
- **element-summary/{id}** → per-player match history (current season).

Fetched JSONs are cached in `data/raw/` and refreshed if older than `CACHE_TTL_DAYS` in `config.py`.

### 2) Features
- Rolling means for **total_points, minutes, goals, assists, CS, influence, creativity, threat, ict_index** over windows in `ROLLING_WINDOWS`.
- **Lag-1** for the same.
- Home/away, team strengths, opponent overall strength.
- EMA **bias features** (`player_bias`, `pos_bias`) read from `state.json`.

Training rows require at least `MIN_MATCHES_FOR_FEATURES` previous matches.

### 3) Models
- **Classifier** (`HistGradientBoostingClassifier`) estimates `P(start≥60)` from features.
- **Regressor** (`HistGradientBoostingRegressor`) predicts raw points.
- Expected points (EP) = `P(start≥60) × predicted_points`.
- Apply bias **corrections**: `+ player_bias + pos_bias` (clipped at 0).

### 4) Error-driven update
For the **last finished GW**, we:
- build “gw-1” features,
- predict that GW,
- compute residuals (= actual − predicted),
- update EMA **player** and **position** biases,
- persist to `models/state.json`.

### 5) Double/Blank GWs
Approximate scaling: if a team has multiple fixtures in the next GW, **multiply EP by #fixtures**.

### 6) Best XI
Solve an ILP to pick the top XI and captain under:
- budget `BUDGET_MILLIONS` (default 100.0),
- formation (`FORMATION`, default **3-4-3**),
- max **3 per team**.

---

## Tuning & Extensions

- Swap models (e.g., LightGBM/XGBoost) in `model.py`.
- Add injury/starts probability from text via an LLM or web features—store as `extra_minutes_boost` feature.
- Replace the double-GW approximation with **fixture-level** EP (per-fixture modeling and summation).
- Add a **15-man squad** optimizer with bench and sub order.

---

## Notes
- This code uses only the **official FPL API** and avoids scraping.
- First run may take a few minutes as it fetches all player histories.
- If ILP fails due to strict constraints, reduce `FORMATION` demands or increase budget temporarily.
