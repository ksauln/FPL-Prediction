from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from fplmodel.config import PROJECT_ROOT, RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, MODELS_DIR
from fplmodel.config import MAX_TRAIN_GW
from fplmodel.data_pull import fetch_bootstrap_static, fetch_fixtures_all, bulk_fetch_player_histories
from fplmodel.data_cleaning import normalize_bootstrap, histories_to_df
from fplmodel.features import build_training_and_pred_frames, expand_for_double_gw
from fplmodel.model import train_models, load_models, predict_expected_points
from fplmodel.evaluation import evaluate_last_finished_gw_and_update_state
from fplmodel.state import ModelState
from fplmodel.utils import get_current_and_last_finished_gw

from fplmodel.team_picker import pick_best_xi

def run_pipeline(force_refetch: bool = False):
    # 1) Pull data
    bootstrap = fetch_bootstrap_static(force=force_refetch)
    fixtures = fetch_fixtures_all(force=force_refetch)
    norms = normalize_bootstrap(bootstrap)
    elements_df, teams_df, events_df = norms["elements"], norms["teams"], norms["events"]

    # Which GW?
    next_gw, last_finished_gw = get_current_and_last_finished_gw(events_df)
    if MAX_TRAIN_GW is not None:
        last_finished_gw = min(last_finished_gw, int(MAX_TRAIN_GW))

    # 2) Player histories (bulk)
    player_ids = elements_df["player_id"].tolist()
    bulk_fetch_player_histories(player_ids, force=force_refetch, sleep_s=0.0)

    # 3) Load histories from disk into one DF
    #    Avoid re-reading thousands of files into memory at once by streaming
    import json, os
    rows = []
    for fn in os.listdir(RAW_DIR):
        if fn.startswith("player_") and fn.endswith(".json"):
            with open(RAW_DIR / fn, "r", encoding="utf-8") as f:
                data = json.load(f)
            pid = int(fn.split("_")[1].split(".")[0])
            for h in data.get("history", []):
                h["player_id"] = pid
                rows.append(h)
    histories_df = pd.DataFrame(rows)

    if histories_df.empty:
        raise RuntimeError("No player history data found.")

    # 4) Build training and next-gw prediction frames
    state = ModelState()
    X_train, y_train, X_pred = build_training_and_pred_frames(
        elements_df, teams_df, histories_df, next_gw, last_finished_gw, state
    )

    # 5) Train (or retrain) models
    clf, reg = train_models(X_train.drop(columns=["player_id","full_name","now_cost_millions","team_id","element_type"], errors="ignore"), y_train)

    # 6) Predict EP for next GW
    predictions = predict_expected_points(X_pred, clf, reg, state)

    # 7) Double/Blank GW scaling (approximate): multiply EP by number of fixtures
    fixtures_df = pd.DataFrame(fixtures)
    predictions = expand_for_double_gw(predictions, fixtures_df, next_gw)
    predictions["expected_points"] = predictions["expected_points"] * predictions["fixture_multiplier"]

    # 8) Evaluate last finished GW and update biases (EMA)
    # Build train-like features (meta + feats) for evaluation function:
    # We'll reuse X_pred-like frame for players: it already contains features for last gw (since built up to last_finished_gw).
    # But for simplicity, we pass a frame containing only features + identifiers required.
    train_like = X_pred[["player_id","element_type"] + [c for c in X_pred.columns if c.endswith("_lag1") or "_ma" in c or c in ["was_home","team_strength_overall","team_attack_home","team_attack_away","team_def_home","team_def_away","opp_strength_overall","player_bias","pos_bias"]]].copy()
    res_df = evaluate_last_finished_gw_and_update_state(
        clf, reg, train_like, histories_df, last_finished_gw, state
    )

    # 9) Best XI selection
    team = pick_best_xi(predictions[["player_id","full_name","team_id","element_type","now_cost_millions","expected_points"]])

    # 10) Save artifacts
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_csv = OUTPUTS_DIR / f"predictions_gw{next_gw}.csv"
    predictions.sort_values("expected_points", ascending=False).to_csv(predictions_csv, index=False)

    team_json = OUTPUTS_DIR / f"best_xi_gw{next_gw}.json"
    with open(team_json, "w", encoding="utf-8") as f:
        json.dump(team, f, indent=2)

    # Residuals summary
    if res_df is not None and len(res_df):
        res_csv = OUTPUTS_DIR / f"residuals_gw{last_finished_gw}.csv"
        res_df.to_csv(res_csv, index=False)

    return {
        "next_gw": int(next_gw),
        "last_finished_gw": int(last_finished_gw),
        "predictions_csv": str(predictions_csv),
        "team_json": str(team_json),
        "residuals_csv": str(OUTPUTS_DIR / f"residuals_gw{last_finished_gw}.csv") if (res_df is not None and len(res_df)) else None,
    }

if __name__ == "__main__":
    out = run_pipeline(force_refetch=False)
    print(json.dumps(out, indent=2))
