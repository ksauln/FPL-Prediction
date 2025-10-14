from __future__ import annotations
import json
import os
import pandas as pd

from fplmodel.config import RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, MODELS_DIR, FORMATION_OPTIONS
from fplmodel.config import MAX_TRAIN_GW
from fplmodel.data_pull import fetch_bootstrap_static, fetch_fixtures_all, bulk_fetch_player_histories
from fplmodel.data_cleaning import normalize_bootstrap
from fplmodel.features import (
    build_training_and_pred_frames,
    expand_for_double_gw,
    log_model_feature_weights,
)
from fplmodel.model import train_models, predict_expected_points
from fplmodel.evaluation import evaluate_last_finished_gw_and_update_state
from fplmodel.state import ModelState
from fplmodel.utils import get_current_and_last_finished_gw
from fplmodel.logging_utils import configure_run_logger, update_log_filename_for_gameweek

from fplmodel.team_picker import pick_best_xi
from fplmodel.display import create_best_xi_graphic

def run_pipeline(force_refetch: bool = False):
    logger, file_handler, log_path = configure_run_logger()
    logger.info("Starting pipeline run (force_refetch=%s)", force_refetch)

    try:
        # 1) Pull data
        logger.info("Fetching bootstrap static data...")
        bootstrap = fetch_bootstrap_static(force=force_refetch)

        logger.info("Fetching fixtures data...")
        fixtures = fetch_fixtures_all(force=force_refetch)

        norms = normalize_bootstrap(bootstrap)
        elements_df, teams_df, events_df = norms["elements"], norms["teams"], norms["events"]
        logger.info(
            "Loaded normalised frames: %d elements, %d teams, %d events",
            len(elements_df),
            len(teams_df),
            len(events_df),
        )

        # Which GW?
        next_gw, last_finished_gw = get_current_and_last_finished_gw(events_df)
        if MAX_TRAIN_GW is not None:
            capped = min(last_finished_gw, int(MAX_TRAIN_GW))
            if capped != last_finished_gw:
                logger.info(
                    "Capping last finished GW from %s to MAX_TRAIN_GW=%s",
                    last_finished_gw,
                    MAX_TRAIN_GW,
                )
                last_finished_gw = capped
        logger.info("Next gameweek: %s | Last finished gameweek: %s", next_gw, last_finished_gw)
        file_handler, log_path = update_log_filename_for_gameweek(logger, file_handler, log_path, next_gw)

        # 2) Player histories (bulk)
        player_ids = elements_df["player_id"].tolist()
        logger.info("Fetching player histories for %d players (force_refetch=%s)", len(player_ids), force_refetch)
        bulk_fetch_player_histories(player_ids, force=force_refetch, sleep_s=0.0)

        # 3) Load histories from disk into one DF
        #    Avoid re-reading thousands of files into memory at once by streaming
        raw_files = [
            fn
            for fn in os.listdir(RAW_DIR)
            if fn.startswith("player_") and fn.endswith(".json")
        ]
        logger.info("Loading %d player history files from %s", len(raw_files), RAW_DIR)

        rows = []
        for fn in raw_files:
            with open(RAW_DIR / fn, "r", encoding="utf-8") as f:
                data = json.load(f)
            pid = int(fn.split("_")[1].split(".")[0])
            for h in data.get("history", []):
                h["player_id"] = pid
                rows.append(h)
        histories_df = pd.DataFrame(rows)
        logger.info("Built player histories dataframe with %d rows", len(histories_df))

        if histories_df.empty:
            logger.error("No player history data found in %s", RAW_DIR)
            raise RuntimeError("No player history data found.")

        # 4) Build training and next-gw prediction frames
        state = ModelState()
        X_train, y_train, X_pred = build_training_and_pred_frames(
            elements_df, teams_df, histories_df, next_gw, last_finished_gw, state
        )
        logger.info(
            "Prepared features: X_train=%s, y_train=%d, X_pred=%s",
            tuple(X_train.shape),
            len(y_train),
            tuple(X_pred.shape),
        )

        # 5) Train (or retrain) models
        train_features = X_train.drop(
            columns=["player_id", "full_name", "now_cost_millions", "team_id", "element_type"],
            errors="ignore",
        )
        logger.info("Training models with %d features", train_features.shape[1])
        logger.info(
            "Training feature columns: %s",
            ", ".join(train_features.columns.astype(str)),
        )
        clf, reg = train_models(train_features, y_train)
        log_model_feature_weights(logger, train_features.columns, reg, model_label="regressor")
        log_model_feature_weights(logger, train_features.columns, clf, model_label="classifier")
        logger.info("Model training complete")

        # 6) Predict EP for next GW
        predictions = predict_expected_points(X_pred, clf, reg, state)
        logger.info("Generated baseline predictions for %d players", len(predictions))

        # 7) Double/Blank GW scaling (approximate): multiply EP by number of fixtures
        fixtures_df = pd.DataFrame(fixtures)
        predictions = expand_for_double_gw(predictions, fixtures_df, next_gw)
        predictions["expected_points"] = predictions["expected_points"] * predictions["fixture_multiplier"]
        logger.info("Applied fixture multipliers; average EP now %.2f", predictions["expected_points"].mean())

        top_preds = (
            predictions.sort_values("expected_points", ascending=False)
            .head(5)
            .apply(lambda row: f"{row['full_name']} ({row['expected_points']:.2f})", axis=1)
            .tolist()
        )
        if top_preds:
            logger.info("Top 5 expected point predictions: %s", "; ".join(top_preds))

        # 8) Evaluate last finished GW and update biases (EMA)
        train_like = X_pred[
            ["player_id", "element_type"]
            + [
                c
                for c in X_pred.columns
                if c.endswith("_lag1")
                or "_ma" in c
                or c
                in [
                    "was_home",
                    "team_strength_overall",
                    "team_attack_home",
                    "team_attack_away",
                    "team_def_home",
                    "team_def_away",
                    "opp_strength_overall",
                    "player_bias",
                    "pos_bias",
                ]
            ]
        ].copy()
        res_df = evaluate_last_finished_gw_and_update_state(
            clf, reg, train_like, histories_df, last_finished_gw, state
        )
        if res_df is not None and len(res_df):
            logger.info("Residuals computed for %d players in GW %s", len(res_df), last_finished_gw)
        else:
            logger.info("No residuals available for GW %s", last_finished_gw)

        # 9) Best XI selection
        logger.info("Selecting best XI from %d candidates", len(predictions))
        team = pick_best_xi(
            predictions[["player_id", "full_name", "team_name", "team_id", "element_type", "now_cost_millions", "expected_points"]],
            formations=FORMATION_OPTIONS,
        )
        logger.info(
            "Selected squad: total cost %.1fM | starting cost %.1fM | bench cost %.1fM",
            team["total_cost"],
            team["starting_cost"],
            team["bench_cost"],
        )
        logger.info(
            "Projected points: XI %.2f | Captain included %.2f | Bench %.2f",
            team["expected_points_without_captain"],
            team["total_expected_points_with_captain"],
            team["bench_expected_points"],
        )
        logger.info("Captain: %s", team.get("captain"))

        for player in team.get("squad", []):
            logger.info(
                "XI | %s (%s) - %.2f pts | %.1fM",
                player["full_name"],
                player["team_name"],
                player["expected_points"],
                player["now_cost_millions"],
            )
        for bench_player in team.get("bench", []):
            logger.info(
                "Bench %d | %s (%s) - %.2f pts",
                bench_player.get("bench_order", 0),
                bench_player["full_name"],
                bench_player["team_name"],
                bench_player["expected_points"],
            )

        # 10) Save artifacts
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        predictions_csv = OUTPUTS_DIR / f"predictions_gw{next_gw}.csv"
        predictions.sort_values("expected_points", ascending=False).to_csv(predictions_csv, index=False)
        logger.info("Predictions saved to %s", predictions_csv)

        xi_csv = OUTPUTS_DIR / f"starting_xi_gw{next_gw}.csv"
        squad_df = pd.DataFrame(team.get("squad", []))
        if not squad_df.empty:
            squad_df.to_csv(xi_csv, index=False)
            logger.info("Starting XI saved to %s", xi_csv)
        else:
            xi_csv = None
            logger.warning("No starting XI to save for GW %s", next_gw)

        bench_csv = OUTPUTS_DIR / f"bench_gw{next_gw}.csv"
        bench_df = pd.DataFrame(team.get("bench", []))
        if not bench_df.empty:
            bench_df.to_csv(bench_csv, index=False)
            logger.info("Bench saved to %s", bench_csv)
        else:
            bench_csv = None
            logger.warning("No bench to save for GW %s", next_gw)

        team_json = OUTPUTS_DIR / f"best_xi_gw{next_gw}.json"
        with open(team_json, "w", encoding="utf-8") as f:
            json.dump(team, f, indent=2)
        logger.info("Best XI JSON saved to %s", team_json)

        team_image = create_best_xi_graphic(team, gameweek=next_gw)
        logger.info("Best XI graphic generated at %s", team_image)

        # Residuals summary
        residuals_csv = None
        if res_df is not None and len(res_df):
            residuals_csv = OUTPUTS_DIR / f"residuals_gw{last_finished_gw}.csv"
            res_df.to_csv(residuals_csv, index=False)
            logger.info("Residuals saved to %s", residuals_csv)

        logger.info("Pipeline complete for GW %s", next_gw)
        return {
            "next_gw": int(next_gw),
            "last_finished_gw": int(last_finished_gw),
            "predictions_csv": str(predictions_csv),
            "team_json": str(team_json),
            "team_graphic": str(team_image),
            "starting_xi_csv": str(xi_csv) if xi_csv is not None else None,
            "bench_csv": str(bench_csv) if bench_csv is not None else None,
            "residuals_csv": str(residuals_csv) if residuals_csv is not None else None,
            "log_file": str(log_path),
        }
    except Exception:
        logger.exception("Pipeline execution failed")
        raise
    finally:
        try:
            logger.removeHandler(file_handler)
        except ValueError:
            pass
        file_handler.close()

if __name__ == "__main__":
    out = run_pipeline(force_refetch=False)
    print(json.dumps(out, indent=2))
