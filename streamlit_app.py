"""Streamlit front-end for FPL prediction utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import requests

from fplmodel.config import BUDGET_MILLIONS, DATA_DIR, OUTPUTS_DIR
from fplmodel.team_picker import pick_best_xi
from fplmodel.team_analysis import compare_team_to_optimal, summarise_team
from fplmodel.transfer_recommender import (
    aggregate_expected_points,
    recommend_transfers,
)
from fplmodel.display_metrics import PlayerComparisonDependencies, render_player_comparison_page
from fplmodel.utils import get_current_and_last_finished_gw


POSITION_LABELS = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}
POSITION_SLOTS = (
    {"type_id": 1, "label": "Goalkeeper", "count": 2, "starters": 1},
    {"type_id": 2, "label": "Defender", "count": 5, "starters": 3},
    {"type_id": 3, "label": "Midfielder", "count": 5, "starters": 4},
    {"type_id": 4, "label": "Forward", "count": 3, "starters": 3},
)


st.set_page_config(page_title="FPL Optimization Toolkit", layout="wide")

TAB_STYLE = """
<style>
    button[data-baseweb="tab"] p {
        font-size: 1.5rem;
    }
</style>
"""

SESSION_USER_TEAM_KEY = "user_team_df"
SESSION_CAPTAIN_OVERRIDE_KEY = "user_team_captain_override"
SESSION_FPL_TEAM_CACHE = "fpl_team_cache"


def _extract_gw_from_path(path: Path) -> Optional[int]:
    match = re.search(r"_gw(\d+)", path.stem)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _discover_prediction_files() -> Dict[int, Path]:
    files: Dict[int, Path] = {}
    for path in OUTPUTS_DIR.glob("predictions_gw*.csv"):
        gw = _extract_gw_from_path(path)
        if gw is not None:
            files[gw] = path
    return files


def _available_prediction_gameweeks() -> List[int]:
    return sorted(_discover_prediction_files())


def _load_latest_predictions() -> Tuple[int, Path, pd.DataFrame]:
    files = _discover_prediction_files()
    if not files:
        raise FileNotFoundError(
            "No prediction files found in the outputs directory. Run the pipeline first."
        )
    latest_gw = max(files)
    predictions_path = files[latest_gw]
    df = _load_predictions(predictions_path)
    return latest_gw, predictions_path, df


def _load_next_predictions() -> Tuple[int, Path, pd.DataFrame]:
    files = _discover_prediction_files()
    if not files:
        raise FileNotFoundError(
            "No prediction files found in the outputs directory. Run the pipeline first."
        )

    last_finished = _last_finished_gameweek()
    candidate_gws = sorted(files)
    next_gw = None
    if last_finished is not None:
        for gw in candidate_gws:
            if gw > last_finished:
                next_gw = gw
                break
    if next_gw is None:
        next_gw = candidate_gws[0]

    predictions_path = files[next_gw]
    df = _load_predictions(predictions_path)
    return next_gw, predictions_path, df


@st.cache_data(show_spinner=False)
def _fetch_fpl_team_from_api(fpl_id: int, event: int) -> pd.DataFrame:
    """Fetch the user's squad for a specific gameweek via the public FPL API."""
    if fpl_id <= 0:
        raise ValueError("FPL ID must be a positive integer.")

    url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/event/{event}/picks/"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch FPL data (status {response.status_code}). "
            "Double-check your FPL ID and that the gameweek has valid picks."
        )

    payload = response.json()
    picks = payload.get("picks", [])
    if not picks:
        raise ValueError(
            "No picks returned for the requested gameweek. "
            "Ensure your squad has been set for that round."
        )

    records: List[Dict[str, object]] = []
    for pick in picks:
        pid = int(pick.get("element", 0))
        position = int(pick.get("position", 0))
        is_captain = bool(pick.get("is_captain", False))
        is_vice = bool(pick.get("is_vice_captain", False))
        multiplier = int(pick.get("multiplier", 0))
        records.append(
            {
                "player_id": pid,
                "starting": int(position <= 11),
                "bench": int(position > 11),
                "captain": int(is_captain),
                "vice_captain": int(is_vice),
                "multiplier": multiplier,
                "fpl_position": position,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Failed to parse picks for the provided FPL ID.")

    return df


def _load_predictions_for_horizon(
    horizon: int,
) -> Tuple[List[int], Dict[int, pd.DataFrame], List[int]]:
    files = _discover_prediction_files()
    if not files:
        raise FileNotFoundError(
            "No prediction files found in the outputs directory. Run the pipeline first."
        )

    available_gws = sorted(files)
    last_finished = _last_finished_gameweek()
    if last_finished is not None:
        start_gw = next((gw for gw in available_gws if gw > last_finished), available_gws[0])
    else:
        start_gw = available_gws[0]

    target_gws = [start_gw + offset for offset in range(horizon)]
    predictions_by_gw: Dict[int, pd.DataFrame] = {}
    missing: List[int] = []
    for gw in target_gws:
        path = files.get(gw)
        if path is None:
            missing.append(gw)
            continue
        predictions_by_gw[gw] = _load_predictions(path)

    if not predictions_by_gw:
        raise FileNotFoundError(
            "No upcoming prediction files found for the requested horizon. "
            f"Missing gameweeks: {', '.join(str(gw) for gw in target_gws)}"
        )

    loaded_gws = sorted(predictions_by_gw)
    return loaded_gws, predictions_by_gw, missing


def _best_xi_image_for_gw(gameweek: int) -> Optional[Path]:
    candidate = OUTPUTS_DIR / f"best_xi_gw{gameweek}.png"
    return candidate if candidate.exists() else None


def _best_xi_data_for_gw(gameweek: int) -> Optional[Dict[str, object]]:
    path = OUTPUTS_DIR / f"best_xi_gw{gameweek}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def _load_bootstrap_data() -> Dict[str, object]:
    path = DATA_DIR / "raw" / "bootstrap-static.json"
    if not path.exists():
        raise FileNotFoundError(
            "bootstrap-static.json not found in data/raw. Run the pipeline to refresh data."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def _load_bootstrap_events() -> pd.DataFrame:
    data = _load_bootstrap_data()
    events = data.get("events", [])
    if not events:
        raise ValueError("No events data found in bootstrap-static.json.")
    return pd.DataFrame(events)


@st.cache_data(show_spinner=False)
def _load_bootstrap_elements_df() -> pd.DataFrame:
    data = _load_bootstrap_data()
    elements = data.get("elements", [])
    if not elements:
        raise ValueError("No elements data found in bootstrap-static.json.")
    return pd.DataFrame(elements)


@st.cache_data(show_spinner=False)
def _load_bootstrap_teams_df() -> pd.DataFrame:
    data = _load_bootstrap_data()
    teams = data.get("teams", [])
    if not teams:
        raise ValueError("No teams data found in bootstrap-static.json.")
    return pd.DataFrame(teams)


@st.cache_data(show_spinner=False)
def _load_fixtures_df() -> pd.DataFrame:
    path = DATA_DIR / "raw" / "fixtures-all.json"
    if not path.exists():
        raise FileNotFoundError(
            "fixtures-all.json not found in data/raw. Run the pipeline to refresh data."
        )
    with open(path, "r", encoding="utf-8") as handle:
        fixtures = json.load(handle)
    return pd.DataFrame(fixtures)


def _last_finished_gameweek() -> Optional[int]:
    try:
        events_df = _load_bootstrap_events()
    except (FileNotFoundError, ValueError):
        return None
    _, last_finished = get_current_and_last_finished_gw(events_df)
    return last_finished if last_finished > 0 else None


@st.cache_data(show_spinner=False)
def _load_actual_points_for_gw(gameweek: int) -> Dict[int, float]:
    if gameweek <= 0:
        return {}
    records: Dict[int, float] = {}
    players_dir = DATA_DIR / "raw"
    for path in players_dir.glob("player_*.json"):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                player_data = json.load(handle)
        except json.JSONDecodeError:
            continue
        history = player_data.get("history", [])
        for entry in history:
            if int(entry.get("round", -1)) == gameweek:
                pid = int(entry.get("element") or player_data.get("id", 0))
                records[pid] = float(entry.get("total_points", 0.0))
                break
    return records


def _summarise_actual_points(
    team_df: pd.DataFrame, actual_points: Dict[int, float], captain_override: Optional[int] = None
) -> float:
    if team_df.empty:
        return 0.0
    actual_df = team_df.copy()
    actual_df["expected_points"] = actual_df["player_id"].map(lambda pid: actual_points.get(int(pid), 0.0))
    summary = summarise_team(actual_df, captain_id=captain_override)
    return summary.total_expected_points_with_captain


def _enrich_user_team(
    user_team: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    gameweek: Optional[int] = None,
) -> pd.DataFrame:
    enriched = user_team.copy()
    predictions_meta = predictions[
        [
            "player_id",
            "full_name",
            "team_name",
            "team_id",
            "element_type",
            "now_cost_millions",
            "expected_points",
        ]
    ].drop_duplicates("player_id")

    enriched = enriched.merge(predictions_meta, on="player_id", how="left", suffixes=("", "_pred"))

    for col in ["full_name", "team_name", "team_id", "element_type", "now_cost_millions", "expected_points"]:
        pred_col = f"{col}_pred"
        has_col = col in enriched.columns
        has_pred = pred_col in enriched.columns

        if has_col and has_pred:
            enriched[col] = enriched[col].fillna(enriched[pred_col])
            enriched = enriched.drop(columns=pred_col)
        elif not has_col and has_pred:
            enriched[col] = enriched[pred_col]
            enriched = enriched.drop(columns=pred_col)
        elif not has_col and not has_pred:
            enriched[col] = pd.NA

    if "starting" not in enriched.columns:
        enriched["starting"] = 0
        if len(enriched):
            starter_index = enriched.index[:11]
            enriched.loc[starter_index, "starting"] = 1
    if "bench" not in enriched.columns:
        enriched["bench"] = 1 - enriched["starting"]
    if "captain" not in enriched.columns:
        enriched["captain"] = 0

    enriched["player_id"] = enriched["player_id"].astype(int)
    enriched["starting"] = enriched["starting"].astype(int)
    enriched["bench"] = enriched["bench"].astype(int)
    enriched["captain"] = enriched["captain"].astype(int)

    try:
        elements_df = _load_bootstrap_elements_df()
    except (FileNotFoundError, ValueError):
        elements_df = pd.DataFrame()

    if not elements_df.empty:
        element_meta = elements_df[
            [
                "id",
                "expected_goals_per_90",
                "expected_assists_per_90",
                "clean_sheets_per_90",
                "expected_goal_involvements_per_90",
            ]
        ].rename(columns={"id": "player_id"})

        enriched = enriched.merge(element_meta, on="player_id", how="left")

        enriched["expected_goals"] = enriched.pop("expected_goals_per_90")
        enriched["expected_assists"] = enriched.pop("expected_assists_per_90")
        enriched["expected_clean_sheet"] = enriched.pop("clean_sheets_per_90")
        enriched = enriched.drop(columns=["expected_goal_involvements_per_90"], errors="ignore")

    if gameweek is not None:
        fixture_labels = _fixture_labels_for_gw(gameweek)
        if fixture_labels:
            enriched["opponent"] = enriched["team_id"].map(fixture_labels)

    if "opponent" not in enriched.columns:
        enriched["opponent"] = pd.NA

    return enriched


def _fixture_labels_for_gw(gameweek: int) -> Dict[int, str]:
    if gameweek <= 0:
        return {}

    try:
        fixtures_df = _load_fixtures_df()
        teams_df = _load_bootstrap_teams_df()
    except (FileNotFoundError, ValueError):
        return {}

    if fixtures_df is None or fixtures_df.empty or "event" not in fixtures_df.columns:
        return {}

    gw_fixtures = fixtures_df[fixtures_df["event"] == gameweek]
    if gw_fixtures.empty:
        return {}

    name_col = "short_name" if "short_name" in teams_df.columns else "name"
    team_name_map = teams_df.set_index("id")[name_col].to_dict()

    fixtures_map: Dict[int, List[str]] = defaultdict(list)
    for _, fixture in gw_fixtures.iterrows():
        team_h = fixture.get("team_h")
        team_a = fixture.get("team_a")
        if pd.isna(team_h) or pd.isna(team_a):
            continue
        try:
            team_h = int(team_h)
            team_a = int(team_a)
        except (TypeError, ValueError):
            continue
        opponent_home = team_name_map.get(team_a)
        opponent_away = team_name_map.get(team_h)
        if opponent_home:
            fixtures_map[team_h].append(f"{opponent_home} (H)")
        if opponent_away:
            fixtures_map[team_a].append(f"{opponent_away} (A)")

    return {team_id: " / ".join(parts) for team_id, parts in fixtures_map.items()}


def _apply_transfers_to_team(
    user_team: pd.DataFrame, transfers: List[Dict[str, object]]
) -> pd.DataFrame:
    """
    Apply recommended transfers to a user's squad DataFrame and return the new squad.
    Columns present in the original DataFrame are preserved where possible.
    """

    if not transfers or user_team is None or user_team.empty:
        return user_team.copy()

    team = user_team.copy()
    if "player_id" not in team.columns:
        return team

    base_cols = list(team.columns)
    out_ids = {int(t["out_player"]["player_id"]) for t in transfers}
    team = team[~team["player_id"].isin(out_ids)].copy()

    for transfer in transfers:
        pid_in = int(transfer["in_player"]["player_id"])
        if pid_in in team["player_id"].values:
            continue
        new_row = {col: pd.NA for col in base_cols}
        new_row["player_id"] = pid_in
        for flag in ("starting", "bench", "captain", "vice_captain", "multiplier"):
            if flag in new_row:
                new_row[flag] = 0
        team = pd.concat([team, pd.DataFrame([new_row])], ignore_index=True)

    return team


def _infer_captain(player_df: pd.DataFrame) -> Optional[int]:
    if "captain" in player_df.columns and player_df["captain"].sum() > 0:
        captain_series = player_df.loc[player_df["captain"] == 1, "player_id"]
        if not captain_series.empty:
            return int(captain_series.iloc[0])
    return None


def _load_predictions(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    expected_cols = {
        "player_id",
        "full_name",
        "team_name",
        "team_id",
        "element_type",
        "now_cost_millions",
        "expected_points",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing columns: {sorted(missing)}")
    df["player_id"] = df["player_id"].astype(int)
    return df


def _load_user_team(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    if "player_id" not in df.columns:
        raise ValueError("Team file must contain a 'player_id' column")
    df["player_id"] = df["player_id"].astype(int)
    return df


def _format_player_option(player_id: Optional[int], lookup: Dict[int, Dict[str, object]]) -> str:
    if player_id is None:
        return "Select a player"
    player = lookup.get(player_id)
    if player is None:
        return str(player_id)
    position = POSITION_LABELS.get(int(player.get("element_type", 0)), "Unknown")
    return f"{player['full_name']} ({position} - {player['team_name']})"


def _build_team_interactively(
    predictions: pd.DataFrame, *, session_prefix: str
) -> Optional[pd.DataFrame]:
    available = (
        predictions.drop_duplicates("player_id")
        .copy()
        .sort_values("full_name")
        .reset_index(drop=True)
    )
    lookup = available.set_index("player_id").to_dict("index")

    selection_slots: List[Dict[str, object]] = []
    for slot in POSITION_SLOTS:
        group_df = available[available["element_type"] == slot["type_id"]]
        options = [None] + group_df["player_id"].astype(int).tolist()
        st.markdown(f"**{slot['label']}s**")
        for idx in range(slot["count"]):
            select_key = f"{session_prefix}_{slot['type_id']}_{idx}"
            choice = st.selectbox(
                f"{slot['label']} {idx + 1}",
                options=options,
                key=select_key,
                format_func=lambda pid, _lookup=lookup: _format_player_option(pid, _lookup),
            )
            selection_slots.append(
                {
                    "player_id": choice,
                    "type_id": slot["type_id"],
                    "label": slot["label"],
                    "slot_index": idx,
                    "is_starting": idx < slot["starters"],
                }
            )

    if any(slot["player_id"] is None for slot in selection_slots):
        st.info("Select players for every position to continue.")
        return None

    selected_ids = [int(slot["player_id"]) for slot in selection_slots]
    if len(selected_ids) != len(set(selected_ids)):
        st.error("Each player can only be selected once.")
        return None

    starting_map = {
        int(slot["player_id"]): bool(slot["is_starting"]) for slot in selection_slots
    }

    team_df = available.set_index("player_id").loc[selected_ids].reset_index()
    team_df["starting"] = team_df["player_id"].map(lambda pid: int(starting_map[int(pid)]))
    team_df["bench"] = 1 - team_df["starting"]
    team_df["starting"] = team_df["starting"].astype(int)
    team_df["bench"] = team_df["bench"].astype(int)

    starting_ids = [pid for pid in selected_ids if starting_map[pid]]
    if len(starting_ids) != 11:
        st.error("Exactly 11 starters are required. Adjust your selections and try again.")
        return None

    captain_state_key = f"{session_prefix}_captain"
    if captain_state_key in st.session_state and st.session_state[captain_state_key] not in starting_ids:
        st.session_state[captain_state_key] = starting_ids[0]

    captain_id = st.selectbox(
        "Select your captain",
        options=starting_ids,
        key=captain_state_key,
        format_func=lambda pid, _lookup=lookup: _format_player_option(pid, _lookup),
    )

    if captain_id not in starting_ids:
        st.error("Captain must be one of the starting XI.")
        return None

    team_df["captain"] = (team_df["player_id"] == captain_id).astype(int)
    team_df["player_id"] = team_df["player_id"].astype(int)

    bench_count = int(team_df["bench"].sum())
    if bench_count != 4:
        st.error("Exactly four players must be on the bench.")
        return None

    display_df = team_df[
        [
            "full_name",
            "team_name",
            "element_type",
            "now_cost_millions",
            "expected_points",
            "starting",
            "bench",
            "captain",
        ]
    ].copy()
    display_df["position"] = display_df["element_type"].map(POSITION_LABELS)
    display_df = display_df[
        [
            "full_name",
            "team_name",
            "position",
            "now_cost_millions",
            "expected_points",
            "starting",
            "bench",
            "captain",
        ]
    ]
    display_df[["starting", "bench", "captain"]] = display_df[
        ["starting", "bench", "captain"]
    ].astype(bool)
    st.dataframe(display_df, width="stretch")

    return team_df


def _format_team_display(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    display["position"] = display.get("element_type", pd.Series(dtype=int)).map(POSITION_LABELS)
    numeric_cols = [
        "now_cost_millions",
        "expected_clean_sheet",
        "expected_assists",
        "expected_goals",
        "expected_points",
        "bench_order",
    ]
    for col in numeric_cols:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce")

    if "opponent" in display.columns:
        display["opponent"] = display["opponent"].fillna("TBC")

    columns = [
        "position",
        "full_name",
        "team_name",
        "opponent",
        "now_cost_millions",
        "expected_clean_sheet",
        "expected_assists",
        "expected_goals",
        "expected_points",
        "captain",
        "bench_order",
    ]
    existing = [col for col in columns if col in display.columns]
    display = display[existing]
    rename_map = {
        "position": "Pos",
        "full_name": "Name",
        "team_name": "Team",
        "opponent": "Opponent",
        "now_cost_millions": "Cost (£m)",
        "expected_clean_sheet": "Expected CS",
        "expected_assists": "Expected A",
        "expected_goals": "Expected G",
        "expected_points": "Expected Pts",
        "captain": "Captain",
        "bench_order": "Bench Order",
    }
    display = display.rename(columns=rename_map)

    if "Captain" in display.columns:
        display["Captain"] = (
            display["Captain"].fillna(0).astype(int).astype(bool)
        )

    if "Bench Order" in display.columns:
        order = pd.to_numeric(display["Bench Order"], errors="coerce")
        order = order.where(order > 0, pd.NA)
        display["Bench Order"] = (
            order.astype("Int64").astype(str).replace({"<NA>": ""})
        )

    return display


def _display_team(team: Dict[str, object], enriched: Optional[pd.DataFrame] = None) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Starting cost", f"£{team['starting_cost']:.1f}m")
    col2.metric("Bench cost", f"£{team['bench_cost']:.1f}m")
    col3.metric("Total cost", f"£{team['total_cost']:.1f}m")

    if enriched is None or enriched.empty:
        st.subheader("Starting XI")
        st.dataframe(pd.DataFrame(team["squad"]))
        if team.get("bench"):
            st.subheader("Bench")
            st.dataframe(pd.DataFrame(team["bench"]))
        return

    starters = enriched[enriched["starting"] == 1].copy()
    bench = enriched[enriched["bench"] == 1].copy()

    if not starters.empty:
        st.subheader("Starting XI")
        st.dataframe(_format_team_display(starters), width="stretch")

    if not bench.empty:
        if "bench_order" in bench.columns:
            bench = bench.sort_values("bench_order")
        st.subheader("Bench")
        st.dataframe(_format_team_display(bench), width="stretch")


def _optimal_team_page() -> None:
    st.header("Optimal Team")
    try:
        next_gw, predictions_path, predictions_df = _load_next_predictions()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.caption(
        f"Using predictions for upcoming gameweek {next_gw} "
        f"(`{predictions_path.name}` in `outputs/`)."
    )

    try:
        optimal_team = pick_best_xi(predictions_df, budget_m=float(BUDGET_MILLIONS))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to compute optimal team: {exc}")
        return

    st.success("Optimal team generated successfully")

    image_path = _best_xi_image_for_gw(next_gw)
    if image_path is not None:
        st.image(
            str(image_path),
            caption=f"Workflow best XI visual – GW {next_gw}",
            width=1350,
        )
    else:
        st.info("No stored Best XI image found for the latest gameweek.")

    metrics = st.columns(3)
    metrics[0].metric(
        "Expected points (XI)",
        f"{optimal_team['expected_points_without_captain']:.2f}",
    )
    metrics[1].metric(
        "Total points (with captain)",
        f"{optimal_team['total_expected_points_with_captain']:.2f}",
    )
    metrics[2].metric("Captain", optimal_team.get("captain", "N/A"))

    st.caption(f"Formation: {optimal_team.get('formation_name', 'N/A')}")
    st.caption("Budget assumption: £100.0m total for the 15-player squad.")
    optimal_records = optimal_team["squad"] + optimal_team.get("bench", [])
    optimal_df = pd.DataFrame(optimal_records)
    optimal_enriched = _enrich_user_team(optimal_df, predictions_df, gameweek=next_gw)
    _display_team(optimal_team, optimal_enriched)


def _team_comparison_page() -> None:
    st.header("Team Comparison")
    st.markdown(
        "Compare your squad against the optimal XI for the latest available gameweek. "
        "Predictions are loaded automatically from the pipeline outputs."
    )

    try:
        latest_gw, predictions_path, predictions_df = _load_next_predictions()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.caption(
        f"Using predictions for upcoming gameweek {latest_gw} "
        f"(`{predictions_path.name}` in `outputs/`)."
    )

    stored_team_df = st.session_state.get(SESSION_USER_TEAM_KEY)
    stored_captain = st.session_state.get(SESSION_CAPTAIN_OVERRIDE_KEY)
    saved_option = "Use saved team"
    fpl_option = "Load via FPL ID"
    team_options: List[str] = []
    if isinstance(stored_team_df, pd.DataFrame) and not stored_team_df.empty:
        team_options.append(saved_option)
    team_options.extend([fpl_option, "Build interactively"])
    if "comparison_team_mode" in st.session_state and st.session_state["comparison_team_mode"] not in team_options:
        del st.session_state["comparison_team_mode"]
    team_input_method = st.radio(
        "Team input method",
        tuple(team_options),
        key="comparison_team_mode",
    )

    captain_override: Optional[int] = None
    user_team_df: Optional[pd.DataFrame] = None

    last_finished_gw = _last_finished_gameweek()

    if team_input_method == saved_option:
        user_team_df = _enrich_user_team(
            stored_team_df.copy(), predictions_df, gameweek=latest_gw
        )
        captain_override = stored_captain
    elif team_input_method == fpl_option:
        fpl_id_value = st.text_input(
            "Enter your FPL team ID",
            key="comparison_fpl_id",
            placeholder="e.g. 1234567",
        )
        if not fpl_id_value:
            st.info("Enter your FPL ID to load your latest picks.")
            return
        try:
            fpl_id = int(fpl_id_value.strip())
        except ValueError:
            st.error("FPL ID must be an integer.")
            return

        if last_finished_gw is None:
            st.error("Unable to determine the last finished gameweek from bootstrap data.")
            return

        cache: Dict[Tuple[int, int], pd.DataFrame] = st.session_state.setdefault(
            SESSION_FPL_TEAM_CACHE, {}
        )
        cache_key = (fpl_id, last_finished_gw)
        if cache_key in cache:
            user_team_df = cache[cache_key].copy()
        else:
            try:
                user_team_df = _fetch_fpl_team_from_api(fpl_id, last_finished_gw)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
            cache[cache_key] = user_team_df.copy()
        captain_override = _infer_captain(user_team_df)
    else:
        st.markdown(
            "Use the search boxes below to select each position in your 15-player squad."
        )
        user_team_df = _build_team_interactively(
            predictions_df, session_prefix="comparison_team"
        )
        if user_team_df is None:
            return

    if user_team_df is None or user_team_df.empty:
        st.warning("No team selected yet.")
        return

    enriched_user_team = _enrich_user_team(
        user_team_df, predictions_df, gameweek=latest_gw
    )
    if captain_override is None:
        captain_override = _infer_captain(enriched_user_team)

    try:
        result = compare_team_to_optimal(
            predictions_df,
            enriched_user_team,
            captain_id=captain_override,
            budget_m=float(BUDGET_MILLIONS),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to compare teams: {exc}")
        return

    st.session_state[SESSION_USER_TEAM_KEY] = enriched_user_team
    st.session_state[SESSION_CAPTAIN_OVERRIDE_KEY] = captain_override

    comparison = result["comparison"]

    metrics = st.columns(4)
    metrics[0].metric(
        "Your expected points",
        f"{comparison['user_expected_points']:.2f}",
    )
    metrics[1].metric(
        "Optimal expected points",
        f"{comparison['optimal_expected_points']:.2f}",
    )
    metrics[2].metric(
        "Points gap",
        f"{comparison['points_gap']:.2f}",
    )
    metrics[3].metric(
        "Rating",
        f"{comparison['rating']:.1f}%",
    )

    last_finished = _last_finished_gameweek()
    if last_finished is not None:
        actual_points_map = _load_actual_points_for_gw(last_finished)
        user_actual_points = _summarise_actual_points(
            enriched_user_team, actual_points_map, captain_override
        )
        actual_cols = st.columns(2)
        actual_cols[0].metric(
            f"Your GW{last_finished} points",
            f"{user_actual_points:.1f}",
        )
        best_team_data = _best_xi_data_for_gw(last_finished)
        if best_team_data is not None:
            best_df = pd.DataFrame(best_team_data.get("squad", []) + best_team_data.get("bench", []))
            best_captain_id = None
            for record in best_team_data.get("squad", []):
                if record.get("captain", 0) == 1:
                    best_captain_id = int(record["player_id"])
                    break
            best_actual_points = _summarise_actual_points(
                best_df, actual_points_map, best_captain_id
            )
            actual_cols[1].metric(
                f"Optimal GW{last_finished} points",
                f"{best_actual_points:.1f}",
            )
        else:
            actual_cols[1].metric(
                f"Optimal GW{last_finished} points",
                "N/A",
            )
            st.info(
                f"No stored best XI data found for gameweek {last_finished} "
                "to compare actual points."
            )

    optimal_records = result["optimal_team"]["squad"] + result["optimal_team"].get("bench", [])
    optimal_df = pd.DataFrame(optimal_records)
    optimal_enriched = _enrich_user_team(optimal_df, predictions_df, gameweek=latest_gw)

    st.subheader("Your team")
    _display_team(result["user_team"], enriched_user_team)

    st.subheader("Optimal team")
    _display_team(result["optimal_team"], optimal_enriched)

    st.caption(
        "Expected goals/assists/clean sheets are per 90 values sourced from the latest FPL bootstrap data."
    )

def _player_comparison_page() -> None:
    deps = PlayerComparisonDependencies(
        load_predictions_for_horizon=_load_predictions_for_horizon,
        discover_prediction_files=_discover_prediction_files,
        last_finished_gameweek=_last_finished_gameweek,
        load_predictions=_load_predictions,
        load_actual_points_for_gw=_load_actual_points_for_gw,
        load_bootstrap_elements_df=_load_bootstrap_elements_df,
        load_fixtures_df=_load_fixtures_df,
        load_bootstrap_teams_df=_load_bootstrap_teams_df,
    )
    render_player_comparison_page(POSITION_LABELS, deps)

def _transfer_recommender_page() -> None:
    st.header("Transfer Recommender")
    st.markdown(
        "Receive transfer suggestions using the most recent pipeline predictions. "
        "Select your current squad and configure transfer constraints below."
    )

    available_gws = _available_prediction_gameweeks()
    if not available_gws:
        st.error(
            "No prediction files found in the outputs directory. Run the pipeline first."
        )
        return

    st.caption(
        "Available prediction files: "
        + ", ".join(f"GW {gw}" for gw in available_gws)
    )

    max_horizon = len(available_gws)
    default_horizon = min(4, max_horizon)
    horizon = st.number_input(
        "Number of upcoming gameweeks to consider",
        min_value=1,
        max_value=max_horizon,
        value=default_horizon,
        step=1,
    )

    free_transfers = st.number_input(
        "Free transfers available", min_value=0, value=1, step=1
    )
    max_transfers = st.number_input(
        "Maximum transfers to suggest", min_value=0, value=int(free_transfers), step=1
    )

    try:
        selected_gws, predictions_by_gw, missing_gws = _load_predictions_for_horizon(int(horizon))
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    caption_parts = []
    if selected_gws:
        caption_parts.append(
            "Using predictions for: " + ", ".join(f"GW {gw}" for gw in selected_gws)
        )
    if missing_gws:
        caption_parts.append(
            "Missing prediction files for: " + ", ".join(f"GW {gw}" for gw in missing_gws)
        )
    if caption_parts:
        st.caption(" | ".join(caption_parts))

    stored_team_df = st.session_state.get(SESSION_USER_TEAM_KEY)
    stored_team_option = "Use saved team from Team Comparison"
    fpl_team_option = "Load via FPL ID"
    team_input_options: List[str] = []
    if isinstance(stored_team_df, pd.DataFrame) and not stored_team_df.empty:
        team_input_options.append(stored_team_option)
    team_input_options.extend([fpl_team_option, "Build interactively"])
    if "transfer_team_mode" in st.session_state and st.session_state["transfer_team_mode"] not in team_input_options:
        del st.session_state["transfer_team_mode"]

    team_input_method = st.radio(
        "Team input method",
        tuple(team_input_options),
        key="transfer_team_mode",
    )

    base_predictions = predictions_by_gw[selected_gws[0]]

    last_finished_gw = _last_finished_gameweek()

    if team_input_method == stored_team_option:
        user_team_df = _enrich_user_team(
            stored_team_df.copy(), base_predictions, gameweek=selected_gws[0]
        )
    elif team_input_method == fpl_team_option:
        fpl_id_value = st.text_input(
            "Enter your FPL team ID",
            key="transfer_fpl_id",
            placeholder="e.g. 1234567",
        )
        if not fpl_id_value:
            st.info("Enter your FPL ID to load your squad.")
            return
        try:
            fpl_id = int(fpl_id_value.strip())
        except ValueError:
            st.error("FPL ID must be an integer.")
            return

        if last_finished_gw is None:
            st.error("Unable to determine the last finished gameweek from bootstrap data.")
            return

        cache: Dict[Tuple[int, int], pd.DataFrame] = st.session_state.setdefault(
            SESSION_FPL_TEAM_CACHE, {}
        )
        cache_key = (fpl_id, last_finished_gw)
        if cache_key in cache:
            user_team_df = cache[cache_key].copy()
        else:
            try:
                user_team_df = _fetch_fpl_team_from_api(fpl_id, last_finished_gw)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
            cache[cache_key] = user_team_df.copy()
        user_team_df = _enrich_user_team(
            user_team_df, base_predictions, gameweek=selected_gws[0]
        )
    else:
        st.markdown(
            "Use the search boxes below to select your current 15-player squad."
        )
        user_team_df = _build_team_interactively(
            base_predictions, session_prefix="transfer_team"
        )
        if user_team_df is None:
            return
        user_team_df = _enrich_user_team(user_team_df, base_predictions)

    try:
        result = recommend_transfers(
            user_team_df,
            predictions_by_gw,
            gameweeks=selected_gws,
            free_transfers=int(free_transfers),
            max_transfers=int(max_transfers),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to generate transfer recommendations: {exc}")
        return

    metadata = result["metadata"]

    metrics = st.columns(5)
    metrics[0].metric("Transfers suggested", len(result["recommended_transfers"]))
    metrics[1].metric("Free transfers used", metadata["free_transfers_used"])
    metrics[2].metric("Paid transfers (hits)", metadata["additional_transfers"])
    metrics[3].metric("Free transfers remaining", metadata["free_transfers_remaining"])
    metrics[4].metric(
        "Optimal projected points with transfers",
        f"{metadata['total_expected_points_optimal']:.2f}",
    )

    st.subheader("Recommended moves")
    if not result["recommended_transfers"]:
        st.write("Your squad already matches the optimal team for the selected horizon.")
    else:
        for suggestion in result["recommended_transfers"]:
            out_player = suggestion["out_player"]
            in_player = suggestion["in_player"]
            delta = suggestion["expected_points_delta"]
            st.markdown(
                f"**Out:** {out_player['full_name']} (EP {out_player['expected_points']:.2f}) → "
                f"**In:** {in_player['full_name']} (EP {in_player['expected_points']:.2f})"
            )
            st.caption(f"Expected points delta: {delta:+.2f}")

    aggregated = aggregate_expected_points(predictions_by_gw, gameweeks=selected_gws)

    post_transfer_team = _apply_transfers_to_team(
        user_team_df, result["recommended_transfers"]
    )
    post_transfer_team = _enrich_user_team(
        post_transfer_team, base_predictions, gameweek=selected_gws[0]
    )
    gw_cols = [col for col in aggregated.columns if col.startswith("expected_points_gw")]
    rename_gw = {col: f"GW {col.split('gw')[1]}" for col in gw_cols}

    def _projection_table(title: str, player_ids: List[int]) -> None:
        subset = aggregated[aggregated["player_id"].isin(player_ids)].copy()
        if subset.empty:
            return
        subset["position"] = subset["element_type"].map(POSITION_LABELS)
        display_cols = ["full_name", "team_name", "position"] + gw_cols + ["expected_points"]
        subset = subset[display_cols].rename(
            columns={
                "full_name": "Player",
                "team_name": "Team",
                "expected_points": "Total",
                **rename_gw,
            }
        )
        subset = subset.sort_values("Total", ascending=False).reset_index(drop=True)
        st.subheader(title)
        st.dataframe(subset, width="stretch")

    user_player_ids = [int(pid) for pid in user_team_df["player_id"].tolist()]
    post_transfer_ids = [int(pid) for pid in post_transfer_team["player_id"].tolist()]
    optimal_players = result["optimal_team"]["squad"] + result["optimal_team"].get("bench", [])
    optimal_player_ids = [int(player["player_id"]) for player in optimal_players]

    _projection_table(
        f"Your projected points (current squad, next {len(selected_gws)} GWs)",
        user_player_ids,
    )
    if result["recommended_transfers"]:
        _projection_table(
            f"Your projected points with recommended transfers (next {len(selected_gws)} GWs)",
            post_transfer_ids,
        )
    _projection_table(
        f"Optimal projected points (next {len(selected_gws)} GWs)",
        optimal_player_ids,
    )

    st.subheader("Optimal squad over horizon")
    _display_team(result["optimal_team"])


PAGES = {
    "Optimal Team": _optimal_team_page,
    "Team Comparison": _team_comparison_page,
    "Transfer Recommender": _transfer_recommender_page,
    "Player Comparison Lab": _player_comparison_page,
}


def main() -> None:
    st.markdown(TAB_STYLE, unsafe_allow_html=True)
    st.title("FPL Optimization Toolkit")
    tab_labels = list(PAGES)
    tabs = st.tabs(tab_labels)
    for tab, (label, render_page) in zip(tabs, PAGES.items()):
        with tab:
            render_page()


if __name__ == "__main__":
    main()
