"""Streamlit front-end for FPL prediction utilities."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from fplmodel.config import BUDGET_MILLIONS
from fplmodel.team_picker import pick_best_xi
from fplmodel.team_analysis import compare_team_to_optimal
from fplmodel.transfer_recommender import recommend_transfers


POSITION_LABELS = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}
POSITION_SLOTS = (
    {"type_id": 1, "label": "Goalkeeper", "count": 2, "starters": 1},
    {"type_id": 2, "label": "Defender", "count": 5, "starters": 3},
    {"type_id": 3, "label": "Midfielder", "count": 5, "starters": 4},
    {"type_id": 4, "label": "Forward", "count": 3, "starters": 3},
)


st.set_page_config(page_title="FPL Optimisation Toolkit", layout="wide")


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
    st.dataframe(display_df, use_container_width=True)

    return team_df


def _display_team(team: Dict[str, object]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Starting cost", f"£{team['starting_cost']:.1f}m")
    col2.metric("Bench cost", f"£{team['bench_cost']:.1f}m")
    col3.metric("Total cost", f"£{team['total_cost']:.1f}m")

    st.subheader("Starting XI")
    st.dataframe(pd.DataFrame(team["squad"]))

    if team.get("bench"):
        st.subheader("Bench")
        st.dataframe(pd.DataFrame(team["bench"]))


def _optimal_team_page() -> None:
    st.header("Optimal Team")
    st.markdown(
        "Upload a predictions CSV to compute the optimal squad and captain for the selected gameweek."
    )

    predictions_file = st.file_uploader(
        "Predictions CSV", type="csv", key="optimal_predictions"
    )

    budget = st.number_input(
        "Budget (millions)", value=float(BUDGET_MILLIONS), min_value=0.0, step=0.5
    )

    if predictions_file is None:
        st.info("Upload predictions to see the optimal team.")
        return

    try:
        predictions_df = _load_predictions(predictions_file)
        optimal_team = pick_best_xi(predictions_df, budget_m=budget)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to compute optimal team: {exc}")
        return

    st.success("Optimal team generated successfully")

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
    _display_team(optimal_team)


def _team_comparison_page() -> None:
    st.header("Team Comparison")
    st.markdown(
        "Compare your squad against the optimal XI for a gameweek. Upload your predictions and either import a squad CSV or build your team with the search fields."
    )

    predictions_file = st.file_uploader(
        "Predictions CSV", type="csv", key="comparison_predictions"
    )

    if predictions_file is None:
        st.info("Upload predictions to get started.")
        return

    try:
        predictions_df = _load_predictions(predictions_file)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    team_input_method = st.radio(
        "Team input method",
        ("Upload CSV", "Build interactively"),
        key="comparison_team_mode",
    )

    captain_override: Optional[int] = None

    if team_input_method == "Upload CSV":
        team_file = st.file_uploader(
            "Your squad CSV", type="csv", key="comparison_team"
        )
        if team_file is None:
            st.info("Upload your squad CSV or switch to the interactive builder.")
            return
        try:
            user_team_df = _load_user_team(team_file)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return

        if "captain" not in user_team_df.columns or user_team_df["captain"].sum() == 0:
            player_lookup = user_team_df.set_index("player_id")
            options: List[str] = ["Use file data"]
            options.extend(
                [
                    f"{int(pid)} - {player_lookup.loc[pid]['full_name']}"
                    if "full_name" in player_lookup.columns
                    else str(int(pid))
                    for pid in user_team_df["player_id"].tolist()
                ]
            )
            selection = st.selectbox("Captain override", options)
            if selection != "Use file data":
                captain_override = int(selection.split(" - ")[0])
    else:
        st.markdown(
            "Use the search boxes below to select each position in your 15-player squad."
        )
        user_team_df = _build_team_interactively(
            predictions_df, session_prefix="comparison_team"
        )
        if user_team_df is None:
            return

    try:
        result = compare_team_to_optimal(
            predictions_df, user_team_df, captain_id=captain_override
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to compare teams: {exc}")
        return

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

    st.subheader("Your team")
    _display_team(result["user_team"])

    st.subheader("Optimal team")
    _display_team(result["optimal_team"])


def _transfer_recommender_page() -> None:
    st.header("Transfer Recommender")
    st.markdown(
        "Upload predictions for the next four gameweeks to receive transfer suggestions."
    )

    predictions_files = st.file_uploader(
        "Predictions CSVs (ordered by upcoming gameweek)",
        type="csv",
        accept_multiple_files=True,
        key="transfer_predictions",
    )

    starting_gw = st.number_input(
        "Starting gameweek number", min_value=1, value=1, step=1
    )
    free_transfers = st.number_input(
        "Free transfers available", min_value=0, value=1, step=1
    )
    max_transfers = st.number_input(
        "Maximum transfers to suggest", min_value=0, value=int(free_transfers), step=1
    )

    if not predictions_files:
        st.info("Upload at least one predictions CSV to receive recommendations.")
        return

    try:
        ordered_files = sorted(predictions_files, key=lambda f: f.name)
        predictions_by_gw: Dict[int, pd.DataFrame] = {}
        for idx, uploaded in enumerate(ordered_files):
            uploaded.seek(0)
            gw = int(starting_gw + idx)
            predictions_by_gw[gw] = _load_predictions(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load predictions: {exc}")
        return

    team_input_method = st.radio(
        "Team input method",
        ("Upload CSV", "Build interactively"),
        key="transfer_team_mode",
    )

    if team_input_method == "Upload CSV":
        team_file = st.file_uploader(
            "Your current squad CSV", type="csv", key="transfer_team"
        )
        if team_file is None:
            st.info("Upload your squad CSV or switch to the interactive builder.")
            return
        try:
            user_team_df = _load_user_team(team_file)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
    else:
        st.markdown(
            "Use the search boxes below to select your current 15-player squad."
        )
        base_predictions = predictions_by_gw[min(predictions_by_gw)]
        user_team_df = _build_team_interactively(
            base_predictions, session_prefix="transfer_team"
        )
        if user_team_df is None:
            return

    try:
        result = recommend_transfers(
            user_team_df,
            predictions_by_gw,
            gameweeks=sorted(predictions_by_gw),
            free_transfers=int(free_transfers),
            max_transfers=int(max_transfers),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to generate transfer recommendations: {exc}")
        return

    metadata = result["metadata"]

    metrics = st.columns(4)
    metrics[0].metric("Transfers suggested", len(result["recommended_transfers"]))
    metrics[1].metric("Free transfers used", metadata["free_transfers_used"])
    metrics[2].metric("Extra transfers", metadata["additional_transfers"])
    metrics[3].metric(
        "Optimal expected points",
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

    st.subheader("Optimal squad over horizon")
    _display_team(result["optimal_team"])


PAGES = {
    "Optimal Team": _optimal_team_page,
    "Team Comparison": _team_comparison_page,
    "Transfer Recommender": _transfer_recommender_page,
}


def main() -> None:
    st.title("FPL Optimisation Toolkit")
    choice = st.sidebar.radio("Navigation", list(PAGES))
    PAGES[choice]()


if __name__ == "__main__":
    main()

