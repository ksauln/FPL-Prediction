"""Reusable Streamlit components for player comparison and metrics displays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fplmodel.transfer_recommender import aggregate_expected_points


PredictionByGameweek = Dict[int, pd.DataFrame]


@dataclass(frozen=True)
class PlayerComparisonDependencies:
    """Collection of callables required to render the player comparison page."""

    load_predictions_for_horizon: Callable[[int], Tuple[List[int], PredictionByGameweek, List[int]]]
    discover_prediction_files: Callable[[], Dict[int, Path]]
    last_finished_gameweek: Callable[[], Optional[int]]
    load_predictions: Callable[[Path], pd.DataFrame]
    load_actual_points_for_gw: Callable[[int], Dict[int, float]]
    load_bootstrap_elements_df: Callable[[], pd.DataFrame]
    load_fixtures_df: Callable[[], pd.DataFrame]
    load_bootstrap_teams_df: Callable[[], pd.DataFrame]


def render_player_comparison_page(
    position_labels: Dict[int, str], deps: PlayerComparisonDependencies
) -> None:
    """Render the Player Comparison Lab page using the provided dependencies."""

    st.header("Player Comparison Lab")
    st.markdown(
        "Interactively compare player projections, historical performance, and upcoming "
        "fixtures to find the perfect transfers for your squad."
    )

    horizon = st.slider("Projection horizon (gameweeks)", 1, 5, 3)
    try:
        loaded_gws, predictions_by_gw, missing = deps.load_predictions_for_horizon(horizon)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    if missing:
        st.warning(
            "Missing prediction files for some requested gameweeks: "
            + ", ".join(str(gw) for gw in missing)
        )

    aggregated = aggregate_expected_points(predictions_by_gw, gameweeks=loaded_gws)
    aggregated["position"] = aggregated["element_type"].map(position_labels)

    try:
        elements_df = deps.load_bootstrap_elements_df()
    except (FileNotFoundError, ValueError):
        elements_df = pd.DataFrame()

    if not elements_df.empty:
        metric_cols = [
            "id",
            "expected_goals_per_90",
            "expected_assists_per_90",
            "clean_sheets_per_90",
        ]
        elements_metrics = elements_df[metric_cols].rename(columns={"id": "player_id"})
        aggregated = aggregated.merge(elements_metrics, on="player_id", how="left")

    player_lookup = (
        aggregated[["player_id", "full_name", "team_name", "position"]]
        .set_index("player_id")
        .to_dict("index")
    )

    def _format_player(pid: int) -> str:
        data = player_lookup.get(pid)
        if not data:
            return str(pid)
        return f"{data['full_name']} ({data['position']} – {data['team_name']})"

    default_players = aggregated.sort_values("expected_points", ascending=False)[
        "player_id"
    ].head(3)
    selected_players = st.multiselect(
        "Players to analyse",
        options=aggregated["player_id"].tolist(),
        default=default_players.tolist(),
        format_func=_format_player,
    )

    if not selected_players:
        st.info("Select at least one player to begin the comparison.")
        return

    selected_df = aggregated[aggregated["player_id"].isin(selected_players)].copy()
    selected_df = selected_df.sort_values("expected_points", ascending=False)
    gw_cols = [col for col in selected_df.columns if col.startswith("expected_points_gw")]
    selected_df["expected_points_per_gw"] = selected_df[gw_cols].sum(axis=1) / max(
        len(gw_cols), 1
    )

    _render_projection_summary(selected_df, loaded_gws)
    _render_skill_radar(selected_df)
    _render_expected_vs_actual(selected_players, deps)
    _render_gameweek_timeline(selected_df, gw_cols)
    _render_fixture_difficulty_table(selected_df, loaded_gws, deps)


def _render_projection_summary(selected_df: pd.DataFrame, loaded_gws: List[int]) -> None:
    summary_cols = [
        "full_name",
        "team_name",
        "position",
        "expected_points",
        "expected_points_per_gw",
    ]
    gw_cols = [col for col in selected_df.columns if col.startswith("expected_points_gw")]
    summary_cols.extend(gw_cols)

    summary = selected_df[summary_cols].rename(
        columns={
            "full_name": "Player",
            "team_name": "Team",
            "position": "Position",
            "expected_points": f"Total EP (GW {loaded_gws[0]}-{loaded_gws[-1]})",
            "expected_points_per_gw": "Avg EP per GW",
        }
    ).reset_index(drop=True)

    st.subheader("Projection summary")
    st.dataframe(summary, width="stretch")


def _render_skill_radar(selected_df: pd.DataFrame) -> None:
    st.subheader("Skill radar")
    radar_metrics = {
        "Expected Goals/90": "expected_goals_per_90",
        "Expected Assists/90": "expected_assists_per_90",
        "Clean Sheets/90": "clean_sheets_per_90",
        "Avg Expected Points": "expected_points_per_gw",
    }
    radar_fig = go.Figure()
    for _, row in selected_df.iterrows():
        values = [float(row.get(col, 0) or 0) for col in radar_metrics.values()]
        radar_fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=list(radar_metrics.keys()),
                fill="toself",
                name=row["full_name"],
            )
        )
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=700,
        margin=dict(l=70, r=70, t=90, b=70),
    )
    st.plotly_chart(radar_fig)


def _render_expected_vs_actual(selected_players: List[int], deps: PlayerComparisonDependencies) -> None:
    st.subheader("Expected vs actual points")
    prediction_files = deps.discover_prediction_files()
    if not prediction_files:
        st.info("Prediction files are required to show expected vs actual comparisons.")
        return

    available_gws = sorted(prediction_files)
    last_finished = deps.last_finished_gameweek()
    default_index = max(
        (i for i, gw in enumerate(available_gws) if last_finished and gw <= last_finished),
        default=len(available_gws) - 1,
    )
    gw_choice = st.selectbox(
        "Gameweek to review",
        options=available_gws,
        index=default_index,
        format_func=lambda gw: f"GW {gw}",
    )

    scatter_predictions = deps.load_predictions(prediction_files[gw_choice])
    scatter_df = scatter_predictions[[
        "player_id",
        "full_name",
        "team_name",
        "expected_points",
    ]].copy()
    actual_points = deps.load_actual_points_for_gw(gw_choice)
    scatter_df["actual_points"] = scatter_df["player_id"].map(actual_points)
    scatter_selected = scatter_df[scatter_df["player_id"].isin(selected_players)]

    if scatter_selected["actual_points"].isna().all():
        st.info("Actual scores for the selected gameweek are not yet available.")
        return

    scatter_selected = scatter_selected.fillna(0.0)
    scatter_fig = px.scatter(
        scatter_selected,
        x="expected_points",
        y="actual_points",
        color="full_name",
        hover_data={"team_name": True, "player_id": True},
        labels={
            "expected_points": "Expected points",
            "actual_points": "Actual points",
            "full_name": "Player",
        },
    )
    scatter_fig.update_layout(legend_title="Player")
    scatter_fig.update_traces(marker=dict(size=28, line=dict(width=1, color="rgba(0,0,0,0.4)")))
    st.plotly_chart(scatter_fig)


def _render_gameweek_timeline(selected_df: pd.DataFrame, gw_cols: List[str]) -> None:
    st.subheader("Gameweek timeline race")
    timeline_df = selected_df[["player_id", "full_name"] + gw_cols].copy()
    timeline_df = timeline_df.melt(
        id_vars=["player_id", "full_name"],
        value_vars=gw_cols,
        var_name="gameweek",
        value_name="expected_points",
    )
    timeline_df["gameweek"] = timeline_df["gameweek"].str.extract(r"gw(\d+)").astype(int)
    timeline_df = timeline_df.sort_values("gameweek")

    if timeline_df["expected_points"].abs().sum() == 0:
        st.info("Not enough prediction data to display the timeline race.")
        return

    bar_fig = px.bar(
        timeline_df,
        x="full_name",
        y="expected_points",
        color="full_name",
        animation_frame="gameweek",
        labels={"full_name": "Player", "expected_points": "Expected points"},
    )
    bar_fig.update_layout(legend_title="Player")
    st.plotly_chart(bar_fig)


def _render_fixture_difficulty_table(
    selected_df: pd.DataFrame, loaded_gws: List[int], deps: PlayerComparisonDependencies
) -> None:
    st.subheader("Next five fixtures difficulty")
    try:
        fixtures_df = deps.load_fixtures_df()
    except (FileNotFoundError, ValueError):
        fixtures_df = pd.DataFrame()
    try:
        teams_df = deps.load_bootstrap_teams_df()
    except (FileNotFoundError, ValueError):
        teams_df = pd.DataFrame()

    if fixtures_df.empty or teams_df.empty:
        st.info("Fixture data unavailable. Run the data pipeline to refresh local files.")
        return

    teams_lookup = teams_df.set_index("id")["name"].to_dict()
    start_gw = loaded_gws[0]
    fixture_columns = [f"Fixture {idx + 1}" for idx in range(5)]
    table_rows: List[Dict[str, object]] = []
    diff_rows: List[Dict[str, float]] = []
    index_labels: List[str] = []

    upcoming_df = fixtures_df.dropna(subset=["event"]).copy()
    upcoming_df["event"] = upcoming_df["event"].astype(int)
    upcoming_df = upcoming_df[upcoming_df["event"] >= start_gw]

    for _, player in selected_df.iterrows():
        team_id = int(player.get("team_id", 0))
        fixtures: List[str] = []
        difficulties: List[float] = []
        player_fixtures = (
            upcoming_df[
                (upcoming_df["team_h"] == team_id) | (upcoming_df["team_a"] == team_id)
            ].sort_values("event")
        )
        for _, fixture in player_fixtures.head(10).iterrows():
            gw = int(fixture["event"])
            if int(fixture["team_h"]) == team_id:
                opponent_id = int(fixture["team_a"])
                venue = "H"
                difficulty = fixture.get("team_h_difficulty")
            else:
                opponent_id = int(fixture["team_h"])
                venue = "A"
                difficulty = fixture.get("team_a_difficulty")
            opponent_name = teams_lookup.get(opponent_id, f"Team {opponent_id}")
            fixtures.append(f"GW {gw}: {opponent_name} ({venue})")
            difficulties.append(float(difficulty) if difficulty is not None else float("nan"))
            if len(fixtures) == 5:
                break
        while len(fixtures) < 5:
            fixtures.append("TBC")
            difficulties.append(float("nan"))
        table_rows.append(dict(zip(fixture_columns, fixtures)))
        diff_rows.append(dict(zip(fixture_columns, difficulties)))
        index_labels.append(player["full_name"])

    fixtures_table = pd.DataFrame(table_rows, index=index_labels)
    difficulty_table = pd.DataFrame(diff_rows, index=index_labels)

    palette = {
        1: "#1F9D55",
        2: "#51CF66",
        3: "#FFD43B",
        4: "#FFA94D",
        5: "#FA5252",
    }

    def _style_row(row: pd.Series) -> List[str]:
        styles: List[str] = []
        diffs = difficulty_table.loc[row.name]
        for col in row.index:
            diff_value = diffs.get(col)
            if pd.isna(diff_value):
                styles.append("")
                continue
            diff_int = int(diff_value)
            bg = palette.get(diff_int, "#CED4DA")
            text_color = "#000000" if diff_int <= 3 else "#FFFFFF"
            styles.append(
                f"background-color: {bg}; color: {text_color}; font-weight: 600;"
            )
        return styles

    fixtures_styled = fixtures_table.style.apply(_style_row, axis=1)
    st.dataframe(fixtures_styled, width="stretch")
