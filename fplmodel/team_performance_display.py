"""Streamlit page components for analysing a user's historical FPL performance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


POSITION_LABELS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

TABLE_HEADER_BG = "rgba(128, 128, 128, 0.12)"
TABLE_ROW_BORDER = "rgba(128, 128, 128, 0.16)"
PRIMARY_ROW_BG = "rgba(56, 128, 255, 0.20)"
PRIMARY_ROW_BORDER = "rgba(56, 128, 255, 0.45)"
DANGER_ROW_BG = "rgba(220, 76, 70, 0.24)"
DANGER_ROW_BORDER = "rgba(220, 76, 70, 0.45)"
WARNING_ROW_BG = "rgba(255, 193, 7, 0.26)"
WARNING_ROW_BORDER = "rgba(255, 193, 7, 0.45)"

BASE_TABLE_STYLES = [
    {"selector": "thead th", "props": f"background-color: {TABLE_HEADER_BG}; font-weight: 600; border-bottom: 1px solid {TABLE_ROW_BORDER};"},
    {"selector": "tbody td", "props": f"border-bottom: 1px solid {TABLE_ROW_BORDER};"},
    {"selector": "tbody tr:last-child td", "props": "border-bottom: none;"},
]


def _apply_base_table_style(styler: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
    return styler.set_table_styles(BASE_TABLE_STYLES, overwrite=False)


@dataclass(frozen=True)
class TeamPerformanceDependencies:
    """Callable hooks supplied by the main Streamlit app."""

    load_bootstrap_events: Callable[[], pd.DataFrame]
    last_finished_gameweek: Callable[[], Optional[int]]
    load_actual_points_for_gw: Callable[[int], Dict[int, float]]
    load_bootstrap_elements_df: Callable[[], pd.DataFrame]
    load_bootstrap_teams_df: Callable[[], pd.DataFrame]


@st.cache_data(show_spinner=False)
def _fetch_entry_history(fpl_id: int) -> Dict[str, object]:
    if fpl_id <= 0:
        raise ValueError("FPL ID must be a positive integer.")
    url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/history/"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch entry history (status {response.status_code}).")
    payload = response.json()
    if not payload or "current" not in payload:
        raise ValueError("Entry history payload missing expected data.")
    return payload


@st.cache_data(show_spinner=False)
def _fetch_entry_overview(fpl_id: int) -> Dict[str, object]:
    if fpl_id <= 0:
        raise ValueError("FPL ID must be a positive integer.")
    url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch entry overview (status {response.status_code}).")
    return response.json()


@st.cache_data(show_spinner=False)
def _fetch_entry_picks(fpl_id: int, event: int) -> List[Dict[str, object]]:
    if fpl_id <= 0 or event <= 0:
        raise ValueError("FPL ID and event must be positive integers.")
    url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/event/{event}/picks/"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch picks for team {fpl_id} in GW {event} (status {response.status_code})."
        )
    payload = response.json()
    picks = payload.get("picks", [])
    if not picks:
        raise ValueError("No picks data returned for the requested gameweek.")
    return picks


@st.cache_data(show_spinner=False)
def _fetch_classic_league_standings(league_id: int) -> List[Dict[str, object]]:
    if league_id <= 0:
        raise ValueError("League ID must be a positive integer.")
    results: List[Dict[str, object]] = []
    page = 1
    # The classic league endpoint paginates; fetch up to a sensible cap to avoid excessive calls.
    for _ in range(10):
        url = (
            f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
            f"?page_standings={page}"
        )
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise ValueError(
                f"Failed to fetch standings for league {league_id} (status {response.status_code})."
            )
        payload = response.json()
        standings = payload.get("standings", {})
        page_results = standings.get("results", [])
        if not page_results:
            break
        results.extend(page_results)
        if not standings.get("has_next"):
            break
        page += 1
    return results


def _prepare_history_df(history: Dict[str, object], events_df: pd.DataFrame) -> pd.DataFrame:
    current = history.get("current", [])
    if not current:
        raise ValueError("No gameweek records found for this team.")

    history_df = pd.DataFrame(current).copy()
    history_df = history_df.sort_values("event").reset_index(drop=True)
    history_df["event"] = history_df["event"].astype(int)
    history_df["points"] = history_df["points"].astype(float)
    history_df["total_points"] = history_df["total_points"].astype(float)
    if "overall_rank" in history_df.columns:
        history_df["overall_rank"] = history_df["overall_rank"].astype(float)
    if "value" in history_df.columns:
        history_df["value"] = history_df["value"].astype(float) / 10.0

    if events_df is not None and not events_df.empty:
        merge_cols = ["id"]
        for optional_col in ("average_entry_score", "highest_score"):
            if optional_col in events_df.columns:
                merge_cols.append(optional_col)
        events_subset = events_df[merge_cols].rename(columns={"id": "event"})
        events_subset["event"] = events_subset["event"].astype(int)
        history_df = history_df.merge(events_subset, on="event", how="left")
    else:
        history_df["average_entry_score"] = pd.NA
        history_df["highest_score"] = pd.NA

    for col in ("average_entry_score", "highest_score"):
        if col not in history_df.columns:
            history_df[col] = pd.NA

    history_df["points_vs_average"] = history_df["points"] - history_df["average_entry_score"]
    history_df["points_vs_highest"] = history_df["points"] - history_df["highest_score"]
    history_df["overall_rank_delta"] = history_df["overall_rank"].shift(1) - history_df["overall_rank"]
    history_df["overall_rank_delta"] = history_df["overall_rank_delta"].fillna(0).astype(float)
    history_df["season_points_average"] = history_df["points"].expanding().mean()
    return history_df


def _format_int(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def _render_summary(history_df: pd.DataFrame) -> None:
    latest = history_df.iloc[-1]
    gw = int(latest["event"])
    prev_points = history_df.iloc[-2]["points"] if len(history_df) > 1 else None

    season_avg_points = float(history_df["season_points_average"].iloc[-1])

    col1, col2, col3, col4, col5 = st.columns(5)
    delta_points = None
    if prev_points is not None:
        delta_points = f"{latest['points'] - prev_points:+.1f}"
    col1.metric(f"GW {gw} points", f"{latest['points']:.1f}", delta=delta_points)

    avg_points = latest.get("average_entry_score")
    if pd.notna(avg_points):
        col2.metric(
            "Global average",
            f"{avg_points:.1f}",
            delta=f"{latest['points_vs_average']:+.1f}",
        )
    else:
        col2.metric("Global average", "N/A")

    highest_points = latest.get("highest_score")
    if pd.notna(highest_points):
        col3.metric(
            "Top manager score",
            f"{highest_points:.1f}",
            delta=f"{latest['points_vs_highest']:+.1f}",
        )
    else:
        col3.metric("Top manager score", "N/A")

    overall_rank = latest.get("overall_rank")
    rank_delta = latest.get("overall_rank_delta")
    delta_str = None
    if pd.notna(rank_delta):
        delta_str = f"{rank_delta:+.0f}"
    col4.metric(
        "Overall rank",
        _format_int(overall_rank),
        delta=delta_str,
    )

    avg_delta = latest["points"] - season_avg_points
    col5.metric(
        "Season avg points",
        f"{season_avg_points:.1f}",
        delta=f"{avg_delta:+.1f}",
    )

    st.caption(
        f"Total points: {_format_int(latest.get('total_points'))}  •  "
        f"Transfers used: {int(latest.get('event_transfers', 0))}  •  "
        f"Hits taken: {int(latest.get('event_transfers_cost', 0))}"
    )


def _render_history_charts(history_df: pd.DataFrame) -> None:
    point_color = "#4C78A8"
    average_color = "#F58518"
    rank_color = "#54A24B"

    points_fig = go.Figure()
    points_fig.add_trace(
        go.Scatter(
            x=history_df["event"],
            y=history_df["points"],
            mode="lines+markers",
            name="Your points",
            line=dict(color=point_color, width=3),
            marker=dict(size=8, symbol="circle", line=dict(color="white", width=1)),
            hovertemplate="GW %{x}<br>Your points: %{y:.1f}<extra></extra>",
        )
    )

    if history_df["average_entry_score"].notna().any():
        points_fig.add_trace(
            go.Scatter(
                x=history_df["event"],
                y=history_df["average_entry_score"],
                mode="lines",
                name="Global average",
                line=dict(color=average_color, width=2, dash="dash"),
                hovertemplate="GW %{x}<br>Global average: %{y:.1f}<extra></extra>",
            )
        )

    points_fig.update_layout(
        title="Points vs global average",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
        xaxis=dict(title="Gameweek", showgrid=False),
        yaxis=dict(title="Points", gridcolor="rgba(200,200,200,0.25)"),
        margin=dict(l=40, r=20, t=70, b=40),
    )
    st.plotly_chart(points_fig, use_container_width=True)

    if "overall_rank" in history_df.columns and history_df["overall_rank"].notna().any():
        rank_fig = go.Figure()
        rank_fig.add_trace(
            go.Scatter(
                x=history_df["event"],
                y=history_df["overall_rank"],
                mode="lines+markers",
                name="Overall rank",
                line=dict(color=rank_color, width=3),
                fill="tozeroy",
                fillcolor="rgba(84,162,75,0.15)",
                marker=dict(size=7, symbol="square", line=dict(color="white", width=1)),
                hovertemplate="GW %{x}<br>Overall rank: %{y:,}<extra></extra>",
            )
        )
        rank_fig.update_layout(
            title="Overall rank trend",
            template="plotly_white",
            hovermode="x unified",
            showlegend=False,
            xaxis=dict(title="Gameweek", showgrid=False),
            yaxis=dict(
                title="Overall rank",
                autorange="reversed",
                gridcolor="rgba(200,200,200,0.25)",
            ),
            margin=dict(l=40, r=20, t=70, b=40),
        )
        st.plotly_chart(rank_fig, use_container_width=True)


def _render_history_table(history_df: pd.DataFrame) -> None:
    table_df = history_df[
        [
            "event",
            "points",
            "average_entry_score",
            "points_vs_average",
            "points_vs_highest",
            "season_points_average",
            "overall_rank",
            "total_points",
            "value",
            "event_transfers",
            "event_transfers_cost",
            "points_on_bench",
        ]
    ].rename(
        columns={
            "event": "GW",
            "points": "Points",
            "average_entry_score": "Global avg",
            "points_vs_average": "Vs avg",
            "points_vs_highest": "Vs top score",
            "season_points_average": "Season avg",
            "overall_rank": "Overall rank",
            "total_points": "Total points",
            "value": "Team value",
            "event_transfers": "Transfers",
            "event_transfers_cost": "Hits",
            "points_on_bench": "Bench points",
        }
    )
    table_df["Overall rank"] = table_df["Overall rank"].apply(_format_int)
    table_df["Total points"] = table_df["Total points"].apply(_format_int)
    table_df["Season avg"] = table_df["Season avg"].apply(
        lambda value: f"{float(value):.1f}" if pd.notna(value) else "N/A"
    )
    table_df["Team value"] = table_df["Team value"].apply(
        lambda value: f"£{float(value):.1f}m" if pd.notna(value) else "N/A"
    )
    st.subheader("Gameweek history")
    st.dataframe(table_df, use_container_width=True)


def _build_player_points_table(
    fpl_id: int,
    gameweek: int,
    element_lookup: Dict[int, Dict[str, object]],
    team_lookup: Dict[int, str],
    actual_points: Dict[int, float],
) -> Tuple[pd.DataFrame, List[str]]:
    try:
        picks = _fetch_entry_picks(fpl_id, gameweek)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Unable to load picks for team {fpl_id}: {exc}")
        return pd.DataFrame(), []

    records: List[Dict[str, object]] = []
    weakness_notes: List[str] = []
    for pick in picks:
        player_id = int(pick.get("element", 0))
        multiplier = int(pick.get("multiplier", 0))
        is_captain = bool(pick.get("is_captain", False))
        is_vice = bool(pick.get("is_vice_captain", False))

        element_meta = element_lookup.get(player_id, {})
        team_id = int(element_meta.get("team", 0)) if element_meta else 0
        player_name = element_meta.get("web_name") or element_meta.get("second_name") or f"Player {player_id}"
        position_id = int(element_meta.get("element_type", 0)) if element_meta else 0
        position_label = POSITION_LABELS.get(position_id, "N/A")
        club_name = team_lookup.get(team_id, "Unknown")

        actual = float(actual_points.get(player_id, 0.0))
        role_parts: List[str] = []
        if multiplier > 0:
            role_parts.append("Starter")
            if multiplier > 1:
                role_parts.append(f"x{multiplier}")
        else:
            role_parts.append("Bench")
        if is_captain:
            role_parts.append("C")
        elif is_vice:
            role_parts.append("VC")
        role_label = " ".join(role_parts)

        note = ""
        flag_style = ""
        if multiplier > 0 and actual <= 2.0:
            note = "🔻 Low impact"
            flag_style = "negative"
            weakness_notes.append(f"{player_name} ({actual:.1f} pts) underperformed in the XI.")
        elif multiplier == 0 and actual >= 6.0:
            note = "❗ Bench haul"
            flag_style = "warning"
            weakness_notes.append(f"{player_name} ({actual:.1f} pts) left on the bench.")

        records.append(
            {
                "Player": player_name,
                "Pos": position_label,
                "Club": club_name,
                "Role": role_label,
                "Actual points": round(actual, 1),
                "Notes": note,
                "__flag_style": flag_style,
            }
        )

    if not records:
        return pd.DataFrame(), weakness_notes

    df = pd.DataFrame(records)
    df = df.sort_values(
        by=["Role", "Actual points", "Player"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    display_columns = ["Player", "Pos", "Club", "Role", "Actual points", "Notes", "__flag_style"]
    df = df[display_columns]

    return df, weakness_notes


def _style_player_points_table(df: pd.DataFrame):
    if df.empty:
        return df

    if "__flag_style" in df.columns:
        flag_styles = df["__flag_style"].tolist()
        display_df = df.drop(columns="__flag_style").copy()
    else:
        flag_styles = ["" for _ in range(len(df))]
        display_df = df.copy()

    def _highlight(row: pd.Series) -> List[str]:
        flag_style = flag_styles[row.name]
        if flag_style == "negative":
            color = DANGER_ROW_BG
            border = DANGER_ROW_BORDER
        elif flag_style == "warning":
            color = WARNING_ROW_BG
            border = WARNING_ROW_BORDER
        else:
            return [""] * len(row)
        return [
            f"background-color: {color}; border-left: 3px solid {border};"
            for _ in row
        ]

    styler = (
        display_df.style.apply(_highlight, axis=1)
        .format(subset=["Actual points"], formatter="{:.1f}")
        .format(
            subset=["Notes"],
            formatter=lambda value: value if value else "",
        )
        .hide(axis="index")
    )
    styler = _apply_base_table_style(styler)
    return styler


def _style_generic_table(
    df: pd.DataFrame,
    *,
    formatters: Optional[Dict[str, Callable[[object], str]]] = None,
    highlight_column: Optional[str] = None,
    highlight_color: str = PRIMARY_ROW_BG,
    highlight_border: str = PRIMARY_ROW_BORDER,
) -> pd.io.formats.style.Styler:
    def _highlight(row: pd.Series) -> List[str]:
        if highlight_column and bool(row.get(highlight_column)):
            return [
                f"background-color: {highlight_color}; border-left: 3px solid {highlight_border};"
                for _ in row
            ]
        return [""] * len(row)

    styler = df.style
    if highlight_column:
        styler = styler.apply(_highlight, axis=1)
    if formatters:
        styler = styler.format(formatters)
    styler = styler.hide(axis="index")
    styler = _apply_base_table_style(styler)
    return styler


def _format_optional_int(value: Optional[object]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def _format_optional_signed_int(value: Optional[object]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):+d}"


def _blank_if_empty(value: Optional[object]) -> str:
    if value is None:
        return ""
    value_str = str(value)
    return value_str if value_str.lower() != "nan" else ""


def _render_mini_leagues(
    fpl_id: int,
    overview: Dict[str, object],
    user_history: pd.DataFrame,
    events_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    load_actual_points_for_gw: Callable[[int], Dict[int, float]],
) -> None:
    leagues = overview.get("leagues", {})
    classic_leagues = leagues.get("classic", []) if isinstance(leagues, dict) else []

    st.subheader("Mini leagues")
    if not classic_leagues:
        st.info("No classic mini leagues found for this team.")
        return

    if user_history is None or user_history.empty:
        st.info("Your gameweek history is required to compare against league opponents.")
        return

    element_lookup: Dict[int, Dict[str, object]] = {}
    if elements_df is not None and not elements_df.empty:
        element_lookup = (
            elements_df[
                ["id", "web_name", "second_name", "team", "element_type"]
            ]
            .rename(columns={"id": "player_id"})
            .set_index("player_id")
            .to_dict("index")
        )

    team_lookup: Dict[int, str] = {}
    if teams_df is not None and not teams_df.empty:
        name_col = "short_name" if "short_name" in teams_df.columns else "name"
        team_lookup = teams_df.set_index("id")[name_col].to_dict()

    summary_df = pd.DataFrame(classic_leagues)
    summary_df = summary_df.rename(
        columns={
            "name": "League",
            "entry_rank": "Current rank",
            "entry_last_rank": "Previous rank",
        }
    )
    for col in ("Current rank", "Previous rank"):
        summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
    summary_df["Movement"] = summary_df["Previous rank"] - summary_df["Current rank"]
    summary_df["Movement"] = summary_df["Movement"].fillna(0).astype(int)

    summary_view = summary_df[["League", "Current rank", "Previous rank", "Movement"]].copy()

    summary_style = _style_generic_table(
        summary_view,
        formatters={
            "Current rank": _format_optional_int,
            "Previous rank": _format_optional_int,
            "Movement": _format_optional_signed_int,
        },
    )
    st.dataframe(summary_style, use_container_width=True)

    league_lookup: Dict[str, int] = {}
    for _, row in summary_df.iterrows():
        rank_value = row.get("Current rank")
        if pd.isna(rank_value):
            rank_label = "N/A"
        else:
            rank_label = str(int(rank_value))
        league_lookup[f"{row['League']} (Rank {rank_label})"] = int(row["id"])
    selection = st.selectbox(
        "Inspect standings",
        options=list(league_lookup.keys()),
        key="team_perf_league_select",
    )
    selected_league_id = league_lookup.get(selection)
    if selected_league_id is None:
        return
    try:
        standings = _fetch_classic_league_standings(selected_league_id)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return
    if not standings:
        st.info("No standings data returned for the selected league.")
        return
    standings_df = pd.DataFrame(standings)
    standings_df = standings_df.rename(
        columns={
            "entry_name": "Team",
            "player_name": "Manager",
            "rank": "Rank",
            "total": "Total points",
            "event_total": "GW points",
        }
    )
    entry_ids = None
    if "entry" in standings_df.columns:
        entry_ids = pd.to_numeric(standings_df["entry"], errors="coerce").astype("Int64")
        standings_df["entry"] = entry_ids

    standings_df["Rank"] = pd.to_numeric(standings_df["Rank"], errors="coerce").astype("Int64")
    standings_df = standings_df.sort_values("Rank").reset_index(drop=True)
    standings_df["total_points_raw"] = pd.to_numeric(standings_df["Total points"], errors="coerce")
    standings_df["gw_points_raw"] = pd.to_numeric(standings_df["GW points"], errors="coerce")

    if entry_ids is not None:
        standings_df["Your team"] = standings_df["entry"] == fpl_id
    else:
        standings_df["Your team"] = False

    display_df = standings_df[
        ["Rank", "Team", "Manager", "gw_points_raw", "total_points_raw", "Your team"]
    ].rename(
        columns={
            "gw_points_raw": "GW points",
            "total_points_raw": "Total points",
        }
    ).copy()

    league_table_style = _style_generic_table(
        display_df,
        formatters={
            "Rank": _format_optional_int,
            "GW points": lambda value: "" if value is None or pd.isna(value) else f"{int(value):,}",
            "Total points": lambda value: "" if value is None or pd.isna(value) else f"{int(value):,}",
            "Your team": lambda flag: "Yes" if bool(flag) else "",
        },
        highlight_column="Your team",
    )

    st.dataframe(league_table_style, use_container_width=True)

    leader_row = standings_df.iloc[0]
    leader_points = leader_row["total_points_raw"]
    user_league_row = standings_df.loc[standings_df["Your team"]]
    if not user_league_row.empty and pd.notna(leader_points):
        user_total = float(user_league_row.iloc[0]["total_points_raw"])
        diff_vs_leader = user_total - float(leader_points)
        st.caption(f"Points vs league leader: {diff_vs_leader:+,.0f}")

    if entry_ids is None or standings_df["entry"].isna().all():
        st.info("Opponent comparison unavailable for this league (missing entry identifiers).")
        return

    opponent_options: List[Tuple[str, int]] = []
    for _, row in standings_df.iterrows():
        entry_id = row.get("entry")
        if pd.isna(entry_id):
            continue
        rank_value = row.get("Rank")
        rank_label = f"Rank {int(rank_value)}" if pd.notna(rank_value) else "Unranked"
        team_label = f"{row['Team']} – {row['Manager']} ({rank_label}, ID {int(entry_id)})"
        opponent_options.append((team_label, int(entry_id)))

    if not opponent_options:
        st.info("No opponent data available for comparison.")
        return

    opponent_lookup = {label: entry for label, entry in opponent_options}
    opponent_labels = list(opponent_lookup.keys())
    default_index = 0
    for idx, entry_id in enumerate(opponent_lookup.values()):
        if entry_id != fpl_id:
            default_index = idx
            break

    opponent_select_key = f"team_perf_opponent_{selected_league_id}"
    selected_opponent_label = st.selectbox(
        "Opponent team",
        options=opponent_labels,
        index=min(default_index, len(opponent_labels) - 1),
        key=opponent_select_key,
    )
    opponent_entry_id = opponent_lookup[selected_opponent_label]

    try:
        opponent_history_raw = _fetch_entry_history(opponent_entry_id)
        opponent_history = _prepare_history_df(opponent_history_raw, events_df)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Unable to load opponent history: {exc}")
        return

    available_gws = sorted(
        set(user_history["event"].tolist()).intersection(opponent_history["event"].tolist())
    )
    if not available_gws:
        available_gws = sorted(user_history["event"].tolist())
    if not available_gws:
        st.info("No overlapping gameweeks available for comparison.")
        return

    gw_select_key = f"team_perf_gw_{selected_league_id}_{opponent_entry_id}"
    gw_index = max(len(available_gws) - 1, 0)
    selected_gw = st.selectbox(
        "Gameweek to review",
        options=available_gws,
        index=gw_index,
        key=gw_select_key,
        format_func=lambda gw: f"GW {gw}",
    )

    user_gw_row = user_history[user_history["event"] == selected_gw]
    opponent_gw_row = opponent_history[opponent_history["event"] == selected_gw]
    if user_gw_row.empty or opponent_gw_row.empty:
        st.info("Selected gameweek data is unavailable for one of the teams.")
        return

    user_gw = user_gw_row.iloc[0]
    opponent_gw = opponent_gw_row.iloc[0]

    def _safe_float(value: Optional[object]) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    user_gw_points = _safe_float(user_gw.get("points"))
    opponent_gw_points = _safe_float(opponent_gw.get("points"))
    if user_gw_points is None or opponent_gw_points is None:
        st.info("Missing gameweek points data for one of the teams.")
        return

    gw_points_diff = user_gw_points - opponent_gw_points

    user_total_points = _safe_float(user_gw.get("total_points"))
    opponent_total_points = _safe_float(opponent_gw.get("total_points"))
    total_diff: Optional[float] = None
    if user_total_points is not None and opponent_total_points is not None:
        total_diff = user_total_points - opponent_total_points

    cols = st.columns(4)
    cols[0].metric("Your GW points", f"{user_gw_points:.1f}")
    cols[1].metric("Opponent GW points", f"{opponent_gw_points:.1f}")
    cols[2].metric("GW points diff", f"{gw_points_diff:+.1f}")
    cols[3].metric(
        "Season total diff",
        f"{total_diff:+,.0f}" if total_diff is not None else "N/A",
    )

    def _format_signed(value: Optional[float]) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):+.1f}"

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "GW points",
                "Points vs global avg",
                "Season total points",
                "Overall rank",
                "Season avg points",
                "Points on bench",
            ],
            "Your team": [
                f"{user_gw_points:.1f}",
                _format_signed(user_gw.get("points_vs_average")),
                _format_int(user_gw.get("total_points")),
                _format_int(user_gw.get("overall_rank")),
                f"{float(user_gw.get('season_points_average', 0.0)):.1f}"
                if pd.notna(user_gw.get("season_points_average"))
                else "N/A",
                f"{float(user_gw.get('points_on_bench', 0.0)):.1f}"
                if pd.notna(user_gw.get("points_on_bench"))
                else "N/A",
            ],
            selected_opponent_label: [
                f"{opponent_gw_points:.1f}",
                _format_signed(opponent_gw.get("points_vs_average")),
                _format_int(opponent_gw.get("total_points")),
                _format_int(opponent_gw.get("overall_rank")),
                f"{float(opponent_gw.get('season_points_average', 0.0)):.1f}"
                if pd.notna(opponent_gw.get("season_points_average"))
                else "N/A",
                f"{float(opponent_gw.get('points_on_bench', 0.0)):.1f}"
                if pd.notna(opponent_gw.get("points_on_bench"))
                else "N/A",
            ],
        }
    ).set_index("Metric")

    comparison_style = _style_generic_table(
        comparison_df,
        formatters={column: _blank_if_empty for column in comparison_df.columns},
    )
    st.dataframe(comparison_style, use_container_width=True)

    try:
        actual_points_map = load_actual_points_for_gw(selected_gw)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Unable to load player scores for GW {selected_gw}: {exc}")
        actual_points_map = {}

    user_players_df, user_flags = _build_player_points_table(
        fpl_id, selected_gw, element_lookup, team_lookup, actual_points_map
    )
    opponent_players_df, opponent_flags = _build_player_points_table(
        opponent_entry_id, selected_gw, element_lookup, team_lookup, actual_points_map
    )

    player_cols = st.columns(2)
    user_col, opponent_col = player_cols

    user_col.subheader(f"Your squad – GW {selected_gw}")
    if user_players_df.empty:
        user_col.info("No player detail available for your squad.")
    else:
        user_col.dataframe(_style_player_points_table(user_players_df), use_container_width=True)
        if user_flags:
            user_col.caption("Weak spots: " + "; ".join(user_flags))

    opponent_col.subheader("Opponent squad")
    if opponent_players_df.empty:
        opponent_col.info("No player detail available for the opponent.")
    else:
        opponent_col.dataframe(_style_player_points_table(opponent_players_df), use_container_width=True)
        if opponent_flags:
            opponent_col.caption("Weak spots: " + "; ".join(opponent_flags))

    caption_parts = []
    caption_parts.append(
        f"GW {selected_gw}: {gw_points_diff:+.1f} point swing in your favour."
        if gw_points_diff
        else f"GW {selected_gw}: level on points."
    )
    if total_diff is not None:
        caption_parts.append(f"Season total advantage: {total_diff:+,.0f} points.")
    if caption_parts:
        st.caption(" ".join(caption_parts))


def render_team_performance_page(
    deps: TeamPerformanceDependencies, *, default_fpl_id: Optional[int] = None
) -> Optional[int]:
    st.header("Team Points Comparison")
    st.markdown(
        "Review how your team has performed each gameweek, benchmarked against the global "
        "average and your classic mini leagues. Enter your FPL ID to begin."
    )

    if default_fpl_id is not None:
        default_value = str(default_fpl_id)
        if (
            "team_perf_fpl_id" not in st.session_state
            or st.session_state["team_perf_fpl_id"] != default_value
        ):
            st.session_state["team_perf_fpl_id"] = default_value

    fpl_id_value = st.text_input(
        "FPL team ID",
        key="team_perf_fpl_id",
        placeholder="e.g. 1234567",
    )
    if not fpl_id_value:
        st.info("Provide your FPL ID to load history and mini league standings.")
        return None

    try:
        fpl_id = int(fpl_id_value.strip())
    except ValueError:
        st.error("FPL ID must be a whole number.")
        return None

    try:
        events_df = deps.load_bootstrap_events()
    except Exception as exc:  # noqa: BLE001
        events_df = pd.DataFrame()
        st.warning(f"Unable to load bootstrap events data: {exc}")

    try:
        history = _fetch_entry_history(fpl_id)
        history_df = _prepare_history_df(history, events_df)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    _render_summary(history_df)
    _render_history_charts(history_df)
    _render_history_table(history_df)

    try:
        elements_df = deps.load_bootstrap_elements_df()
    except Exception:  # noqa: BLE001
        elements_df = pd.DataFrame()

    try:
        teams_df = deps.load_bootstrap_teams_df()
    except Exception:  # noqa: BLE001
        teams_df = pd.DataFrame()

    try:
        overview = _fetch_entry_overview(fpl_id)
        _render_mini_leagues(
            fpl_id,
            overview,
            history_df,
            events_df,
            elements_df,
            teams_df,
            deps.load_actual_points_for_gw,
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Unable to load mini league information: {exc}")

    last_finished = deps.last_finished_gameweek()
    if last_finished:
        st.caption(f"Last finished gameweek detected: GW {last_finished}.")

    return fpl_id


__all__ = [
    "TeamPerformanceDependencies",
    "render_team_performance_page",
]
