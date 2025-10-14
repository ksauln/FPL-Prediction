from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .config import FORMATION, OUTPUTS_DIR

# Position metadata reused by the optimiser and renderer
POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_ROWS = ["GK", "DEF", "MID", "FWD"]
POSITION_COLOURS = {
    "GK": "#1B5E20",   # dark green
    "DEF": "#0D47A1",  # blue
    "MID": "#EF6C00",  # orange
    "FWD": "#B71C1C",  # red
}


def _to_dataframe(squad: Iterable[Mapping]) -> pd.DataFrame:
    """Convert the squad list/dict to a tidy DataFrame with pos_name column."""
    df = squad if isinstance(squad, pd.DataFrame) else pd.DataFrame(list(squad))
    if df.empty:
        raise ValueError("Squad is empty; nothing to display.")
    if "element_type" not in df.columns:
        raise KeyError("Squad entries must include 'element_type'.")
    df = df.copy()
    df["pos_name"] = df["element_type"].map(POS_MAP)
    return df


def _draw_pitch(ax: plt.Axes) -> None:
    """Draw a simple football pitch background."""
    ax.set_facecolor("#2E7D32")
    # Outer lines
    ax.plot([5, 95, 95, 5, 5], [5, 5, 95, 95, 5], color="white", linewidth=2)
    # Halfway line and centre circle
    ax.plot([5, 95], [50, 50], color="white", linewidth=1.5)
    centre_circle = plt.Circle((50, 50), 9.15, edgecolor="white", facecolor="none", linewidth=1.5)
    ax.add_patch(centre_circle)
    ax.scatter(50, 50, color="white", s=20)
    # Penalty boxes and six-yard boxes
    left_penalty = plt.Rectangle((5, 26), 16, 48, linewidth=1.5, edgecolor="white", facecolor="none")
    right_penalty = plt.Rectangle((79, 26), 16, 48, linewidth=1.5, edgecolor="white", facecolor="none")
    left_six = plt.Rectangle((5, 38), 6, 24, linewidth=1.2, edgecolor="white", facecolor="none")
    right_six = plt.Rectangle((89, 38), 6, 24, linewidth=1.2, edgecolor="white", facecolor="none")
    ax.add_patch(left_penalty)
    ax.add_patch(right_penalty)
    ax.add_patch(left_six)
    ax.add_patch(right_six)
    # Penalty spots
    ax.scatter([16], [50], color="white", s=15)
    ax.scatter([84], [50], color="white", s=15)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")


def _player_positions(formation: Mapping[str, int]) -> Mapping[str, Sequence[float]]:
    """Return evenly spaced x,y coordinates per position row."""
    y_coords = {"GK": 12, "DEF": 32, "MID": 54, "FWD": 76}
    pos_xy = {}
    for pos in POSITION_ROWS:
        count = formation.get(pos, 0)
        if count <= 0:
            continue
        xs = [100 * (i + 1) / (count + 1) for i in range(count)]
        ys = [y_coords[pos]] * count
        pos_xy[pos] = list(zip(xs, ys))
    return pos_xy


def _format_label(player: Mapping, mark_captain: bool) -> str:
    name = player.get("full_name", "Unknown")
    team = player.get("team_name", "")
    cost = player.get("now_cost_millions")
    ep = player.get("expected_points")
    captain_suffix = " (C)" if mark_captain else ""
    cost_part = f"Cost: £{cost:.1f}m" if isinstance(cost, (int, float)) else "Cost: N/A"
    ep_part = f"Predicted: {ep:.2f} pts" if isinstance(ep, (int, float)) else "Predicted: N/A"
    if isinstance(team, str) and team:
        team_part = f"Team: {team}"
    else:
        team_part = "Team: Unknown"
    return f"{name}{captain_suffix}\n{team_part}\n{cost_part}\n{ep_part}"


def create_best_xi_graphic(
    team: Mapping,
    gameweek: Optional[int] = None,
    output_path: Optional[Path] = None,
    formation: Optional[Mapping[str, int]] = None,
    show: bool = False,
) -> Path:
    """
    Render the best XI squad graphic and save it to disk.

    Parameters
    ----------
    team : Mapping
        Output from pick_best_xi containing a "squad" key with player dicts.
    gameweek : Optional[int]
        Gameweek number to include in the title (if provided).
    output_path : Optional[Path]
        Where to save the generated image. Defaults to OUTPUTS_DIR/best_xi_gw{gameweek}.png.
    formation : Optional[Mapping[str, int]]
        Formation specification, defaults to config.FORMATION.
    show : bool
        When True, display the figure interactively (useful in notebooks).
    """
    if "squad" not in team:
        raise KeyError("Team mapping must include a 'squad' key.")

    squad_df = _to_dataframe(team["squad"])
    bench_df = None
    if team.get("bench"):
        bench_df = _to_dataframe(team["bench"])
        if "bench_order" in bench_df.columns:
            bench_df = bench_df.sort_values("bench_order")
        else:
            bench_df = bench_df.sort_values("expected_points", ascending=False)
    formation = formation or team.get("formation") or FORMATION
    pos_xy = _player_positions(formation)

    has_bench = bench_df is not None
    if has_bench:
        fig = plt.figure(figsize=(12, 9))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
        ax = fig.add_subplot(gs[0])
        bench_ax = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        bench_ax = None

    _draw_pitch(ax)

    # Add title and summary details
    total_ep = team.get("total_expected_points_with_captain")
    starting_cost = team.get("starting_cost")
    bench_cost = team.get("bench_cost")
    total_cost = team.get("total_cost")
    captain = team.get("captain")
    title_parts = ["Best XI"]
    if gameweek is not None:
        title_parts.append(f"GW{gameweek}")
    ax.set_title(" - ".join(title_parts), fontsize=22, color="white", pad=20)
    subtitle_parts = []
    if total_ep is not None:
        subtitle_parts.append(f"XI EP (C): {total_ep:.2f}")
    if starting_cost is not None:
        subtitle_parts.append(f"XI Cost: £{starting_cost:.1f}m")
    if bench_cost is not None:
        subtitle_parts.append(f"Bench Cost: £{bench_cost:.1f}m")
    elif total_cost is not None:
        subtitle_parts.append(f"Cost: £{total_cost:.1f}m")
    formation_name = team.get("formation_name")
    if not formation_name and formation:
        formation_name = f"{formation.get('DEF', 0)}-{formation.get('MID', 0)}-{formation.get('FWD', 0)}"
    if formation_name:
        subtitle_parts.append(f"Formation: {formation_name}")
    if captain:
        subtitle_parts.append(f"Captain: {captain}")
    if subtitle_parts:
        ax.text(50, 96, " | ".join(subtitle_parts), ha="center", va="center", color="white", fontsize=14)

    # Plot players
    for pos in POSITION_ROWS:
        players = squad_df[squad_df["pos_name"] == pos]
        coords = pos_xy.get(pos, [])
        if not len(players) or not coords:
            continue
        for (x, y), (_, player) in zip(coords, players.iterrows()):
            is_captain = bool(player.get("captain", 0))
            colour = POSITION_COLOURS.get(pos, "#FFFFFF")
            edge_colour = "#FFD600" if is_captain else "white"
            ax.scatter(x, y, s=1800, color=colour, edgecolor=edge_colour, linewidth=3, alpha=0.9, zorder=3)
            ax.text(
                x,
                y + 12,
                _format_label(player, is_captain),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold" if is_captain else "normal",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#000000", alpha=0.6, edgecolor="none"),
                linespacing=1.3,
                zorder=4,
            )

    if bench_ax is not None:
        bench_ax.set_facecolor("#1B5E20")
        bench_ax.set_xlim(0, 100)
        bench_ax.set_ylim(0, 100)
        bench_ax.axis("off")
        bench_title = ["Bench"]
        bench_ep = team.get("bench_expected_points")
        if bench_ep is not None:
            bench_title.append(f"EP: {bench_ep:.2f}")
        if total_cost is not None and bench_cost is not None:
            bench_title.append(f"Total Cost: £{total_cost:.1f}m")
        bench_ax.text(
            50,
            90,
            " | ".join(bench_title),
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        bench_count = len(bench_df)
        if bench_count:
            xs = [100 * (i + 1) / (bench_count + 1) for i in range(bench_count)]
            for idx, ((_, player), x) in enumerate(zip(bench_df.iterrows(), xs)):
                order = int(player.get("bench_order", idx + 1))
                name = player.get("full_name", "Unknown")
                pos = player.get("pos_name", "N/A")
                team_name = player.get("team_name", "Unknown")
                cost = player.get("now_cost_millions")
                ep = player.get("expected_points")
                cost_txt = f"£{cost:.1f}m" if isinstance(cost, (int, float)) else "£?"
                ep_txt = f"{ep:.2f} pts" if isinstance(ep, (int, float)) else "?"
                colour = POSITION_COLOURS.get(pos, "#546E7A")
                bench_ax.scatter(
                    x,
                    35,
                    s=1200,
                    color=colour,
                    edgecolor="white",
                    linewidth=2,
                    alpha=0.9,
                )
                bench_ax.text(
                    x,
                    35,
                    str(order),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=14,
                    fontweight="bold",
                    zorder=3,
                )
                bench_ax.text(
                    x,
                    68,
                    f"{name}\n{pos} — {team_name}\n{cost_txt} | {ep_txt}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.35",
                        facecolor="#000000",
                        alpha=0.6,
                        edgecolor="none",
                    ),
                    linespacing=1.2,
                )

    # Save figure
    if gameweek is None and output_path is None:
        raise ValueError("Provide either gameweek or output_path so the image can be named.")
    if output_path is None:
        output_path = OUTPUTS_DIR / f"best_xi_gw{gameweek}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    return output_path


__all__ = ["create_best_xi_graphic"]
