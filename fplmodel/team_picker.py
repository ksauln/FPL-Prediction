from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd
from pulp import LpBinary, LpMaximize, LpProblem, LpStatusOptimal, LpVariable, lpSum

from .config import (
    BENCH_GK_COUNT,
    BENCH_SIZE,
    BUDGET_MILLIONS,
    FORMATION,
    MAX_PER_TEAM,
    SQUAD_POSITION_LIMITS,
)

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _solve_for_formation(
    df: pd.DataFrame,
    budget_m: float,
    formation: Dict[str, int],
) -> Dict[str, object]:
    """Solve the ILP for a specific formation."""
    # Decision variables
    start_vars = {
        pid: LpVariable(f"x_{pid}", lowBound=0, upBound=1, cat=LpBinary)
        for pid in df["player_id"]
    }
    bench_vars = {
        pid: LpVariable(f"b_{pid}", lowBound=0, upBound=1, cat=LpBinary)
        for pid in df["player_id"]
    }
    captain_vars = {
        pid: LpVariable(f"c_{pid}", lowBound=0, upBound=1, cat=LpBinary)
        for pid in df["player_id"]
    }
    prob = LpProblem("FPL_Best_Squad", LpMaximize)

    # Objective: starting XI EP + captain's double (bench does not score)
    ep_map = df.set_index("player_id")["expected_points"].to_dict()
    prob += lpSum(start_vars[pid] * ep_map[pid] for pid in ep_map) + lpSum(
        captain_vars[pid] * ep_map[pid] for pid in ep_map
    )

    # Exactly 11 starters based on formation
    desired_xi = sum(formation.values())
    total_slots = sum(SQUAD_POSITION_LIMITS.values())
    if total_slots != desired_xi + BENCH_SIZE:
        raise ValueError(
            "Formation and squad constraints misaligned: "
            f"XI requires {desired_xi} slots but squad totals {total_slots} "
            f"with bench size {BENCH_SIZE}."
        )
    prob += lpSum(start_vars[pid] for pid in ep_map) == desired_xi

    # Bench size (includes reserve GK)
    prob += lpSum(bench_vars[pid] for pid in ep_map) == BENCH_SIZE

    # Captain must be a starter; exactly one captain
    prob += lpSum(captain_vars[pid] for pid in ep_map) == 1
    for pid in ep_map:
        prob += captain_vars[pid] <= start_vars[pid]
        prob += bench_vars[pid] + start_vars[pid] <= 1

    # Formation constraints for the XI
    for pos, need in formation.items():
        pids = df[df["pos_name"] == pos]["player_id"].tolist()
        prob += lpSum(start_vars[pid] for pid in pids) == need

    # Total squad composition (XI + bench)
    for pos, limit in SQUAD_POSITION_LIMITS.items():
        pids = df[df["pos_name"] == pos]["player_id"].tolist()
        if not pids:
            continue
        prob += lpSum(start_vars[pid] + bench_vars[pid] for pid in pids) == limit

    # Bench composition: enforce GK count and outfield balance
    gk_pids = df[df["pos_name"] == "GK"]["player_id"].tolist()
    if gk_pids:
        prob += lpSum(bench_vars[pid] for pid in gk_pids) == BENCH_GK_COUNT
    outfield_slots = BENCH_SIZE - BENCH_GK_COUNT
    if outfield_slots > 0:
        outfield_pids = df[df["pos_name"] != "GK"]["player_id"].tolist()
        prob += lpSum(bench_vars[pid] for pid in outfield_pids) == outfield_slots

    # Max players per team across full squad
    for team_id, grp in df.groupby("team_id"):
        pids = grp["player_id"].tolist()
        prob += lpSum(start_vars[pid] + bench_vars[pid] for pid in pids) <= MAX_PER_TEAM

    # Budget constraint for the full squad
    cost_map = df.set_index("player_id")["now_cost_millions"].to_dict()
    prob += lpSum(
        (start_vars[pid] + bench_vars[pid]) * cost_map[pid] for pid in ep_map
    ) <= budget_m

    status = prob.solve()
    if status != LpStatusOptimal:
        raise RuntimeError("No optimal XI found. Try adjusting budget or formation.")

    base_columns = [
        "player_id",
        "full_name",
        "team_name",
        "team_id",
        "element_type",
        "now_cost_millions",
        "expected_points",
    ]
    selected = df[base_columns].copy()
    selected["starting"] = selected["player_id"].apply(
        lambda pid: int(start_vars[pid].value() or 0)
    )
    selected["bench"] = selected["player_id"].apply(
        lambda pid: int(bench_vars[pid].value() or 0)
    )
    selected["captain"] = selected["player_id"].apply(
        lambda pid: int(captain_vars[pid].value() or 0)
    )

    squad = selected[selected["starting"] == 1].sort_values(
        ["element_type", "expected_points"], ascending=[True, False]
    )

    bench = selected[selected["bench"] == 1].copy()
    bench_outfield = bench[bench["element_type"] != 1].sort_values(
        "expected_points", ascending=False
    )
    bench_gk = bench[bench["element_type"] == 1].sort_values(
        "expected_points", ascending=False
    )
    bench_ordered = pd.concat([bench_outfield, bench_gk], ignore_index=True)
    bench_ordered["bench_order"] = bench_ordered.index + 1

    starting_cost = squad["now_cost_millions"].sum()
    bench_cost = bench_ordered["now_cost_millions"].sum()
    total_cost = starting_cost + bench_cost

    base_ep = squad["expected_points"].sum()
    cap_ep = (
        squad.loc[squad["captain"] == 1, "expected_points"].sum()
        if (squad["captain"] == 1).any()
        else 0.0
    )
    total_ep = base_ep + cap_ep

    result = {
        "squad": squad.to_dict(orient="records"),
        "bench": bench_ordered.to_dict(orient="records"),
        "total_cost": float(total_cost),
        "starting_cost": float(starting_cost),
        "bench_cost": float(bench_cost),
        "expected_points_without_captain": float(base_ep),
        "total_expected_points_with_captain": float(total_ep),
        "bench_expected_points": float(bench_ordered["expected_points"].sum()),
        "captain": squad.loc[squad["captain"] == 1, "full_name"].iloc[0]
        if (squad["captain"] == 1).any()
        else None,
    }
    return result


def pick_best_xi(
    pred_df: pd.DataFrame,
    budget_m: float = BUDGET_MILLIONS,
    formation: Optional[Dict[str, int]] = None,
    formations: Optional[Iterable[Dict[str, int]]] = None,
) -> Dict[str, object]:
    """
    ILP: pick best XI with bench under budget, formation, per-team, and squad constraints.

    Also selects a captain (one of the XI) to maximise EP (captain doubles).
    `pred_df` must include: player_id, team_id, team_name, element_type,
    now_cost_millions, expected_points, full_name.
    """
    if formation is not None and formations is not None:
        raise ValueError("Provide either formation or formations, not both.")

    if formations is None:
        formations = [formation or FORMATION]
    else:
        formations = list(formations)
        if not formations:
            raise ValueError("formations must be a non-empty iterable of dicts.")

    df = pred_df.copy()
    df["pos_name"] = df["element_type"].map(POS_MAP)

    best_result: Optional[Dict[str, object]] = None
    best_points = float("-inf")
    last_error: Optional[Exception] = None

    for current in formations:
        try:
            result = _solve_for_formation(df, budget_m, current)
        except RuntimeError as exc:
            last_error = exc
            continue

        current_points = result["total_expected_points_with_captain"]
        if current_points > best_points:
            best_points = current_points
            result["formation"] = current.copy()
            result["formation_name"] = f"{current.get('DEF', 0)}-{current.get('MID', 0)}-{current.get('FWD', 0)}"
            best_result = result

    if best_result is None:
        if last_error is not None:
            raise RuntimeError(
                f"Unable to find optimal XI for provided formations: {last_error}"
            ) from last_error
        raise RuntimeError("No formation produced a valid XI.")

    return best_result


__all__ = ["pick_best_xi"]
