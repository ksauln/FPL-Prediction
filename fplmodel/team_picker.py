from __future__ import annotations
from typing import Dict
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatusOptimal

from .config import BUDGET_MILLIONS, FORMATION, MAX_PER_TEAM

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

def pick_best_xi(pred_df: pd.DataFrame, budget_m: float = BUDGET_MILLIONS, formation: Dict[str, int] = FORMATION):
    """
    ILP: pick best XI under budget with formation and <=3 per team constraints.
    Also selects a captain (one of the XI) to maximize EP (captain doubles).
    pred_df must include: player_id, team_id, element_type, now_cost_millions, expected_points, full_name
    Returns dict with selected players, total cost, expected points (including captain boost), and captain.
    """
    df = pred_df.copy()
    df["pos_name"] = df["element_type"].map(POS_MAP)
    # Decision variables
    x = {pid: LpVariable(f"x_{pid}", lowBound=0, upBound=1, cat=LpBinary) for pid in df["player_id"]}
    c = {pid: LpVariable(f"c_{pid}", lowBound=0, upBound=1, cat=LpBinary) for pid in df["player_id"]}
    prob = LpProblem("FPL_Best_XI", LpMaximize)

    # Objective: maximize EP + captain's extra EP (i.e., + ep again)
    ep_map = df.set_index("player_id")["expected_points"].to_dict()
    prob += lpSum([x[pid]*ep_map[pid] for pid in ep_map]) + lpSum([c[pid]*ep_map[pid] for pid in ep_map])

    # Exactly 11 players
    prob += lpSum([x[pid] for pid in ep_map]) == 11

    # Captain must be one of the selected players; exactly 1 captain
    prob += lpSum([c[pid] for pid in ep_map]) == 1
    for pid in ep_map:
        prob += c[pid] <= x[pid]

    # Formation constraints
    for pos, need in formation.items():
        pids = df[df["pos_name"] == pos]["player_id"].tolist()
        prob += lpSum([x[pid] for pid in pids]) == need

    # Team constraint
    for team_id, grp in df.groupby("team_id"):
        pids = grp["player_id"].tolist()
        prob += lpSum([x[pid] for pid in pids]) <= MAX_PER_TEAM

    # Budget
    cost_map = df.set_index("player_id")["now_cost_millions"].to_dict()
    prob += lpSum([x[pid]*cost_map[pid] for pid in ep_map]) <= budget_m

    status = prob.solve()
    if status != LpStatusOptimal:
        raise RuntimeError("No optimal XI found. Try adjusting budget or formation.")

    selected = df[[ "player_id","full_name","team_id","element_type","now_cost_millions","expected_points"]].copy()
    selected["selected"] = selected["player_id"].apply(lambda pid: int(x[pid].value() or 0))
    selected["captain"] = selected["player_id"].apply(lambda pid: int(c[pid].value() or 0))
    squad = selected[selected["selected"] == 1].sort_values("element_type")
    total_cost = squad["now_cost_millions"].sum()
    base_ep = squad["expected_points"].sum()
    cap_ep = (squad[squad["captain"] == 1]["expected_points"].sum() if (squad["captain"] == 1).any() else 0.0)
    total_ep = base_ep + cap_ep

    result = {
        "squad": squad.to_dict(orient="records"),
        "total_cost": float(total_cost),
        "expected_points_without_captain": float(base_ep),
        "total_expected_points_with_captain": float(total_ep),
        "captain": squad.loc[squad["captain"] == 1, "full_name"].iloc[0] if (squad["captain"] == 1).any() else None,
    }
    return result
