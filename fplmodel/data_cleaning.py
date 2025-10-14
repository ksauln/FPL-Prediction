from __future__ import annotations
import pandas as pd
from typing import Dict, Any

def normalize_bootstrap(bootstrap: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    From bootstrap-static JSON, create tidy dataframes for elements, teams, events.
    """
    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    events = pd.DataFrame(bootstrap["events"])
    # Select/rename a few common fields
    elements = elements.rename(columns={
        "id": "player_id",
        "team": "team_id",
        "now_cost": "cost_tenths",
        "element_type": "element_type"
    })
    # Helpful human labels
    elements["full_name"] = elements["first_name"] + " " + elements["second_name"]
    # Costs in millions
    elements["now_cost_millions"] = elements["cost_tenths"] / 10.0

    teams = teams.rename(columns={"id":"team_id"})
    events = events.rename(columns={"id":"event_id"})

    return {"elements": elements, "teams": teams, "events": events}

def histories_to_df(player_histories: Dict[int, Any]) -> pd.DataFrame:
    """
    Combine per-player 'history' arrays into one tidy DF.
    Returns columns including: player_id, round (gw), total_points, minutes, was_home, opponent_team, kickoff_time, etc.
    """
    rows = []
    for pid, payload in player_histories.items():
        for h in payload.get("history", []):
            r = dict(h)
            r["player_id"] = pid
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Coerce types
    if "round" in df.columns:
        df["round"] = df["round"].astype(int)
    # Sometimes kickoff_time can be null for very old data
    return df
