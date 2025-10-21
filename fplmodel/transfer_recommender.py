"""Tools for recommending Fantasy Premier League transfers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

from .team_analysis import summarise_team
from .team_picker import pick_best_xi


def _ensure_expected_points(df: pd.DataFrame) -> pd.DataFrame:
    if "expected_points" not in df.columns:
        raise ValueError("DataFrame must contain an 'expected_points' column")
    return df


def aggregate_expected_points(
    predictions_by_gw: Dict[int, pd.DataFrame],
    gameweeks: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Aggregate expected points across multiple gameweeks for each player."""

    if not predictions_by_gw:
        raise ValueError("predictions_by_gw cannot be empty")

    if gameweeks is None:
        gameweeks = sorted(predictions_by_gw.keys())
    else:
        gameweeks = list(gameweeks)
        missing = [gw for gw in gameweeks if gw not in predictions_by_gw]
        if missing:
            raise KeyError(f"Missing predictions for gameweeks: {missing}")

    meta_frames = []
    ep_frames = []
    for gw in gameweeks:
        df = predictions_by_gw[gw].copy()
        _ensure_expected_points(df)
        meta_frames.append(
            df[
                [
                    "player_id",
                    "full_name",
                    "team_name",
                    "team_id",
                    "element_type",
                    "now_cost_millions",
                ]
            ]
        )
        ep_frame = df[["player_id", "expected_points"]].copy()
        ep_frame = ep_frame.rename(columns={"expected_points": f"expected_points_gw{gw}"})
        ep_frames.append(ep_frame)

    metadata = (
        pd.concat(meta_frames, ignore_index=True)
        .sort_values("player_id")
        .drop_duplicates("player_id", keep="last")
    )

    combined = metadata
    for frame in ep_frames:
        combined = combined.merge(frame, on="player_id", how="left")

    ep_cols = [col for col in combined.columns if col.startswith("expected_points_gw")]
    combined[ep_cols] = combined[ep_cols].fillna(0.0)
    combined["expected_points"] = combined[ep_cols].sum(axis=1)

    return combined


def recommend_transfers(
    user_team: pd.DataFrame,
    predictions_by_gw: Dict[int, pd.DataFrame],
    *,
    gameweeks: Optional[Iterable[int]] = None,
    free_transfers: int = 1,
    max_transfers: Optional[int] = None,
    budget_m: Optional[float] = None,
    formation: Optional[Dict[str, int]] = None,
    formations: Optional[Iterable[Dict[str, int]]] = None,
) -> Dict[str, object]:
    """Recommend transfers to maximise points across multiple gameweeks."""

    if free_transfers < 0:
        raise ValueError("free_transfers must be non-negative")

    transfers_limit = max_transfers if max_transfers is not None else free_transfers
    if transfers_limit < 0:
        raise ValueError("max_transfers must be non-negative")

    aggregated = aggregate_expected_points(predictions_by_gw, gameweeks=gameweeks)

    user_team_with_ep = user_team.merge(
        aggregated[
            [
                "player_id",
                "expected_points",
                "full_name",
                "team_name",
                "team_id",
                "element_type",
                "now_cost_millions",
            ]
        ],
        on="player_id",
        how="left",
        suffixes=("", "_pred"),
    )

    if user_team_with_ep["expected_points"].isna().any():
        missing = user_team_with_ep.loc[user_team_with_ep["expected_points"].isna(), "player_id"].tolist()
        raise ValueError(f"Missing predictions for user players: {missing}")

    for col in ["full_name", "team_name", "team_id", "element_type", "now_cost_millions"]:
        pred_col = f"{col}_pred"
        if pred_col in user_team_with_ep.columns:
            if col not in user_team_with_ep.columns:
                user_team_with_ep[col] = user_team_with_ep[pred_col]
            else:
                user_team_with_ep[col] = user_team_with_ep[col].fillna(user_team_with_ep[pred_col])
            user_team_with_ep = user_team_with_ep.drop(columns=pred_col)

    user_summary = summarise_team(user_team_with_ep)

    squad_budget = budget_m if budget_m is not None else user_summary.total_cost
    optimal_team = pick_best_xi(
        aggregated.rename(columns={"expected_points": "expected_points_total"}).assign(
            expected_points=lambda df: df.pop("expected_points_total")
        ),
        budget_m=squad_budget,
        formation=formation,
        formations=formations,
    )

    optimal_players: List[Dict[str, object]] = optimal_team["squad"] + optimal_team.get("bench", [])
    optimal_ids = {player["player_id"] for player in optimal_players}

    user_ids = set(user_team_with_ep["player_id"].tolist())

    outgoing_ids = list(user_ids - optimal_ids)
    incoming_ids = list(optimal_ids - user_ids)

    player_lookup = aggregated.set_index("player_id")

    def _player_record(pid: int) -> Dict[str, object]:
        data = player_lookup.loc[pid]
        per_gw = {
            col: float(data[col])
            for col in player_lookup.columns
            if col.startswith("expected_points_gw")
        }
        return {
            "player_id": int(pid),
            "full_name": data["full_name"],
            "team_name": data["team_name"],
            "element_type": int(data["element_type"]),
            "now_cost_millions": float(data["now_cost_millions"]),
            "expected_points": float(data["expected_points"]),
            "expected_points_by_gw": per_gw,
        }

    transfer_suggestions = []
    if transfers_limit > 0:
        # Group players by position to keep replacements position-aligned where possible.
        def _sort_outgoing(ids):
            return sorted(ids, key=lambda pid: player_lookup.loc[pid, "expected_points"])

        def _sort_incoming(ids):
            return sorted(
                ids,
                key=lambda pid: player_lookup.loc[pid, "expected_points"],
                reverse=True,
            )

        outgoing_by_pos = {}
        incoming_by_pos = {}
        for pid in outgoing_ids:
            pos = int(player_lookup.loc[pid, "element_type"])
            outgoing_by_pos.setdefault(pos, []).append(pid)
        for pid in incoming_ids:
            pos = int(player_lookup.loc[pid, "element_type"])
            incoming_by_pos.setdefault(pos, []).append(pid)

        remaining_outgoing: List[int] = []
        remaining_incoming: List[int] = []

        for pos in sorted(set(outgoing_by_pos) | set(incoming_by_pos)):
            outs = _sort_outgoing(outgoing_by_pos.get(pos, []))
            ins = _sort_incoming(incoming_by_pos.get(pos, []))
            pair_count = min(len(outs), len(ins))
            for idx in range(pair_count):
                if len(transfer_suggestions) >= transfers_limit:
                    break
                pid_out = outs[idx]
                pid_in = ins[idx]
                out_ep = float(player_lookup.loc[pid_out, "expected_points"])
                in_ep = float(player_lookup.loc[pid_in, "expected_points"])
                transfer_suggestions.append(
                    {
                        "out_player": _player_record(pid_out),
                        "in_player": _player_record(pid_in),
                        "expected_points_delta": in_ep - out_ep,
                    }
                )
            remaining_outgoing.extend(outs[pair_count:])
            remaining_incoming.extend(ins[pair_count:])
            if len(transfer_suggestions) >= transfers_limit:
                break

        if len(transfer_suggestions) < transfers_limit:
            remaining_outgoing = _sort_outgoing(remaining_outgoing)
            remaining_incoming = _sort_incoming(remaining_incoming)
            for pid_out, pid_in in zip(remaining_outgoing, remaining_incoming):
                if len(transfer_suggestions) >= transfers_limit:
                    break
                out_ep = float(player_lookup.loc[pid_out, "expected_points"])
                in_ep = float(player_lookup.loc[pid_in, "expected_points"])
                transfer_suggestions.append(
                    {
                        "out_player": _player_record(pid_out),
                        "in_player": _player_record(pid_in),
                        "expected_points_delta": in_ep - out_ep,
                    }
                )

    total_transfers = len(transfer_suggestions)
    free_transfers_used = min(total_transfers, free_transfers)
    paid_transfers_used = max(0, total_transfers - free_transfers)
    free_transfers_remaining = max(0, free_transfers - free_transfers_used)

    return {
        "user_team": user_summary.as_dict(),
        "optimal_team": optimal_team,
        "recommended_transfers": transfer_suggestions,
        "metadata": {
            "free_transfers_used": free_transfers_used,
            "additional_transfers": paid_transfers_used,
            "paid_transfers_used": paid_transfers_used,
            "free_transfers_remaining": free_transfers_remaining,
            "total_expected_points_current": user_summary.total_expected_points_with_captain,
            "total_expected_points_optimal": float(optimal_team["total_expected_points_with_captain"]),
        },
    }


__all__ = ["aggregate_expected_points", "recommend_transfers"]
