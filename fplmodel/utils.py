import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .config import RAW_DIR

def save_json(path: Path, obj: Any, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_current_and_last_finished_gw(events_df: pd.DataFrame) -> Tuple[int, int]:
    """
    From events (bootstrap 'events'), infer current and last finished GW.
    Returns (next_gw, last_finished_gw).
    """
    # 'finished' indicates if GW is finished; 'is_next' indicates next GW
    next_rows = events_df[events_df["is_next"] == True]
    if len(next_rows):
        next_gw = int(next_rows.iloc[0]["id"])
    else:
        # If season completed, pick last+1 to indicate no next GW
        next_gw = int(events_df["id"].max()) + 1

    finished = events_df[events_df["finished"] == True]
    last_finished_gw = int(finished["id"].max()) if len(finished) else 0
    return next_gw, last_finished_gw

def unix_now():
    return int(time.time())
