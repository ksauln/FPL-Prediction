from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

from .config import MODELS_DIR, EMA_ALPHA

class ModelState:
    """
    Maintains persistent state for bias-correction based on latest GW residuals.
    Stores:
      - player_bias: dict[player_id] = float (EMA of residuals)
      - position_bias: dict[pos_code] = float (EMA)
      - last_evaluated_gw: int
    """
    def __init__(self, path: Path | None = None):
        self.path = path or (MODELS_DIR / "state.json")
        self.player_bias: Dict[str, float] = {}
        self.position_bias: Dict[str, float] = {}
        self.last_evaluated_gw: int = 0
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.player_bias = data.get("player_bias", {})
                self.position_bias = data.get("position_bias", {})
                self.last_evaluated_gw = data.get("last_evaluated_gw", 0)
            except Exception:
                # start fresh if corrupted
                self.player_bias, self.position_bias, self.last_evaluated_gw = {}, {}, 0

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "player_bias": self.player_bias,
                    "position_bias": self.position_bias,
                    "last_evaluated_gw": self.last_evaluated_gw,
                },
                f,
                indent=2,
            )

    def get_player_bias(self, player_id: int) -> float:
        return float(self.player_bias.get(str(player_id), 0.0))

    def get_position_bias(self, pos_code: int) -> float:
        # pos_code: 1 GK, 2 DEF, 3 MID, 4 FWD
        return float(self.position_bias.get(str(pos_code), 0.0))

    def update_from_residuals(self, df_residuals):
        """
        df_residuals columns expected:
          - player_id
          - element_type (1/2/3/4)
          - residual (actual - predicted)
          - gw (gameweek evaluated)
        Applies EMA update.
        """
        alpha = EMA_ALPHA
        # Per-player EMA
        for pid, r in df_residuals.groupby("player_id")["residual"].mean().items():
            key = str(int(pid))
            prev = self.player_bias.get(key, 0.0)
            self.player_bias[key] = (1 - alpha) * prev + alpha * float(r)
        # Per-position EMA
        for pos, r in df_residuals.groupby("element_type")["residual"].mean().items():
            key = str(int(pos))
            prev = self.position_bias.get(key, 0.0)
            self.position_bias[key] = (1 - alpha) * prev + alpha * float(r)
        # last evaluated gw
        if "gw" in df_residuals.columns and len(df_residuals["gw"]):
            self.last_evaluated_gw = int(df_residuals["gw"].max())
        self.save()
