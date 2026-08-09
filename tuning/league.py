"""Opponent league: hand-crafted archetypes plus self-play checkpoints.

The league is the opponent pool used by tuning and benchmarking.  It mixes
permanently retained hand-crafted strategies with ``checkpoint`` entries that
record past champion hyperparameters, so self-play supplies increasingly strong
opponents without ever dropping the archetypes that keep the pool diverse.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

_ELO_K = 32.0


class League:
    """A persistent registry of named opponents with Elo ratings."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            return []
        return list(data.get("opponents", []))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"opponents": self.entries}
        self.path.write_text(json.dumps(payload, indent=2))

    def names(self) -> List[str]:
        return [e["name"] for e in self.entries]

    def get(self, name: str) -> Optional[Dict]:
        for e in self.entries:
            if e["name"] == name:
                return e
        return None

    def rating(self, name: str) -> float:
        entry = self.get(name)
        return float(entry["rating"]) if entry else 1500.0

    def register(self, name: str, kind: str, params: Optional[Dict] = None, note: str = "") -> None:
        """Add an opponent without duplicate entries (upserts in place)."""
        for e in self.entries:
            if e["name"] == name:
                e.update({"kind": kind, "params": params or {}, "note": note})
                self.save()
                return
        self.entries.append({
            "name": name,
            "kind": kind,
            "params": params or {},
            "rating": 1500.0,
            "note": note,
        })
        self.save()

    def update_elo(self, winner: str, loser: str) -> None:
        """Apply a single two-player outcome to the league ratings."""
        rw = self.rating(winner)
        rl = self.rating(loser)
        ew = 1.0 / (1.0 + math.pow(10.0, (rl - rw) / 400.0))
        el = 1.0 - ew
        self._set_rating(winner, rw + _ELO_K * (1.0 - ew))
        self._set_rating(loser, rl + _ELO_K * (0.0 - el))

    def _set_rating(self, name: str, rating: float) -> None:
        for e in self.entries:
            if e["name"] == name:
                e["rating"] = round(rating, 1)
                self.save()
                return

    def top(self, k: int) -> List[Dict]:
        return sorted(self.entries, key=lambda e: -e["rating"])[:k]

    def checkpoints(self) -> List[Dict]:
        return [e for e in self.entries if e.get("kind") == "checkpoint"]
