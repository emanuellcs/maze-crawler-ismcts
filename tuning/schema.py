"""Hyperparameter schema loaded from the single source of truth.

``config/hyperparameters.json`` defines the champion values and the search-space
bounds for every tunable.  This module is the only place that knows how to map
those definitions onto Optuna ``Trial`` suggestions, so the Optuna space, the
champion defaults, and the validation rules can never drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Param:
    """One tunable hyperparameter with its search-space bounds."""

    name: str
    type: str  # "int" | "float"
    min: float
    max: float
    step: Optional[float] = None
    log: bool = False
    default: float = 0.0
    group: str = ""

    def suggest(self, trial) -> float:
        """Ask Optuna to sample this parameter for one trial."""
        if self.type == "int":
            step = int(self.step) if self.step else 1
            return trial.suggest_int(self.name, int(self.min), int(self.max), step=step, log=self.log)
        return trial.suggest_float(self.name, self.min, self.max, log=self.log)


class Schema:
    """Ordered set of tunable parameters with champion defaults."""

    def __init__(self, params: Dict[str, Param]):
        self.params = params

    @classmethod
    def load(cls, path: Path) -> "Schema":
        """Build a schema from a ``config/hyperparameters.json`` file."""
        data = json.loads(Path(path).read_text())
        params: Dict[str, Param] = {}
        for name, spec in data.get("schema", {}).items():
            params[name] = Param(
                name=name,
                type=str(spec.get("type", "float")),
                min=float(spec.get("min", 0.0)),
                max=float(spec.get("max", 1.0)),
                step=spec.get("step"),
                log=bool(spec.get("log", False)),
                default=float(spec.get("default", 0.0)),
                group=str(spec.get("group", "")),
            )
        return cls(params)

    def names(self, group: Optional[str] = None) -> List[str]:
        return [n for n, p in self.params.items() if group is None or p.group == group]

    def defaults(self) -> Dict[str, float]:
        return {n: p.default for n, p in self.params.items()}

    def sample_trial(self, trial) -> Dict[str, float]:
        """Sample every parameter from one Optuna trial."""
        return {n: p.suggest(trial) for n, p in self.params.items()}

    def clamp(self, hp: Dict[str, float]) -> Dict[str, float]:
        """Clamp a candidate set back inside the schema bounds."""
        out = {}
        for n, p in self.params.items():
            value = float(hp.get(n, p.default))
            out[n] = min(p.max, max(p.min, value))
        return out
