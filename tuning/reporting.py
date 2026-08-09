"""Champion write-back and leaderboard reporting.

The tuned champion is persisted back into ``config/hyperparameters.json`` so
``main.py`` picks it up on the next import — closing the loop on the single
source of truth.  Self-play checkpoints are also registered in the league.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

from tuning.league import League


def read_champion(config_path: Path, champion_key: str = "champion") -> Dict[str, object]:
    """Return a champion param dict from the hyperparameter config file."""
    try:
        data = json.loads(Path(config_path).read_text())
        return dict(data.get(champion_key, {}))
    except Exception:
        return {}


def write_champion(
    config_path: Path, params: Dict[str, object], note: str = "", champion_key: str = "champion"
) -> None:
    """Merge ``params`` into a champion entry of the hyperparameter config file."""
    config_path = Path(config_path)
    try:
        data = json.loads(config_path.read_text())
    except Exception:
        data = {}
    champion = dict(data.get(champion_key, {}))
    champion.update(params)
    data[champion_key] = champion
    if note:
        data[f"{champion_key}_note"] = note
    data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    config_path.write_text(json.dumps(data, indent=2))


def promote_champion(
    config_path: Path,
    league_path: Path,
    name: str,
    params: Dict[str, object],
    note: str = "",
    champion_key: str = "champion",
) -> None:
    """Persist a new champion and register it as a self-play checkpoint."""
    write_champion(config_path, params, note=note, champion_key=champion_key)
    League(league_path).register(name=name, kind="checkpoint", params=params, note=note)
