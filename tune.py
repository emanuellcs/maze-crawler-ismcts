"""Optuna tuning against the strong baseline benchmark.

Each trial samples C++ ISMCTS hyperparameters, injects them into the native
engine via main.set_hyperparameters, and evaluates the candidate against the
benchmark policy using side-swapped seeds to reduce variance.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# CRITICAL: Force spawn to ensure clean native state in workers.
multiprocessing.set_start_method("spawn", force=True)

import optuna
from kaggle_environments import make

# Add the root to sys.path so we can import main and opponents.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
import opponents.baseline as baseline_opponent

LOGGER = logging.getLogger("tune")
FAIL_SCORE = -1.0e6

MACRO_PRIOR_KEYS = (
    "IDLE",
    "FACTORY_SUPPORT_WORKER",
    "FACTORY_SAFE_ADVANCE",
    "FACTORY_BUILD_WORKER",
    "FACTORY_BUILD_SCOUT",
    "FACTORY_BUILD_MINER",
    "FACTORY_JUMP_OBSTACLE",
    "WORKER_OPEN_NORTH_WALL",
    "WORKER_ESCORT_FACTORY",
    "WORKER_ADVANCE",
    "SCOUT_HUNT_CRYSTAL",
    "SCOUT_EXPLORE_NORTH",
    "SCOUT_RETURN_ENERGY",
    "MINER_SEEK_NODE",
    "MINER_TRANSFORM",
)

@dataclass(frozen=True)
class EvalConfig:
    """Serializable match settings."""
    seeds: int
    base_seed: int
    time_budget_ms: int
    debug: bool

def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def _final_own_energy(final_state: Any, owner: int) -> int:
    observation = _get(final_state, "observation", {}) or {}
    robots = _get(observation, "robots", {}) or {}
    total = 0
    for data in robots.values():
        if len(data) >= 5 and int(data[4]) == owner:
            total += int(data[3])
    return total

def _state_failed(final_state: Any) -> bool:
    status = str(_get(final_state, "status", "")).upper()
    return any(token in status for token in ("ERROR", "INVALID", "TIMEOUT"))

def run_match(seed: int, candidate_player: int, config: EvalConfig) -> tuple[float, float]:
    """Run one game and return (win_score, energy_margin)."""
    opponent_player = 1 - candidate_player
    agents = [None, None]
    agents[candidate_player] = main.agent
    agents[opponent_player] = baseline_opponent.agent

    env = make("crawl", configuration={"randomSeed": seed}, debug=config.debug)
    steps = env.run(agents)
    final = steps[-1]

    if _state_failed(final[candidate_player]):
        return 0.0, -10000.0  # Heavy penalty for failure.
    if _state_failed(final[opponent_player]):
        return 1.0, 10000.0   # Opponent failure is a win.

    candidate_energy = _final_own_energy(final[candidate_player], candidate_player)
    opponent_energy = _final_own_energy(final[opponent_player], opponent_player)
    
    margin = float(candidate_energy - opponent_energy)
    reward = _get(final[candidate_player], "reward", 0)
    
    # Win = 1.0, Draw = 0.5, Loss = 0.0 (based on reward normalization)
    win_score = 1.0 if reward > 0.5 else (0.5 if reward == 0.5 else 0.0)
    
    return win_score, margin

def objective(trial: optuna.trial.Trial, config: EvalConfig) -> float:
    """Sample parameters and return composite score."""
    hp = {
        "C_puct": trial.suggest_float("C_puct", 0.5, 3.0),
        "baseline_prior_multiplier": trial.suggest_float("baseline_prior_multiplier", 0.75, 2.0),
        "rollout_depth": trial.suggest_int("rollout_depth", 16, 96, step=8),
        "search_threads": 1, # PREVENT OVERSUBSCRIPTION
    }
    for key in MACRO_PRIOR_KEYS:
        low, high = (0.05, 0.75) if key == "IDLE" else (0.25, 2.5)
        hp[key] = trial.suggest_float(key, low, high)

    # Apply parameters and clear cache in this worker process.
    main.set_hyperparameters(**hp)

    try:
        total_win_score = 0.0
        total_margin = 0.0
        for i in range(config.seeds):
            seed = config.base_seed + i
            # Side-swapping
            w0, m0 = run_match(seed, 0, config)
            w1, m1 = run_match(seed, 1, config)
            total_win_score += (w0 + w1)
            total_margin += (m0 + m1)
        
        avg_win_rate = total_win_score / (2 * config.seeds)
        avg_margin = total_margin / (2 * config.seeds)
        
        # Composite score: win rate is primary, margin provides continuous gradient.
        return avg_win_rate + (avg_margin / 10000.0)
    except Exception:
        LOGGER.error(f"Trial {trial.number} crashed:\n{traceback.format_exc()}")
        return FAIL_SCORE

def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--time-budget", type=int, default=300)
    parser.add_argument("--storage", default="sqlite:///tune.db")
    parser.add_argument("--study-name", default="crawl-vs-opponent")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = EvalConfig(
        seeds=args.seeds,
        base_seed=args.base_seed,
        time_budget_ms=args.time_budget,
        debug=args.debug
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True
    )
    
    # Enqueue current best as first trial if study is new.
    if len(study.trials) == 0:
        study.enqueue_trial(main.BEST_PARAMS)

    study.optimize(lambda t: objective(t, config), n_trials=args.trials, n_jobs=args.n_jobs)

    if study.best_trial:
        print(f"Best trial: {study.best_trial.number}")
        print(f"  Value: {study.best_trial.value}")
        print(f"  Params: {study.best_trial.params}")

if __name__ == "__main__":
    main_cli()
