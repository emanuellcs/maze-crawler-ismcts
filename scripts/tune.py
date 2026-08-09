"""Optuna tuning for the Maze Crawler ISMCTS engine.

Each trial samples the engine hyperparameters, injects them via
``main.set_hyperparameters``, and scores the candidate against a fixed opponent
pool (hand-crafted archetypes plus self-play checkpoints) over fixed seeds with
full seat rotation.  Trials run in spawned worker processes so the native engine
state stays clean.  On success the champion is written back to
``config/hyperparameters.json`` and registered in ``config/league.json``.
"""

from __future__ import annotations

import argparse
import functools
import logging
import multiprocessing
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

# Spawn is mandatory: fork() after pybind11 + std::thread initialization can
# deadlock on Linux.
multiprocessing.set_start_method("spawn", force=True)

import optuna  # noqa: E402
from kaggle_environments import make  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main  # noqa: E402
from tuning import evaluation, league as league_mod, optuna_runner, reporting, schema  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("tune")

HANDCRAFTED_OPPONENTS = ("baseline", "miner_rush", "worker_rush", "wall_turtle")
FAILURE_SCORE = evaluation.FAILURE_SCORE


@dataclass(frozen=True)
class EvalConfig:
    """Serializable match settings."""

    seeds: int
    base_seed: int
    max_steps: int
    budget_ms: int
    timeout_per_match: float
    debug: bool
    opponent_pool: tuple  # (kind, name) pairs; checkpoints carry params


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _make_opponent_agent(kind, name, params):
    """Return a callable agent for one league/handcrafted opponent."""
    if kind == "handcrafted":
        module = __import__(f"opponents.{name}", fromlist=["agent"])
        return module.agent
    # Self-play checkpoint: our own engine driven by the saved parameters.
    def checkpoint_agent(obs, config):
        main.set_hyperparameters(**params)
        return main.agent(obs, config)

    return checkpoint_agent


def _final_energy(final_state, owner):
    robots = _get(_get(final_state, "observation", {}) or {}, "robots", {}) or {}
    return sum(int(d[3]) for d in robots.values() if len(d) >= 5 and int(d[4]) == owner)


def run_match(seed, candidate_seat, opponent_agent, config: EvalConfig):
    """Run one side of a paired match and return (win_score, margin)."""
    agents = [None, None]
    agents[candidate_seat] = main.agent
    agents[1 - candidate_seat] = opponent_agent

    env = make("crawl", configuration={"randomSeed": seed, "episodeSteps": config.max_steps}, debug=config.debug)

    steps = evaluation.run_with_timeout(lambda: env.run(agents), config.timeout_per_match)
    if steps is None:
        return 0.0, -10000.0

    final = steps[-1]
    if evaluation.state_failed(_get(final[candidate_seat], "status", "")):
        return 0.0, -10000.0
    if evaluation.state_failed(_get(final[1 - candidate_seat], "status", "")):
        return 1.0, 10000.0

    margin = _final_energy(final[candidate_seat], candidate_seat) - _final_energy(
        final[1 - candidate_seat], 1 - candidate_seat
    )
    reward = _get(final[candidate_seat], "reward", 0)
    win = 1.0 if reward > 0.5 else (0.5 if reward == 0.5 else 0.0)
    return win, float(margin)


def evaluate_hp(hp, config: EvalConfig) -> float:
    """Score one hyperparameter set in a worker process."""
    try:
        full_hp = dict(hp)
        full_hp["search_time_ms"] = config.budget_ms
        main.set_hyperparameters(**full_hp)
        total_wins = 0.0
        total_games = 0
        for kind, name, params in config.opponent_pool:
            opponent = _make_opponent_agent(kind, name, params)
            main.set_hyperparameters(**full_hp)  # re-assert candidate params
            for i in range(config.seeds):
                seed = config.base_seed + i
                for seat in (0, 1):
                    w, _ = run_match(seed, seat, opponent, config)
                    total_wins += w
                    total_games += 1
        win_rate = total_wins / max(1, total_games)
        _LOG.info("hp win_rate=%.3f games=%d", win_rate, total_games)
        return win_rate
    except Exception:  # noqa: BLE001
        _LOG.error("evaluation failed:\n%s", traceback.format_exc())
        return FAILURE_SCORE


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=499, help="Episode step cap for matches")
    parser.add_argument("--time-budget", type=int, default=3600, help="Study wall-clock seconds")
    parser.add_argument("--storage", default="sqlite:///tune.db")
    parser.add_argument("--study-name", default="crawl-vs-league")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-match wall-clock seconds")
    parser.add_argument("--budget-ms", type=int, default=500, help="Per-turn native search budget")
    parser.add_argument("--promote", action="store_true", help="Write the champion back on success")
    parser.add_argument("--debug", action="store_true")
    return parser


def main_cli(argv=None) -> int:
    args = build_parser().parse_args(argv)

    hyper_config = _ROOT / "config" / "hyperparameters.json"
    league_path = _ROOT / "config" / "league.json"
    sch = schema.Schema.load(hyper_config)

    # Opponent pool: all hand-crafted archetypes plus the top self-play checkpoints.
    lg = league_mod.League(league_path)
    pool = [(kind, name, params) for kind, name, params in
            [("handcrafted", n, None) for n in HANDCRAFTED_OPPONENTS] +
            [("checkpoint", e["name"], e.get("params", {})) for e in lg.checkpoints()
             if e["name"] not in HANDCRAFTED_OPPONENTS]]
    if not pool:
        _LOG.warning("no opponents registered; defaulting to baseline")
        pool = [("handcrafted", "baseline", None)]

    config = EvalConfig(
        seeds=args.seeds,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
        budget_ms=args.budget_ms,
        timeout_per_match=args.timeout,
        debug=args.debug,
        opponent_pool=tuple(pool),
    )
    champion = dict(sch.defaults())
    champion.update(reporting.read_champion(hyper_config))

    best = optuna_runner.run_study(
        objective=functools.partial(evaluate_hp, config=config),
        sample=sch.sample_trial,
        trials=args.trials,
        n_jobs=args.n_jobs,
        storage=args.storage,
        study_name=args.study_name,
        time_budget_s=args.time_budget,
        champion=champion,
        seed_offset=args.base_seed,
    )
    if best is None:
        print("No successful trials; nothing to promote.")
        return 1

    if args.promote:
        reporting.promote_champion(
            hyper_config,
            league_path,
            name=f"champion-{args.study_name}",
            params=best,
            note=f"promoted from {args.study_name}",
        )
        print(f"Promoted champion -> {hyper_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
