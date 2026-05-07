"""Optuna hyperparameter tuning for the C++ Crawl engine."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from dataclasses import dataclass
from typing import Any

import optuna
from kaggle_environments import make

import crawler_engine

MACRO_PRIOR_KEYS = (
    "IDLE",
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
    seeds: int
    base_seed: int
    time_budget_ms: int
    debug: bool


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _engine_seed(step: int, player: int, game_seed: int) -> int:
    return ((step + 1) * 1_315_423_911 + player + game_seed * 2_654_435_761) & (
        (1 << 64) - 1
    )


class EngineAgent:
    def __init__(
        self,
        hyperparameters: dict[str, float | int] | None,
        time_budget_ms: int,
        game_seed: int,
    ):
        self.hyperparameters = hyperparameters
        self.time_budget_ms = time_budget_ms
        self.game_seed = game_seed
        self.engines: dict[int, crawler_engine.Engine] = {}

    def __call__(self, obs: Any, config: Any) -> dict[str, str]:
        player = int(_get(obs, "player", 0))
        engine = self.engines.get(player)
        if engine is None:
            engine = crawler_engine.Engine(player)
            if self.hyperparameters is not None:
                engine.set_hyperparameters(self.hyperparameters)
            self.engines[player] = engine

        step = int(_get(obs, "step", -1))
        engine.update_observation(
            player,
            _get(obs, "walls", []),
            _get(obs, "crystals", {}) or {},
            _get(obs, "robots", {}) or {},
            _get(obs, "mines", {}) or {},
            _get(obs, "miningNodes", {}) or {},
            int(_get(obs, "southBound", 0)),
            int(_get(obs, "northBound", 19)),
            step,
        )
        return engine.choose_actions(
            self.time_budget_ms, seed=_engine_seed(step, player, self.game_seed)
        )


def suggest_hyperparameters(trial: optuna.trial.Trial) -> dict[str, float | int]:
    params: dict[str, float | int] = {
        "C_puct": trial.suggest_float("C_puct", 0.5, 3.0),
        "baseline_prior_multiplier": trial.suggest_float(
            "baseline_prior_multiplier", 0.75, 2.0
        ),
        "rollout_depth": trial.suggest_int("rollout_depth", 16, 96, step=8),
    }
    for key in MACRO_PRIOR_KEYS:
        low, high = (0.05, 0.75) if key == "IDLE" else (0.25, 2.5)
        params[key] = trial.suggest_float(key, low, high)
    return params


def _final_own_energy(final_state: Any, owner: int) -> int:
    observation = _get(final_state, "observation", {}) or {}
    robots = _get(observation, "robots", {}) or {}
    total = 0
    for data in robots.values():
        if len(data) >= 5 and int(data[4]) == owner:
            total += int(data[3])
    return total


def run_match(
    params: dict[str, float | int], seed: int, candidate_player: int, config: EvalConfig
) -> float:
    baseline_player = 1 - candidate_player
    agents = [None, None]
    agents[candidate_player] = EngineAgent(params, config.time_budget_ms, seed)
    agents[baseline_player] = EngineAgent(None, config.time_budget_ms, seed)

    env = make("crawl", configuration={"randomSeed": seed}, debug=config.debug)
    steps = env.run(agents)
    final = steps[-1]
    energy = [
        _final_own_energy(final[0], 0),
        _final_own_energy(final[1], 1),
    ]
    return float(energy[candidate_player] - energy[baseline_player])


def evaluate_params(params: dict[str, float | int], config: EvalConfig) -> float:
    margins: list[float] = []
    for offset in range(config.seeds):
        seed = config.base_seed + offset
        margins.append(run_match(params, seed, candidate_player=0, config=config))
        margins.append(run_match(params, seed, candidate_player=1, config=config))
    return sum(margins) / len(margins)


def objective(trial: optuna.trial.Trial, config: EvalConfig) -> float:
    return evaluate_params(suggest_hyperparameters(trial), config)


def _resolve_workers(workers: int) -> int:
    if workers == -1:
        return max(1, os.cpu_count() or 1)
    if workers <= 0:
        raise ValueError("--workers must be -1 or a positive integer")
    return workers


def optimize(args: argparse.Namespace) -> optuna.Study:
    config = EvalConfig(
        seeds=args.seeds,
        base_seed=args.base_seed,
        time_budget_ms=args.time_budget_ms,
        debug=args.debug,
    )
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
    )

    workers = min(_resolve_workers(args.workers), max(1, args.trials))
    submitted = 0
    completed = 0
    in_flight: dict[concurrent.futures.Future[float], optuna.trial.Trial] = {}

    def submit_next(pool: concurrent.futures.ProcessPoolExecutor) -> None:
        nonlocal submitted
        trial = study.ask()
        params = suggest_hyperparameters(trial)
        future = pool.submit(evaluate_params, params, config)
        in_flight[future] = trial
        submitted += 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        while submitted < min(workers, args.trials):
            submit_next(pool)

        while in_flight:
            done, _ = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                trial = in_flight.pop(future)
                try:
                    value = future.result()
                except Exception:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise

                study.tell(trial, value)
                completed += 1
                print(
                    f"trial={trial.number} value={value:.3f} completed={completed}/{args.trials}",
                    flush=True,
                )

                if submitted < args.trials:
                    submit_next(pool)

    if len(study.trials) > 0:
        best = study.best_trial
        print(f"best_trial={best.number} best_value={best.value:.3f}", flush=True)
        print(f"best_params={best.params}", flush=True)
    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", "--n-jobs", dest="workers", type=int, default=-1)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--time-budget-ms", type=int, default=50)
    parser.add_argument("--storage", default="sqlite:///tune.db")
    parser.add_argument("--study-name", default="crawl-hparams")
    parser.add_argument("--sampler-seed", type=int, default=20240507)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.time_budget_ms <= 0:
        raise ValueError("--time-budget-ms must be positive")
    return args


if __name__ == "__main__":
    optimize(parse_args())
