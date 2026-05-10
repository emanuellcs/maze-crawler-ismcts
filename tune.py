"""Optuna tuning against the strong `opponent.py` benchmark.

Each trial samples C++ ISMCTS hyperparameters, injects them into a fresh
`crawler_engine.Engine` wrapper, and evaluates the candidate against the
leaderboard-grade Python opponent on paired seeds with side swapping.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import importlib.util
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import optuna
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
OPPONENT_PATH = ROOT / "opponent.py"
LOGGER = logging.getLogger("tune")
FAIL_MARGIN = -1.0e9

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

DEFAULT_PARAMS = {
    "C_puct": 2.0884330868271443,
    "baseline_prior_multiplier": 1.8863044112273712,
    "rollout_depth": 80,
    "IDLE": 0.49504719444108913,
    "FACTORY_SUPPORT_WORKER": 1.0390992283842135,
    "FACTORY_SAFE_ADVANCE": 2.161864767112469,
    "FACTORY_BUILD_WORKER": 0.997532683560502,
    "FACTORY_BUILD_SCOUT": 1.8649154683704814,
    "FACTORY_BUILD_MINER": 0.8269752415957998,
    "FACTORY_JUMP_OBSTACLE": 1.4925105256980329,
    "WORKER_OPEN_NORTH_WALL": 0.7351332485632837,
    "WORKER_ESCORT_FACTORY": 1.583596636264722,
    "WORKER_ADVANCE": 0.8874633406271473,
    "SCOUT_HUNT_CRYSTAL": 1.083268581072123,
    "SCOUT_EXPLORE_NORTH": 1.6851712172373043,
    "SCOUT_RETURN_ENERGY": 1.1040824358243229,
    "MINER_SEEK_NODE": 0.6983213636231111,
    "MINER_TRANSFORM": 0.5309500821275892,
}


@dataclass(frozen=True)
class EvalConfig:
    """Serializable match settings passed to process-pool workers."""

    seeds: int
    base_seed: int
    time_budget_ms: int
    debug: bool
    fail_value: float


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read fields from either Kaggle objects or dict-like test stubs."""

    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _engine_seed(step: int, player: int, game_seed: int) -> int:
    """Derive deterministic per-turn search seeds without sharing RNG state."""

    return ((step + 1) * 1_315_423_911 + player + game_seed * 2_654_435_761) & (
        (1 << 64) - 1
    )


def _load_crawler_engine() -> Any:
    """Import crawler_engine, falling back to main.py's JIT compiler if needed."""

    try:
        return importlib.import_module("crawler_engine")
    except Exception as first_exc:
        try:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            main = importlib.import_module("main")
            ensure_native = getattr(main, "_ensure_native_engine", None)
            if ensure_native is None or not ensure_native():
                raise RuntimeError("main._ensure_native_engine() failed")
            module = getattr(main, "crawler_engine", None)
            return (
                module
                if module is not None
                else importlib.import_module("crawler_engine")
            )
        except Exception as second_exc:
            raise RuntimeError(
                "could not import or JIT-compile crawler_engine\n"
                f"initial import error: {first_exc}\n"
                f"fallback error: {second_exc}"
            ) from second_exc


def _load_opponent_agent() -> Callable[[Any, Any], dict[str, str]]:
    """Load opponent.py without mutating it or relying on package installation."""

    if not OPPONENT_PATH.exists():
        raise RuntimeError(f"opponent.py not found at {OPPONENT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "maze_crawler_tuning_opponent", OPPONENT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for {OPPONENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if agent is None or not callable(agent):
        raise RuntimeError("opponent.py must expose callable agent(obs, config)")
    return agent


class CandidateAgent:
    """Kaggle-compatible callable that owns fresh C++ engines for one match."""

    def __init__(
        self,
        hyperparameters: dict[str, float | int],
        time_budget_ms: int,
        game_seed: int,
    ):
        self.hyperparameters = hyperparameters
        self.time_budget_ms = time_budget_ms
        self.game_seed = game_seed
        self.crawler_engine = _load_crawler_engine()
        self.engines: dict[int, Any] = {}

    def __call__(self, obs: Any, config: Any) -> dict[str, str]:
        player = int(_get(obs, "player", 0))
        engine = self.engines.get(player)
        if engine is None:
            engine = self.crawler_engine.Engine(player)
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


def make_candidate_agent(
    hyperparameters: dict[str, float | int], time_budget_ms: int, game_seed: int
) -> Callable[[Any, Any], dict[str, str]]:
    """Factory required by tuning workers to build a fresh C++ candidate agent."""

    return CandidateAgent(hyperparameters, time_budget_ms, game_seed)


def make_opponent_agent() -> Callable[[Any, Any], dict[str, str]]:
    """Factory for a fresh opponent wrapper per match."""

    opponent_agent = _load_opponent_agent()

    def agent(obs: Any, config: Any) -> dict[str, str]:
        return opponent_agent(obs, config)

    return agent


def suggest_hyperparameters(trial: optuna.trial.Trial) -> dict[str, float | int]:
    """Sample the C++ hyperparameter surface exposed through pybind11."""

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
    """Read final energy from the owning player's observation.

    Enemy robots are fogged, so player 0's final observation is the reliable
    source for owner 0 energy and player 1's final observation is the reliable
    source for owner 1 energy.
    """

    observation = _get(final_state, "observation", {}) or {}
    robots = _get(observation, "robots", {}) or {}
    total = 0
    for data in robots.values():
        if len(data) >= 5 and int(data[4]) == owner:
            total += int(data[3])
    return total


def _state_failed(final_state: Any) -> bool:
    """Detect common Kaggle agent failure statuses without depending on enums."""

    status = str(_get(final_state, "status", "")).upper()
    return any(token in status for token in ("ERROR", "INVALID", "TIMEOUT"))


def run_match(
    params: dict[str, float | int], seed: int, candidate_player: int, config: EvalConfig
) -> float:
    """Run one candidate-vs-opponent game and return candidate energy margin."""

    opponent_player = 1 - candidate_player
    agents: list[Callable[[Any, Any], dict[str, str]] | None] = [None, None]
    agents[candidate_player] = make_candidate_agent(params, config.time_budget_ms, seed)
    agents[opponent_player] = make_opponent_agent()

    env = make("crawl", configuration={"randomSeed": seed}, debug=config.debug)
    steps = env.run(agents)
    final = steps[-1]

    if _state_failed(final[candidate_player]):
        return config.fail_value
    if _state_failed(final[opponent_player]):
        return -config.fail_value

    candidate_energy = _final_own_energy(final[candidate_player], candidate_player)
    opponent_energy = _final_own_energy(final[opponent_player], opponent_player)
    return float(candidate_energy - opponent_energy)


def evaluate_params(params: dict[str, float | int], config: EvalConfig) -> float:
    """Average candidate margins across seeds and swapped player order."""

    try:
        margins: list[float] = []
        for offset in range(config.seeds):
            seed = config.base_seed + offset
            margins.append(run_match(params, seed, candidate_player=0, config=config))
            margins.append(run_match(params, seed, candidate_player=1, config=config))
        return sum(margins) / len(margins)
    except Exception as exc:
        raise RuntimeError(
            f"trial evaluation failed:\n{traceback.format_exc()}"
        ) from exc


def objective(trial: optuna.trial.Trial, config: EvalConfig) -> float:
    """Optuna objective: sample params and return average opponent energy margin."""

    return evaluate_params(suggest_hyperparameters(trial), config)


def _resolve_n_jobs(n_jobs: int, trials: int) -> int:
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs <= 0:
        raise ValueError("--n-jobs must be -1 or a positive integer")
    return min(n_jobs, max(1, trials))


def optimize(args: argparse.Namespace) -> optuna.Study:
    """Run process-parallel Optuna ask/tell optimization."""

    config = EvalConfig(
        seeds=args.seeds,
        base_seed=args.base_seed,
        time_budget_ms=args.time_budget,
        debug=args.debug,
        fail_value=args.fail_value,
    )
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
    )
    study.enqueue_trial(DEFAULT_PARAMS)

    n_jobs = _resolve_n_jobs(args.n_jobs, args.trials)
    LOGGER.info(
        "starting study=%s trials=%d n_jobs=%d seeds=%d time_budget_ms=%d storage=%s",
        args.study_name,
        args.trials,
        n_jobs,
        args.seeds,
        args.time_budget,
        args.storage,
    )

    submitted = 0
    completed = 0
    failed = 0
    in_flight: dict[concurrent.futures.Future[float], optuna.trial.Trial] = {}

    def submit_next(pool: concurrent.futures.ProcessPoolExecutor) -> None:
        nonlocal submitted
        trial = study.ask()
        params = suggest_hyperparameters(trial)
        in_flight[pool.submit(evaluate_params, params, config)] = trial
        submitted += 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as pool:
        while submitted < min(n_jobs, args.trials):
            submit_next(pool)

        while in_flight:
            done, _ = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                trial = in_flight.pop(future)
                try:
                    value = future.result()
                except Exception as exc:
                    failed += 1
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    LOGGER.exception("trial=%d failed: %s", trial.number, exc)
                else:
                    completed += 1
                    study.tell(trial, value)
                    LOGGER.info(
                        "trial=%d value=%.3f completed=%d/%d failed=%d",
                        trial.number,
                        value,
                        completed,
                        args.trials,
                        failed,
                    )

                if submitted < args.trials:
                    submit_next(pool)

    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if complete_trials:
        best = study.best_trial
        LOGGER.info("best_trial=%d best_value=%.3f", best.number, best.value)
        LOGGER.info("best_params=%s", best.params)
    else:
        LOGGER.warning("no completed trials; inspect logs and failed trial states")
    return study


def parse_args() -> argparse.Namespace:
    """Parse CLI settings and reject invalid self-play budgets early."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--time-budget", "--time-budget-ms", type=int, default=300)
    parser.add_argument("--n-jobs", "--workers", dest="n_jobs", type=int, default=16)
    parser.add_argument("--storage", default="sqlite:///tune.db")
    parser.add_argument("--study-name", default="crawl-vs-opponent")
    parser.add_argument("--sampler-seed", type=int, default=20260507)
    parser.add_argument("--fail-value", type=float, default=FAIL_MARGIN)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.time_budget <= 0:
        raise ValueError("--time-budget must be positive")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
    )
    optimize(args)


if __name__ == "__main__":
    main()
