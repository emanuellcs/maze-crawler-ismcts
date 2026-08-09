"""Optuna study orchestration shared by the game tuning harnesses.

Implements the ask-and-tell loop with process isolation, a wall-clock deadline,
graceful handling of worker crashes and timeouts, resumable SQLite storage, and
the incumbent champion enqueued as the first trial.
"""

from __future__ import annotations

import concurrent.futures
import logging
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Dict, Optional, Sequence

import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState

_LOG = logging.getLogger("tuning.runner")

FAILURE_SCORE = -1.0e6


def run_study(
    *,
    objective: Callable[[Dict[str, Any]], float],
    sample: Callable[[optuna.trial.Trial], Dict[str, Any]],
    trials: int,
    n_jobs: int,
    storage: str,
    study_name: str,
    time_budget_s: int,
    champion: Optional[Dict[str, Any]] = None,
    seed_offset: int = 0,
) -> Optional[Dict[str, Any]]:
    """Run an ask-and-tell Optuna study and return the best params.

    Args:
        objective: Serializable worker callable scoring one hyperparameter set.
        sample: Serializable sampler mapping an Optuna trial to a param dict.
        trials: Total number of trials to run.
        n_jobs: Parallel worker processes.
        storage: Optuna storage URL (``sqlite:///...``).
        study_name: Stable study identifier for resumption.
        time_budget_s: Wall-clock cap for the whole study.
        champion: Optional incumbent enqueued as trial zero.
        seed_offset: RNG offset added to trial numbers.

    Returns:
        Best parameter dict, or ``None`` when nothing completed.
    """
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=min(20, max(1, trials // 50)),
        seed=42 + seed_offset,
    )
    db_storage = RDBStorage(
        url=storage,
        engine_kwargs={"connect_args": {"timeout": 60.0}},
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=db_storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    if len(study.trials) == 0 and champion:
        study.enqueue_trial(champion)

    deadline = _Deadline(time_budget_s)
    submitted = 0
    completed = 0
    active: Dict[concurrent.futures.Future, optuna.trial.Trial] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        try:
            while completed < trials and not deadline.exceeded():
                while len(active) < n_jobs and submitted < trials and not deadline.exceeded():
                    trial = study.ask()
                    hp = sample(trial)
                    future = executor.submit(objective, hp)
                    active[future] = trial
                    submitted += 1

                if not active:
                    break

                done, _ = concurrent.futures.wait(
                    list(active.keys()), timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED
                )

                pool_broken = False
                for future in done:
                    trial = active.pop(future)
                    try:
                        value = future.result()
                    except BrokenProcessPool:
                        _LOG.critical("process pool broken; failing outstanding trials")
                        for f, t in active.items():
                            study.tell(t, state=TrialState.FAIL)
                        active.clear()
                        pool_broken = True
                        break
                    except Exception:
                        _LOG.exception("trial %d worker raised", trial.number)
                        study.tell(trial, state=TrialState.FAIL)
                        completed += 1
                        continue

                    if value is None or value == FAILURE_SCORE:
                        study.tell(trial, state=TrialState.FAIL)
                    else:
                        study.tell(trial, value)
                    completed += 1

                if pool_broken:
                    break
        except KeyboardInterrupt:
            _LOG.warning("study interrupted; partial results preserved")
            for f in active:
                f.cancel()
            for t in active.values():
                study.tell(t, state=TrialState.FAIL)
            active.clear()

    if completed == 0:
        return None

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed_trials:
        return None
    best = study.best_params
    _LOG.info("best value=%.4f params=%s", study.best_value, best)
    return dict(best)


class _Deadline:
    """Simple monotonic wall-clock deadline."""

    def __init__(self, seconds: float) -> None:
        import time

        self._end = time.monotonic() + max(0.0, seconds)

    def exceeded(self) -> bool:
        import time

        return time.monotonic() >= self._end
