"""Tuning infrastructure for the native Kaggle agents.

The package provides the reusable machinery behind ``scripts/tune.py`` and
``scripts/bench.py``:

- :mod:`schema`   parameter schema loaded from ``config/hyperparameters.json``
- :mod:`league`   opponent registry (hand-crafted + self-play checkpoints) with Elo
- :mod:`evaluation`  wall-clock-bounded match execution helpers
- :mod:`optuna_runner`  process-parallel Optuna study orchestration
- :mod:`reporting`  champion write-back to the single source of truth

Game-specific match scoring and seat rotation live in each repository's
``scripts/tune.py`` / ``scripts/bench.py`` adapters.
"""
