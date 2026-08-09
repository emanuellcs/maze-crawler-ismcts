# Tuning System

This document describes the tuning pipeline for the Maze Crawler engine: the hyperparameter schema, the Optuna runner, the evaluation protocol, the opponent league, and champion promotion.

## Overview

Tuning optimizes the engine hyperparameters against a league of opponents. The system is designed around three properties:

- A single source of truth for tuned values and search-space bounds.
- A reliable evaluation signal based on fixed seeds and full seat rotation.
- A promotion step that writes the best configuration back to the config file and records it as a self-play checkpoint.

The harness is `scripts/tune.py`. It uses the reusable package under `tuning/`.

## Single Source of Truth

`config/hyperparameters.json` holds two blocks:

- `champion`: the current best tuned values, loaded by `main.py` at import time.
- `schema`: one entry per tunable parameter with its type, bounds, step, and default.

The `tuning/schema.py` module maps these definitions onto Optuna `Trial` suggestions. Because the schema drives both the search space and the champion defaults, the two cannot drift apart.

### Parameter Groups

| Group | Parameters |
| --- | --- |
| search | `C_puct`, `baseline_prior_multiplier`, `rollout_depth` |
| macro_prior | `IDLE` and the per-macro priors (`FACTORY_*`, `WORKER_*`, `SCOUT_*`, `MINER_*`) |

Numeric values are not duplicated in documentation. See `config/hyperparameters.json` for the authoritative champion values and bounds.

## Optuna Runner

`tuning/optuna_runner.py` implements the ask-and-tell loop:

- A multivariate TPE sampler explores the search space.
- Trials run in spawned worker processes so the native engine state stays clean.
- The incumbent champion is enqueued as the first trial when the study is new.
- Studies persist to SQLite and resume when rerun with the same `--storage` and `--study-name`.
- A wall-clock deadline bounds the whole study.
- Worker crashes, timeouts, and `BrokenProcessPool` are handled without killing the study.
- Trials that fail or return a sentinel failure score are marked `FAIL` so the sampler avoids that region.

## Evaluation Protocol

Each trial scores one hyperparameter set in a worker process. The protocol controls variance:

- A fixed seed set is shared across all trials, so trials play the same maps.
- Every match is played in both seats (the candidate as player 0 and as player 1).
- The opponent pool is fixed for the study: every hand-crafted archetype plus the self-play checkpoints in `config/league.json`.
- The per-turn search budget is a runtime knob set by `--budget-ms`, applied through `main.set_hyperparameters(search_time_ms=...)`. It is not part of the Optuna search space.

The trial score is the win rate over all matches. Draws score half a win.

## Opponent League

`config/league.json` records named opponents with an Elo rating. Two kinds exist:

- `handcrafted`: deterministic Python policies in `opponents/`.
- `checkpoint`: saved engine hyperparameter sets, which make self-play opponents.

Promoting a champion registers it as a checkpoint. The league is never pruned of hand-crafted archetypes, which keeps the pool diverse and prevents mode collapse.

## Champion Promotion

`scripts/tune.py --promote` runs after a successful study and:

1. Writes the best parameters into the `champion` block of `config/hyperparameters.json`.
2. Registers the champion as a checkpoint in `config/league.json`.
3. Records a promotion note and timestamp.

`main.py` picks up the new champion on the next import.

## Budget Knob

`main.set_hyperparameters(search_time_ms=N)` caps the per-turn native search time. Tuning uses a reduced budget for cheap exploration and the full budget for final studies. The Kaggle submission keeps the full budget.

## CLI Reference

```text
usage: scripts/tune.py [options]

  --trials N            Total Optuna trials (default 200)
  --n-jobs N            Parallel worker processes (default 8)
  --seeds N             Seed count per trial (default 3)
  --max-steps N         Episode step cap for matches (default 499)
  --time-budget S       Study wall-clock seconds (default 3600)
  --storage URL         Optuna storage URL (default sqlite:///tune.db)
  --study-name NAME     Stable study identifier
  --base-seed N         Seed set base offset (default 42)
  --timeout S           Per-match wall-clock cap (default 60)
  --budget-ms N         Per-turn native search budget (default 500)
  --promote             Write the champion back on success
  --debug               Enable environment debug logging
```

## Worked Examples

Smoke study:

```bash
PYTHONPATH=build python scripts/tune.py \
  --trials 4 --n-jobs 2 --seeds 1 --max-steps 40 \
  --time-budget 300 --budget-ms 60 --timeout 30 \
  --storage sqlite:////tmp/maze-smoke.db --study-name smoke
```

Production study:

```bash
PYTHONPATH=build python scripts/tune.py \
  --trials 1000 --n-jobs 16 --seeds 5 \
  --time-budget 43200 --budget-ms 1500 --timeout 90 \
  --storage sqlite:///tune.db --study-name crawl-vs-league --promote
```

Use `make tune` to run the defaults.
