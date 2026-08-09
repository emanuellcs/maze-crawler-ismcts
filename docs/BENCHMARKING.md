# Benchmarking

This document describes how the Maze Crawler engine is measured: the benchmark harness, the evaluation protocol, the confidence interval, the Elo ladder, and the promotion gate.

## Overview

`scripts/bench.py` is the propose-measure-keep loop. It plays the current champion against every opponent on held-out seeds with full seat rotation, reports win rates with a binomial confidence interval, and can gate a promotion against the incumbent.

## Running the Harness

```bash
make bench
```

or:

```bash
PYTHONPATH=build python scripts/bench.py \
  --seeds 50 --base-seed 9000 --budget-ms 1000 --timeout 90
```

The champion is loaded from `config/hyperparameters.json`. Opponents are the hand-crafted archetypes plus the self-play checkpoints in `config/league.json`.

## CLI Reference

```text
usage: scripts/bench.py [options]

  --seeds N            Held-out seed count (default 20)
  --base-seed N        Held-out seed offset (default 9000)
  --budget-ms N        Per-turn search budget (default 1000)
  --max-steps N        Episode step cap for matches (default 499)
  --timeout S          Per-match wall-clock cap (default 90)
  --elo                Update league Elo after measuring
  --gate F             Promote only if WR vs incumbent exceeds F
```

## Evaluation Protocol

- A held-out seed set is separate from the tuning seed set, so promotion decisions are not overfit to tuning maps.
- Every opponent is played in both seats on every seed.
- The champion uses the parameters from `config/hyperparameters.json` and the budget from `--budget-ms`.
- Matches are wall-clock bounded so a hung worker cannot stall the run.

## Confidence Interval

Win rates are reported with a Wilson score interval at 95 percent confidence:

```math
\frac{
  \hat{p} + \frac{z^2}{2n}
  \pm z\sqrt{\frac{\hat{p}(1 - \hat{p}) + z^2/(4n)}{n}}
}{1 + z^2/n}
```

where `p_hat` is the observed win rate and `n` is the number of games.

The interval makes the sample size explicit. A win rate from two games has a wide interval; a promotion should be gated on enough games for the interval to exclude the target threshold.

## Elo Ladder

With `--elo`, the harness updates league ratings in `config/league.json` after each opponent block. A standard Elo update with `K = 32` is applied based on the observed win rate. The ladder doubles as a progress meter across versions.

## Promotion Gate

The `--gate` flag compares the champion to the incumbent (the highest-rated checkpoint other than `champion`). When the head-to-head win rate meets or exceeds the gate, the champion is promoted:

1. `config/hyperparameters.json` `champion` is updated.
2. The champion is registered in `config/league.json`.

A sensible gate is 0.55 to 0.60 with enough games that the confidence interval excludes 0.50.

## CI Gate

The continuous integration workflow runs a small benchmark smoke (`--seeds 2 --budget-ms 80 --max-steps 40`) to confirm the harness runs. It also runs a profiling perf gate (`scripts/profiling.py --max-p95`) to catch latency regressions.

## Report Output

Reports can be written to `docs/benchmarks/` with the date, seed set, budget, and per-opponent results so measurements remain reproducible and comparable.
