"""Per-turn latency profiling for the Maze Crawler agent.

Runs the native agent against a compact observation fixture for ``--calls``
invocations under a fixed per-turn budget and reports the latency distribution
(p50/p95/max) plus the most expensive Python-level hotspots.  ``--max-p95``
turns the p95 latency into a hard exit-code gate for CI.

Run:
    PYTHONPATH=build python scripts/profile.py --calls 30 --budget-ms 200
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main  # noqa: E402


def _fixture():
    return SimpleNamespace(
        player=0,
        walls=[0] * 400,
        crystals={},
        robots={
            "f0": [0, 5, 2, 1000, 0, 0, 0, 0],
            "w0": [2, 5, 4, 180, 0, 0, 0, 0],
            "s0": [1, 6, 4, 80, 0, 0, 0, 0],
        },
        mines={},
        miningNodes={"5,6": 1},
        southBound=0,
        northBound=19,
        step=0,
    )


def main_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--budget-ms", type=int, default=200)
    parser.add_argument("--max-p95", type=float, default=0.0, help="Fail if p95 exceeds this (ms)")
    parser.add_argument("--profile", action="store_true", help="Also dump a cProfile hotspot report")
    args = parser.parse_args(argv)

    main.set_hyperparameters(search_time_ms=args.budget_ms)
    obs = _fixture()
    config = SimpleNamespace(width=20, workerCost=200, wallRemoveCost=100)

    latencies = []
    for _ in range(args.calls):
        start = time.perf_counter()
        actions = main.agent(obs, config)
        latencies.append((time.perf_counter() - start) * 1000.0)
        assert isinstance(actions, dict) and "f0" in actions

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    peak = latencies[-1]
    print(f"calls={args.calls} budget_ms={args.budget_ms} "
          f"p50={p50:.1f}ms p95={p95:.1f}ms max={peak:.1f}ms")

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(5):
            main.agent(obs, config)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats(12)

    if args.max_p95 > 0 and p95 > args.max_p95:
        print(f"perf gate failed: p95={p95:.1f}ms > {args.max_p95}ms")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
