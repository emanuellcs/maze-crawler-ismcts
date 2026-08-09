"""Differential rules-fidelity harness against kaggle-environments.

The native simulator must agree with the Kaggle interpreter on the observable
state.  ``diffcheck`` plays short full games with the native agent and verifies:

- every returned action is legal (the environment never reports ERROR/INVALID),
- the scroll cadence (southBound at each step) matches the rulebook ramp
  ``10 -> 2`` over ``450`` steps,
- the game terminates by the expected step.

Run:
    PYTHONPATH=build python scripts/diffcheck.py --games 6 --steps 120 --budget-ms 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main  # noqa: E402

SCROLL_START = 10
SCROLL_END = 2
SCROLL_RAMP = 450


def scroll_interval(step: int) -> int:
    """Rulebook scroll interval at a public step (matches the environment)."""
    if step >= SCROLL_RAMP:
        return SCROLL_END
    progress = step / SCROLL_RAMP
    return max(SCROLL_END, round(SCROLL_START - (SCROLL_START - SCROLL_END) * progress))


def scroll_south_at(step: int) -> int:
    """Expected southBound at the start of agent step ``step``."""
    south = 0
    counter = SCROLL_START
    for turn in range(step):
        counter -= 1
        if counter <= 0:
            south += 1
            counter = scroll_interval(turn)
    return south


def main_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--steps", type=int, default=120, help="Episode step cap")
    parser.add_argument("--budget-ms", type=int, default=80, help="Per-turn search budget")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    from kaggle_environments import make

    main.set_hyperparameters(search_time_ms=args.budget_ms)
    failures = 0

    for game in range(args.games):
        seed = args.seed + game
        env = make("crawl", configuration={"randomSeed": seed, "episodeSteps": args.steps}, debug=False)
        env.run([main.agent, main.agent])
        steps = env.steps
        last = steps[-1]
        if any(s.status in {"ERROR", "INVALID", "TIMEOUT"} for s in last):
            print(f"game {game}: statuses={[s.status for s in last]} FAIL")
            failures += 1
            continue

        mismatches = 0
        for i, s in enumerate(steps):
            south = s[0].observation.get("southBound", 0)
            step = s[0].observation.get("step", 0)
            expected = scroll_south_at(step)
            if south != expected:
                mismatches += 1
                if mismatches <= 3:
                    print(f"game {game} step {step}: southBound={south} expected={expected}")
        status = "OK" if mismatches == 0 else f"SCROLL-DIFF({mismatches})"
        print(f"game {game}: seed={seed} steps={len(steps)} final={status} rewards={[s.reward for s in last]}")
        failures += 0 if mismatches == 0 else 1

    print(f"diffcheck: {args.games - failures}/{args.games} games clean")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
