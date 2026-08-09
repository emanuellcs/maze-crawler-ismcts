"""Benchmark harness: measure the champion against the league.

``bench.py`` is the propose-measure-keep loop.  It plays the current champion
against a held-out seed set with full seat rotation against every opponent in
``config/league.json`` plus the hand-crafted archetypes, reports win rates with
a binomial confidence interval, updates league Elo ratings, and can gate a
promotion against the incumbent.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import math
from pathlib import Path

multiprocessing.set_start_method("spawn", force=True)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main  # noqa: E402
from tuning import evaluation, league as league_mod, reporting, schema  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("bench")

HANDCRAFTED_OPPONENTS = ("baseline", "miner_rush", "worker_rush", "wall_turtle")


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _make_opponent_agent(kind, name, params):
    if kind == "handcrafted":
        module = __import__(f"opponents.{name}", fromlist=["agent"])
        return module.agent

    def checkpoint_agent(obs, config):
        main.set_hyperparameters(**params)
        return main.agent(obs, config)

    return checkpoint_agent


def _final_energy(final_state, owner):
    robots = _get(_get(final_state, "observation", {}) or {}, "robots", {}) or {}
    return sum(int(d[3]) for d in robots.values() if len(d) >= 5 and int(d[4]) == owner)


def _outcome(seed, candidate_seat, opponent_agent, timeout, max_steps=499):
    from kaggle_environments import make

    agents = [None, None]
    agents[candidate_seat] = main.agent
    agents[1 - candidate_seat] = opponent_agent
    env = make("crawl", configuration={"randomSeed": seed, "episodeSteps": max_steps}, debug=False)
    steps = evaluation.run_with_timeout(lambda: env.run(agents), timeout)
    if steps is None:
        return "error", 0.0
    final = steps[-1]
    if evaluation.state_failed(_get(final[candidate_seat], "status", "")):
        return "loss", 0.0
    if evaluation.state_failed(_get(final[1 - candidate_seat], "status", "")):
        return "win", 1.0
    reward = _get(final[candidate_seat], "reward", 0)
    return ("win" if reward > 0.5 else "draw" if reward == 0.5 else "loss",
            float(reward > 0.5) + 0.5 * float(reward == 0.5))


def wilson_ci(wins, games, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if games == 0:
        return 0.0, 0.0, 0.0
    p = wins / games
    denom = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def main_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20, help="Held-out seed count")
    parser.add_argument("--base-seed", type=int, default=9000, help="Held-out seed offset")
    parser.add_argument("--budget-ms", type=int, default=1000, help="Per-turn search budget")
    parser.add_argument("--max-steps", type=int, default=499, help="Episode step cap for matches")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--elo", action="store_true", help="Update league Elo after measuring")
    parser.add_argument("--gate", type=float, default=0.0,
                        help="Promote champion only if WR vs incumbent exceeds this")
    args = parser.parse_args(argv)

    hyper_config = _ROOT / "config" / "hyperparameters.json"
    league_path = _ROOT / "config" / "league.json"
    sch = schema.Schema.load(hyper_config)
    champion_params = dict(sch.defaults())
    champion_params.update(reporting.read_champion(hyper_config))

    lg = league_mod.League(league_path)
    opponents = [("handcrafted", n, None) for n in HANDCRAFTED_OPPONENTS]
    opponents += [("checkpoint", e["name"], e.get("params", {})) for e in lg.entries
                  if e["name"] not in HANDCRAFTED_OPPONENTS]

    champion_params["search_time_ms"] = args.budget_ms
    main.set_hyperparameters(**champion_params)

    _LOG.info("benchmarking champion vs %d opponents on %d seeds (budget %d ms)",
              len(opponents), args.seeds, args.budget_ms)
    for kind, name, params in opponents:
        opponent = _make_opponent_agent(kind, name, params)
        wins = games = 0
        for i in range(args.seeds):
            main.set_hyperparameters(**champion_params)
            for seat in (0, 1):
                _, score = _outcome(args.base_seed + i, seat, opponent, args.timeout, args.max_steps)
                wins += score
                games += 1
            if args.elo and i % 2 == 1 and games:
                _update_elo(lg, "champion", name, wins, games)
        p, lo, hi = wilson_ci(wins, games)
        print(f"vs {name:<24} {games:4d} games  WR={p:.3f}  CI=[{lo:.3f},{hi:.3f}]")

    if args.gate and args.elo:
        incumbent = _pick_incumbent(lg)
        if incumbent:
            champ_wr = _head_to_head(lg, "champion", incumbent, args)
            print(f"champion vs incumbent {incumbent}: WR={champ_wr:.3f}")
            if champ_wr < args.gate:
                print(f"gate not met ({champ_wr:.3f} < {args.gate}); champion NOT promoted")
                return 1
            print("gate met; promoting champion")
            reporting.promote_champion(hyper_config, league_path, "champion",
                                       champion_params, note="bench-gated promotion")
    return 0


def _read_champion(config_path):
    import json

    try:
        data = json.loads(Path(config_path).read_text())
        return dict(data.get("champion", {}))
    except Exception:
        return {}


def _pick_incumbent(lg):
    """Highest-rated checkpoint other than 'champion'."""
    cps = [e for e in lg.checkpoints() if e["name"] != "champion"]
    return sorted(cps, key=lambda e: -e["rating"])[0]["name"] if cps else None


def _update_elo(lg, champ, opp, wins, games):
    if games and wins / games >= 0.5:
        lg.update_elo(champ, opp)
    elif games:
        lg.update_elo(opp, champ)


def _head_to_head(lg, champ, opp, args):
    opponent = _make_opponent_agent("checkpoint", opp, lg.get(opp).get("params", {}))
    wins = games = 0
    champion_params = dict(schema.Schema.load(_ROOT / "config" / "hyperparameters.json").defaults())
    champion_params.update(reporting.read_champion(_ROOT / "config" / "hyperparameters.json"))
    for i in range(args.seeds):
        main.set_hyperparameters(**champion_params)
        for seat in (0, 1):
            _, score = _outcome(args.base_seed + i, seat, opponent, args.timeout, args.max_steps)
            wins += score
            games += 1
    return wins / max(1, games)


if __name__ == "__main__":
    raise SystemExit(main_cli())
