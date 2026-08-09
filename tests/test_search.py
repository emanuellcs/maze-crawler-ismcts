"""ISMCTS search behavior tests."""

from __future__ import annotations

import time

from _fixtures import VALID_ACTIONS, make_engine


def test_mcts_tiebreak_value_uses_energy_margin():
    """Verify the rollout evaluator follows endgame energy tiebreak priority."""

    high_margin = make_engine(
        {
            "f0": [0, 5, 5, 2000, 0, 0, 0, 0],
            "f1": [0, 14, 5, 1000, 1, 0, 0, 0],
        },
        step=500,
    )
    low_margin = make_engine(
        {
            "f0": [0, 5, 5, 1200, 0, 0, 0, 0],
            "f1": [0, 14, 5, 1000, 1, 0, 0, 0],
        },
        step=500,
    )
    assert high_margin.debug_mcts_value(0) > low_margin.debug_mcts_value(0) > 0.0
    assert high_margin.debug_mcts_value(1) < 0.0


def test_mcts_respects_small_time_budget_and_returns_valid_actions():
    """Verify ISMCTS respects tiny budgets and returns controlled legal actions."""

    robots = {
        "f0": [0, 5, 2, 1000, 0, 0, 0, 0],
        "w0": [2, 5, 4, 180, 0, 0, 0, 0],
        "s0": [1, 6, 4, 80, 0, 0, 0, 0],
        "f1": [0, 14, 2, 1000, 1, 0, 0, 0],
        "s1": [1, 14, 5, 80, 1, 0, 0, 0],
    }
    engine = make_engine(robots, crystals={"6,5": 30}, nodes={"5,6": 1})
    start = time.perf_counter()
    actions = engine.choose_actions(5, seed=12345)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert elapsed_ms < 100.0
    assert set(actions) == {"f0", "w0", "s0"}
    assert set(actions.values()) <= VALID_ACTIONS
