"""Deterministic baseline policy tests for the macro library."""

from __future__ import annotations

from _fixtures import WIDTH, WALL_N, open_walls, make_engine


def test_opponent_policy_factory_supports_adjacent_worker():
    """Protect the Factory support macro that transfers energy to Workers."""

    engine = make_engine(
        {
            "f0": [0, 5, 2, 1000, 0, 0, 0, 0],
            "w0": [2, 5, 3, 50, 0, 0, 0, 0],
        }
    )

    actions = engine.choose_actions(0, seed=1)
    assert actions["f0"] == "TRANSFER_NORTH"


def test_opponent_policy_factory_emergency_jumps_north():
    """Protect the emergency Factory jump used near the scrolling south bound."""

    engine = make_engine(
        {"f0": [0, 5, 6, 1000, 0, 0, 0, 0]},
        south_bound=5,
        north_bound=24,
    )

    actions = engine.choose_actions(0, seed=1)
    assert actions["f0"] == "JUMP_NORTH"


def test_opponent_policy_worker_removes_blocking_north_wall():
    """Protect the Worker wall-opening macro used by the baseline policy."""

    walls = open_walls()
    walls[2 * WIDTH + 5] |= WALL_N
    engine = make_engine({"w0": [2, 5, 2, 150, 0, 0, 0, 0]}, walls=walls)

    actions = engine.choose_actions(0, seed=1)
    assert actions["w0"] == "REMOVE_NORTH"


def test_opponent_policy_scout_transfers_to_adjacent_factory():
    """Protect Scout-to-Factory energy return when adjacent."""

    engine = make_engine(
        {
            "f0": [0, 5, 2, 1000, 0, 0, 0, 0],
            "s0": [1, 5, 3, 50, 0, 0, 0, 0],
        }
    )

    actions = engine.choose_actions(0, seed=1)
    assert actions["s0"] == "TRANSFER_SOUTH"


def test_opponent_policy_scout_returns_when_loaded():
    """Protect Scout return routing when carrying high energy."""

    engine = make_engine(
        {
            "f0": [0, 5, 2, 1000, 0, 0, 0, 0],
            "s0": [1, 6, 3, 90, 0, 0, 0, 0],
        }
    )

    actions = engine.choose_actions(0, seed=1)
    assert actions["s0"] == "SOUTH"


def test_opponent_policy_scout_hunts_visible_crystal():
    """Protect Scout crystal targeting from sparse visible crystal observations."""

    engine = make_engine(
        {"s0": [1, 5, 2, 20, 0, 0, 0, 0]},
        crystals={"5,4": 25},
    )

    actions = engine.choose_actions(0, seed=1)
    assert actions["s0"] == "NORTH"


def test_opponent_policy_scout_explores_north_without_crystals():
    """Protect Scout exploration when no crystals are visible."""

    engine = make_engine({"s0": [1, 5, 2, 20, 0, 0, 0, 0]})

    actions = engine.choose_actions(0, seed=1)
    assert actions["s0"] == "NORTH"
