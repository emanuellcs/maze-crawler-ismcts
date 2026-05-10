"""Regression tests for rules fidelity, search behavior, pybind, and packaging."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np
import pytest

import crawler_engine
import package_submission

WIDTH = 20
HEIGHT = 20
WALL_N, WALL_E, WALL_S, WALL_W = 1, 2, 4, 8
VALID_ACTIONS = {
    "IDLE",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "BUILD_SCOUT",
    "BUILD_WORKER",
    "BUILD_MINER",
    "JUMP_NORTH",
    "JUMP_SOUTH",
    "JUMP_EAST",
    "JUMP_WEST",
    "BUILD_NORTH",
    "BUILD_SOUTH",
    "BUILD_EAST",
    "BUILD_WEST",
    "REMOVE_NORTH",
    "REMOVE_SOUTH",
    "REMOVE_EAST",
    "REMOVE_WEST",
    "TRANSFORM",
    "TRANSFER_NORTH",
    "TRANSFER_SOUTH",
    "TRANSFER_EAST",
    "TRANSFER_WEST",
}


def open_walls():
    """Build a deterministic mostly-open wall array with fixed border/center walls."""

    walls = np.zeros(WIDTH * HEIGHT, dtype=np.int16)
    for r in range(HEIGHT):
        walls[r * WIDTH] |= WALL_W
        walls[r * WIDTH + WIDTH - 1] |= WALL_E
        walls[r * WIDTH + 9] |= WALL_E
        walls[r * WIDTH + 10] |= WALL_W
    for c in range(WIDTH):
        walls[c] |= WALL_S
    return walls


def make_engine(
    robots,
    walls=None,
    crystals=None,
    mines=None,
    nodes=None,
    step=0,
    south_bound=0,
    north_bound=19,
):
    """Create an engine from a compact Kaggle-style observation fixture."""

    engine = crawler_engine.Engine(0)
    engine.update_observation(
        0,
        open_walls() if walls is None else walls,
        crystals or {},
        robots,
        mines or {},
        nodes or {},
        south_bound,
        north_bound,
        step,
    )
    return engine


def robot_by_uid(state, uid):
    """Find a robot in debug_state output and fail with a useful test message."""

    for robot in state["robotList"]:
        if robot["uid"] == uid:
            return robot
    raise AssertionError(f"missing robot {uid}")


def test_bridge_smoke():
    engine = make_engine({"f0": [0, 5, 2, 1000, 0, 0, 0, 0]})
    actions = engine.choose_actions(10, seed=1)
    assert isinstance(actions, dict)
    assert actions["f0"] in {
        "BUILD_WORKER",
        "BUILD_SCOUT",
        "NORTH",
        "JUMP_NORTH",
        "IDLE",
    }
    assert engine.determinize(1)["southBound"] == 0


def test_hyperparameters_roundtrip_validation_and_action_generation():
    engine = make_engine({"f0": [0, 5, 2, 1000, 0, 0, 0, 0]})
    defaults = engine.get_hyperparameters()
    assert defaults["C_puct"] == pytest.approx(2.0884330868271443)
    assert defaults["baseline_prior_multiplier"] == pytest.approx(1.8863044112273712)
    assert defaults["rollout_depth"] == 80
    assert defaults["FACTORY_SUPPORT_WORKER"] == pytest.approx(1.0390992283842135)
    assert defaults["FACTORY_BUILD_WORKER"] == pytest.approx(0.997532683560502)

    engine.set_hyperparameters(
        {
            "C_puct": 2.0,
            "baseline_prior_multiplier": 1.1,
            "rollout_depth": 16,
            "FACTORY_BUILD_WORKER": 0.5,
        }
    )
    updated = engine.get_hyperparameters()
    assert updated["C_puct"] == pytest.approx(2.0)
    assert updated["baseline_prior_multiplier"] == pytest.approx(1.1)
    assert updated["rollout_depth"] == 16
    assert updated["FACTORY_BUILD_WORKER"] == pytest.approx(0.5)

    with pytest.raises(KeyError):
        engine.set_hyperparameters({"NOT_A_PARAMETER": 1.0})
    with pytest.raises(ValueError):
        engine.set_hyperparameters({"C_puct": 0.0})
    with pytest.raises(ValueError):
        engine.set_hyperparameters({"rollout_depth": 0})

    actions = engine.choose_actions(5, seed=7)
    assert actions["f0"] in VALID_ACTIONS


def test_opponent_policy_factory_supports_adjacent_worker():
    engine = make_engine(
        {
            "f0": [0, 5, 2, 1000, 0, 0, 0, 0],
            "w0": [2, 5, 3, 50, 0, 0, 0, 0],
        }
    )

    actions = engine.choose_actions(0, seed=1)
    assert actions["f0"] == "TRANSFER_NORTH"


def test_opponent_policy_factory_emergency_jumps_north():
    engine = make_engine(
        {"f0": [0, 5, 6, 1000, 0, 0, 0, 0]},
        south_bound=5,
        north_bound=24,
    )

    actions = engine.choose_actions(0, seed=1)
    assert actions["f0"] == "JUMP_NORTH"


def test_opponent_policy_worker_removes_blocking_north_wall():
    walls = open_walls()
    walls[2 * WIDTH + 5] |= WALL_N
    engine = make_engine({"w0": [2, 5, 2, 150, 0, 0, 0, 0]}, walls=walls)

    actions = engine.choose_actions(0, seed=1)
    assert actions["w0"] == "REMOVE_NORTH"


def test_opponent_policy_scout_transfers_to_adjacent_factory():
    engine = make_engine(
        {
            "f0": [0, 5, 2, 1000, 0, 0, 0, 0],
            "s0": [1, 5, 3, 50, 0, 0, 0, 0],
        }
    )

    actions = engine.choose_actions(0, seed=1)
    assert actions["s0"] == "TRANSFER_SOUTH"


def test_opponent_policy_scout_returns_when_loaded():
    engine = make_engine(
        {
            "f0": [0, 5, 2, 1000, 0, 0, 0, 0],
            "s0": [1, 6, 3, 90, 0, 0, 0, 0],
        }
    )

    actions = engine.choose_actions(0, seed=1)
    assert actions["s0"] == "SOUTH"


def test_opponent_policy_scout_hunts_visible_crystal():
    engine = make_engine(
        {"s0": [1, 5, 2, 20, 0, 0, 0, 0]},
        crystals={"5,4": 25},
    )

    actions = engine.choose_actions(0, seed=1)
    assert actions["s0"] == "NORTH"


def test_opponent_policy_scout_explores_north_without_crystals():
    engine = make_engine({"s0": [1, 5, 2, 20, 0, 0, 0, 0]})

    actions = engine.choose_actions(0, seed=1)
    assert actions["s0"] == "NORTH"


def test_policy_source_avoids_dynamic_allocation_primitives():
    with open(
        os.path.join(os.path.dirname(__file__), "src", "crawler_engine_policy.cpp"),
        encoding="utf-8",
    ) as policy_file:
        policy_source = policy_file.read()

    for forbidden in ("std::vector", "std::deque", "std::set", "new"):
        assert forbidden not in policy_source


def test_factory_build_spawn_before_combat():
    engine = make_engine({"f0": [0, 5, 2, 1000, 0, 0, 0, 0]})
    engine.step({"f0": "BUILD_WORKER"})
    state = engine.debug_state()
    assert state["robots"] == 2
    assert any(
        r["type"] == 2 and r["col"] == 5 and r["row"] == 3 for r in state["robotList"]
    )
    worker = next(r for r in state["robotList"] if r["type"] == 2)
    assert worker["move_cd"] == 2


def test_same_type_annihilation_and_crystal_consumption():
    robots = {
        "s0": [1, 5, 4, 50, 0, 0, 0, 0],
        "s1": [1, 5, 6, 50, 1, 0, 0, 0],
    }
    engine = make_engine(robots, crystals={"5,5": 30})
    engine.step({"s0": "NORTH", "s1": "SOUTH"})
    state = engine.debug_state()
    assert not any(r["uid"] in {"s0", "s1"} for r in state["robotList"])


def test_enemy_factories_mutually_annihilate():
    robots = {
        "f0": [0, 5, 4, 1000, 0, 0, 0, 0],
        "f1": [0, 5, 6, 1000, 1, 0, 0, 0],
    }
    engine = make_engine(robots)
    engine.step({"f0": "NORTH", "f1": "SOUTH"})
    state = engine.debug_state()
    assert not any(r["uid"] in {"f0", "f1"} for r in state["robotList"])
    assert state["done"]


def test_worker_crushes_scout_and_gets_crystal():
    robots = {
        "w0": [2, 5, 4, 100, 0, 0, 0, 0],
        "s1": [1, 5, 6, 50, 1, 0, 0, 0],
    }
    engine = make_engine(robots, crystals={"5,5": 30})
    engine.step({"w0": "NORTH", "s1": "SOUTH"})
    state = engine.debug_state()
    worker = robot_by_uid(state, "w0")
    assert worker["row"] == 5
    assert worker["energy"] == 129
    assert not any(r["uid"] == "s1" for r in state["robotList"])


def test_transfer_drains_source_and_caps_target():
    robots = {
        "s0": [1, 5, 5, 99, 0, 0, 0, 0],
        "w0": [2, 5, 6, 50, 0, 0, 0, 0],
    }
    engine = make_engine(robots)
    engine.step({"w0": "TRANSFER_SOUTH"})
    state = engine.debug_state()
    scout = robot_by_uid(state, "s0")
    worker = robot_by_uid(state, "w0")
    assert scout["energy"] == 100
    assert worker["energy"] == 0


def test_transfer_to_factory_does_not_overflow_int16_range():
    robots = {
        "f0": [0, 5, 5, 40000, 0, 0, 0, 0],
        "w0": [2, 5, 6, 300, 0, 0, 0, 0],
    }
    engine = make_engine(robots)
    engine.step({"w0": "TRANSFER_SOUTH"})
    state = engine.debug_state()
    factory = robot_by_uid(state, "f0")
    worker = robot_by_uid(state, "w0")
    assert factory["energy"] == 40298
    assert worker["energy"] == 0


def test_factory_spawn_is_stationary_combat_participant():
    robots = {
        "f0": [0, 5, 2, 1000, 0, 0, 0, 0],
        "w1": [2, 5, 4, 300, 1, 0, 0, 0],
    }
    engine = make_engine(robots)
    engine.step({"f0": "BUILD_SCOUT", "w1": "SOUTH"})
    state = engine.debug_state()
    enemy_worker = robot_by_uid(state, "w1")
    assert enemy_worker["row"] == 3
    assert not any(
        r["owner"] == 0 and r["type"] == 1 and r["row"] == 3 for r in state["robotList"]
    )


def test_scroll_counter_reconstructed_for_high_step_observation():
    robots = {
        "f0": [0, 5, 0, 1000, 0, 0, 0, 0],
        "f1": [0, 14, 5, 1000, 1, 0, 0, 0],
    }
    engine = make_engine(robots, step=400)
    engine.step({})
    state = engine.debug_state()
    assert state["southBound"] == 1
    assert not any(r["uid"] == "f0" for r in state["robotList"])


def test_mcts_tiebreak_value_uses_energy_margin():
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


def test_period_two_move_cooldown_blocks_next_turn():
    robots = {
        "f0": [0, 2, 2, 1000, 0, 0, 0, 0],
        "f1": [0, 15, 2, 1000, 1, 0, 0, 0],
        "w0": [2, 5, 5, 100, 0, 0, 0, 0],
    }
    engine = make_engine(robots)
    engine.step({"w0": "NORTH"})
    state = engine.debug_state()
    worker = robot_by_uid(state, "w0")
    assert worker["row"] == 6
    assert worker["move_cd"] == 2

    engine.step({"w0": "NORTH"})
    state = engine.debug_state()
    worker = robot_by_uid(state, "w0")
    assert worker["row"] == 6
    assert worker["move_cd"] == 1

    engine.step({"w0": "NORTH"})
    state = engine.debug_state()
    worker = robot_by_uid(state, "w0")
    assert worker["row"] == 7
    assert worker["move_cd"] == 2


def test_jump_sets_move_and_jump_cooldowns():
    engine = make_engine({"f0": [0, 5, 5, 1000, 0, 0, 0, 0]})
    engine.step({"f0": "JUMP_NORTH"})
    state = engine.debug_state()
    factory = robot_by_uid(state, "f0")
    assert factory["row"] == 7
    assert factory["move_cd"] == 2
    assert factory["jump_cd"] == 20


def test_fixed_center_wall_remove_costs_but_does_not_open():
    robots = {
        "f0": [0, 2, 2, 1000, 0, 0, 0, 0],
        "f1": [0, 15, 2, 1000, 1, 0, 0, 0],
        "w0": [2, 9, 5, 150, 0, 0, 0, 0],
    }
    engine = make_engine(robots)
    engine.step({"w0": "REMOVE_EAST"})
    state = engine.debug_state()
    worker = robot_by_uid(state, "w0")
    assert worker["energy"] == 49

    engine.step({"w0": "EAST"})
    state = engine.debug_state()
    worker = robot_by_uid(state, "w0")
    assert worker["col"] == 9


def test_offboard_jump_death():
    engine = make_engine({"f0": [0, 5, 19, 1000, 0, 0, 0, 0]})
    engine.step({"f0": "JUMP_NORTH"})
    state = engine.debug_state()
    assert not any(r["uid"] == "f0" for r in state["robotList"])


def test_mcts_respects_small_time_budget_and_returns_valid_actions():
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


def test_packaged_submission_jit_compiles_in_extracted_directory():
    package_path = package_submission.build_package()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(package_path, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        smoke = subprocess.run(
            [sys.executable, "main.py"],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert smoke.returncode == 0, smoke.stdout + smoke.stderr
        assert "crawler_engine native available: True" in smoke.stdout

        call = subprocess.run(
            [
                sys.executable,
                "-c",
                "from types import SimpleNamespace\n"
                "from main import agent\n"
                "obs = SimpleNamespace(player=0, walls=[0] * 400, crystals={}, "
                "robots={'f0': [0, 5, 2, 1000, 0, 0, 0, 0]}, mines={}, "
                "miningNodes={}, southBound=0, northBound=19, step=0)\n"
                "config = SimpleNamespace(width=20, workerCost=200, wallRemoveCost=100)\n"
                "actions = agent(obs, config)\n"
                "assert 'f0' in actions\n"
                "print(actions)\n",
            ],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert call.returncode == 0, call.stdout + call.stderr


if __name__ == "__main__":
    test_bridge_smoke()
    test_hyperparameters_roundtrip_validation_and_action_generation()
    test_factory_build_spawn_before_combat()
    test_same_type_annihilation_and_crystal_consumption()
    test_enemy_factories_mutually_annihilate()
    test_worker_crushes_scout_and_gets_crystal()
    test_transfer_drains_source_and_caps_target()
    test_transfer_to_factory_does_not_overflow_int16_range()
    test_factory_spawn_is_stationary_combat_participant()
    test_scroll_counter_reconstructed_for_high_step_observation()
    test_mcts_tiebreak_value_uses_energy_margin()
    test_period_two_move_cooldown_blocks_next_turn()
    test_jump_sets_move_and_jump_cooldowns()
    test_fixed_center_wall_remove_costs_but_does_not_open()
    test_offboard_jump_death()
    test_mcts_respects_small_time_budget_and_returns_valid_actions()
    test_packaged_submission_jit_compiles_in_extracted_directory()
    print("crawler_engine smoke tests passed")
