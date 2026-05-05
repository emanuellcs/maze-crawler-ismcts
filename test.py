from __future__ import annotations

import numpy as np

import crawler_engine


WIDTH = 20
HEIGHT = 20
WALL_N, WALL_E, WALL_S, WALL_W = 1, 2, 4, 8


def open_walls():
    walls = np.zeros(WIDTH * HEIGHT, dtype=np.int16)
    for r in range(HEIGHT):
        walls[r * WIDTH] |= WALL_W
        walls[r * WIDTH + WIDTH - 1] |= WALL_E
        walls[r * WIDTH + 9] |= WALL_E
        walls[r * WIDTH + 10] |= WALL_W
    for c in range(WIDTH):
        walls[c] |= WALL_S
    return walls


def make_engine(robots, walls=None, crystals=None, mines=None, nodes=None, step=0):
    engine = crawler_engine.Engine(0)
    engine.update_observation(
        0,
        open_walls() if walls is None else walls,
        crystals or {},
        robots,
        mines or {},
        nodes or {},
        0,
        19,
        step,
    )
    return engine


def robot_by_uid(state, uid):
    for robot in state["robotList"]:
        if robot["uid"] == uid:
            return robot
    raise AssertionError(f"missing robot {uid}")


def test_bridge_smoke():
    engine = make_engine({"f0": [0, 5, 2, 1000, 0, 0, 0, 0]})
    actions = engine.choose_actions(10, seed=1)
    assert isinstance(actions, dict)
    assert actions["f0"] in {"BUILD_WORKER", "BUILD_SCOUT", "NORTH", "JUMP_NORTH", "IDLE"}
    assert engine.determinize(1)["southBound"] == 0


def test_factory_build_spawn_before_combat():
    engine = make_engine({"f0": [0, 5, 2, 1000, 0, 0, 0, 0]})
    engine.step({"f0": "BUILD_WORKER"})
    state = engine.debug_state()
    assert state["robots"] == 2
    assert any(r["type"] == 2 and r["col"] == 5 and r["row"] == 3 for r in state["robotList"])
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


if __name__ == "__main__":
    test_bridge_smoke()
    test_factory_build_spawn_before_combat()
    test_same_type_annihilation_and_crystal_consumption()
    test_enemy_factories_mutually_annihilate()
    test_worker_crushes_scout_and_gets_crystal()
    test_transfer_drains_source_and_caps_target()
    test_period_two_move_cooldown_blocks_next_turn()
    test_jump_sets_move_and_jump_cooldowns()
    test_fixed_center_wall_remove_costs_but_does_not_open()
    test_offboard_jump_death()
    print("crawler_engine smoke tests passed")
