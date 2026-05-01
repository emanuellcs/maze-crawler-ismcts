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


def test_same_type_annihilation_and_crystal_consumption():
    robots = {
        "s0": [1, 5, 4, 50, 0, 0, 0, 0],
        "s1": [1, 5, 6, 50, 1, 0, 0, 0],
    }
    engine = make_engine(robots, crystals={"5,5": 30})
    engine.step({"s0": "NORTH", "s1": "SOUTH"})
    state = engine.debug_state()
    assert not any(r["uid"] in {"s0", "s1"} for r in state["robotList"])


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


def test_transfer_cap_and_mine_generation():
    robots = {
        "s0": [1, 5, 5, 90, 0, 0, 0, 0],
        "w0": [2, 5, 6, 10, 0, 0, 0, 0],
    }
    engine = make_engine(robots, mines={"5,5": [20, 1000, 0]})
    engine.step({"w0": "TRANSFER_SOUTH"})
    state = engine.debug_state()
    scout = robot_by_uid(state, "s0")
    worker = robot_by_uid(state, "w0")
    assert scout["energy"] == 100
    assert worker["energy"] == 0


def test_offboard_jump_death():
    engine = make_engine({"f0": [0, 5, 19, 1000, 0, 0, 0, 0]})
    engine.step({"f0": "JUMP_NORTH"})
    state = engine.debug_state()
    assert not any(r["uid"] == "f0" for r in state["robotList"])


if __name__ == "__main__":
    test_bridge_smoke()
    test_factory_build_spawn_before_combat()
    test_same_type_annihilation_and_crystal_consumption()
    test_worker_crushes_scout_and_gets_crystal()
    test_transfer_cap_and_mine_generation()
    test_offboard_jump_death()
    print("crawler_engine smoke tests passed")
