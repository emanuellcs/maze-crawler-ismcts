"""Rule simulator tests: turn phases, combat, transfers, scroll, and jumps."""

from __future__ import annotations

from _fixtures import WIDTH, WALL_N, make_engine, robot_by_uid


def test_factory_build_spawn_before_combat():
    """Verify Factory spawns participate as stationary combatants immediately."""

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
    """Verify same-type collision annihilates both robots and consumes crystals."""

    robots = {
        "s0": [1, 5, 4, 50, 0, 0, 0, 0],
        "s1": [1, 5, 6, 50, 1, 0, 0, 0],
    }
    engine = make_engine(robots, crystals={"5,5": 30})
    engine.step({"s0": "NORTH", "s1": "SOUTH"})
    state = engine.debug_state()
    assert not any(r["uid"] in {"s0", "s1"} for r in state["robotList"])


def test_enemy_factories_mutually_annihilate():
    """Verify opposing Factory collision removes both factories and ends the game."""

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
    """Verify crush hierarchy and survivor crystal collection."""

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
    """Verify transfers drain the source and cap non-Factory targets."""

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
    """Verify Factory energy storage handles values beyond int16 range."""

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
    """Verify spawned units can be crushed on the spawn cell during movement."""

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
    """Verify observation reload reconstructs scroll timing at late steps."""

    robots = {
        "f0": [0, 5, 0, 1000, 0, 0, 0, 0],
        "f1": [0, 14, 5, 1000, 1, 0, 0, 0],
    }
    # At step 400 the env cadence (start 10 / ramp 450 / end 2) has two turns
    # until the next boundary advance, so a single step does not scroll.
    engine = make_engine(robots, step=400)
    engine.step({})
    state = engine.debug_state()
    assert state["southBound"] == 0
    assert any(r["uid"] == "f0" for r in state["robotList"])

    # At step 401 exactly one turn remains, so the next step scrolls.
    engine = make_engine(robots, step=401)
    engine.step({})
    state = engine.debug_state()
    assert state["southBound"] == 1
    assert not any(r["uid"] == "f0" for r in state["robotList"])


def test_period_two_move_cooldown_blocks_next_turn():
    """Verify period-two units must wait one turn before moving again."""

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
    """Verify Factory jumps set both movement and jump cooldowns."""

    engine = make_engine({"f0": [0, 5, 5, 1000, 0, 0, 0, 0]})
    engine.step({"f0": "JUMP_NORTH"})
    state = engine.debug_state()
    factory = robot_by_uid(state, "f0")
    assert factory["row"] == 7
    assert factory["move_cd"] == 2
    assert factory["jump_cd"] == 20


def test_fixed_center_wall_remove_costs_but_does_not_open():
    """Verify fixed mirror-axis wall edits cost energy but preserve geometry."""

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
    """Verify Factory jumps landing outside the active board destroy the unit."""

    engine = make_engine({"f0": [0, 5, 19, 1000, 0, 0, 0, 0]})
    engine.step({"f0": "JUMP_NORTH"})
    state = engine.debug_state()
    assert not any(r["uid"] == "f0" for r in state["robotList"])
