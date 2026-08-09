"""Pybind bridge tests: observation ingestion, hyperparameters, determinization."""

from __future__ import annotations

import pytest

from _fixtures import VALID_ACTIONS, make_engine


def test_bridge_smoke():
    """Verify the pybind Engine can ingest an observation and return actions."""

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


def test_determinize_does_not_duplicate_visible_enemies():
    """Visible enemy robots must be re-inserted once, never sampled twice."""

    # Factory at (5,2) with vision 4 sees (7,4); an enemy there is reported by
    # the environment and must be re-inserted exactly once.
    robots = {
        "f0": [0, 5, 2, 1000, 0, 0, 0, 0],
        "e0": [1, 7, 4, 50, 1, 0, 0, 0],
    }
    engine = make_engine(robots)
    for seed in range(1, 100):
        state = engine.determinize(seed)
        enemy_list = [r for r in state["robotList"] if r["owner"] == 1]
        visible = [r for r in enemy_list if r["uid"] == "e0"]
        assert len(visible) == 1, f"seed {seed}: visible enemy duplicated or missing"
        for r in enemy_list:
            if r["uid"] != "e0":
                assert r["uid"].startswith("sim-")
                assert not (r["col"] == 7 and r["row"] == 4), f"seed {seed}: synthetic enemy on visible cell"


def test_determinize_dedups_out_of_vision_reported_enemy():
    """Even a reported enemy outside the vision mask is never doubled."""

    # An enemy at (10,5) is outside the factory's vision. The environment never
    # reports such robots, but the dedup safety net must still prevent a
    # synthetic duplicate from surviving.
    robots = {
        "f0": [0, 5, 2, 1000, 0, 0, 0, 0],
        "e0": [1, 10, 5, 50, 1, 0, 0, 0],
    }
    engine = make_engine(robots)
    for seed in range(1, 100):
        state = engine.determinize(seed)
        enemy_list = [r for r in state["robotList"] if r["owner"] == 1]
        at_cell = [r for r in enemy_list if r["col"] == 10 and r["row"] == 5]
        assert len(at_cell) == 1, f"seed {seed}: enemy at (10,5) duplicated or missing"
        assert at_cell[0]["uid"] == "e0"


def test_hyperparameters_roundtrip_validation_and_action_generation():
    """Validate pybind hyperparameter IO and ensure search still returns actions."""

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
