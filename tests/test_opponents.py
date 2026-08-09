"""Opponent archetype registration tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "name",
    ["baseline", "miner_rush", "worker_rush", "wall_turtle"],
)
def test_archetype_imports_and_returns_action_dict(name):
    """Every opponent archetype must expose a callable that returns a dict."""

    module = __import__(f"opponents.{name}", fromlist=["agent"])
    assert callable(module.agent)

    obs = SimpleNamespace(
        player=0,
        walls=[0] * 400,
        crystals={},
        robots={"f0": [0, 5, 2, 1000, 0, 0, 0, 0]},
        mines={},
        miningNodes={},
        southBound=0,
        northBound=19,
    )
    config = SimpleNamespace(
        width=20,
        workerCost=200,
        wallRemoveCost=100,
        minerCost=300,
        scoutCost=50,
    )
    actions = module.agent(obs, config)
    assert isinstance(actions, dict)
    for value in actions.values():
        assert isinstance(value, str)
