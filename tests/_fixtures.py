"""Shared fixtures for the Maze Crawler test modules.

The engine is exercised through the pybind bridge with compact Kaggle-style
observations: ``walls`` is a flat 400-cell active-window array, ``robots`` maps
UID to ``[type, col, row, energy, owner, move_cd, jump_cd, build_cd]``,
``crystals`` and ``nodes`` use ``{"col,row": value}``, and ``mines`` uses
``{"col,row": [energy, maxEnergy, owner]}``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import crawler_engine

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    """Build a deterministic open wall array with fixed border and center walls."""

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
    """Find a robot in ``debug_state`` output and fail with a useful message."""

    for robot in state["robotList"]:
        if robot["uid"] == uid:
            return robot
    raise AssertionError(f"missing robot {uid}")
