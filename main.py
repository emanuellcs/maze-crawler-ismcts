"""Kaggle entrypoint for the Maze Crawler C++ engine."""

from __future__ import annotations

from random import choice

try:
    import crawler_engine
except Exception:  # pragma: no cover - fallback is for submission diagnostics.
    crawler_engine = None


_ENGINES = {}


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _cfg(config, name, default):
    return _get(config, name, default)


def _fallback_agent(obs, config):
    actions = {}
    width = _cfg(config, "width", 20)
    player = _get(obs, "player", 0)
    robots = _get(obs, "robots", {}) or {}
    walls = _get(obs, "walls", []) or []
    south_bound = _get(obs, "southBound", 0)

    my_robots = {uid: data for uid, data in robots.items() if data[4] == player}
    for uid, data in my_robots.items():
        rtype, col, row, energy = data[0], data[1], data[2], data[3]
        build_cd = data[7] if len(data) > 7 else 0
        idx = (row - south_bound) * width + col
        w = walls[idx] if 0 <= idx < len(walls) and walls[idx] != -1 else 0

        if rtype == 0:
            if w & 1:
                actions[uid] = "JUMP_NORTH"
            elif energy >= _cfg(config, "workerCost", 200) and build_cd == 0:
                actions[uid] = "BUILD_WORKER"
            else:
                actions[uid] = "NORTH"
        elif rtype == 2 and (w & 1) and energy >= _cfg(config, "wallRemoveCost", 100):
            actions[uid] = "REMOVE_NORTH"
        else:
            passable = []
            if not (w & 1):
                passable.append("NORTH")
            if not (w & 2):
                passable.append("EAST")
            if not (w & 4):
                passable.append("SOUTH")
            if not (w & 8):
                passable.append("WEST")
            actions[uid] = "NORTH" if "NORTH" in passable else (choice(passable) if passable else "IDLE")
    return actions


def agent(obs, config):
    if crawler_engine is None:
        return _fallback_agent(obs, config)

    player = int(_get(obs, "player", 0))
    engine = _ENGINES.get(player)
    if engine is None:
        engine = crawler_engine.Engine(player)
        _ENGINES[player] = engine

    step = int(_get(obs, "step", -1))
    engine.update_observation(
        player,
        _get(obs, "walls", []),
        _get(obs, "crystals", {}) or {},
        _get(obs, "robots", {}) or {},
        _get(obs, "mines", {}) or {},
        _get(obs, "miningNodes", {}) or {},
        int(_get(obs, "southBound", 0)),
        int(_get(obs, "northBound", 19)),
        step,
    )
    return engine.choose_actions(2000, seed=(step + 1) * 1315423911 + player)
