"""Miner-rush archetype: snowballs a mine economy, the 'AI TOOK MY JOB' style.

Opens with a scout to reveal mining nodes, commits a miner to the nearest node,
transforms it into a mine, and detours the factory to drain the mine's energy
before the scroll takes it.  Trades early army strength for compounding income.
"""

from opponents._maze_common import (
    FACTORY,
    MINER,
    OFFSETS,
    SCOUT,
    WORKER,
    can_move,
    path_to,
    split_robots,
    visible_crystals,
    visible_nodes,
)


def _north_open(obs, config, c, r):
    idx = (r - obs.southBound) * config.width + c
    if 0 <= idx < len(obs.walls) and obs.walls[idx] != -1:
        return not (obs.walls[idx] & 1)
    return True


def agent(obs, config):
    actions = {}
    mine, enemy_cells = split_robots(obs, obs.player)
    nodes = visible_nodes(obs)
    reserved = set()

    units = sorted(mine.items(), key=lambda x: (x[1][0], x[0]))
    factory = next((u for u, d in units if d[0] == FACTORY), None)
    miners = [u for u, d in units if d[0] == MINER]
    scouts = [u for u, d in units if d[0] == SCOUT]
    workers = [u for u, d in units if d[0] == WORKER]

    if factory is not None:
        fc, fr, fe = mine[factory][1:4]
        fm, fj, fb = mine[factory][5:8] if len(mine[factory]) > 7 else (0, 0, 0)
        act = None

        for key, data in obs.mines.items():
            if data[2] == obs.player and int(key.split(",")[1]) > obs.southBound:
                mc, mr = map(int, key.split(","))
                if abs(fc - mc) + abs(fr - mr) <= 6:
                    step = path_to(obs, config, (fc, fr), [(mc, mr)], enemy_cells, 30, True, fj)
                    if step:
                        act = step
                        break

        if act is None and fm <= 1 and fb == 0 and fr + 1 <= obs.northBound:
            spawn = (fc, fr + 1)
            occupied = {(d[1], d[2]) for d in mine.values()}
            if spawn not in occupied and _north_open(obs, config, fc, fr):
                if len(miners) < 1 and fe >= config.minerCost + 300:
                    act = "BUILD_MINER"
                elif len(scouts) < 1 and fe >= config.scoutCost + 300:
                    act = "BUILD_SCOUT"
                elif len(workers) < 1 and fe >= config.workerCost:
                    act = "BUILD_WORKER"

        if act is None and fm <= 1:
            step = path_to(obs, config, (fc, fr), [(fc, min(obs.northBound, fr + 25))],
                           enemy_cells, 40, True, fj)
            if step:
                act = step
        if act is None and fr - obs.southBound <= 3 and fj == 0 and obs.southBound > 0:
            act = "JUMP_NORTH"

        actions[factory] = act or "IDLE"
        if act and act in OFFSETS:
            reserved.add((fc + OFFSETS[act][0], fr + OFFSETS[act][1]))
        elif act and act.startswith("JUMP_"):
            d = act.split("_")[1]
            reserved.add((fc + 2 * OFFSETS[d][0], fr + 2 * OFFSETS[d][1]))
        else:
            reserved.add((fc, fr))

    for uid in miners:
        c, r, e = mine[uid][1:4]
        act = None
        if f"{c},{r}" in obs.miningNodes and e >= config.transformCost:
            act = "TRANSFORM"
        elif nodes:
            nearest = min(nodes, key=lambda p: abs(p[0] - c) + abs(p[1] - r))
            if abs(nearest[0] - c) + abs(nearest[1] - r) <= 20:
                step = path_to(obs, config, (c, r), [nearest], reserved | enemy_cells, 25)
                if step:
                    act = step
        actions[uid] = act or "IDLE"

    for uid in scouts:
        c, r = mine[uid][1:3]
        act = None
        goals = visible_crystals(obs)
        if goals:
            step = path_to(obs, config, (c, r), list(goals), reserved | enemy_cells, 30)
            if step:
                act = step
        if act is None:
            step = path_to(obs, config, (c, r), [(c, min(obs.northBound, r + 15))],
                           reserved | enemy_cells, 30)
            act = step
        actions[uid] = act or "IDLE"

    for uid in workers:
        actions[uid] = "IDLE"

    return actions


def act(obs, config):
    try:
        return agent(obs, config)
    except Exception:
        return {}
