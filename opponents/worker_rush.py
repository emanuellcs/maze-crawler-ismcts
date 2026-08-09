"""Worker-rush archetype: early workers clear the lane and push north hard.

Builds up to three workers, opens north walls, and rushes vertical progress to
deny the opponent time.  Aggressive and tempo-driven; wins by cutting off the
opponent's economy rather than out-mining it.
"""

from opponents._maze_common import (
    FACTORY,
    OFFSETS,
    SCOUT,
    WORKER,
    path_to,
    split_robots,
)


def agent(obs, config):
    actions = {}
    mine, enemy_cells = split_robots(obs, obs.player)
    reserved = set()

    units = sorted(mine.items(), key=lambda x: (x[1][0], x[0]))
    factory = next((u for u, d in units if d[0] == FACTORY), None)
    workers = [u for u, d in units if d[0] == WORKER]
    scouts = [u for u, d in units if d[0] == SCOUT]

    if factory is not None:
        fc, fr, fe = mine[factory][1:4]
        fm, fj, fb = mine[factory][5:8] if len(mine[factory]) > 7 else (0, 0, 0)
        act = None

        if act is None and fb == 0 and fr + 1 <= obs.northBound:
            spawn = (fc, fr + 1)
            occupied = {(d[1], d[2]) for d in mine.values()}
            idx = (fr - obs.southBound) * config.width + fc
            north_open = not (0 <= idx < len(obs.walls) and obs.walls[idx] != -1 and (obs.walls[idx] & 1))
            if spawn not in occupied and north_open:
                if len(workers) < 3 and fe >= config.workerCost + 100:
                    act = "BUILD_WORKER"
                elif len(scouts) < 1 and fe >= config.scoutCost + 300:
                    act = "BUILD_SCOUT"

        if act is None and fm <= 1:
            step = path_to(obs, config, (fc, fr), [(fc, min(obs.northBound, fr + 30))],
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

    for uid in workers:
        wc, wr, we = mine[uid][1:4]
        wm = mine[uid][5] if len(mine[uid]) > 5 else 0
        act = None
        idx = (wr - obs.southBound) * config.width + wc
        wall_n = 0 <= idx < len(obs.walls) and obs.walls[idx] != -1 and (obs.walls[idx] & 1)
        if wall_n and we >= config.wallRemoveCost:
            act = "REMOVE_NORTH"
        elif wm <= 1:
            goal = (wc, min(obs.northBound, wr + 5))
            step = path_to(obs, config, (wc, wr), [goal], reserved | enemy_cells, 25)
            if step:
                act = step
        actions[uid] = act or "IDLE"
        if act in OFFSETS:
            reserved.add((wc + OFFSETS[act][0], wr + OFFSETS[act][1]))
        else:
            reserved.add((wc, wr))

    for uid in scouts:
        sc, sr = mine[uid][1:3]
        step = path_to(obs, config, (sc, sr), [(sc, min(obs.northBound, sr + 20))],
                       reserved | enemy_cells, 30)
        actions[uid] = step or "IDLE"

    return actions


def act(obs, config):
    try:
        return agent(obs, config)
    except Exception:
        return {}
