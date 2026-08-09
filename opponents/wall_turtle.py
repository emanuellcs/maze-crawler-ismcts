"""Wall-turtle archetype: defensive walls, lean economy, plays the tiebreak.

Builds and keeps one worker, fortifies the factory lane with walls, hoards
energy instead of over-building, and aims to out-survive the opponent into the
energy tiebreak.  Opposes tempo and economy rushes with restraint.
"""

from opponents._maze_common import (
    FACTORY,
    OFFSETS,
    WORKER,
    can_move,
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

    if factory is not None:
        fc, fr, fe = mine[factory][1:4]
        fm, fj, fb = mine[factory][5:8] if len(mine[factory]) > 7 else (0, 0, 0)
        act = None

        # Build exactly one worker for defense, then hoard.
        if act is None and fb == 0 and fr + 1 <= obs.northBound:
            spawn = (fc, fr + 1)
            occupied = {(d[1], d[2]) for d in mine.values()}
            if spawn not in occupied and len(workers) < 1 and fe >= config.workerCost + 500:
                act = "BUILD_WORKER"

        # Keep a healthy scroll margin; move north only when needed.
        if act is None and fm <= 1:
            if fr - obs.southBound <= 5:
                step = path_to(obs, config, (fc, fr), [(fc, min(obs.northBound, fr + 8))],
                               enemy_cells, 30, True, fj)
                if step:
                    act = step
        if act is None and fr - obs.southBound <= 2 and fj == 0 and obs.southBound > 0:
            act = "JUMP_NORTH"

        actions[factory] = act or "IDLE"
        if act in OFFSETS:
            reserved.add((fc + OFFSETS[act][0], fr + OFFSETS[act][1]))
        else:
            reserved.add((fc, fr))

    for uid in workers:
        wc, wr, we = mine[uid][1:4]
        wm = mine[uid][5] if len(mine[uid]) > 5 else 0
        act = None

        # Fortify: if the factory is south of us, wall it in; otherwise clear north.
        if factory is not None and (wc, wr) != (mine[factory][1], mine[factory][2]):
            fc, fr = mine[factory][1], mine[factory][2]
            if abs(wc - fc) + abs(wr - fr) == 1 and we >= config.wallBuildCost:
                for d, (dc, dr) in OFFSETS.items():
                    if wc + dc == fc and wr + dr == fr:
                        act = f"BUILD_{d}"
                        break

        if act is None:
            idx = (wr - obs.southBound) * config.width + wc
            wall_n = 0 <= idx < len(obs.walls) and obs.walls[idx] != -1 and (obs.walls[idx] & 1)
            if wall_n and we >= config.wallRemoveCost:
                act = "REMOVE_NORTH"
            elif wm <= 1 and (wr - obs.southBound) <= 6:
                step = path_to(obs, config, (wc, wr), [(wc, min(obs.northBound, wr + 3))],
                               reserved | enemy_cells, 20)
                if step:
                    act = step
        actions[uid] = act or "IDLE"
        if act and act in OFFSETS:
            reserved.add((wc + OFFSETS[act][0], wr + OFFSETS[act][1]))
        else:
            reserved.add((wc, wr))

    return actions


def act(obs, config):
    try:
        return agent(obs, config)
    except Exception:
        return {}
