"""Crawl starter agent: build a worker, march north, knock down walls."""

from random import choice


def agent(obs, config):
    actions = {}
    width = config.width
    my_robots = {
        uid: data for uid, data in obs.robots.items()
        if data[4] == obs.player
    }

    for uid, data in my_robots.items():
        rtype, col, row, energy = data[0], data[1], data[2], data[3]
        build_cd = data[7] if len(data) > 7 else 0

        idx = (row - obs.southBound) * width + col
        w = obs.walls[idx] if 0 <= idx < len(obs.walls) and obs.walls[idx] != -1 else 0

        if rtype == 0:  # Factory
            if w & 1:
                actions[uid] = "JUMP_NORTH"
            elif energy >= config.workerCost and build_cd == 0:
                actions[uid] = "BUILD_WORKER"
            else:
                actions[uid] = "NORTH"
        elif rtype == 2 and (w & 1) and energy >= config.wallRemoveCost:
            actions[uid] = "REMOVE_NORTH"
        else:
            passable = []
            if not (w & 1): passable.append("NORTH")
            if not (w & 2): passable.append("EAST")
            if not (w & 4): passable.append("SOUTH")
            if not (w & 8): passable.append("WEST")
            actions[uid] = "NORTH" if "NORTH" in passable else (choice(passable) if passable else "IDLE")

    return actions
