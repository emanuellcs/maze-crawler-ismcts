"""Deterministic Python benchmark opponent for tuning the native ISMCTS agent.

The policy mirrors the C++ baseline strategy at a higher level: Factory survival
and support first, Worker wall opening, Scout crystal harvesting and return, and
simple Bitboard-free reservation logic.  ``tune.py`` evaluates candidate
hyperparameters against this opponent with paired seeds and side swapping.
"""

import sys
from collections import deque

# Kaggle robot type encoding shared with the C++ engine.
FACTORY, SCOUT, WORKER, MINER = 0, 1, 2, 3

# Direction order intentionally favors northward progress before lateral detours.
DIRS = ("NORTH", "EAST", "WEST", "SOUTH")
OFFSETS = {"NORTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0), "SOUTH": (0, -1)}
WALL_BITS = {"NORTH": 1, "EAST": 2, "SOUTH": 4, "WEST": 8}


def agent(obs, config):
    """Return deterministic benchmark actions for one Maze Crawler turn.

    Parameters
    ----------
    obs:
        Kaggle observation object with ``walls`` as a flat active-window array,
        ``robots`` as ``{"uid": [type, col, row, energy, owner, move_cd,
        jump_cd, build_cd]}``, and sparse ``crystals`` keyed by ``"col,row"``.
    config:
        Kaggle configuration object containing rule constants such as width and
        robot costs.
    """

    actions = {}
    # Active-window geometry is needed to decode wall-array indices.
    width, south, north = config.width, obs.southBound, obs.northBound

    # Split visible robots by ownership and reserve destinations to avoid friendly collisions.
    my_robots = {uid: d for uid, d in obs.robots.items() if d[4] == obs.player}
    enemy_robots = {uid: d for uid, d in obs.robots.items() if d[4] != obs.player}
    occupied_enemy = {(d[1], d[2]) for d in enemy_robots.values()}
    reserved = set()

    def get_w(c, r):
        """Return a wall bitfield, using east/west symmetry when direct vision lacks it."""

        idx = (r - south) * width + c
        # Directly observed walls are authoritative.
        if 0 <= c < width and 0 <= idx < len(obs.walls) and obs.walls[idx] != -1:
            return obs.walls[idx]

        # The maze mirrors east/west, so the opposite column can predict layout.
        oc = width - 1 - c
        oidx = (r - south) * width + oc
        if 0 <= oc < width and 0 <= oidx < len(obs.walls) and obs.walls[oidx] != -1:
            v = obs.walls[oidx]
            res = v & 5
            if v & 2:
                res |= 8
            if v & 8:
                res |= 2
            return res
        return 0

    def can_move(c, r, d):
        """Return whether a one-cell move is in bounds and wall-passable."""

        dc, dr = OFFSETS[d]
        if not (0 <= c + dc < width and south <= r + dr <= north):
            return False
        if get_w(c, r) & WALL_BITS[d]:
            return False
        return True

    def can_jump(c, r, d):
        """Return whether a Factory jump has an in-bounds, non-trap landing cell."""

        dc, dr = OFFSETS[d]
        nc, nr = c + 2 * dc, r + 2 * dr
        if not (0 <= nc < width and south <= nr <= north):
            return False
        if get_w(nc, nr) == 15:
            return False
        return True

    def get_path(start, goals, avoid, depth, init_j_cd):
        """Find the first action on a cooldown-aware BFS path to any goal."""

        if not goals:
            return None
        g_set = set(goals)
        q = deque([(start[0], start[1], 0, None, init_j_cd)])
        visited = {(start[0], start[1], init_j_cd)}

        while q:
            c, r, d, first, j_cd = q.popleft()

            # The start cell is not considered a successful path by itself.
            if (c, r) in g_set and d > 0:
                return first

            if d >= depth:
                continue

            # Movement edges decrement jump cooldown.
            for dr in DIRS:
                nc, nr = c + OFFSETS[dr][0], r + OFFSETS[dr][1]
                if (nc, nr) not in avoid and can_move(c, r, dr):
                    nj = max(0, j_cd - 1)
                    if (nc, nr, nj) not in visited:
                        visited.add((nc, nr, nj))
                        q.append((nc, nr, d + 1, first or dr, nj))

            # Jump edges move two cells and reset Factory jump cooldown.
            if j_cd == 0:
                for dr in DIRS:
                    nc, nr = c + 2 * OFFSETS[dr][0], r + 2 * OFFSETS[dr][1]
                    if (nc, nr) not in avoid and can_jump(c, r, dr):
                        if (nc, nr, 20) not in visited:
                            visited.add((nc, nr, 20))
                            q.append((nc, nr, d + 1, first or f"JUMP_{dr}", 20))
        return None

    # UID-sorted role groups make the policy deterministic for tuning comparisons.
    units = sorted(my_robots.items(), key=lambda x: (x[1][0], x[0]))
    f_uid, f_data = next(((u, d) for u, d in units if d[0] == FACTORY), (None, None))
    workers = [u for u, d in units if d[0] == WORKER]
    scouts = [u for u, d in units if d[0] == SCOUT]
    crystals = {tuple(map(int, k.split(","))) for k, v in obs.crystals.items() if v > 0}

    # Factory strategy: support, survive scroll pressure, advance, then build economy.
    if f_uid is not None:
        fc, fr, fe = f_data[1:4]
        fm, fj, fb = f_data[5:8] if len(f_data) > 7 else (0, 0, 0)
        f_act = None

        # Keep adjacent Workers funded so they can clear north walls.
        for w_uid in workers:
            wd = my_robots[w_uid]
            if abs(fc - wd[1]) + abs(fr - wd[2]) == 1 and wd[3] < 200:
                for d in DIRS:
                    if fc + OFFSETS[d][0] == wd[1] and fr + OFFSETS[d][1] == wd[2]:
                        f_act = f"TRANSFER_{d}"
                        break
            if f_act:
                break

        # Escape the scrolling boundary before normal movement or build choices.
        if (
            not f_act
            and fr - south <= 3
            and south > 0
            and fj == 0
            and can_jump(fc, fr, "NORTH")
        ):
            f_act = "JUMP_NORTH"

        # Route the Factory north with a polite path before accepting collision risk.
        if not f_act and fm <= 1:
            goals = [(c, min(north, fr + 25)) for c in range(width)]
            avoid_p = occupied_enemy | {
                (my_robots[u][1], my_robots[u][2]) for u in workers + scouts
            }
            step = get_path((fc, fr), goals, avoid_p, 40, fj)
            if not step:
                step = get_path((fc, fr), goals, occupied_enemy, 40, fj)
            if step:
                f_act = step

        # Build support units only when the north spawn cell is open and unoccupied.
        if not f_act and fb == 0:
            spawn = (fc, fr + 1)
            if not (get_w(fc, fr) & 1) and spawn not in {
                (d[1], d[2]) for d in my_robots.values()
            }:
                if len(workers) < 2 and fe >= config.workerCost:
                    f_act = "BUILD_WORKER"
                elif len(scouts) < 1 and fe >= config.scoutCost + 300:
                    f_act = "BUILD_SCOUT"

        # Reserve the Factory destination so later friendly units route around it.
        actions[f_uid] = f_act or "IDLE"
        if f_act in DIRS:
            reserved.add((fc + OFFSETS[f_act][0], fr + OFFSETS[f_act][1]))
        elif f_act and f_act.startswith("JUMP_"):
            d = f_act.split("_")[1]
            reserved.add((fc + 2 * OFFSETS[d][0], fr + 2 * OFFSETS[d][1]))
        else:
            reserved.add((fc, fr))

    # Worker strategy: open north walls, then escort ahead of the Factory.
    for w_uid in workers:
        wc, wr, we = my_robots[w_uid][1:4]
        wm = my_robots[w_uid][5] if len(my_robots[w_uid]) > 5 else 0
        w_act = None

        # Opening a north wall is often higher leverage than a blind move.
        if (get_w(wc, wr) & 1) and we >= 100:
            w_act = "REMOVE_NORTH"

        # Escort the Factory when present; otherwise keep moving north.
        if not w_act and wm <= 1:
            target = (
                (fc, min(north, fr + 5))
                if f_uid is not None
                else (wc, min(north, wr + 5))
            )
            step = get_path((wc, wr), [target], reserved | occupied_enemy, 25, 999)
            if not step:
                step = get_path(
                    (wc, wr),
                    [(wc, min(north, wr + 5))],
                    reserved | occupied_enemy,
                    20,
                    999,
                )
            if step:
                w_act = step

        actions[w_uid] = w_act or "IDLE"
        if w_act in DIRS:
            reserved.add((wc + OFFSETS[w_act][0], wr + OFFSETS[w_act][1]))
        else:
            reserved.add((wc, wr))

    # Scout strategy: return loaded energy, harvest visible crystals, then explore.
    for s_uid in scouts:
        sc, sr, se = my_robots[s_uid][1:4]
        sm = my_robots[s_uid][5] if len(my_robots[s_uid]) > 5 else 0
        s_act = None

        # Adjacent transfer turns collected energy into Factory survival margin.
        if f_uid is not None and abs(sc - fc) + abs(sr - fr) == 1 and se >= 50:
            for d in DIRS:
                if sc + OFFSETS[d][0] == fc and sr + OFFSETS[d][1] == fr:
                    s_act = f"TRANSFER_{d}"
                    break

        # Route to Factory when loaded, visible crystals when available, else north.
        if not s_act and sm <= 0:
            if se > 80 and f_uid is not None:
                goals = [(fc, fr)]
            elif crystals:
                goals = sorted(crystals, key=lambda p: abs(p[0] - sc) + abs(p[1] - sr))[
                    :3
                ]
            else:
                goals = [(c, min(north, sr + 15)) for c in range(width)]

            step = get_path((sc, sr), goals, reserved | occupied_enemy, 30, 999)
            if step:
                s_act = step

        actions[s_uid] = s_act or "IDLE"
        if s_act in DIRS:
            reserved.add((sc + OFFSETS[s_act][0], sr + OFFSETS[s_act][1]))
        else:
            reserved.add((sc, sr))

    return actions


def act(obs, config):
    """Return benchmark actions and swallow unexpected policy exceptions."""

    try:
        return agent(obs, config)
    except:
        return {}
