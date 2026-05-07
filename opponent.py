"""Opponent script to assist in tuning the hyperparameters of the ISMCTS agent. This file should not be modified."""

import sys
from collections import deque

# Unit type constants
FACTORY, SCOUT, WORKER, MINER = 0, 1, 2, 3
# Navigation constants
DIRS = ("NORTH", "EAST", "WEST", "SOUTH")
OFFSETS = {"NORTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0), "SOUTH": (0, -1)}
WALL_BITS = {"NORTH": 1, "EAST": 2, "SOUTH": 4, "WEST": 8}


def agent(obs, config):
    actions = {}
    # Environment dimensions and bounds
    width, south, north = config.width, obs.southBound, obs.northBound

    # Unit tracking
    my_robots = {uid: d for uid, d in obs.robots.items() if d[4] == obs.player}
    enemy_robots = {uid: d for uid, d in obs.robots.items() if d[4] != obs.player}
    occupied_enemy = {(d[1], d[2]) for d in enemy_robots.values()}
    reserved = set()  # Tracking destinations to prevent collisions

    def get_w(c, r):
        """Retrieves wall bitmask, exploiting maze symmetry for predicted vision."""
        idx = (r - south) * width + c
        # Direct vision check
        if 0 <= c < width and 0 <= idx < len(obs.walls) and obs.walls[idx] != -1:
            return obs.walls[idx]

        # Maze East-West Symmetry check
        oc = width - 1 - c
        oidx = (r - south) * width + oc
        if 0 <= oc < width and 0 <= oidx < len(obs.walls) and obs.walls[oidx] != -1:
            v = obs.walls[oidx]
            res = v & 5  # North and South bits stay the same
            if v & 2:
                res |= 8  # Symmetric East is West
            if v & 8:
                res |= 2  # Symmetric West is East
            return res
        return 0

    def can_move(c, r, d):
        """Checks if a standard move in direction d is valid."""
        dc, dr = OFFSETS[d]
        if not (0 <= c + dc < width and south <= r + dr <= north):
            return False
        if get_w(c, r) & WALL_BITS[d]:
            return False
        return True

    def can_jump(c, r, d):
        """Checks if a jump in direction d is valid and safe."""
        dc, dr = OFFSETS[d]
        nc, nr = c + 2 * dc, r + 2 * dr
        if not (0 <= nc < width and south <= nr <= north):
            return False
        if get_w(nc, nr) == 15:
            return False  # Don't jump into a fully enclosed trap
        return True

    def get_path(start, goals, avoid, depth, init_j_cd):
        """
        Unified BFS for pathfinding.
        Tracks (col, row, jump_cd) to plan moves and jumps in a single search.
        """
        if not goals:
            return None
        g_set = set(goals)
        q = deque([(start[0], start[1], 0, None, init_j_cd)])
        visited = {(start[0], start[1], init_j_cd)}

        while q:
            c, r, d, first, j_cd = q.popleft()

            # Goal reached
            if (c, r) in g_set and d > 0:
                return first

            if d >= depth:
                continue

            # Explore moves
            for dr in DIRS:
                nc, nr = c + OFFSETS[dr][0], r + OFFSETS[dr][1]
                if (nc, nr) not in avoid and can_move(c, r, dr):
                    nj = max(0, j_cd - 1)
                    if (nc, nr, nj) not in visited:
                        visited.add((nc, nr, nj))
                        q.append((nc, nr, d + 1, first or dr, nj))

            # Explore jumps
            if j_cd == 0:
                for dr in DIRS:
                    nc, nr = c + 2 * OFFSETS[dr][0], r + 2 * OFFSETS[dr][1]
                    if (nc, nr) not in avoid and can_jump(c, r, dr):
                        if (nc, nr, 20) not in visited:
                            visited.add((nc, nr, 20))
                            q.append((nc, nr, d + 1, first or f"JUMP_{dr}", 20))
        return None

    # Sort units by role priority
    units = sorted(my_robots.items(), key=lambda x: (x[1][0], x[0]))
    f_uid, f_data = next(((u, d) for u, d in units if d[0] == FACTORY), (None, None))
    workers = [u for u, d in units if d[0] == WORKER]
    scouts = [u for u, d in units if d[0] == SCOUT]
    crystals = {tuple(map(int, k.split(","))) for k, v in obs.crystals.items() if v > 0}

    # 1. FACTORY STRATEGY
    if f_uid is not None:
        fc, fr, fe = f_data[1:4]
        fm, fj, fb = f_data[5:8] if len(f_data) > 7 else (0, 0, 0)
        f_act = None

        # Support Workers by transferring excess energy
        for w_uid in workers:
            wd = my_robots[w_uid]
            if abs(fc - wd[1]) + abs(fr - wd[2]) == 1 and wd[3] < 200:
                for d in DIRS:
                    if fc + OFFSETS[d][0] == wd[1] and fr + OFFSETS[d][1] == wd[2]:
                        f_act = f"TRANSFER_{d}"
                        break
            if f_act:
                break

        # Emergency jump if too close to southern scrolling bound
        if (
            not f_act
            and fr - south <= 3
            and south > 0
            and fj == 0
            and can_jump(fc, fr, "NORTH")
        ):
            f_act = "JUMP_NORTH"

        # Core Northward Movement
        if not f_act and fm <= 1:
            goals = [(c, min(north, fr + 25)) for c in range(width)]
            # First try a polite path (avoiding friendly units)
            avoid_p = occupied_enemy | {
                (my_robots[u][1], my_robots[u][2]) for u in workers + scouts
            }
            step = get_path((fc, fr), goals, avoid_p, 40, fj)
            # If blocked, take the aggressive path (survival first)
            if not step:
                step = get_path((fc, fr), goals, occupied_enemy, 40, fj)
            if step:
                f_act = step

        # Economy: Build support units
        if not f_act and fb == 0:
            spawn = (fc, fr + 1)
            if not (get_w(fc, fr) & 1) and spawn not in {
                (d[1], d[2]) for d in my_robots.values()
            }:
                if len(workers) < 2 and fe >= config.workerCost:
                    f_act = "BUILD_WORKER"
                elif len(scouts) < 1 and fe >= config.scoutCost + 300:
                    f_act = "BUILD_SCOUT"

        # Execute and reserve destination
        actions[f_uid] = f_act or "IDLE"
        if f_act in DIRS:
            reserved.add((fc + OFFSETS[f_act][0], fr + OFFSETS[f_act][1]))
        elif f_act and f_act.startswith("JUMP_"):
            d = f_act.split("_")[1]
            reserved.add((fc + 2 * OFFSETS[d][0], fr + 2 * OFFSETS[d][1]))
        else:
            reserved.add((fc, fr))

    # 2. WORKER VANGUARD STRATEGY
    for w_uid in workers:
        wc, wr, we = my_robots[w_uid][1:4]
        wm = my_robots[w_uid][5] if len(my_robots[w_uid]) > 5 else 0
        w_act = None

        # Remove walls blocking the path
        if (get_w(wc, wr) & 1) and we >= 100:
            w_act = "REMOVE_NORTH"

        # Move ahead of Factory to clear obstacles
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

    # 3. SCOUT HARVESTER STRATEGY
    for s_uid in scouts:
        sc, sr, se = my_robots[s_uid][1:4]
        sm = my_robots[s_uid][5] if len(my_robots[s_uid]) > 5 else 0
        s_act = None

        # Transfer scavenged energy to Factory
        if f_uid is not None and abs(sc - fc) + abs(sr - fr) == 1 and se >= 50:
            for d in DIRS:
                if sc + OFFSETS[d][0] == fc and sr + OFFSETS[d][1] == fr:
                    s_act = f"TRANSFER_{d}"
                    break

        # Resource collection and Exploration
        if not s_act and sm <= 0:
            # If full, return to Factory
            if se > 80 and f_uid is not None:
                goals = [(fc, fr)]
            # If crystals visible, collect them
            elif crystals:
                goals = sorted(crystals, key=lambda p: abs(p[0] - sc) + abs(p[1] - sr))[
                    :3
                ]
            # Otherwise, scout northward
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
    """Safety wrapper to ensure agent always returns a valid action dictionary."""
    try:
        return agent(obs, config)
    except:
        return {}
