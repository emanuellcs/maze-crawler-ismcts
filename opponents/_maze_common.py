"""Shared helpers for hand-crafted Maze Crawler opponent archetypes."""

import sys
from collections import deque

FACTORY, SCOUT, WORKER, MINER = 0, 1, 2, 3
DIRS = ("NORTH", "EAST", "WEST", "SOUTH")
OFFSETS = {"NORTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0), "SOUTH": (0, -1)}
WALL_BITS = {"NORTH": 1, "EAST": 2, "SOUTH": 4, "WEST": 8}
OPPOSITE = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}


def get_wall(obs, config, c, r):
    """Wall bitfield for a cell, using east/west symmetry as a fallback."""
    width = config.width
    south = obs.southBound
    idx = (r - south) * width + c
    if 0 <= c < width and 0 <= idx < len(obs.walls) and obs.walls[idx] != -1:
        return obs.walls[idx]
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


def can_move(obs, config, c, r, d):
    dc, dr = OFFSETS[d]
    if not (0 <= c + dc < config.width and obs.southBound <= r + dr <= obs.northBound):
        return False
    return not (get_wall(obs, config, c, r) & WALL_BITS[d])


def path_to(obs, config, start, goals, avoid, depth, allow_jump=False, jump_cd=0):
    """BFS returning the first action toward any goal (or None)."""
    g_set = set(goals)
    q = deque([(start[0], start[1], 0, None, jump_cd)])
    visited = {(start[0], start[1], jump_cd)}
    while q:
        c, r, d, first, jc = q.popleft()
        if (c, r) in g_set and d > 0:
            return first
        if d >= depth:
            continue
        for nd in DIRS:
            nc, nr = c + OFFSETS[nd][0], r + OFFSETS[nd][1]
            if (nc, nr) in avoid:
                continue
            if can_move(obs, config, c, r, nd):
                nj = max(0, jc - 1)
                if (nc, nr, nj) not in visited:
                    visited.add((nc, nr, nj))
                    q.append((nc, nr, d + 1, first or nd, nj))
        if allow_jump and jc == 0:
            for nd in DIRS:
                nc, nr = c + 2 * OFFSETS[nd][0], r + 2 * OFFSETS[nd][1]
                if (nc, nr) in avoid:
                    continue
                if 0 <= nc < config.width and obs.southBound <= nr <= obs.northBound and get_wall(obs, config, nc, nr) != 15:
                    if (nc, nr, 20) not in visited:
                        visited.add((nc, nr, 20))
                        q.append((nc, nr, d + 1, first or f"JUMP_{nd}", 20))
    return None


def visible_nodes(obs):
    return {tuple(map(int, k.split(","))) for k in obs.miningNodes if obs.miningNodes[k]}


def visible_crystals(obs):
    return {tuple(map(int, k.split(","))) for k, v in obs.crystals.items() if v > 0}


def split_robots(obs, player):
    mine = {u: d for u, d in obs.robots.items() if d[4] == player}
    enemy = {u: d for u, d in obs.robots.items() if d[4] != player}
    enemy_cells = {(d[1], d[2]) for d in enemy.values()}
    return mine, enemy_cells
