#include "crawler_engine_internal.hpp"

// Deterministic fallback policy and macro-action translation. Full ISMCTS should
// replace selection, but keep using these bounded macro generators.

#include <array>
#include <cmath>
#include <limits>

namespace crawler {
namespace {

// Conservative primitive fallback used by both direct heuristics and macro translation.
Action best_passable_direction(const BoardState& state, int robot_index) {
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    if (state.can_move_through(c, r, DIR_NORTH) && r + 1 <= state.north_bound) {
        return ACT_NORTH;
    }
    if (state.can_move_through(c, r, DIR_EAST)) {
        return ACT_EAST;
    }
    if (state.can_move_through(c, r, DIR_WEST)) {
        return ACT_WEST;
    }
    if (state.can_move_through(c, r, DIR_SOUTH) && r - 1 >= state.south_bound) {
        return ACT_SOUTH;
    }
    return ACT_IDLE;
}

// Macro translation validates affordability before the simulator's energy-drain phase.
bool can_pay_after_drain(int energy, int cost) {
    return energy >= cost + ENERGY_PER_TURN;
}

// Cooldowns tick before action execution, so a cooldown of 1 is effectively ready.
bool ready_after_tick(int cooldown) {
    return cooldown <= 1;
}

// Convert a direction back into the primitive action family needed by a macro.
Action movement_action(Direction direction) {
    switch (direction) {
        case DIR_NORTH: return ACT_NORTH;
        case DIR_SOUTH: return ACT_SOUTH;
        case DIR_EAST: return ACT_EAST;
        case DIR_WEST: return ACT_WEST;
        default: return ACT_IDLE;
    }
}

Action transfer_action(Direction direction) {
    switch (direction) {
        case DIR_NORTH: return ACT_TRANSFER_NORTH;
        case DIR_SOUTH: return ACT_TRANSFER_SOUTH;
        case DIR_EAST: return ACT_TRANSFER_EAST;
        case DIR_WEST: return ACT_TRANSFER_WEST;
        default: return ACT_IDLE;
    }
}

bool can_step_in_active(const BoardState& state, int c, int r, Direction direction) {
    const int nc = c + direction_dc(direction);
    const int nr = r + direction_dr(direction);
    return state.in_active(nc, nr) && state.can_move_through(c, r, direction);
}

// Move greedily toward a known target cell without pathfinding allocation.
Action step_toward_cell(const BoardState& state, int robot_index, int target_c, int target_r) {
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    std::array<Direction, 4> candidates{DIR_NONE, DIR_NONE, DIR_NONE, DIR_NONE};
    int count = 0;

    const int dc = target_c - c;
    const int dr = target_r - r;
    if (std::abs(dr) >= std::abs(dc)) {
        if (dr > 0) candidates[static_cast<size_t>(count++)] = DIR_NORTH;
        if (dr < 0) candidates[static_cast<size_t>(count++)] = DIR_SOUTH;
        if (dc > 0) candidates[static_cast<size_t>(count++)] = DIR_EAST;
        if (dc < 0) candidates[static_cast<size_t>(count++)] = DIR_WEST;
    } else {
        if (dc > 0) candidates[static_cast<size_t>(count++)] = DIR_EAST;
        if (dc < 0) candidates[static_cast<size_t>(count++)] = DIR_WEST;
        if (dr > 0) candidates[static_cast<size_t>(count++)] = DIR_NORTH;
        if (dr < 0) candidates[static_cast<size_t>(count++)] = DIR_SOUTH;
    }

    for (int i = 0; i < count; ++i) {
        const Direction d = candidates[static_cast<size_t>(i)];
        if (can_step_in_active(state, c, r, d)) {
            return movement_action(d);
        }
    }
    return best_passable_direction(state, robot_index);
}

// Locate support/return targets using fixed scans over the robot store.
int nearest_friendly_factory(const BoardState& state, int robot_index) {
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    const int owner = state.robots.owner[static_cast<size_t>(robot_index)];
    int best = -1;
    int best_dist = std::numeric_limits<int>::max();
    for (int i = 0; i < state.robots.used; ++i) {
        if (i == robot_index || state.robots.alive[static_cast<size_t>(i)] == 0 ||
            state.robots.owner[static_cast<size_t>(i)] != owner ||
            state.robots.type[static_cast<size_t>(i)] != FACTORY) {
            continue;
        }
        const int dist = std::abs(c - state.robots.col[static_cast<size_t>(i)]) +
                         std::abs(r - state.robots.row[static_cast<size_t>(i)]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

Action transfer_to_adjacent_factory(const BoardState& state, int robot_index) {
    const int factory = nearest_friendly_factory(state, robot_index);
    if (factory < 0) {
        return ACT_IDLE;
    }
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    const int fc = state.robots.col[static_cast<size_t>(factory)];
    const int fr = state.robots.row[static_cast<size_t>(factory)];
    Direction d = DIR_NONE;
    if (fc == c && fr == r + 1) d = DIR_NORTH;
    if (fc == c && fr == r - 1) d = DIR_SOUTH;
    if (fc == c + 1 && fr == r) d = DIR_EAST;
    if (fc == c - 1 && fr == r) d = DIR_WEST;
    if (d != DIR_NONE && state.can_move_through(c, r, d)) {
        return transfer_action(d);
    }
    return ACT_IDLE;
}

// Locate visible tactical objectives using fixed scans over the active window.
int nearest_crystal_cell(const BoardState& state, int robot_index) {
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    int best_idx = -1;
    int best_dist = std::numeric_limits<int>::max();
    for (int rr = state.south_bound; rr <= state.north_bound; ++rr) {
        for (int cc = 0; cc < WIDTH; ++cc) {
            const int idx = state.abs_index(cc, rr);
            if (idx < 0 || state.crystal_energy[static_cast<size_t>(idx)] <= 0) {
                continue;
            }
            const int dist = std::abs(c - cc) + std::abs(r - rr);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = idx;
            }
        }
    }
    return best_idx;
}

int nearest_mining_node_cell(const BoardState& state, int robot_index) {
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    int best_idx = -1;
    int best_dist = std::numeric_limits<int>::max();
    for (int rr = state.south_bound; rr <= state.north_bound; ++rr) {
        for (int cc = 0; cc < WIDTH; ++cc) {
            const int idx = state.abs_index(cc, rr);
            if (idx < 0 || state.mining_node[static_cast<size_t>(idx)] == 0) {
                continue;
            }
            const int dist = std::abs(c - cc) + std::abs(r - rr);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = idx;
            }
        }
    }
    return best_idx;
}

// Append one macro while preserving MAX_MACROS as a hard cap.
void add_macro(MacroList& list, MacroAction macro) {
    if (list.count >= MAX_MACROS) {
        return;
    }
    list.macros[static_cast<size_t>(list.count++)] = macro;
}

}  // namespace

// Fast deterministic fallback policy while full ISMCTS rollout selection is still being built.
Action CrawlerSim::heuristic_action_for(int robot_index) const {
    if (robot_index < 0 || robot_index >= state.robots.used ||
        state.robots.alive[static_cast<size_t>(robot_index)] == 0 ||
        state.robots.owner[static_cast<size_t>(robot_index)] != state.player) {
        return ACT_IDLE;
    }
    const uint8_t type = state.robots.type[static_cast<size_t>(robot_index)];
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    const int e = state.robots.energy[static_cast<size_t>(robot_index)];
    const uint8_t wall = state.wall_at(c, r);

    if (type == FACTORY) {
        int workers = 0;
        int scouts = 0;
        for (int i = 0; i < state.robots.used; ++i) {
            if (state.robots.alive[static_cast<size_t>(i)] != 0 &&
                state.robots.owner[static_cast<size_t>(i)] == state.player) {
                workers += state.robots.type[static_cast<size_t>(i)] == WORKER ? 1 : 0;
                scouts += state.robots.type[static_cast<size_t>(i)] == SCOUT ? 1 : 0;
            }
        }
        if ((wall & WALL_N) != 0 &&
            ready_after_tick(state.robots.jump_cd[static_cast<size_t>(robot_index)]) &&
            ready_after_tick(state.robots.move_cd[static_cast<size_t>(robot_index)])) {
            return ACT_JUMP_NORTH;
        }
        if (ready_after_tick(state.robots.build_cd[static_cast<size_t>(robot_index)]) &&
            state.can_move_through(c, r, DIR_NORTH)) {
            if (workers < 2 && can_pay_after_drain(e, WORKER_COST)) {
                return ACT_BUILD_WORKER;
            }
            if (scouts < 3 && can_pay_after_drain(e, SCOUT_COST)) {
                return ACT_BUILD_SCOUT;
            }
        }
        return best_passable_direction(state, robot_index);
    }

    if (type == WORKER) {
        if ((wall & WALL_N) != 0 && can_pay_after_drain(e, WALL_REMOVE_COST)) {
            return ACT_REMOVE_NORTH;
        }
        return best_passable_direction(state, robot_index);
    }

    if (type == MINER) {
        const int idx = state.abs_index(c, r);
        if (idx >= 0 && state.mining_node[static_cast<size_t>(idx)] != 0 && can_pay_after_drain(e, TRANSFORM_COST)) {
            return ACT_TRANSFORM;
        }
        return best_passable_direction(state, robot_index);
    }

    return best_passable_direction(state, robot_index);
}

// Generate bounded per-robot intent choices. The search layer consumes these instead
// of branching over every primitive action for every robot.
MacroList CrawlerSim::generate_macros_for(int robot_index) const {
    MacroList list{};
    if (robot_index < 0 || robot_index >= state.robots.used ||
        state.robots.alive[static_cast<size_t>(robot_index)] == 0) {
        return list;
    }
    const uint8_t type = state.robots.type[static_cast<size_t>(robot_index)];
    add_macro(list, MACRO_IDLE);
    if (type == FACTORY) {
        add_macro(list, MACRO_FACTORY_SAFE_ADVANCE);
        add_macro(list, MACRO_FACTORY_BUILD_WORKER);
        add_macro(list, MACRO_FACTORY_BUILD_SCOUT);
        add_macro(list, MACRO_FACTORY_BUILD_MINER);
        add_macro(list, MACRO_FACTORY_JUMP_OBSTACLE);
    } else if (type == WORKER) {
        add_macro(list, MACRO_WORKER_OPEN_NORTH_WALL);
        add_macro(list, MACRO_WORKER_ESCORT_FACTORY);
        add_macro(list, MACRO_WORKER_ADVANCE);
    } else if (type == SCOUT) {
        add_macro(list, MACRO_SCOUT_HUNT_CRYSTAL);
        add_macro(list, MACRO_SCOUT_EXPLORE_NORTH);
        add_macro(list, MACRO_SCOUT_RETURN_ENERGY);
    } else if (type == MINER) {
        add_macro(list, MACRO_MINER_SEEK_NODE);
        add_macro(list, MACRO_MINER_TRANSFORM);
    }
    return list;
}

// Translate one macro intent into the current best legal primitive. Invalid or
// unaffordable macros deliberately degrade to IDLE.
Action CrawlerSim::primitive_for_macro(int robot_index, MacroAction macro) const {
    if (robot_index < 0 || robot_index >= state.robots.used ||
        state.robots.alive[static_cast<size_t>(robot_index)] == 0) {
        return ACT_IDLE;
    }

    const uint8_t type = state.robots.type[static_cast<size_t>(robot_index)];
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    const int e = state.robots.energy[static_cast<size_t>(robot_index)];
    const uint8_t wall = state.wall_at(c, r);

    switch (macro) {
        case MACRO_FACTORY_BUILD_WORKER:
            if (type == FACTORY && ready_after_tick(state.robots.build_cd[static_cast<size_t>(robot_index)]) &&
                can_pay_after_drain(e, WORKER_COST) && r + 1 <= state.north_bound &&
                state.can_move_through(c, r, DIR_NORTH)) {
                return ACT_BUILD_WORKER;
            }
            break;
        case MACRO_FACTORY_BUILD_SCOUT:
            if (type == FACTORY && ready_after_tick(state.robots.build_cd[static_cast<size_t>(robot_index)]) &&
                can_pay_after_drain(e, SCOUT_COST) && r + 1 <= state.north_bound &&
                state.can_move_through(c, r, DIR_NORTH)) {
                return ACT_BUILD_SCOUT;
            }
            break;
        case MACRO_FACTORY_BUILD_MINER:
            if (type == FACTORY && ready_after_tick(state.robots.build_cd[static_cast<size_t>(robot_index)]) &&
                can_pay_after_drain(e, MINER_COST) && r + 1 <= state.north_bound &&
                state.can_move_through(c, r, DIR_NORTH)) {
                return ACT_BUILD_MINER;
            }
            break;
        case MACRO_FACTORY_JUMP_OBSTACLE:
            if (type == FACTORY && ready_after_tick(state.robots.jump_cd[static_cast<size_t>(robot_index)]) &&
                ready_after_tick(state.robots.move_cd[static_cast<size_t>(robot_index)]) &&
                (wall & WALL_N) != 0 && r + 2 <= state.north_bound) {
                return ACT_JUMP_NORTH;
            }
            break;
        case MACRO_FACTORY_SAFE_ADVANCE:
            if (type == FACTORY) {
                if ((wall & WALL_N) != 0 &&
                    ready_after_tick(state.robots.jump_cd[static_cast<size_t>(robot_index)]) &&
                    ready_after_tick(state.robots.move_cd[static_cast<size_t>(robot_index)]) &&
                    r + 2 <= state.north_bound) {
                    return ACT_JUMP_NORTH;
                }
                return best_passable_direction(state, robot_index);
            }
            break;
        case MACRO_WORKER_OPEN_NORTH_WALL:
            if (type == WORKER && (wall & WALL_N) != 0 && can_pay_after_drain(e, WALL_REMOVE_COST)) {
                return ACT_REMOVE_NORTH;
            }
            break;
        case MACRO_WORKER_ESCORT_FACTORY:
            if (type == WORKER) {
                const int factory = nearest_friendly_factory(state, robot_index);
                if (factory >= 0) {
                    return step_toward_cell(state, robot_index,
                                            state.robots.col[static_cast<size_t>(factory)],
                                            state.robots.row[static_cast<size_t>(factory)]);
                }
                return best_passable_direction(state, robot_index);
            }
            break;
        case MACRO_WORKER_ADVANCE:
            if (type == WORKER) {
                if ((wall & WALL_N) != 0 && can_pay_after_drain(e, WALL_REMOVE_COST)) {
                    return ACT_REMOVE_NORTH;
                }
                return best_passable_direction(state, robot_index);
            }
            break;
        case MACRO_SCOUT_HUNT_CRYSTAL:
            if (type == SCOUT) {
                const int idx = nearest_crystal_cell(state, robot_index);
                if (idx >= 0) {
                    return step_toward_cell(state, robot_index, detail::cell_col(idx), detail::cell_row(idx));
                }
                return best_passable_direction(state, robot_index);
            }
            break;
        case MACRO_SCOUT_EXPLORE_NORTH:
            if (type == SCOUT) {
                if (can_step_in_active(state, c, r, DIR_NORTH)) {
                    return ACT_NORTH;
                }
                return best_passable_direction(state, robot_index);
            }
            break;
        case MACRO_SCOUT_RETURN_ENERGY:
            if (type == SCOUT) {
                const Action transfer = transfer_to_adjacent_factory(state, robot_index);
                if (transfer != ACT_IDLE) {
                    return transfer;
                }
                const int factory = nearest_friendly_factory(state, robot_index);
                if (factory >= 0) {
                    return step_toward_cell(state, robot_index,
                                            state.robots.col[static_cast<size_t>(factory)],
                                            state.robots.row[static_cast<size_t>(factory)]);
                }
                return best_passable_direction(state, robot_index);
            }
            break;
        case MACRO_MINER_SEEK_NODE:
            if (type == MINER) {
                const int here = state.abs_index(c, r);
                if (here >= 0 && state.mining_node[static_cast<size_t>(here)] != 0 &&
                    can_pay_after_drain(e, TRANSFORM_COST)) {
                    return ACT_TRANSFORM;
                }
                const int idx = nearest_mining_node_cell(state, robot_index);
                if (idx >= 0) {
                    return step_toward_cell(state, robot_index, detail::cell_col(idx), detail::cell_row(idx));
                }
                return best_passable_direction(state, robot_index);
            }
            break;
        case MACRO_MINER_TRANSFORM:
            if (type == MINER) {
                const int idx = state.abs_index(c, r);
                if (idx >= 0 && state.mining_node[static_cast<size_t>(idx)] != 0 &&
                    can_pay_after_drain(e, TRANSFORM_COST)) {
                    return ACT_TRANSFORM;
                }
            }
            break;
        case MACRO_IDLE:
        default:
            break;
    }

    return ACT_IDLE;
}

}  // namespace crawler
