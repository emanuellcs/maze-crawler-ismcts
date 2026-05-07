#include "crawler_engine_internal.hpp"

// Deterministic fallback policy and macro-action translation. The factory,
// worker, and scout branches mirror opponent.py, but all scratch state stays in
// fixed arrays.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace crawler {
namespace {

constexpr int BFS_CD_STATES = FACTORY_JUMP_COOLDOWN + 2;
constexpr int BFS_BLOCKED_CD = BFS_CD_STATES - 1;
constexpr int BFS_CAPACITY = ACTIVE_CELLS * BFS_CD_STATES;
constexpr int NO_JUMP_PATH_CD = 999;

constexpr std::array<Direction, 4> POLICY_DIRS{DIR_NORTH, DIR_EAST, DIR_WEST, DIR_SOUTH};

struct BFSNode {
    int16_t col = 0;
    int16_t row = 0;
    uint8_t depth = 0;
    uint8_t jump_cd = 0;
    Action first = ACT_IDLE;
};

Action movement_action(Direction direction) {
    switch (direction) {
        case DIR_NORTH: return ACT_NORTH;
        case DIR_SOUTH: return ACT_SOUTH;
        case DIR_EAST: return ACT_EAST;
        case DIR_WEST: return ACT_WEST;
        default: return ACT_IDLE;
    }
}

Action jump_action(Direction direction) {
    switch (direction) {
        case DIR_NORTH: return ACT_JUMP_NORTH;
        case DIR_SOUTH: return ACT_JUMP_SOUTH;
        case DIR_EAST: return ACT_JUMP_EAST;
        case DIR_WEST: return ACT_JUMP_WEST;
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

bool can_pay_after_drain(int energy, int cost) {
    return energy >= cost + ENERGY_PER_TURN;
}

bool is_move_action(Action action) {
    return action >= ACT_NORTH && action <= ACT_WEST;
}

bool is_jump_action(Action action) {
    return action >= ACT_JUMP_NORTH && action <= ACT_JUMP_WEST;
}

bool uid_less(const RobotStore& robots, int lhs, int rhs) {
    for (int i = 0; i < UID_LEN; ++i) {
        const unsigned char a = static_cast<unsigned char>(robots.uid[static_cast<size_t>(lhs)][static_cast<size_t>(i)]);
        const unsigned char b = static_cast<unsigned char>(robots.uid[static_cast<size_t>(rhs)][static_cast<size_t>(i)]);
        if (a != b) {
            return a < b;
        }
    }
    return lhs < rhs;
}

void insert_uid_sorted(const RobotStore& robots, std::array<int, MAX_ROBOTS>& indices, int& count, int robot_index) {
    if (count >= MAX_ROBOTS) {
        return;
    }
    int pos = count;
    while (pos > 0 && uid_less(robots, robot_index, indices[static_cast<size_t>(pos - 1)])) {
        indices[static_cast<size_t>(pos)] = indices[static_cast<size_t>(pos - 1)];
        --pos;
    }
    indices[static_cast<size_t>(pos)] = robot_index;
    ++count;
}

int primary_factory_for_owner(const BoardState& state, int owner) {
    int best = -1;
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0 ||
            state.robots.owner[static_cast<size_t>(i)] != owner ||
            state.robots.type[static_cast<size_t>(i)] != FACTORY) {
            continue;
        }
        if (best < 0 || uid_less(state.robots, i, best)) {
            best = i;
        }
    }
    return best;
}

bool friendly_cell_occupied(const BoardState& state, int owner, int c, int r) {
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] != 0 &&
            state.robots.owner[static_cast<size_t>(i)] == owner &&
            state.robots.col[static_cast<size_t>(i)] == c &&
            state.robots.row[static_cast<size_t>(i)] == r) {
            return true;
        }
    }
    return false;
}

void mark_cell(const BoardState& state, BitBoard& mask, int c, int r) {
    const int active = state.active_index(c, r);
    if (active >= 0) {
        mask.set(active);
    }
}

void merge_mask(BitBoard& dst, const BitBoard& src) {
    for (int i = 0; i < ACTIVE_WORDS; ++i) {
        dst.words[static_cast<size_t>(i)] |= src.words[static_cast<size_t>(i)];
    }
}

BitBoard merged_mask(const BitBoard& a, const BitBoard& b) {
    BitBoard out = a;
    merge_mask(out, b);
    return out;
}

void reserve_destination(const BoardState& state, BitBoard& reserved, int c, int r, Action action) {
    Direction direction = action_direction(action);
    int dc = 0;
    int dr = 0;
    if (is_move_action(action)) {
        dc = direction_dc(direction);
        dr = direction_dr(direction);
    } else if (is_jump_action(action)) {
        dc = direction_dc(direction) * 2;
        dr = direction_dr(direction) * 2;
    } else {
        direction = DIR_NONE;
    }
    (void)direction;
    mark_cell(state, reserved, c + dc, r + dr);
}

uint8_t mirror_wall(uint8_t wall) {
    uint8_t out = wall & static_cast<uint8_t>(WALL_N | WALL_S);
    if ((wall & WALL_E) != 0) {
        out |= WALL_W;
    }
    if ((wall & WALL_W) != 0) {
        out |= WALL_E;
    }
    return out;
}

uint8_t policy_wall_at(const BoardState& state, int c, int r) {
    if (state.in_active(c, r)) {
        const int idx = state.abs_index(c, r);
        if (idx >= 0 && state.wall_known[static_cast<size_t>(idx)] != 0) {
            return state.walls[static_cast<size_t>(idx)];
        }
    }

    const int oc = WIDTH - 1 - c;
    if (state.in_active(oc, r)) {
        const int idx = state.abs_index(oc, r);
        if (idx >= 0 && state.wall_known[static_cast<size_t>(idx)] != 0) {
            return mirror_wall(state.walls[static_cast<size_t>(idx)]);
        }
    }
    return 0;
}

bool can_move_policy(const BoardState& state, int c, int r, Direction direction) {
    const int nc = c + direction_dc(direction);
    const int nr = r + direction_dr(direction);
    if (!state.in_active(nc, nr)) {
        return false;
    }
    return (policy_wall_at(state, c, r) & direction_wall_bit(direction)) == 0;
}

bool can_jump_policy(const BoardState& state, int c, int r, Direction direction) {
    const int nc = c + direction_dc(direction) * 2;
    const int nr = r + direction_dr(direction) * 2;
    if (!state.in_active(nc, nr)) {
        return false;
    }
    return policy_wall_at(state, nc, nr) != static_cast<uint8_t>(WALL_N | WALL_E | WALL_S | WALL_W);
}

int normalize_jump_cd(int jump_cd) {
    if (jump_cd <= 0) {
        return 0;
    }
    if (jump_cd > FACTORY_JUMP_COOLDOWN) {
        return BFS_BLOCKED_CD;
    }
    return jump_cd;
}

int bfs_move_jump_cd(int jump_cd) {
    if (jump_cd >= BFS_BLOCKED_CD) {
        return BFS_BLOCKED_CD;
    }
    return std::max(0, jump_cd - 1);
}

int visited_index(int active_index, int jump_cd) {
    return active_index * BFS_CD_STATES + jump_cd;
}

Action get_path_policy(const BoardState& state, int start_c, int start_r, const BitBoard& goals,
                       const BitBoard& avoid, int depth_limit, int init_jump_cd) {
    if (!goals.any()) {
        return ACT_IDLE;
    }
    const int start_active = state.active_index(start_c, start_r);
    if (start_active < 0) {
        return ACT_IDLE;
    }

    std::array<BFSNode, BFS_CAPACITY> queue{};
    std::array<uint8_t, BFS_CAPACITY> visited{};
    int head = 0;
    int tail = 0;
    int count = 0;

    const int start_cd = normalize_jump_cd(init_jump_cd);
    visited[static_cast<size_t>(visited_index(start_active, start_cd))] = 1;
    queue[static_cast<size_t>(tail)] = BFSNode{static_cast<int16_t>(start_c), static_cast<int16_t>(start_r),
                                               0, static_cast<uint8_t>(start_cd), ACT_IDLE};
    tail = (tail + 1) % BFS_CAPACITY;
    ++count;

    while (count > 0) {
        const BFSNode node = queue[static_cast<size_t>(head)];
        head = (head + 1) % BFS_CAPACITY;
        --count;

        const int active = state.active_index(node.col, node.row);
        if (active >= 0 && node.depth > 0 && goals.test(active)) {
            return node.first;
        }
        if (node.depth >= depth_limit) {
            continue;
        }

        for (Direction direction : POLICY_DIRS) {
            const int nc = node.col + direction_dc(direction);
            const int nr = node.row + direction_dr(direction);
            const int next_active = state.active_index(nc, nr);
            if (next_active < 0 || avoid.test(next_active) ||
                !can_move_policy(state, node.col, node.row, direction)) {
                continue;
            }
            const int next_cd = bfs_move_jump_cd(node.jump_cd);
            const int vi = visited_index(next_active, next_cd);
            if (visited[static_cast<size_t>(vi)] != 0 || count >= BFS_CAPACITY) {
                continue;
            }
            visited[static_cast<size_t>(vi)] = 1;
            queue[static_cast<size_t>(tail)] =
                BFSNode{static_cast<int16_t>(nc), static_cast<int16_t>(nr),
                        static_cast<uint8_t>(node.depth + 1), static_cast<uint8_t>(next_cd),
                        node.first == ACT_IDLE ? movement_action(direction) : node.first};
            tail = (tail + 1) % BFS_CAPACITY;
            ++count;
        }

        if (node.jump_cd != 0) {
            continue;
        }
        for (Direction direction : POLICY_DIRS) {
            const int nc = node.col + direction_dc(direction) * 2;
            const int nr = node.row + direction_dr(direction) * 2;
            const int next_active = state.active_index(nc, nr);
            if (next_active < 0 || avoid.test(next_active) ||
                !can_jump_policy(state, node.col, node.row, direction)) {
                continue;
            }
            const int next_cd = FACTORY_JUMP_COOLDOWN;
            const int vi = visited_index(next_active, next_cd);
            if (visited[static_cast<size_t>(vi)] != 0 || count >= BFS_CAPACITY) {
                continue;
            }
            visited[static_cast<size_t>(vi)] = 1;
            queue[static_cast<size_t>(tail)] =
                BFSNode{static_cast<int16_t>(nc), static_cast<int16_t>(nr),
                        static_cast<uint8_t>(node.depth + 1), static_cast<uint8_t>(next_cd),
                        node.first == ACT_IDLE ? jump_action(direction) : node.first};
            tail = (tail + 1) % BFS_CAPACITY;
            ++count;
        }
    }

    return ACT_IDLE;
}

BitBoard row_goals(const BoardState& state, int row) {
    BitBoard goals{};
    goals.clear();
    for (int c = 0; c < WIDTH; ++c) {
        mark_cell(state, goals, c, row);
    }
    return goals;
}

BitBoard single_goal(const BoardState& state, int c, int r) {
    BitBoard goals{};
    goals.clear();
    mark_cell(state, goals, c, r);
    return goals;
}

BitBoard nearest_crystal_goals(const BoardState& state, int robot_index) {
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    std::array<int, 3> best_idx{-1, -1, -1};
    std::array<int, 3> best_dist{std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
                                 std::numeric_limits<int>::max()};

    for (int rr = state.south_bound; rr <= state.north_bound; ++rr) {
        for (int cc = 0; cc < WIDTH; ++cc) {
            const int idx = state.abs_index(cc, rr);
            if (idx < 0 || state.crystal_energy[static_cast<size_t>(idx)] <= 0) {
                continue;
            }
            const int dist = std::abs(c - cc) + std::abs(r - rr);
            for (int pos = 0; pos < 3; ++pos) {
                if (dist < best_dist[static_cast<size_t>(pos)] ||
                    (dist == best_dist[static_cast<size_t>(pos)] &&
                     (best_idx[static_cast<size_t>(pos)] < 0 || idx < best_idx[static_cast<size_t>(pos)]))) {
                    for (int shift = 2; shift > pos; --shift) {
                        best_dist[static_cast<size_t>(shift)] = best_dist[static_cast<size_t>(shift - 1)];
                        best_idx[static_cast<size_t>(shift)] = best_idx[static_cast<size_t>(shift - 1)];
                    }
                    best_dist[static_cast<size_t>(pos)] = dist;
                    best_idx[static_cast<size_t>(pos)] = idx;
                    break;
                }
            }
        }
    }

    BitBoard goals{};
    goals.clear();
    for (int i = 0; i < 3; ++i) {
        const int idx = best_idx[static_cast<size_t>(i)];
        if (idx >= 0) {
            mark_cell(state, goals, detail::cell_col(idx), detail::cell_row(idx));
        }
    }
    return goals;
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

bool can_step_in_active(const BoardState& state, int c, int r, Direction direction) {
    const int nc = c + direction_dc(direction);
    const int nr = r + direction_dr(direction);
    return state.in_active(nc, nr) && state.can_move_through(c, r, direction);
}

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

Action adjacent_transfer_action(int from_c, int from_r, int to_c, int to_r) {
    for (Direction direction : POLICY_DIRS) {
        if (from_c + direction_dc(direction) == to_c &&
            from_r + direction_dr(direction) == to_r) {
            return transfer_action(direction);
        }
    }
    return ACT_IDLE;
}

Action factory_support_worker_action(const BoardState& state, int factory, int owner) {
    std::array<int, MAX_ROBOTS> workers{};
    int worker_count = 0;
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] != 0 &&
            state.robots.owner[static_cast<size_t>(i)] == owner &&
            state.robots.type[static_cast<size_t>(i)] == WORKER) {
            insert_uid_sorted(state.robots, workers, worker_count, i);
        }
    }

    const int fc = state.robots.col[static_cast<size_t>(factory)];
    const int fr = state.robots.row[static_cast<size_t>(factory)];
    for (int i = 0; i < worker_count; ++i) {
        const int worker = workers[static_cast<size_t>(i)];
        const int wc = state.robots.col[static_cast<size_t>(worker)];
        const int wr = state.robots.row[static_cast<size_t>(worker)];
        if (std::abs(fc - wc) + std::abs(fr - wr) == 1 &&
            state.robots.energy[static_cast<size_t>(worker)] < WORKER_MAX_ENERGY - 100) {
            return adjacent_transfer_action(fc, fr, wc, wr);
        }
    }
    return ACT_IDLE;
}

Action transfer_to_adjacent_factory(const BoardState& state, int robot_index, int factory) {
    if (factory < 0) {
        return ACT_IDLE;
    }
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    const int fc = state.robots.col[static_cast<size_t>(factory)];
    const int fr = state.robots.row[static_cast<size_t>(factory)];
    if (std::abs(c - fc) + std::abs(r - fr) != 1) {
        return ACT_IDLE;
    }
    return adjacent_transfer_action(c, r, fc, fr);
}

void collect_policy_units(const BoardState& state, int owner, std::array<int, MAX_ROBOTS>& workers, int& worker_count,
                          std::array<int, MAX_ROBOTS>& scouts, int& scout_count,
                          std::array<int, MAX_ROBOTS>& miners, int& miner_count,
                          BitBoard& occupied_enemy) {
    worker_count = 0;
    scout_count = 0;
    miner_count = 0;
    occupied_enemy.clear();
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        if (state.robots.owner[static_cast<size_t>(i)] != owner) {
            mark_cell(state, occupied_enemy, state.robots.col[static_cast<size_t>(i)],
                      state.robots.row[static_cast<size_t>(i)]);
            continue;
        }
        const uint8_t type = state.robots.type[static_cast<size_t>(i)];
        if (type == WORKER) {
            insert_uid_sorted(state.robots, workers, worker_count, i);
        } else if (type == SCOUT) {
            insert_uid_sorted(state.robots, scouts, scout_count, i);
        } else if (type == MINER) {
            insert_uid_sorted(state.robots, miners, miner_count, i);
        }
    }
}

void add_macro(MacroList& list, MacroAction macro) {
    if (list.count >= MAX_MACROS) {
        return;
    }
    list.macros[static_cast<size_t>(list.count++)] = macro;
}

Action factory_advance_action(const BoardState& state, int robot_index, const BitBoard& avoid, int depth) {
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    const int goal_row = std::min(state.north_bound, r + 25);
    const BitBoard goals = row_goals(state, goal_row);
    return get_path_policy(state, c, r, goals, avoid, depth,
                           state.robots.jump_cd[static_cast<size_t>(robot_index)]);
}

}  // namespace

Action CrawlerSim::heuristic_action_for(int robot_index) const {
    return heuristic_action_for_owner(robot_index, state.player);
}

Action CrawlerSim::heuristic_action_for_owner(int robot_index, int for_owner) const {
    if (robot_index < 0 || robot_index >= state.robots.used ||
        state.robots.alive[static_cast<size_t>(robot_index)] == 0 ||
        state.robots.owner[static_cast<size_t>(robot_index)] != for_owner) {
        return ACT_IDLE;
    }
    PrimitiveActions actions{};
    actions.clear();
    fill_heuristic_plan_for_owner(for_owner, actions, nullptr);
    return actions.actions[static_cast<size_t>(robot_index)];
}

void CrawlerSim::fill_heuristic_plan_for_owner(int owner, PrimitiveActions& actions,
                                               std::array<MacroAction, MAX_ROBOTS>* baseline_macros) const {
    if (owner < 0 || owner > 1) {
        return;
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] != 0 &&
            state.robots.owner[static_cast<size_t>(i)] == owner) {
            actions.actions[static_cast<size_t>(i)] = ACT_IDLE;
            if (baseline_macros != nullptr) {
                (*baseline_macros)[static_cast<size_t>(i)] = MACRO_IDLE;
            }
        }
    }

    std::array<int, MAX_ROBOTS> workers{};
    std::array<int, MAX_ROBOTS> scouts{};
    std::array<int, MAX_ROBOTS> miners{};
    int worker_count = 0;
    int scout_count = 0;
    int miner_count = 0;
    BitBoard occupied_enemy{};
    collect_policy_units(state, owner, workers, worker_count, scouts, scout_count, miners, miner_count,
                         occupied_enemy);

    BitBoard reserved{};
    reserved.clear();
    const int factory = primary_factory_for_owner(state, owner);
    if (factory >= 0) {
        const int fc = state.robots.col[static_cast<size_t>(factory)];
        const int fr = state.robots.row[static_cast<size_t>(factory)];
        const int fe = state.robots.energy[static_cast<size_t>(factory)];
        Action action = ACT_IDLE;
        MacroAction macro = MACRO_IDLE;

        action = factory_support_worker_action(state, factory, owner);
        if (action != ACT_IDLE) {
            macro = MACRO_FACTORY_SUPPORT_WORKER;
        }

        if (action == ACT_IDLE && fr - state.south_bound <= 3 && state.south_bound > 0 &&
            state.robots.jump_cd[static_cast<size_t>(factory)] == 0 &&
            can_jump_policy(state, fc, fr, DIR_NORTH)) {
            action = ACT_JUMP_NORTH;
            macro = MACRO_FACTORY_JUMP_OBSTACLE;
        }

        if (action == ACT_IDLE && state.robots.move_cd[static_cast<size_t>(factory)] <= 1) {
            BitBoard polite = occupied_enemy;
            for (int i = 0; i < worker_count; ++i) {
                const int robot = workers[static_cast<size_t>(i)];
                mark_cell(state, polite, state.robots.col[static_cast<size_t>(robot)],
                          state.robots.row[static_cast<size_t>(robot)]);
            }
            for (int i = 0; i < scout_count; ++i) {
                const int robot = scouts[static_cast<size_t>(i)];
                mark_cell(state, polite, state.robots.col[static_cast<size_t>(robot)],
                          state.robots.row[static_cast<size_t>(robot)]);
            }
            action = factory_advance_action(state, factory, polite, 40);
            if (action == ACT_IDLE) {
                action = factory_advance_action(state, factory, occupied_enemy, 40);
            }
            if (action != ACT_IDLE) {
                macro = MACRO_FACTORY_SAFE_ADVANCE;
            }
        }

        if (action == ACT_IDLE && state.robots.build_cd[static_cast<size_t>(factory)] == 0 &&
            fr + 1 <= state.north_bound && (policy_wall_at(state, fc, fr) & WALL_N) == 0 &&
            !friendly_cell_occupied(state, owner, fc, fr + 1)) {
            if (worker_count < 2 && can_pay_after_drain(fe, WORKER_COST)) {
                action = ACT_BUILD_WORKER;
                macro = MACRO_FACTORY_BUILD_WORKER;
            } else if (scout_count < 1 && fe >= SCOUT_COST + 300 + ENERGY_PER_TURN) {
                action = ACT_BUILD_SCOUT;
                macro = MACRO_FACTORY_BUILD_SCOUT;
            }
        }

        actions.actions[static_cast<size_t>(factory)] = action;
        if (baseline_macros != nullptr) {
            (*baseline_macros)[static_cast<size_t>(factory)] = macro;
        }
        reserve_destination(state, reserved, fc, fr, action);
    }

    for (int i = 0; i < worker_count; ++i) {
        const int worker = workers[static_cast<size_t>(i)];
        const int wc = state.robots.col[static_cast<size_t>(worker)];
        const int wr = state.robots.row[static_cast<size_t>(worker)];
        const int we = state.robots.energy[static_cast<size_t>(worker)];
        Action action = ACT_IDLE;
        MacroAction macro = MACRO_IDLE;

        if ((policy_wall_at(state, wc, wr) & WALL_N) != 0 && can_pay_after_drain(we, WALL_REMOVE_COST)) {
            action = ACT_REMOVE_NORTH;
            macro = MACRO_WORKER_OPEN_NORTH_WALL;
        }

        if (action == ACT_IDLE && state.robots.move_cd[static_cast<size_t>(worker)] <= 1) {
            BitBoard avoid = merged_mask(reserved, occupied_enemy);
            if (factory >= 0) {
                const int fc = state.robots.col[static_cast<size_t>(factory)];
                const int fr = state.robots.row[static_cast<size_t>(factory)];
                const BitBoard goals = single_goal(state, fc, std::min(state.north_bound, fr + 5));
                action = get_path_policy(state, wc, wr, goals, avoid, 25, NO_JUMP_PATH_CD);
                if (action != ACT_IDLE) {
                    macro = MACRO_WORKER_ESCORT_FACTORY;
                }
            } else {
                const BitBoard goals = single_goal(state, wc, std::min(state.north_bound, wr + 5));
                action = get_path_policy(state, wc, wr, goals, avoid, 25, NO_JUMP_PATH_CD);
                if (action != ACT_IDLE) {
                    macro = MACRO_WORKER_ADVANCE;
                }
            }
            if (action == ACT_IDLE) {
                const BitBoard goals = single_goal(state, wc, std::min(state.north_bound, wr + 5));
                action = get_path_policy(state, wc, wr, goals, avoid, 20, NO_JUMP_PATH_CD);
                if (action != ACT_IDLE) {
                    macro = MACRO_WORKER_ADVANCE;
                }
            }
        }

        actions.actions[static_cast<size_t>(worker)] = action;
        if (baseline_macros != nullptr) {
            (*baseline_macros)[static_cast<size_t>(worker)] = macro;
        }
        reserve_destination(state, reserved, wc, wr, action);
    }

    for (int i = 0; i < scout_count; ++i) {
        const int scout = scouts[static_cast<size_t>(i)];
        const int sc = state.robots.col[static_cast<size_t>(scout)];
        const int sr = state.robots.row[static_cast<size_t>(scout)];
        const int se = state.robots.energy[static_cast<size_t>(scout)];
        Action action = ACT_IDLE;
        MacroAction macro = MACRO_IDLE;

        if (factory >= 0 && se >= 50) {
            action = transfer_to_adjacent_factory(state, scout, factory);
            if (action != ACT_IDLE) {
                macro = MACRO_SCOUT_RETURN_ENERGY;
            }
        }

        if (action == ACT_IDLE && state.robots.move_cd[static_cast<size_t>(scout)] <= 0) {
            BitBoard goals{};
            goals.clear();
            BitBoard avoid = merged_mask(reserved, occupied_enemy);
            int depth = 30;
            if (se > 80 && factory >= 0) {
                goals = single_goal(state, state.robots.col[static_cast<size_t>(factory)],
                                    state.robots.row[static_cast<size_t>(factory)]);
                macro = MACRO_SCOUT_RETURN_ENERGY;
            } else {
                goals = nearest_crystal_goals(state, scout);
                if (goals.any()) {
                    macro = MACRO_SCOUT_HUNT_CRYSTAL;
                } else {
                    goals = row_goals(state, std::min(state.north_bound, sr + 15));
                    macro = MACRO_SCOUT_EXPLORE_NORTH;
                }
            }
            action = get_path_policy(state, sc, sr, goals, avoid, depth, NO_JUMP_PATH_CD);
            if (action == ACT_IDLE) {
                macro = MACRO_IDLE;
            }
        }

        actions.actions[static_cast<size_t>(scout)] = action;
        if (baseline_macros != nullptr) {
            (*baseline_macros)[static_cast<size_t>(scout)] = macro;
        }
        reserve_destination(state, reserved, sc, sr, action);
    }

    for (int i = 0; i < miner_count; ++i) {
        const int miner = miners[static_cast<size_t>(i)];
        const int c = state.robots.col[static_cast<size_t>(miner)];
        const int r = state.robots.row[static_cast<size_t>(miner)];
        const int e = state.robots.energy[static_cast<size_t>(miner)];
        const int idx = state.abs_index(c, r);
        Action action = ACT_IDLE;
        MacroAction macro = MACRO_IDLE;
        if (idx >= 0 && state.mining_node[static_cast<size_t>(idx)] != 0 &&
            can_pay_after_drain(e, TRANSFORM_COST)) {
            action = ACT_TRANSFORM;
            macro = MACRO_MINER_TRANSFORM;
        } else {
            action = best_passable_direction(state, miner);
            if (action != ACT_IDLE) {
                macro = MACRO_MINER_SEEK_NODE;
            }
        }
        actions.actions[static_cast<size_t>(miner)] = action;
        if (baseline_macros != nullptr) {
            (*baseline_macros)[static_cast<size_t>(miner)] = macro;
        }
    }
}

MacroList CrawlerSim::generate_macros_for(int robot_index) const {
    MacroList list{};
    if (robot_index < 0 || robot_index >= state.robots.used ||
        state.robots.alive[static_cast<size_t>(robot_index)] == 0) {
        return list;
    }
    const uint8_t type = state.robots.type[static_cast<size_t>(robot_index)];
    add_macro(list, MACRO_IDLE);
    if (type == FACTORY) {
        add_macro(list, MACRO_FACTORY_SUPPORT_WORKER);
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

Action CrawlerSim::primitive_for_macro(int robot_index, MacroAction macro) const {
    if (robot_index < 0 || robot_index >= state.robots.used ||
        state.robots.alive[static_cast<size_t>(robot_index)] == 0) {
        return ACT_IDLE;
    }

    const uint8_t type = state.robots.type[static_cast<size_t>(robot_index)];
    const int owner = state.robots.owner[static_cast<size_t>(robot_index)];
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    const int e = state.robots.energy[static_cast<size_t>(robot_index)];

    BitBoard occupied_enemy{};
    occupied_enemy.clear();
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] != 0 &&
            state.robots.owner[static_cast<size_t>(i)] != owner) {
            mark_cell(state, occupied_enemy, state.robots.col[static_cast<size_t>(i)],
                      state.robots.row[static_cast<size_t>(i)]);
        }
    }

    switch (macro) {
        case MACRO_FACTORY_SUPPORT_WORKER:
            if (type == FACTORY) {
                return factory_support_worker_action(state, robot_index, owner);
            }
            break;
        case MACRO_FACTORY_BUILD_WORKER:
            if (type == FACTORY && state.robots.build_cd[static_cast<size_t>(robot_index)] == 0 &&
                can_pay_after_drain(e, WORKER_COST) && r + 1 <= state.north_bound &&
                (policy_wall_at(state, c, r) & WALL_N) == 0 &&
                !friendly_cell_occupied(state, owner, c, r + 1)) {
                return ACT_BUILD_WORKER;
            }
            break;
        case MACRO_FACTORY_BUILD_SCOUT:
            if (type == FACTORY && state.robots.build_cd[static_cast<size_t>(robot_index)] == 0 &&
                e >= SCOUT_COST + 300 + ENERGY_PER_TURN && r + 1 <= state.north_bound &&
                (policy_wall_at(state, c, r) & WALL_N) == 0 &&
                !friendly_cell_occupied(state, owner, c, r + 1)) {
                return ACT_BUILD_SCOUT;
            }
            break;
        case MACRO_FACTORY_BUILD_MINER:
            if (type == FACTORY && state.robots.build_cd[static_cast<size_t>(robot_index)] == 0 &&
                can_pay_after_drain(e, MINER_COST) && r + 1 <= state.north_bound &&
                state.can_move_through(c, r, DIR_NORTH)) {
                return ACT_BUILD_MINER;
            }
            break;
        case MACRO_FACTORY_JUMP_OBSTACLE:
            if (type == FACTORY && state.robots.jump_cd[static_cast<size_t>(robot_index)] == 0 &&
                can_jump_policy(state, c, r, DIR_NORTH)) {
                return ACT_JUMP_NORTH;
            }
            break;
        case MACRO_FACTORY_SAFE_ADVANCE:
            if (type == FACTORY && state.robots.move_cd[static_cast<size_t>(robot_index)] <= 1) {
                return factory_advance_action(state, robot_index, occupied_enemy, 40);
            }
            break;
        case MACRO_WORKER_OPEN_NORTH_WALL:
            if (type == WORKER && (policy_wall_at(state, c, r) & WALL_N) != 0 &&
                can_pay_after_drain(e, WALL_REMOVE_COST)) {
                return ACT_REMOVE_NORTH;
            }
            break;
        case MACRO_WORKER_ESCORT_FACTORY:
            if (type == WORKER && state.robots.move_cd[static_cast<size_t>(robot_index)] <= 1) {
                const int factory = primary_factory_for_owner(state, owner);
                const BitBoard goals = factory >= 0
                                           ? single_goal(state, state.robots.col[static_cast<size_t>(factory)],
                                                         std::min(state.north_bound,
                                                                  state.robots.row[static_cast<size_t>(factory)] + 5))
                                           : single_goal(state, c, std::min(state.north_bound, r + 5));
                return get_path_policy(state, c, r, goals, occupied_enemy, 25, NO_JUMP_PATH_CD);
            }
            break;
        case MACRO_WORKER_ADVANCE:
            if (type == WORKER) {
                if ((policy_wall_at(state, c, r) & WALL_N) != 0 && can_pay_after_drain(e, WALL_REMOVE_COST)) {
                    return ACT_REMOVE_NORTH;
                }
                if (state.robots.move_cd[static_cast<size_t>(robot_index)] <= 1) {
                    const BitBoard goals = single_goal(state, c, std::min(state.north_bound, r + 5));
                    return get_path_policy(state, c, r, goals, occupied_enemy, 20, NO_JUMP_PATH_CD);
                }
            }
            break;
        case MACRO_SCOUT_HUNT_CRYSTAL:
            if (type == SCOUT && state.robots.move_cd[static_cast<size_t>(robot_index)] <= 0) {
                const BitBoard goals = nearest_crystal_goals(state, robot_index);
                return get_path_policy(state, c, r, goals, occupied_enemy, 30, NO_JUMP_PATH_CD);
            }
            break;
        case MACRO_SCOUT_EXPLORE_NORTH:
            if (type == SCOUT && state.robots.move_cd[static_cast<size_t>(robot_index)] <= 0) {
                const BitBoard goals = row_goals(state, std::min(state.north_bound, r + 15));
                return get_path_policy(state, c, r, goals, occupied_enemy, 30, NO_JUMP_PATH_CD);
            }
            break;
        case MACRO_SCOUT_RETURN_ENERGY:
            if (type == SCOUT) {
                const int factory = primary_factory_for_owner(state, owner);
                if (e >= 50) {
                    const Action transfer = transfer_to_adjacent_factory(state, robot_index, factory);
                    if (transfer != ACT_IDLE) {
                        return transfer;
                    }
                }
                if (factory >= 0 && state.robots.move_cd[static_cast<size_t>(robot_index)] <= 0) {
                    const BitBoard goals = single_goal(state, state.robots.col[static_cast<size_t>(factory)],
                                                       state.robots.row[static_cast<size_t>(factory)]);
                    return get_path_policy(state, c, r, goals, occupied_enemy, 30, NO_JUMP_PATH_CD);
                }
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
