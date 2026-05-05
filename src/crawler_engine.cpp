#include "crawler_engine.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace crawler {
namespace {

constexpr uint64_t RNG_MUL = 0xbf58476d1ce4e5b9ULL;
constexpr float EPS = 1.0e-6F;

uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

uint32_t next_u32(uint64_t& state) {
    state = state * RNG_MUL + 0x94d049bb133111ebULL;
    return static_cast<uint32_t>(mix64(state) >> 32U);
}

bool chance(uint64_t& state, int numerator, int denominator) {
    return static_cast<int>(next_u32(state) % static_cast<uint32_t>(denominator)) < numerator;
}

int active_word(int active_index) {
    return active_index >> 6;
}

uint64_t active_mask(int active_index) {
    return 1ULL << (active_index & 63);
}

void copy_uid(std::array<char, UID_LEN>& dst, std::string_view src) {
    dst.fill('\0');
    const int n = std::min<int>(static_cast<int>(src.size()), UID_LEN - 1);
    if (n > 0) {
        std::memcpy(dst.data(), src.data(), static_cast<size_t>(n));
    }
}

bool uid_equal(const std::array<char, UID_LEN>& uid, std::string_view value) {
    const int n = std::min<int>(static_cast<int>(value.size()), UID_LEN - 1);
    if (n == UID_LEN - 1 && static_cast<int>(value.size()) >= UID_LEN) {
        return false;
    }
    return std::strncmp(uid.data(), value.data(), static_cast<size_t>(n)) == 0 && uid[n] == '\0';
}

uint8_t reciprocal_wall(Direction direction) {
    return direction_wall_bit(opposite_direction(direction));
}

void set_or_clear_wall(BoardState& state, int c, int r, Direction direction, bool set_wall) {
    const int idx = state.abs_index(c, r);
    if (idx < 0) {
        return;
    }
    const uint8_t bit = direction_wall_bit(direction);
    if (set_wall) {
        state.walls[idx] = static_cast<uint8_t>(state.walls[idx] | bit);
    } else {
        state.walls[idx] = static_cast<uint8_t>(state.walls[idx] & static_cast<uint8_t>(~bit));
    }
    state.wall_known[idx] = 1;

    const int nc = c + direction_dc(direction);
    const int nr = r + direction_dr(direction);
    const int nidx = state.abs_index(nc, nr);
    if (nidx < 0) {
        return;
    }
    const uint8_t opp = reciprocal_wall(direction);
    if (set_wall) {
        state.walls[nidx] = static_cast<uint8_t>(state.walls[nidx] | opp);
    } else {
        state.walls[nidx] = static_cast<uint8_t>(state.walls[nidx] & static_cast<uint8_t>(~opp));
    }
    state.wall_known[nidx] = 1;
}

int py_round_interval(double value) {
    return static_cast<int>(std::nearbyint(value));
}

int scroll_interval(int step) {
    if (step >= SCROLL_RAMP_STEPS) {
        return SCROLL_END_INTERVAL;
    }
    const double progress = static_cast<double>(step) / static_cast<double>(SCROLL_RAMP_STEPS);
    const double value = static_cast<double>(SCROLL_START_INTERVAL) -
                         static_cast<double>(SCROLL_START_INTERVAL - SCROLL_END_INTERVAL) * progress;
    return std::max(SCROLL_END_INTERVAL, py_round_interval(value));
}

void generate_optimistic_row(BoardState& state, int row, uint64_t seed) {
    if (row < 0 || row >= MAX_ROWS) {
        return;
    }

    uint64_t rng = mix64(seed ^ static_cast<uint64_t>(row + 1) * 0x9e3779b97f4a7c15ULL);
    const int half = WIDTH / 2;
    std::array<uint8_t, WIDTH> row_walls{};

    for (int c = 0; c < WIDTH; ++c) {
        row_walls[c] = 0;
    }
    row_walls[0] = static_cast<uint8_t>(row_walls[0] | WALL_W);
    row_walls[WIDTH - 1] = static_cast<uint8_t>(row_walls[WIDTH - 1] | WALL_E);

    for (int c = 0; c < half - 1; ++c) {
        if (chance(rng, 24, 100)) {
            row_walls[c] = static_cast<uint8_t>(row_walls[c] | WALL_E);
            row_walls[c + 1] = static_cast<uint8_t>(row_walls[c + 1] | WALL_W);
        }
    }

    for (int c = 0; c < half; ++c) {
        const int mc = WIDTH - 1 - c;
        const uint8_t w = row_walls[c];
        uint8_t mirrored = 0;
        if ((w & WALL_N) != 0) {
            mirrored = static_cast<uint8_t>(mirrored | WALL_N);
        }
        if ((w & WALL_S) != 0) {
            mirrored = static_cast<uint8_t>(mirrored | WALL_S);
        }
        if ((w & WALL_E) != 0) {
            mirrored = static_cast<uint8_t>(mirrored | WALL_W);
        }
        if ((w & WALL_W) != 0) {
            mirrored = static_cast<uint8_t>(mirrored | WALL_E);
        }
        row_walls[mc] = mirrored;
    }

    if (chance(rng, 92, 100)) {
        row_walls[half - 1] = static_cast<uint8_t>(row_walls[half - 1] | WALL_E);
        row_walls[half] = static_cast<uint8_t>(row_walls[half] | WALL_W);
    } else {
        row_walls[half - 1] = static_cast<uint8_t>(row_walls[half - 1] & static_cast<uint8_t>(~WALL_E));
        row_walls[half] = static_cast<uint8_t>(row_walls[half] & static_cast<uint8_t>(~WALL_W));
    }

    if (row > 0) {
        for (int c = 0; c < WIDTH; ++c) {
            const int prev_idx = state.abs_index(c, row - 1);
            if (prev_idx >= 0 && (state.walls[prev_idx] & WALL_N) != 0) {
                row_walls[c] = static_cast<uint8_t>(row_walls[c] | WALL_S);
            } else {
                row_walls[c] = static_cast<uint8_t>(row_walls[c] & static_cast<uint8_t>(~WALL_S));
            }
        }
    }

    for (int c = 0; c < WIDTH; ++c) {
        const int idx = state.abs_index(c, row);
        state.walls[idx] = row_walls[c];
        state.wall_known[idx] = 1;
        state.crystal_energy[idx] = 0;
        state.mining_node[idx] = 0;
        if (chance(rng, 6, 100)) {
            state.crystal_energy[idx] = static_cast<int16_t>(10 + static_cast<int>(next_u32(rng) % 41U));
        } else if (chance(rng, 3, 100)) {
            state.mining_node[idx] = 1;
        }
    }
}

int cell_row(int abs_index) {
    return abs_index / WIDTH;
}

int cell_col(int abs_index) {
    return abs_index % WIDTH;
}

bool can_diffuse_through(const BeliefState& belief, int c, int r, Direction direction) {
    if (c < 0 || c >= WIDTH || r < 0 || r >= MAX_ROWS) {
        return false;
    }
    const int idx = r * WIDTH + c;
    if (belief.known_wall[idx] != 0 && (belief.wall[idx] & direction_wall_bit(direction)) != 0) {
        return false;
    }
    const int nc = c + direction_dc(direction);
    const int nr = r + direction_dr(direction);
    return nc >= 0 && nc < WIDTH && nr >= belief.south_bound && nr <= belief.north_bound && nr < MAX_ROWS;
}

void diffuse_enemy_type(BeliefState& belief, int type, int elapsed) {
    const int period = move_period(static_cast<uint8_t>(type));
    const int radius = (type == SCOUT) ? elapsed : ((elapsed + period - 1) / period);
    if (radius <= 0) {
        return;
    }

    std::array<float, MAX_CELLS> cur{};
    std::array<float, MAX_CELLS> next{};
    cur = belief.enemy_prob[static_cast<size_t>(type)];

    for (int step = 0; step < radius; ++step) {
        next.fill(0.0F);
        for (int idx = 0; idx < MAX_CELLS; ++idx) {
            const float p = cur[static_cast<size_t>(idx)];
            if (p <= EPS) {
                continue;
            }
            const int c = cell_col(idx);
            const int r = cell_row(idx);
            int choices = 1;
            std::array<Direction, 4> dirs{DIR_NORTH, DIR_SOUTH, DIR_EAST, DIR_WEST};
            for (Direction d : dirs) {
                if (can_diffuse_through(belief, c, r, d)) {
                    ++choices;
                }
            }
            const float share = p / static_cast<float>(choices);
            next[static_cast<size_t>(idx)] += share;
            for (Direction d : dirs) {
                if (!can_diffuse_through(belief, c, r, d)) {
                    continue;
                }
                const int ni = (r + direction_dr(d)) * WIDTH + (c + direction_dc(d));
                next[static_cast<size_t>(ni)] += share;
            }
        }
        cur = next;
    }

    belief.enemy_prob[static_cast<size_t>(type)] = cur;
}

void compute_rewards(BoardState& state) {
    int factory_count[2] = {0, 0};
    int energy[2] = {0, 0};
    int units[2] = {0, 0};
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const int owner = state.robots.owner[static_cast<size_t>(i)];
        if (owner < 0 || owner > 1) {
            continue;
        }
        energy[owner] += state.robots.energy[static_cast<size_t>(i)];
        units[owner] += 1;
        if (state.robots.type[static_cast<size_t>(i)] == FACTORY) {
            factory_count[owner] += 1;
        }
    }

    const bool dead0 = factory_count[0] == 0;
    const bool dead1 = factory_count[1] == 0;
    if (!dead0 && !dead1 && state.step + 1 < EPISODE_STEPS) {
        state.reward0 = static_cast<float>(energy[0]);
        state.reward1 = static_cast<float>(energy[1]);
        return;
    }

    if (dead0 && !dead1) {
        state.done = true;
        state.winner = 1;
        state.reward0 = static_cast<float>(state.step - EPISODE_STEPS - 1);
        state.reward1 = static_cast<float>(energy[1]);
        return;
    }
    if (dead1 && !dead0) {
        state.done = true;
        state.winner = 0;
        state.reward0 = static_cast<float>(energy[0]);
        state.reward1 = static_cast<float>(state.step - EPISODE_STEPS - 1);
        return;
    }

    state.done = true;
    if (energy[0] > energy[1]) {
        state.winner = 0;
        state.reward0 = 1.0F;
        state.reward1 = 0.0F;
    } else if (energy[1] > energy[0]) {
        state.winner = 1;
        state.reward0 = 0.0F;
        state.reward1 = 1.0F;
    } else if (units[0] > units[1]) {
        state.winner = 0;
        state.reward0 = 1.0F;
        state.reward1 = 0.0F;
    } else if (units[1] > units[0]) {
        state.winner = 1;
        state.reward0 = 0.0F;
        state.reward1 = 1.0F;
    } else {
        state.winner = -1;
        state.reward0 = 0.5F;
        state.reward1 = 0.5F;
    }
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

bool can_pay_after_drain(int energy, int cost) {
    return energy >= cost + ENERGY_PER_TURN;
}

bool ready_after_tick(int cooldown) {
    return cooldown <= 1;
}

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

void add_macro(MacroList& list, MacroAction macro) {
    if (list.count >= MAX_MACROS) {
        return;
    }
    list.macros[static_cast<size_t>(list.count++)] = macro;
}

}  // namespace

void BitBoard::clear() {
    words.fill(0ULL);
}

void BitBoard::set(int active_index) {
    if (active_index < 0 || active_index >= ACTIVE_CELLS) {
        return;
    }
    words[static_cast<size_t>(active_word(active_index))] |= active_mask(active_index);
}

void BitBoard::reset(int active_index) {
    if (active_index < 0 || active_index >= ACTIVE_CELLS) {
        return;
    }
    words[static_cast<size_t>(active_word(active_index))] &= ~active_mask(active_index);
}

bool BitBoard::test(int active_index) const {
    if (active_index < 0 || active_index >= ACTIVE_CELLS) {
        return false;
    }
    return (words[static_cast<size_t>(active_word(active_index))] & active_mask(active_index)) != 0;
}

bool BitBoard::any() const {
    for (uint64_t word : words) {
        if (word != 0ULL) {
            return true;
        }
    }
    return false;
}

int pop_lsb(uint64_t& bits) {
    const int offset = static_cast<int>(std::countr_zero(bits));
    bits &= bits - 1ULL;
    return offset;
}

const char* action_name(Action action) {
    switch (action) {
        case ACT_NORTH: return "NORTH";
        case ACT_SOUTH: return "SOUTH";
        case ACT_EAST: return "EAST";
        case ACT_WEST: return "WEST";
        case ACT_BUILD_SCOUT: return "BUILD_SCOUT";
        case ACT_BUILD_WORKER: return "BUILD_WORKER";
        case ACT_BUILD_MINER: return "BUILD_MINER";
        case ACT_JUMP_NORTH: return "JUMP_NORTH";
        case ACT_JUMP_SOUTH: return "JUMP_SOUTH";
        case ACT_JUMP_EAST: return "JUMP_EAST";
        case ACT_JUMP_WEST: return "JUMP_WEST";
        case ACT_BUILD_NORTH: return "BUILD_NORTH";
        case ACT_BUILD_SOUTH: return "BUILD_SOUTH";
        case ACT_BUILD_EAST: return "BUILD_EAST";
        case ACT_BUILD_WEST: return "BUILD_WEST";
        case ACT_REMOVE_NORTH: return "REMOVE_NORTH";
        case ACT_REMOVE_SOUTH: return "REMOVE_SOUTH";
        case ACT_REMOVE_EAST: return "REMOVE_EAST";
        case ACT_REMOVE_WEST: return "REMOVE_WEST";
        case ACT_TRANSFORM: return "TRANSFORM";
        case ACT_TRANSFER_NORTH: return "TRANSFER_NORTH";
        case ACT_TRANSFER_SOUTH: return "TRANSFER_SOUTH";
        case ACT_TRANSFER_EAST: return "TRANSFER_EAST";
        case ACT_TRANSFER_WEST: return "TRANSFER_WEST";
        case ACT_IDLE:
        default: return "IDLE";
    }
}

const char* macro_action_name(MacroAction macro) {
    switch (macro) {
        case MACRO_FACTORY_SAFE_ADVANCE: return "FACTORY_SAFE_ADVANCE";
        case MACRO_FACTORY_BUILD_WORKER: return "FACTORY_BUILD_WORKER";
        case MACRO_FACTORY_BUILD_SCOUT: return "FACTORY_BUILD_SCOUT";
        case MACRO_FACTORY_BUILD_MINER: return "FACTORY_BUILD_MINER";
        case MACRO_FACTORY_JUMP_OBSTACLE: return "FACTORY_JUMP_OBSTACLE";
        case MACRO_WORKER_OPEN_NORTH_WALL: return "WORKER_OPEN_NORTH_WALL";
        case MACRO_WORKER_ESCORT_FACTORY: return "WORKER_ESCORT_FACTORY";
        case MACRO_WORKER_ADVANCE: return "WORKER_ADVANCE";
        case MACRO_SCOUT_HUNT_CRYSTAL: return "SCOUT_HUNT_CRYSTAL";
        case MACRO_SCOUT_EXPLORE_NORTH: return "SCOUT_EXPLORE_NORTH";
        case MACRO_SCOUT_RETURN_ENERGY: return "SCOUT_RETURN_ENERGY";
        case MACRO_MINER_SEEK_NODE: return "MINER_SEEK_NODE";
        case MACRO_MINER_TRANSFORM: return "MINER_TRANSFORM";
        case MACRO_IDLE:
        default: return "IDLE";
    }
}

Action parse_action(std::string_view value) {
    if (value == "NORTH") return ACT_NORTH;
    if (value == "SOUTH") return ACT_SOUTH;
    if (value == "EAST") return ACT_EAST;
    if (value == "WEST") return ACT_WEST;
    if (value == "BUILD_SCOUT") return ACT_BUILD_SCOUT;
    if (value == "BUILD_WORKER") return ACT_BUILD_WORKER;
    if (value == "BUILD_MINER") return ACT_BUILD_MINER;
    if (value == "JUMP_NORTH") return ACT_JUMP_NORTH;
    if (value == "JUMP_SOUTH") return ACT_JUMP_SOUTH;
    if (value == "JUMP_EAST") return ACT_JUMP_EAST;
    if (value == "JUMP_WEST") return ACT_JUMP_WEST;
    if (value == "BUILD_NORTH") return ACT_BUILD_NORTH;
    if (value == "BUILD_SOUTH") return ACT_BUILD_SOUTH;
    if (value == "BUILD_EAST") return ACT_BUILD_EAST;
    if (value == "BUILD_WEST") return ACT_BUILD_WEST;
    if (value == "REMOVE_NORTH") return ACT_REMOVE_NORTH;
    if (value == "REMOVE_SOUTH") return ACT_REMOVE_SOUTH;
    if (value == "REMOVE_EAST") return ACT_REMOVE_EAST;
    if (value == "REMOVE_WEST") return ACT_REMOVE_WEST;
    if (value == "TRANSFORM") return ACT_TRANSFORM;
    if (value == "TRANSFER_NORTH") return ACT_TRANSFER_NORTH;
    if (value == "TRANSFER_SOUTH") return ACT_TRANSFER_SOUTH;
    if (value == "TRANSFER_EAST") return ACT_TRANSFER_EAST;
    if (value == "TRANSFER_WEST") return ACT_TRANSFER_WEST;
    return ACT_IDLE;
}

Direction action_direction(Action action) {
    switch (action) {
        case ACT_NORTH:
        case ACT_JUMP_NORTH:
        case ACT_BUILD_NORTH:
        case ACT_REMOVE_NORTH:
        case ACT_TRANSFER_NORTH:
            return DIR_NORTH;
        case ACT_SOUTH:
        case ACT_JUMP_SOUTH:
        case ACT_BUILD_SOUTH:
        case ACT_REMOVE_SOUTH:
        case ACT_TRANSFER_SOUTH:
            return DIR_SOUTH;
        case ACT_EAST:
        case ACT_JUMP_EAST:
        case ACT_BUILD_EAST:
        case ACT_REMOVE_EAST:
        case ACT_TRANSFER_EAST:
            return DIR_EAST;
        case ACT_WEST:
        case ACT_JUMP_WEST:
        case ACT_BUILD_WEST:
        case ACT_REMOVE_WEST:
        case ACT_TRANSFER_WEST:
            return DIR_WEST;
        default:
            return DIR_NONE;
    }
}

uint8_t direction_wall_bit(Direction direction) {
    switch (direction) {
        case DIR_NORTH: return WALL_N;
        case DIR_SOUTH: return WALL_S;
        case DIR_EAST: return WALL_E;
        case DIR_WEST: return WALL_W;
        default: return 0;
    }
}

Direction opposite_direction(Direction direction) {
    switch (direction) {
        case DIR_NORTH: return DIR_SOUTH;
        case DIR_SOUTH: return DIR_NORTH;
        case DIR_EAST: return DIR_WEST;
        case DIR_WEST: return DIR_EAST;
        default: return DIR_NONE;
    }
}

int direction_dc(Direction direction) {
    if (direction == DIR_EAST) return 1;
    if (direction == DIR_WEST) return -1;
    return 0;
}

int direction_dr(Direction direction) {
    if (direction == DIR_NORTH) return 1;
    if (direction == DIR_SOUTH) return -1;
    return 0;
}

int move_period(uint8_t type) {
    switch (type) {
        case FACTORY: return FACTORY_MOVE_PERIOD;
        case WORKER: return WORKER_MOVE_PERIOD;
        case MINER: return MINER_MOVE_PERIOD;
        case SCOUT:
        default: return SCOUT_MOVE_PERIOD;
    }
}

int max_energy(uint8_t type) {
    switch (type) {
        case SCOUT: return SCOUT_MAX_ENERGY;
        case WORKER: return WORKER_MAX_ENERGY;
        case MINER: return MINER_MAX_ENERGY;
        case FACTORY:
        default: return std::numeric_limits<int>::max() / 4;
    }
}

int vision_range(uint8_t type) {
    switch (type) {
        case FACTORY: return VISION_FACTORY;
        case SCOUT: return VISION_SCOUT;
        case WORKER: return VISION_WORKER;
        case MINER: return VISION_MINER;
        default: return 0;
    }
}

bool is_fixed_wall(int col, Direction direction) {
    const int half = WIDTH / 2;
    if (direction == DIR_WEST && col == 0) {
        return true;
    }
    if (direction == DIR_EAST && col == WIDTH - 1) {
        return true;
    }
    if (direction == DIR_EAST && col == half - 1) {
        return true;
    }
    if (direction == DIR_WEST && col == half) {
        return true;
    }
    return false;
}

void RobotStore::clear() {
    for (auto& u : uid) {
        u.fill('\0');
    }
    alive.fill(0);
    type.fill(0);
    owner.fill(0);
    col.fill(0);
    row.fill(0);
    energy.fill(0);
    move_cd.fill(0);
    jump_cd.fill(0);
    build_cd.fill(0);
    used = 0;
}

int RobotStore::find_uid(std::string_view value) const {
    for (int i = 0; i < used; ++i) {
        if (alive[static_cast<size_t>(i)] != 0 && uid_equal(uid[static_cast<size_t>(i)], value)) {
            return i;
        }
    }
    return -1;
}

int RobotStore::add_robot(std::string_view uid_value, uint8_t robot_type, uint8_t robot_owner,
                          int robot_col, int robot_row, int robot_energy,
                          int move_cooldown, int jump_cooldown, int build_cooldown) {
    int slot = -1;
    for (int i = 0; i < used; ++i) {
        if (alive[static_cast<size_t>(i)] == 0) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        if (used >= MAX_ROBOTS) {
            return -1;
        }
        slot = used++;
    }

    copy_uid(uid[static_cast<size_t>(slot)], uid_value);
    alive[static_cast<size_t>(slot)] = 1;
    type[static_cast<size_t>(slot)] = robot_type;
    owner[static_cast<size_t>(slot)] = robot_owner;
    col[static_cast<size_t>(slot)] = static_cast<int16_t>(robot_col);
    row[static_cast<size_t>(slot)] = static_cast<int16_t>(robot_row);
    energy[static_cast<size_t>(slot)] = static_cast<int16_t>(robot_energy);
    move_cd[static_cast<size_t>(slot)] = static_cast<int16_t>(move_cooldown);
    jump_cd[static_cast<size_t>(slot)] = static_cast<int16_t>(jump_cooldown);
    build_cd[static_cast<size_t>(slot)] = static_cast<int16_t>(build_cooldown);
    return slot;
}

int RobotStore::add_generated_robot(uint32_t serial, uint8_t robot_type, uint8_t robot_owner,
                                    int robot_col, int robot_row, int robot_energy) {
    std::array<char, UID_LEN> generated{};
    std::snprintf(generated.data(), generated.size(), "sim-%u", serial);
    return add_robot(generated.data(), robot_type, robot_owner, robot_col, robot_row, robot_energy,
                     move_period(robot_type), 0, 0);
}

void RobotStore::remove(int index) {
    if (index < 0 || index >= used) {
        return;
    }
    alive[static_cast<size_t>(index)] = 0;
}

void BoardState::reset() {
    player = 0;
    step = 0;
    south_bound = 0;
    north_bound = HEIGHT - 1;
    scroll_counter = SCROLL_START_INTERVAL;
    next_generated_uid = 1;
    rng_state = 0x9e3779b97f4a7c15ULL;
    done = false;
    winner = -1;
    reward0 = 0.0F;
    reward1 = 0.0F;
    walls.fill(0);
    wall_known.fill(0);
    crystal_energy.fill(0);
    mine_energy.fill(0);
    mine_max.fill(0);
    mine_owner.fill(-1);
    mining_node.fill(0);
    robots.clear();
    own_occupancy.clear();
    enemy_occupancy.clear();
    all_occupancy.clear();
    visibility.clear();
    crystals_active.clear();
    mines_active.clear();
    nodes_active.clear();
}

int BoardState::abs_index(int c, int r) const {
    if (c < 0 || c >= WIDTH || r < 0 || r >= MAX_ROWS) {
        return -1;
    }
    return r * WIDTH + c;
}

int BoardState::active_index(int c, int r) const {
    if (!in_active(c, r)) {
        return -1;
    }
    return (r - south_bound) * WIDTH + c;
}

bool BoardState::in_active(int c, int r) const {
    return c >= 0 && c < WIDTH && r >= south_bound && r <= north_bound && r < MAX_ROWS;
}

uint8_t BoardState::wall_at(int c, int r) const {
    const int idx = abs_index(c, r);
    if (idx < 0) {
        return static_cast<uint8_t>(WALL_N | WALL_E | WALL_S | WALL_W);
    }
    return walls[static_cast<size_t>(idx)];
}

bool BoardState::can_move_through(int c, int r, Direction direction) const {
    const int idx = abs_index(c, r);
    if (idx < 0 || direction == DIR_NONE) {
        return false;
    }
    if ((walls[static_cast<size_t>(idx)] & direction_wall_bit(direction)) != 0) {
        return false;
    }
    const int nc = c + direction_dc(direction);
    const int nr = r + direction_dr(direction);
    return nc >= 0 && nc < WIDTH && nr >= 0 && nr < MAX_ROWS;
}

void BoardState::rebuild_active_bitboards() {
    own_occupancy.clear();
    enemy_occupancy.clear();
    all_occupancy.clear();
    visibility.clear();
    crystals_active.clear();
    mines_active.clear();
    nodes_active.clear();

    for (int i = 0; i < robots.used; ++i) {
        if (robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const int ai = active_index(robots.col[static_cast<size_t>(i)], robots.row[static_cast<size_t>(i)]);
        if (ai < 0) {
            continue;
        }
        all_occupancy.set(ai);
        if (robots.owner[static_cast<size_t>(i)] == player) {
            own_occupancy.set(ai);
            const int range = vision_range(robots.type[static_cast<size_t>(i)]);
            const int rc = robots.col[static_cast<size_t>(i)];
            const int rr = robots.row[static_cast<size_t>(i)];
            for (int dc = -range; dc <= range; ++dc) {
                const int rem = range - std::abs(dc);
                for (int dr = -rem; dr <= rem; ++dr) {
                    const int vi = active_index(rc + dc, rr + dr);
                    if (vi >= 0) {
                        visibility.set(vi);
                    }
                }
            }
        } else {
            enemy_occupancy.set(ai);
        }
    }

    for (int r = south_bound; r <= north_bound; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            const int abs = abs_index(c, r);
            const int ai = active_index(c, r);
            if (abs < 0 || ai < 0) {
                continue;
            }
            if (crystal_energy[static_cast<size_t>(abs)] > 0) {
                crystals_active.set(ai);
            }
            if (mine_max[static_cast<size_t>(abs)] > 0) {
                mines_active.set(ai);
            }
            if (mining_node[static_cast<size_t>(abs)] != 0) {
                nodes_active.set(ai);
            }
        }
    }
}

void PrimitiveActions::clear() {
    actions.fill(ACT_IDLE);
}

void ActionResult::clear() {
    count = 0;
    for (auto& u : uid) {
        u.fill('\0');
    }
    action.fill(ACT_IDLE);
}

void ActionResult::add(std::string_view uid_value, Action primitive) {
    if (count >= MAX_ROBOTS) {
        return;
    }
    copy_uid(uid[static_cast<size_t>(count)], uid_value);
    action[static_cast<size_t>(count)] = primitive;
    ++count;
}

void BeliefState::reset() {
    player = 0;
    turn = 0;
    south_bound = 0;
    north_bound = HEIGHT - 1;
    known_wall.fill(0);
    wall.fill(0);
    visible_crystal.fill(0);
    remembered_mine_energy.fill(0);
    remembered_mine_max.fill(0);
    remembered_mine_owner.fill(-1);
    remembered_node.fill(0);
    currently_visible.fill(0);
    for (auto& p : enemy_prob) {
        p.fill(0.0F);
    }
}

void BeliefState::update_from_observation(const ObservationInput& obs) {
    const int new_turn = obs.step >= 0 ? obs.step : turn + 1;
    const int elapsed = std::max(1, new_turn - turn);
    player = obs.player;
    south_bound = obs.south_bound;
    north_bound = obs.north_bound;

    for (int type = FACTORY; type <= MINER; ++type) {
        diffuse_enemy_type(*this, type, elapsed);
    }

    currently_visible.fill(0);
    for (int i = 0; i < obs.robot_count; ++i) {
        const RobotObservation& robot = obs.robots[static_cast<size_t>(i)];
        if (robot.owner != player) {
            continue;
        }
        const int range = vision_range(static_cast<uint8_t>(robot.type));
        for (int dc = -range; dc <= range; ++dc) {
            const int rem = range - std::abs(dc);
            for (int dr = -rem; dr <= rem; ++dr) {
                const int c = robot.col + dc;
                const int r = robot.row + dr;
                if (c < 0 || c >= WIDTH || r < 0 || r >= MAX_ROWS) {
                    continue;
                }
                currently_visible[static_cast<size_t>(r * WIDTH + c)] = 1;
            }
        }
    }

    for (int r = obs.south_bound; r <= obs.north_bound; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            const int local = (r - obs.south_bound) * WIDTH + c;
            const int abs = r * WIDTH + c;
            if (obs.walls[static_cast<size_t>(local)] >= 0) {
                known_wall[static_cast<size_t>(abs)] = 1;
                wall[static_cast<size_t>(abs)] = static_cast<uint8_t>(obs.walls[static_cast<size_t>(local)]);
            }
            if (currently_visible[static_cast<size_t>(abs)] != 0) {
                visible_crystal[static_cast<size_t>(abs)] = 0;
                remembered_mine_energy[static_cast<size_t>(abs)] = 0;
                remembered_mine_max[static_cast<size_t>(abs)] = 0;
                remembered_mine_owner[static_cast<size_t>(abs)] = -1;
                for (auto& prob : enemy_prob) {
                    prob[static_cast<size_t>(abs)] = 0.0F;
                }
            }
        }
    }

    for (int i = 0; i < obs.crystal_count; ++i) {
        const auto& crystal = obs.crystals[static_cast<size_t>(i)];
        if (crystal.col < 0 || crystal.col >= WIDTH || crystal.row < 0 || crystal.row >= MAX_ROWS) {
            continue;
        }
        visible_crystal[static_cast<size_t>(crystal.row * WIDTH + crystal.col)] =
            static_cast<int16_t>(crystal.energy);
    }

    for (int i = 0; i < obs.mine_count; ++i) {
        const auto& mine = obs.mines[static_cast<size_t>(i)];
        if (mine.col < 0 || mine.col >= WIDTH || mine.row < 0 || mine.row >= MAX_ROWS) {
            continue;
        }
        const int idx = mine.row * WIDTH + mine.col;
        remembered_mine_energy[static_cast<size_t>(idx)] = static_cast<int16_t>(mine.energy);
        remembered_mine_max[static_cast<size_t>(idx)] = static_cast<int16_t>(mine.max_energy);
        remembered_mine_owner[static_cast<size_t>(idx)] = static_cast<int8_t>(mine.owner);
    }

    for (int i = 0; i < obs.mining_node_count; ++i) {
        const auto& node = obs.mining_nodes[static_cast<size_t>(i)];
        if (node.col < 0 || node.col >= WIDTH || node.row < 0 || node.row >= MAX_ROWS) {
            continue;
        }
        remembered_node[static_cast<size_t>(node.row * WIDTH + node.col)] = 1;
    }

    for (int i = 0; i < obs.robot_count; ++i) {
        const RobotObservation& robot = obs.robots[static_cast<size_t>(i)];
        if (robot.owner == player || robot.col < 0 || robot.col >= WIDTH || robot.row < 0 || robot.row >= MAX_ROWS) {
            continue;
        }
        const int idx = robot.row * WIDTH + robot.col;
        for (auto& prob : enemy_prob) {
            prob[static_cast<size_t>(idx)] = 0.0F;
        }
        enemy_prob[static_cast<size_t>(robot.type)][static_cast<size_t>(idx)] = 1.0F;
    }

    turn = new_turn;
}

BoardState BeliefState::determinize(uint64_t seed) const {
    BoardState result{};
    result.reset();
    result.player = player;
    result.step = turn;
    result.south_bound = south_bound;
    result.north_bound = north_bound;
    result.rng_state = mix64(seed);

    result.wall_known = known_wall;
    result.walls = wall;
    result.crystal_energy = visible_crystal;
    result.mine_energy = remembered_mine_energy;
    result.mine_max = remembered_mine_max;
    result.mine_owner = remembered_mine_owner;
    result.mining_node = remembered_node;

    for (int r = south_bound; r <= std::min(MAX_ROWS - 1, north_bound + HEIGHT); ++r) {
        bool row_unknown = true;
        for (int c = 0; c < WIDTH; ++c) {
            if (known_wall[static_cast<size_t>(r * WIDTH + c)] != 0) {
                row_unknown = false;
                break;
            }
        }
        if (row_unknown) {
            generate_optimistic_row(result, r, seed);
        }
    }

    const int enemy_owner = 1 - player;
    for (int type = FACTORY; type <= MINER; ++type) {
        float total = 0.0F;
        for (float p : enemy_prob[static_cast<size_t>(type)]) {
            total += p;
        }
        if (total <= 0.05F) {
            continue;
        }
        float draw = (static_cast<float>(next_u32(result.rng_state) % 100000U) / 100000.0F) * total;
        int selected = -1;
        for (int idx = 0; idx < MAX_CELLS; ++idx) {
            draw -= enemy_prob[static_cast<size_t>(type)][static_cast<size_t>(idx)];
            if (draw <= 0.0F) {
                selected = idx;
                break;
            }
        }
        if (selected >= 0) {
            const int slot = result.robots.add_generated_robot(
                result.next_generated_uid++, static_cast<uint8_t>(type), static_cast<uint8_t>(enemy_owner),
                cell_col(selected), cell_row(selected), max_energy(static_cast<uint8_t>(type)) / 2);
            (void)slot;
        }
    }

    result.rebuild_active_bitboards();
    return result;
}

void CrawlerSim::reset() {
    state.reset();
}

void CrawlerSim::load_from_observation(const ObservationInput& obs, const BeliefState& belief) {
    const int prior_step = state.step;
    const int prior_scroll = state.scroll_counter;
    const uint32_t prior_uid = state.next_generated_uid;
    state.reset();
    state.player = obs.player;
    state.south_bound = obs.south_bound;
    state.north_bound = obs.north_bound;
    state.step = obs.step >= 0 ? obs.step : prior_step + 1;
    state.scroll_counter = prior_scroll > 0 ? prior_scroll : scroll_interval(state.step);
    state.next_generated_uid = prior_uid == 0 ? 1 : prior_uid;
    state.wall_known = belief.known_wall;
    state.walls = belief.wall;
    state.crystal_energy = belief.visible_crystal;
    state.mine_energy = belief.remembered_mine_energy;
    state.mine_max = belief.remembered_mine_max;
    state.mine_owner = belief.remembered_mine_owner;
    state.mining_node = belief.remembered_node;

    for (int r = obs.south_bound; r <= obs.north_bound; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            const int local = (r - obs.south_bound) * WIDTH + c;
            const int abs = state.abs_index(c, r);
            if (abs < 0) {
                continue;
            }
            if (obs.walls[static_cast<size_t>(local)] >= 0) {
                state.wall_known[static_cast<size_t>(abs)] = 1;
                state.walls[static_cast<size_t>(abs)] = static_cast<uint8_t>(obs.walls[static_cast<size_t>(local)]);
            }
        }
    }

    for (int i = 0; i < obs.robot_count; ++i) {
        const auto& r = obs.robots[static_cast<size_t>(i)];
        const int slot = state.robots.add_robot(r.uid.data(), static_cast<uint8_t>(r.type),
                                                static_cast<uint8_t>(r.owner), r.col, r.row, r.energy,
                                                r.move_cd, r.jump_cd, r.build_cd);
        (void)slot;
    }

    state.rebuild_active_bitboards();
}

void CrawlerSim::step(const PrimitiveActions& input_actions) {
    if (state.done) {
        return;
    }

    std::array<Action, MAX_ROBOTS> actions = input_actions.actions;
    std::array<uint8_t, MAX_ROBOTS> destroyed{};
    std::array<uint8_t, MAX_ROBOTS> stationary{};
    std::array<uint8_t, MAX_ROBOTS> moved{};
    std::array<uint8_t, MAX_ROBOTS> offboard{};
    std::array<int16_t, MAX_ROBOTS> target_abs{};
    std::array<int16_t, MAX_CELLS> stationary_first{};
    std::array<int16_t, MAX_CELLS> mover_first{};
    std::array<int16_t, MAX_ROBOTS> stationary_next{};
    std::array<int16_t, MAX_ROBOTS> mover_next{};
    std::array<uint8_t, MAX_CELLS> stationary_count{};
    std::array<uint8_t, MAX_CELLS> mover_count{};
    std::array<uint8_t, MAX_CELLS> combat_cell{};

    target_abs.fill(-1);
    stationary_first.fill(-1);
    mover_first.fill(-1);
    stationary_next.fill(-1);
    mover_next.fill(-1);

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        if (state.robots.move_cd[static_cast<size_t>(i)] > 0) {
            --state.robots.move_cd[static_cast<size_t>(i)];
        }
        if (state.robots.jump_cd[static_cast<size_t>(i)] > 0) {
            --state.robots.jump_cd[static_cast<size_t>(i)];
        }
        if (state.robots.build_cd[static_cast<size_t>(i)] > 0) {
            --state.robots.build_cd[static_cast<size_t>(i)];
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const uint8_t type = state.robots.type[static_cast<size_t>(i)];
        const Action a = actions[static_cast<size_t>(i)];
        bool valid = a == ACT_IDLE || a == ACT_NORTH || a == ACT_SOUTH || a == ACT_EAST || a == ACT_WEST ||
                     (type == FACTORY && (a == ACT_BUILD_SCOUT || a == ACT_BUILD_WORKER || a == ACT_BUILD_MINER ||
                                          a == ACT_JUMP_NORTH || a == ACT_JUMP_SOUTH || a == ACT_JUMP_EAST ||
                                          a == ACT_JUMP_WEST)) ||
                     (type == WORKER && a >= ACT_BUILD_NORTH && a <= ACT_REMOVE_WEST) ||
                     (type == MINER && a == ACT_TRANSFORM) ||
                     (a >= ACT_TRANSFER_NORTH && a <= ACT_TRANSFER_WEST);
        if (!valid) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        state.robots.energy[static_cast<size_t>(i)] =
            static_cast<int16_t>(std::max(0, state.robots.energy[static_cast<size_t>(i)] - ENERGY_PER_TURN));
        if (state.robots.energy[static_cast<size_t>(i)] == 0) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0 || actions[static_cast<size_t>(i)] != ACT_TRANSFORM) {
            continue;
        }
        const int idx = state.abs_index(state.robots.col[static_cast<size_t>(i)],
                                        state.robots.row[static_cast<size_t>(i)]);
        if (idx < 0 || state.mining_node[static_cast<size_t>(idx)] == 0 ||
            state.robots.energy[static_cast<size_t>(i)] < TRANSFORM_COST) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
            continue;
        }
        state.mine_energy[static_cast<size_t>(idx)] =
            static_cast<int16_t>(std::min(MINE_MAX_ENERGY,
                                          state.robots.energy[static_cast<size_t>(i)] - TRANSFORM_COST));
        state.mine_max[static_cast<size_t>(idx)] = MINE_MAX_ENERGY;
        state.mine_owner[static_cast<size_t>(idx)] = static_cast<int8_t>(state.robots.owner[static_cast<size_t>(i)]);
        state.mining_node[static_cast<size_t>(idx)] = 0;
        destroyed[static_cast<size_t>(i)] = 1;
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0 || destroyed[static_cast<size_t>(i)] != 0) {
            continue;
        }
        const Action a = actions[static_cast<size_t>(i)];
        const bool build = a >= ACT_BUILD_NORTH && a <= ACT_BUILD_WEST;
        const bool remove = a >= ACT_REMOVE_NORTH && a <= ACT_REMOVE_WEST;
        if (!build && !remove) {
            continue;
        }
        const int cost = build ? WALL_BUILD_COST : WALL_REMOVE_COST;
        if (state.robots.energy[static_cast<size_t>(i)] < cost) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
            continue;
        }
        state.robots.energy[static_cast<size_t>(i)] =
            static_cast<int16_t>(state.robots.energy[static_cast<size_t>(i)] - cost);
        const Direction d = action_direction(a);
        const int c = state.robots.col[static_cast<size_t>(i)];
        const int r = state.robots.row[static_cast<size_t>(i)];
        if (!is_fixed_wall(c, d)) {
            set_or_clear_wall(state, c, r, d, build);
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0 || destroyed[static_cast<size_t>(i)] != 0) {
            continue;
        }
        const Action a = actions[static_cast<size_t>(i)];
        if (a != ACT_BUILD_SCOUT && a != ACT_BUILD_WORKER && a != ACT_BUILD_MINER) {
            continue;
        }
        int cost = SCOUT_COST;
        uint8_t new_type = SCOUT;
        if (a == ACT_BUILD_WORKER) {
            cost = WORKER_COST;
            new_type = WORKER;
        } else if (a == ACT_BUILD_MINER) {
            cost = MINER_COST;
            new_type = MINER;
        }
        if (state.robots.energy[static_cast<size_t>(i)] < cost ||
            state.robots.build_cd[static_cast<size_t>(i)] > 0) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
            continue;
        }
        const int c = state.robots.col[static_cast<size_t>(i)];
        const int r = state.robots.row[static_cast<size_t>(i)];
        const int sr = r + 1;
        if (sr > state.north_bound || !state.can_move_through(c, r, DIR_NORTH)) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
            continue;
        }
        state.robots.energy[static_cast<size_t>(i)] =
            static_cast<int16_t>(state.robots.energy[static_cast<size_t>(i)] - cost);
        state.robots.build_cd[static_cast<size_t>(i)] = FACTORY_BUILD_COOLDOWN;
        const int slot = state.robots.add_generated_robot(state.next_generated_uid++, new_type,
                                                          state.robots.owner[static_cast<size_t>(i)],
                                                          c, sr, cost);
        if (slot >= 0) {
            actions[static_cast<size_t>(slot)] = ACT_IDLE;
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0 || destroyed[static_cast<size_t>(i)] != 0) {
            continue;
        }
        const Action a = actions[static_cast<size_t>(i)];
        if (a < ACT_TRANSFER_NORTH || a > ACT_TRANSFER_WEST) {
            continue;
        }
        const Direction d = action_direction(a);
        const int c = state.robots.col[static_cast<size_t>(i)];
        const int r = state.robots.row[static_cast<size_t>(i)];
        if (!state.can_move_through(c, r, d)) {
            continue;
        }
        const int tc = c + direction_dc(d);
        const int tr = r + direction_dr(d);
        int target = -1;
        for (int j = 0; j < state.robots.used; ++j) {
            if (i == j || state.robots.alive[static_cast<size_t>(j)] == 0 || destroyed[static_cast<size_t>(j)] != 0) {
                continue;
            }
            if (state.robots.owner[static_cast<size_t>(j)] == state.robots.owner[static_cast<size_t>(i)] &&
                state.robots.col[static_cast<size_t>(j)] == tc && state.robots.row[static_cast<size_t>(j)] == tr) {
                target = j;
                break;
            }
        }
        if (target < 0) {
            continue;
        }
        const int cap = max_energy(state.robots.type[static_cast<size_t>(target)]);
        const int space = std::max(0, cap - state.robots.energy[static_cast<size_t>(target)]);
        const int source_energy = state.robots.energy[static_cast<size_t>(i)];
        const int amount = std::min<int>(source_energy, space);
        state.robots.energy[static_cast<size_t>(target)] =
            static_cast<int16_t>(state.robots.energy[static_cast<size_t>(target)] + amount);
        state.robots.energy[static_cast<size_t>(i)] = 0;
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (destroyed[static_cast<size_t>(i)] != 0) {
            state.robots.remove(i);
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const Action a = actions[static_cast<size_t>(i)];
        if (a >= ACT_NORTH && a <= ACT_WEST) {
            if (state.robots.move_cd[static_cast<size_t>(i)] > 0) {
                stationary[static_cast<size_t>(i)] = 1;
                continue;
            }
            const Direction d = action_direction(a);
            const int c = state.robots.col[static_cast<size_t>(i)];
            const int r = state.robots.row[static_cast<size_t>(i)];
            const int tc = c + direction_dc(d);
            const int tr = r + direction_dr(d);
            if (tc < 0 || tc >= WIDTH) {
                stationary[static_cast<size_t>(i)] = 1;
                continue;
            }
            if (tr < state.south_bound || tr > state.north_bound) {
                const uint8_t source_wall = state.wall_at(c, r);
                if ((source_wall & direction_wall_bit(d)) != 0) {
                    stationary[static_cast<size_t>(i)] = 1;
                } else {
                    offboard[static_cast<size_t>(i)] = 1;
                }
                continue;
            }
            if (state.can_move_through(c, r, d)) {
                target_abs[static_cast<size_t>(i)] = static_cast<int16_t>(state.abs_index(tc, tr));
            } else {
                stationary[static_cast<size_t>(i)] = 1;
            }
        } else if (a >= ACT_JUMP_NORTH && a <= ACT_JUMP_WEST) {
            if (state.robots.move_cd[static_cast<size_t>(i)] > 0 ||
                state.robots.jump_cd[static_cast<size_t>(i)] > 0) {
                stationary[static_cast<size_t>(i)] = 1;
                continue;
            }
            const Direction d = action_direction(a);
            const int tc = state.robots.col[static_cast<size_t>(i)] + direction_dc(d) * 2;
            const int tr = state.robots.row[static_cast<size_t>(i)] + direction_dr(d) * 2;
            state.robots.jump_cd[static_cast<size_t>(i)] = FACTORY_JUMP_COOLDOWN;
            state.robots.move_cd[static_cast<size_t>(i)] =
                static_cast<int16_t>(move_period(state.robots.type[static_cast<size_t>(i)]));
            if (tc >= 0 && tc < WIDTH && tr >= state.south_bound && tr <= state.north_bound) {
                target_abs[static_cast<size_t>(i)] = static_cast<int16_t>(state.abs_index(tc, tr));
            } else {
                offboard[static_cast<size_t>(i)] = 1;
            }
        } else {
            stationary[static_cast<size_t>(i)] = 1;
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (offboard[static_cast<size_t>(i)] != 0) {
            state.robots.remove(i);
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        if (stationary[static_cast<size_t>(i)] != 0) {
            const int idx = state.abs_index(state.robots.col[static_cast<size_t>(i)],
                                            state.robots.row[static_cast<size_t>(i)]);
            if (idx >= 0) {
                stationary_next[static_cast<size_t>(i)] = stationary_first[static_cast<size_t>(idx)];
                stationary_first[static_cast<size_t>(idx)] = static_cast<int16_t>(i);
                ++stationary_count[static_cast<size_t>(idx)];
            }
        }
        if (target_abs[static_cast<size_t>(i)] >= 0) {
            const int idx = target_abs[static_cast<size_t>(i)];
            mover_next[static_cast<size_t>(i)] = mover_first[static_cast<size_t>(idx)];
            mover_first[static_cast<size_t>(idx)] = static_cast<int16_t>(i);
            ++mover_count[static_cast<size_t>(idx)];
        }
    }

    for (int r = state.south_bound; r <= state.north_bound; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            const int idx = state.abs_index(c, r);
            const int total = stationary_count[static_cast<size_t>(idx)] + mover_count[static_cast<size_t>(idx)];
            if (total <= 0) {
                continue;
            }
            if (total == 1) {
                const int mover = mover_first[static_cast<size_t>(idx)];
                if (mover >= 0) {
                    moved[static_cast<size_t>(mover)] = 1;
                }
                continue;
            }

            combat_cell[static_cast<size_t>(idx)] = 1;
            std::array<int16_t, MAX_ROBOTS> participants{};
            std::array<uint8_t, 4> type_count{};
            int count = 0;
            int factory_owner_mask = 0;

            for (int u = mover_first[static_cast<size_t>(idx)]; u >= 0; u = mover_next[static_cast<size_t>(u)]) {
                participants[static_cast<size_t>(count++)] = static_cast<int16_t>(u);
                const uint8_t t = state.robots.type[static_cast<size_t>(u)];
                ++type_count[static_cast<size_t>(t)];
                if (t == FACTORY) {
                    factory_owner_mask |= 1 << state.robots.owner[static_cast<size_t>(u)];
                }
            }
            for (int u = stationary_first[static_cast<size_t>(idx)]; u >= 0;
                 u = stationary_next[static_cast<size_t>(u)]) {
                participants[static_cast<size_t>(count++)] = static_cast<int16_t>(u);
                const uint8_t t = state.robots.type[static_cast<size_t>(u)];
                ++type_count[static_cast<size_t>(t)];
                if (t == FACTORY) {
                    factory_owner_mask |= 1 << state.robots.owner[static_cast<size_t>(u)];
                }
            }

            for (int pi = 0; pi < count; ++pi) {
                const int u = participants[static_cast<size_t>(pi)];
                const uint8_t t = state.robots.type[static_cast<size_t>(u)];
                bool dies = false;
                if (t == FACTORY) {
                    dies = type_count[static_cast<size_t>(FACTORY)] > 1 ||
                           (factory_owner_mask & 0b11) == 0b11;
                } else if (type_count[static_cast<size_t>(t)] > 1) {
                    dies = true;
                } else {
                    for (int stronger = static_cast<int>(t) + 1; stronger <= MINER; ++stronger) {
                        if (type_count[static_cast<size_t>(stronger)] > 0) {
                            dies = true;
                        }
                    }
                    if (type_count[static_cast<size_t>(FACTORY)] > 0) {
                        dies = true;
                    }
                }
                if (dies) {
                    destroyed[static_cast<size_t>(u)] = 1;
                }
            }

            for (int u = mover_first[static_cast<size_t>(idx)]; u >= 0; u = mover_next[static_cast<size_t>(u)]) {
                if (destroyed[static_cast<size_t>(u)] == 0) {
                    moved[static_cast<size_t>(u)] = 1;
                }
            }
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (moved[static_cast<size_t>(i)] == 0 || destroyed[static_cast<size_t>(i)] != 0 ||
            state.robots.alive[static_cast<size_t>(i)] == 0 || target_abs[static_cast<size_t>(i)] < 0) {
            continue;
        }
        const int idx = target_abs[static_cast<size_t>(i)];
        state.robots.col[static_cast<size_t>(i)] = static_cast<int16_t>(cell_col(idx));
        state.robots.row[static_cast<size_t>(i)] = static_cast<int16_t>(cell_row(idx));
        state.robots.move_cd[static_cast<size_t>(i)] =
            static_cast<int16_t>(move_period(state.robots.type[static_cast<size_t>(i)]));
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (destroyed[static_cast<size_t>(i)] != 0) {
            state.robots.remove(i);
        }
    }

    for (int idx = 0; idx < MAX_CELLS; ++idx) {
        if (combat_cell[static_cast<size_t>(idx)] == 0 || state.crystal_energy[static_cast<size_t>(idx)] <= 0) {
            continue;
        }
        bool survivor_here = false;
        for (int i = 0; i < state.robots.used; ++i) {
            if (state.robots.alive[static_cast<size_t>(i)] != 0 &&
                state.abs_index(state.robots.col[static_cast<size_t>(i)],
                                state.robots.row[static_cast<size_t>(i)]) == idx) {
                survivor_here = true;
                break;
            }
        }
        if (!survivor_here) {
            state.crystal_energy[static_cast<size_t>(idx)] = 0;
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const int idx = state.abs_index(state.robots.col[static_cast<size_t>(i)],
                                        state.robots.row[static_cast<size_t>(i)]);
        if (idx >= 0 && state.crystal_energy[static_cast<size_t>(idx)] > 0) {
            const int cap = max_energy(state.robots.type[static_cast<size_t>(i)]);
            const int gain = std::min<int>(state.crystal_energy[static_cast<size_t>(idx)],
                                           std::max(0, cap - state.robots.energy[static_cast<size_t>(i)]));
            state.robots.energy[static_cast<size_t>(i)] =
                static_cast<int16_t>(state.robots.energy[static_cast<size_t>(i)] + gain);
            state.crystal_energy[static_cast<size_t>(idx)] = 0;
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const int idx = state.abs_index(state.robots.col[static_cast<size_t>(i)],
                                        state.robots.row[static_cast<size_t>(i)]);
        if (idx < 0 || state.mine_max[static_cast<size_t>(idx)] <= 0 ||
            state.mine_owner[static_cast<size_t>(idx)] != state.robots.owner[static_cast<size_t>(i)]) {
            continue;
        }
        const int cap = max_energy(state.robots.type[static_cast<size_t>(i)]);
        const int transfer = std::min<int>(state.mine_energy[static_cast<size_t>(idx)],
                                           std::max(0, cap - state.robots.energy[static_cast<size_t>(i)]));
        state.robots.energy[static_cast<size_t>(i)] =
            static_cast<int16_t>(state.robots.energy[static_cast<size_t>(i)] + transfer);
        state.mine_energy[static_cast<size_t>(idx)] =
            static_cast<int16_t>(state.mine_energy[static_cast<size_t>(idx)] - transfer);
    }

    for (int idx = 0; idx < MAX_CELLS; ++idx) {
        if (state.mine_max[static_cast<size_t>(idx)] > 0) {
            state.mine_energy[static_cast<size_t>(idx)] =
                static_cast<int16_t>(std::min<int>(state.mine_energy[static_cast<size_t>(idx)] + MINE_RATE,
                                                   state.mine_max[static_cast<size_t>(idx)]));
        }
    }

    --state.scroll_counter;
    if (state.scroll_counter <= 0) {
        ++state.south_bound;
        ++state.north_bound;
        if (state.north_bound < MAX_ROWS) {
            generate_optimistic_row(state, state.north_bound, state.rng_state ^ static_cast<uint64_t>(state.step));
        }
        state.scroll_counter = scroll_interval(state.step);
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] != 0 &&
            state.robots.row[static_cast<size_t>(i)] < state.south_bound) {
            state.robots.remove(i);
        }
    }

    for (int r = 0; r < state.south_bound && r < MAX_ROWS; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            const int idx = state.abs_index(c, r);
            state.crystal_energy[static_cast<size_t>(idx)] = 0;
            state.mine_energy[static_cast<size_t>(idx)] = 0;
            state.mine_max[static_cast<size_t>(idx)] = 0;
            state.mine_owner[static_cast<size_t>(idx)] = -1;
            state.mining_node[static_cast<size_t>(idx)] = 0;
        }
    }

    compute_rewards(state);
    ++state.step;
    state.rebuild_active_bitboards();
}

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
                    return step_toward_cell(state, robot_index, cell_col(idx), cell_row(idx));
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
                    return step_toward_cell(state, robot_index, cell_col(idx), cell_row(idx));
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

Engine::Engine() {
    belief.reset();
    sim.reset();
}

Engine::Engine(int player) : Engine() {
    belief.player = player;
    sim.state.player = player;
}

void Engine::update_observation(const ObservationInput& obs) {
    belief.update_from_observation(obs);
    sim.load_from_observation(obs, belief);
}

void Engine::step_actions(const PrimitiveActions& actions) {
    sim.step(actions);
}

BoardState Engine::determinize(uint64_t seed) const {
    BoardState sampled = belief.determinize(seed);
    for (int i = 0; i < sim.state.robots.used; ++i) {
        if (sim.state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const int slot = sampled.robots.add_robot(sim.state.robots.uid[static_cast<size_t>(i)].data(),
                                                  sim.state.robots.type[static_cast<size_t>(i)],
                                                  sim.state.robots.owner[static_cast<size_t>(i)],
                                                  sim.state.robots.col[static_cast<size_t>(i)],
                                                  sim.state.robots.row[static_cast<size_t>(i)],
                                                  sim.state.robots.energy[static_cast<size_t>(i)],
                                                  sim.state.robots.move_cd[static_cast<size_t>(i)],
                                                  sim.state.robots.jump_cd[static_cast<size_t>(i)],
                                                  sim.state.robots.build_cd[static_cast<size_t>(i)]);
        (void)slot;
    }
    sampled.rebuild_active_bitboards();
    return sampled;
}

ActionResult Engine::choose_actions(int time_budget_ms, uint64_t seed) {
    (void)seed;
    const int rollout_budget = std::max(1, std::min(MAX_TREE_NODES, time_budget_ms * 2));
    int macro_touch_count = 0;
    for (int i = 0; i < sim.state.robots.used && macro_touch_count < rollout_budget; ++i) {
        if (sim.state.robots.alive[static_cast<size_t>(i)] == 0 ||
            sim.state.robots.owner[static_cast<size_t>(i)] != sim.state.player) {
            continue;
        }
        const MacroList macros = sim.generate_macros_for(i);
        macro_touch_count += macros.count;
    }

    ActionResult result{};
    result.clear();
    for (int i = 0; i < sim.state.robots.used; ++i) {
        if (sim.state.robots.alive[static_cast<size_t>(i)] == 0 ||
            sim.state.robots.owner[static_cast<size_t>(i)] != sim.state.player) {
            continue;
        }
        const Action action = sim.heuristic_action_for(i);
        result.add(sim.state.robots.uid[static_cast<size_t>(i)].data(), action);
    }
    return result;
}

}  // namespace crawler
