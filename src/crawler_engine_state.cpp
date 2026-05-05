#include "crawler_engine_internal.hpp"

// Fixed-buffer state containers and primitive action helpers. These routines are
// deliberately small and allocation-free because they are used by every rollout.

#include <algorithm>
#include <bit>
#include <cstdio>
#include <limits>

namespace crawler {

// ---- Active-window bitboards ------------------------------------------------

void BitBoard::clear() {
    words.fill(0ULL);
}

void BitBoard::set(int active_index) {
    if (active_index < 0 || active_index >= ACTIVE_CELLS) {
        return;
    }
    words[static_cast<size_t>(detail::active_word(active_index))] |= detail::active_mask(active_index);
}

void BitBoard::reset(int active_index) {
    if (active_index < 0 || active_index >= ACTIVE_CELLS) {
        return;
    }
    words[static_cast<size_t>(detail::active_word(active_index))] &= ~detail::active_mask(active_index);
}

bool BitBoard::test(int active_index) const {
    if (active_index < 0 || active_index >= ACTIVE_CELLS) {
        return false;
    }
    return (words[static_cast<size_t>(detail::active_word(active_index))] &
            detail::active_mask(active_index)) != 0;
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

// ---- Action names, parsing, and geometry -----------------------------------

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

// ---- Robot store ------------------------------------------------------------

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
        if (alive[static_cast<size_t>(i)] != 0 && detail::uid_equal(uid[static_cast<size_t>(i)], value)) {
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

    detail::copy_uid(uid[static_cast<size_t>(slot)], uid_value);
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

// ---- Board state and derived tactical masks --------------------------------

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

// ---- Action buffers ---------------------------------------------------------

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
    detail::copy_uid(uid[static_cast<size_t>(count)], uid_value);
    action[static_cast<size_t>(count)] = primitive;
    ++count;
}

}  // namespace crawler
