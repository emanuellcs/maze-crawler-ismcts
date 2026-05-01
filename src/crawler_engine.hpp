#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace crawler {

constexpr int WIDTH = 20;
constexpr int HEIGHT = 20;
constexpr int ACTIVE_CELLS = WIDTH * HEIGHT;
constexpr int MAX_ROWS = 512;
constexpr int MAX_CELLS = WIDTH * MAX_ROWS;
constexpr int ACTIVE_WORDS = (ACTIVE_CELLS + 63) / 64;
constexpr int MAX_ROBOTS = 512;
constexpr int UID_LEN = 24;
constexpr int MAX_OBS_CRYSTALS = ACTIVE_CELLS;
constexpr int MAX_OBS_MINES = ACTIVE_CELLS;
constexpr int MAX_OBS_NODES = ACTIVE_CELLS;
constexpr int MAX_MACROS = 16;
constexpr int MAX_TREE_NODES = 4096;

constexpr int EPISODE_STEPS = 501;
constexpr int FACTORY_ENERGY = 1000;
constexpr int SCOUT_COST = 50;
constexpr int WORKER_COST = 200;
constexpr int MINER_COST = 300;
constexpr int SCOUT_MAX_ENERGY = 100;
constexpr int WORKER_MAX_ENERGY = 300;
constexpr int MINER_MAX_ENERGY = 500;
constexpr int WALL_BUILD_COST = 100;
constexpr int WALL_REMOVE_COST = 100;
constexpr int TRANSFORM_COST = 100;
constexpr int MINE_MAX_ENERGY = 1000;
constexpr int MINE_RATE = 50;
constexpr int ENERGY_PER_TURN = 1;
constexpr int FACTORY_BUILD_COOLDOWN = 10;
constexpr int FACTORY_JUMP_COOLDOWN = 20;
constexpr int FACTORY_MOVE_PERIOD = 2;
constexpr int SCOUT_MOVE_PERIOD = 1;
constexpr int WORKER_MOVE_PERIOD = 2;
constexpr int MINER_MOVE_PERIOD = 2;
constexpr int VISION_FACTORY = 4;
constexpr int VISION_SCOUT = 5;
constexpr int VISION_WORKER = 3;
constexpr int VISION_MINER = 3;
constexpr int SCROLL_START_INTERVAL = 4;
constexpr int SCROLL_END_INTERVAL = 1;
constexpr int SCROLL_RAMP_STEPS = 400;

enum WallBits : uint8_t {
    WALL_N = 1,
    WALL_E = 2,
    WALL_S = 4,
    WALL_W = 8,
};

enum RobotType : uint8_t {
    FACTORY = 0,
    SCOUT = 1,
    WORKER = 2,
    MINER = 3,
};

enum Direction : uint8_t {
    DIR_NONE = 0,
    DIR_NORTH = 1,
    DIR_SOUTH = 2,
    DIR_EAST = 3,
    DIR_WEST = 4,
};

enum Action : uint8_t {
    ACT_IDLE = 0,
    ACT_NORTH,
    ACT_SOUTH,
    ACT_EAST,
    ACT_WEST,
    ACT_BUILD_SCOUT,
    ACT_BUILD_WORKER,
    ACT_BUILD_MINER,
    ACT_JUMP_NORTH,
    ACT_JUMP_SOUTH,
    ACT_JUMP_EAST,
    ACT_JUMP_WEST,
    ACT_BUILD_NORTH,
    ACT_BUILD_SOUTH,
    ACT_BUILD_EAST,
    ACT_BUILD_WEST,
    ACT_REMOVE_NORTH,
    ACT_REMOVE_SOUTH,
    ACT_REMOVE_EAST,
    ACT_REMOVE_WEST,
    ACT_TRANSFORM,
    ACT_TRANSFER_NORTH,
    ACT_TRANSFER_SOUTH,
    ACT_TRANSFER_EAST,
    ACT_TRANSFER_WEST,
};

enum MacroAction : uint8_t {
    MACRO_IDLE = 0,
    MACRO_FACTORY_BUILD_WORKER,
    MACRO_FACTORY_BUILD_SCOUT,
    MACRO_FACTORY_SAFE_ADVANCE,
    MACRO_FACTORY_JUMP_NORTH,
    MACRO_WORKER_OPEN_NORTH,
    MACRO_WORKER_ADVANCE,
    MACRO_SCOUT_EXPLORE_NORTH,
    MACRO_SCOUT_RETURN_ENERGY,
    MACRO_MINER_SEEK_NODE,
    MACRO_MINER_TRANSFORM,
};

struct BitBoard {
    std::array<uint64_t, ACTIVE_WORDS> words{};

    void clear();
    void set(int active_index);
    void reset(int active_index);
    [[nodiscard]] bool test(int active_index) const;
    [[nodiscard]] bool any() const;
};

int pop_lsb(uint64_t& bits);
const char* action_name(Action action);
Action parse_action(std::string_view value);
Direction action_direction(Action action);
uint8_t direction_wall_bit(Direction direction);
Direction opposite_direction(Direction direction);
int direction_dc(Direction direction);
int direction_dr(Direction direction);
int move_period(uint8_t type);
int max_energy(uint8_t type);
int vision_range(uint8_t type);
bool is_fixed_wall(int col, Direction direction);

struct RobotStore {
    std::array<std::array<char, UID_LEN>, MAX_ROBOTS> uid{};
    std::array<uint8_t, MAX_ROBOTS> alive{};
    std::array<uint8_t, MAX_ROBOTS> type{};
    std::array<uint8_t, MAX_ROBOTS> owner{};
    std::array<int16_t, MAX_ROBOTS> col{};
    std::array<int16_t, MAX_ROBOTS> row{};
    std::array<int16_t, MAX_ROBOTS> energy{};
    std::array<int16_t, MAX_ROBOTS> move_cd{};
    std::array<int16_t, MAX_ROBOTS> jump_cd{};
    std::array<int16_t, MAX_ROBOTS> build_cd{};
    int used = 0;

    void clear();
    [[nodiscard]] int find_uid(std::string_view value) const;
    [[nodiscard]] int add_robot(std::string_view uid_value, uint8_t robot_type, uint8_t robot_owner,
                                int robot_col, int robot_row, int robot_energy,
                                int move_cooldown, int jump_cooldown, int build_cooldown);
    [[nodiscard]] int add_generated_robot(uint32_t serial, uint8_t robot_type, uint8_t robot_owner,
                                          int robot_col, int robot_row, int robot_energy);
    void remove(int index);
};

struct BoardState {
    int player = 0;
    int step = 0;
    int south_bound = 0;
    int north_bound = HEIGHT - 1;
    int scroll_counter = SCROLL_START_INTERVAL;
    uint32_t next_generated_uid = 1;
    uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
    bool done = false;
    int winner = -1;
    float reward0 = 0.0F;
    float reward1 = 0.0F;

    std::array<uint8_t, MAX_CELLS> walls{};
    std::array<uint8_t, MAX_CELLS> wall_known{};
    std::array<int16_t, MAX_CELLS> crystal_energy{};
    std::array<int16_t, MAX_CELLS> mine_energy{};
    std::array<int16_t, MAX_CELLS> mine_max{};
    std::array<int8_t, MAX_CELLS> mine_owner{};
    std::array<uint8_t, MAX_CELLS> mining_node{};

    RobotStore robots{};
    BitBoard own_occupancy{};
    BitBoard enemy_occupancy{};
    BitBoard all_occupancy{};
    BitBoard visibility{};
    BitBoard crystals_active{};
    BitBoard mines_active{};
    BitBoard nodes_active{};

    void reset();
    [[nodiscard]] int abs_index(int c, int r) const;
    [[nodiscard]] int active_index(int c, int r) const;
    [[nodiscard]] bool in_active(int c, int r) const;
    [[nodiscard]] uint8_t wall_at(int c, int r) const;
    [[nodiscard]] bool can_move_through(int c, int r, Direction direction) const;
    void rebuild_active_bitboards();
};

struct RobotObservation {
    std::array<char, UID_LEN> uid{};
    int type = 0;
    int col = 0;
    int row = 0;
    int energy = 0;
    int owner = 0;
    int move_cd = 0;
    int jump_cd = 0;
    int build_cd = 0;
};

struct CellEnergyObservation {
    int col = 0;
    int row = 0;
    int energy = 0;
};

struct MineObservation {
    int col = 0;
    int row = 0;
    int energy = 0;
    int max_energy = MINE_MAX_ENERGY;
    int owner = -1;
};

struct CellObservation {
    int col = 0;
    int row = 0;
};

struct ObservationInput {
    int player = 0;
    int south_bound = 0;
    int north_bound = HEIGHT - 1;
    int step = -1;
    std::array<int16_t, ACTIVE_CELLS> walls{};
    int robot_count = 0;
    int crystal_count = 0;
    int mine_count = 0;
    int mining_node_count = 0;
    std::array<RobotObservation, MAX_ROBOTS> robots{};
    std::array<CellEnergyObservation, MAX_OBS_CRYSTALS> crystals{};
    std::array<MineObservation, MAX_OBS_MINES> mines{};
    std::array<CellObservation, MAX_OBS_NODES> mining_nodes{};
};

struct PrimitiveActions {
    std::array<Action, MAX_ROBOTS> actions{};

    void clear();
};

struct ActionResult {
    int count = 0;
    std::array<std::array<char, UID_LEN>, MAX_ROBOTS> uid{};
    std::array<Action, MAX_ROBOTS> action{};

    void clear();
    void add(std::string_view uid_value, Action primitive);
};

struct MacroList {
    int count = 0;
    std::array<MacroAction, MAX_MACROS> macros{};
};

struct BeliefState {
    int player = 0;
    int turn = 0;
    int south_bound = 0;
    int north_bound = HEIGHT - 1;
    std::array<uint8_t, MAX_CELLS> known_wall{};
    std::array<uint8_t, MAX_CELLS> wall{};
    std::array<int16_t, MAX_CELLS> visible_crystal{};
    std::array<int16_t, MAX_CELLS> remembered_mine_energy{};
    std::array<int16_t, MAX_CELLS> remembered_mine_max{};
    std::array<int8_t, MAX_CELLS> remembered_mine_owner{};
    std::array<uint8_t, MAX_CELLS> remembered_node{};
    std::array<uint8_t, MAX_CELLS> currently_visible{};
    std::array<std::array<float, MAX_CELLS>, 4> enemy_prob{};

    void reset();
    void update_from_observation(const ObservationInput& obs);
    [[nodiscard]] BoardState determinize(uint64_t seed) const;
};

class CrawlerSim {
public:
    BoardState state{};

    void reset();
    void load_from_observation(const ObservationInput& obs, const BeliefState& belief);
    void step(const PrimitiveActions& actions);
    [[nodiscard]] Action heuristic_action_for(int robot_index) const;
    [[nodiscard]] MacroList generate_macros_for(int robot_index) const;
};

class Engine {
public:
    BeliefState belief{};
    CrawlerSim sim{};

    Engine();
    explicit Engine(int player);

    void update_observation(const ObservationInput& obs);
    void step_actions(const PrimitiveActions& actions);
    [[nodiscard]] BoardState determinize(uint64_t seed) const;
    [[nodiscard]] ActionResult choose_actions(int time_budget_ms, uint64_t seed);
};

}  // namespace crawler
