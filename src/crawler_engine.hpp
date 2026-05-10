#pragma once

/**
 * @file crawler_engine.hpp
 * @brief Public fixed-buffer API for the Maze Crawler ISMCTS engine.
 *
 * This header is the architectural contract between the Python bridge, the
 * player-centric Belief State, the deterministic simulator, the rollout policy,
 * and the fixed-arena ISMCTS search.  All state-bearing types use bounded
 * `std::array` storage so the simulation and search hot paths can copy,
 * determinize, and roll out board states without heap allocation.
 *
 * The engine uses two coordinate models deliberately:
 * - Absolute board arrays use `row * WIDTH + col` across `MAX_ROWS` so fog-of-war
 *   memory survives scrolling.
 * - Active-window Bitboards use `(row - south_bound) * WIDTH + col` for the
 *   current 20x20 tactical window, enabling O(1) occupancy and feature tests
 *   during Rollouts and macro-policy generation.
 */

#include <array>
#include <cstdint>
#include <string_view>

namespace crawler {

/** @brief Fixed game limits and default Kaggle configuration values. */
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
constexpr int MAX_MCTS_PLAN_ROBOTS = 64;
constexpr int MAX_MCTS_CANDIDATES = 64;
constexpr int MCTS_TREE_DEPTH = 24;
constexpr int MCTS_ROLLOUT_DEPTH = 48;

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

/**
 * @brief Wall bitfield layout used by observations and internal board storage.
 *
 * Each cell owns four directional bits.  Reciprocal wall maintenance writes the
 * opposite bit into the neighboring cell so movement checks can remain a single
 * bit test against the source cell in the simulator hot path.
 */
enum WallBits : uint8_t {
    WALL_N = 1,
    WALL_E = 2,
    WALL_S = 4,
    WALL_W = 8,
};

/**
 * @brief Robot type values matching the Kaggle observation encoding exactly.
 *
 * Keeping these numeric values aligned with the Python environment avoids any
 * translation table in pybind and lets rule code compare raw observation data.
 */
enum RobotType : uint8_t {
    FACTORY = 0,
    SCOUT = 1,
    WORKER = 2,
    MINER = 3,
};

/**
 * @brief Canonical direction enum for movement, wall edits, jumps, and transfers.
 */
enum Direction : uint8_t {
    DIR_NONE = 0,
    DIR_NORTH = 1,
    DIR_SOUTH = 2,
    DIR_EAST = 3,
    DIR_WEST = 4,
};

/**
 * @brief Primitive action enum mapping one-to-one with Kaggle action strings.
 *
 * Primitive actions are the simulator-facing command set.  ISMCTS does not
 * branch directly over their full Cartesian product; it searches over bounded
 * MacroAction joint plans and translates the selected plan back to primitives.
 */
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

/**
 * @brief Search-level intent labels used by the fixed-arena ISMCTS tree.
 *
 * MacroActions compress the simultaneous-move branching factor.  Expansion
 * starts from one deterministic all-robot baseline plan and adds one-robot
 * deviations such as opening a wall, returning Scout energy, or jumping the
 * Factory.  This keeps root branching approximately linear in robot count
 * rather than multiplicative over primitive actions.
 */
enum MacroAction : uint8_t {
    MACRO_IDLE = 0,
    MACRO_FACTORY_SUPPORT_WORKER,
    MACRO_FACTORY_SAFE_ADVANCE,
    MACRO_FACTORY_BUILD_WORKER,
    MACRO_FACTORY_BUILD_SCOUT,
    MACRO_FACTORY_BUILD_MINER,
    MACRO_FACTORY_JUMP_OBSTACLE,
    MACRO_WORKER_OPEN_NORTH_WALL,
    MACRO_WORKER_ESCORT_FACTORY,
    MACRO_WORKER_ADVANCE,
    MACRO_SCOUT_HUNT_CRYSTAL,
    MACRO_SCOUT_EXPLORE_NORTH,
    MACRO_SCOUT_RETURN_ENERGY,
    MACRO_MINER_SEEK_NODE,
    MACRO_MINER_TRANSFORM,
};

/**
 * @brief 20x20 active-window Bitboard backed by 64-bit machine words.
 *
 * Bitboards encode tactical facts for the currently visible/active window:
 * occupancy, visibility, crystals, mines, and mining nodes.  Bitwise shifts and
 * masks replace array lookups in tight policy loops, giving O(1) collision and
 * goal membership tests during Rollouts.
 */
struct BitBoard {
    std::array<uint64_t, ACTIVE_WORDS> words{};

    /**
     * @brief Clear every word in the active-window mask.
     */
    void clear();

    /**
     * @brief Set a cell bit if the active index is inside the 20x20 window.
     * @param active_index Active-window index in `[0, ACTIVE_CELLS)`.
     */
    void set(int active_index);

    /**
     * @brief Clear a cell bit if the active index is inside the 20x20 window.
     * @param active_index Active-window index in `[0, ACTIVE_CELLS)`.
     */
    void reset(int active_index);

    /**
     * @brief Test whether a cell bit is set.
     * @param active_index Active-window index in `[0, ACTIVE_CELLS)`.
     * @return True when the active cell belongs to this Bitboard.
     */
    [[nodiscard]] bool test(int active_index) const;

    /**
     * @brief Report whether any active-window cell is set.
     * @return True when at least one 64-bit word is non-zero.
     */
    [[nodiscard]] bool any() const;
};

/**
 * @brief Pop and return the index of the least significant set bit.
 * @param bits Word to mutate by clearing its least significant set bit.
 * @return Bit offset in `[0, 63]`.
 * @note Callers must only invoke this when `bits != 0`.
 */
int pop_lsb(uint64_t& bits);

/**
 * @brief Convert a primitive action enum to the Kaggle action string.
 * @param action Primitive action enum value.
 * @return Stable string literal accepted by the Python environment.
 */
const char* action_name(Action action);

/**
 * @brief Convert a macro intent enum to its tuning/debug string.
 * @param macro Search-level macro action.
 * @return Stable string literal used by pybind hyperparameter dictionaries.
 */
const char* macro_action_name(MacroAction macro);

/**
 * @brief Parse a Kaggle action string into the primitive enum.
 * @param value Python-side action string.
 * @return Matching Action, or `ACT_IDLE` for unknown values.
 */
Action parse_action(std::string_view value);

/**
 * @brief Extract the direction encoded by a primitive action.
 * @param action Primitive movement, jump, wall, or transfer action.
 * @return Direction associated with the action, or `DIR_NONE`.
 */
Direction action_direction(Action action);

/**
 * @brief Convert a direction into the corresponding wall bit.
 * @param direction Movement direction.
 * @return WallBits mask, or zero for `DIR_NONE`.
 */
uint8_t direction_wall_bit(Direction direction);

/**
 * @brief Return the opposite cardinal direction.
 * @param direction Direction to invert.
 * @return Opposite direction, or `DIR_NONE`.
 */
Direction opposite_direction(Direction direction);

/** @brief Horizontal column delta for one step in the given direction. */
int direction_dc(Direction direction);

/** @brief Vertical row delta for one step in the given direction. */
int direction_dr(Direction direction);

/** @brief Rule-defined movement cooldown period for a robot type. */
int move_period(uint8_t type);

/** @brief Rule-defined energy cap for a robot type. */
int max_energy(uint8_t type);

/** @brief Manhattan vision radius for a robot type. */
int vision_range(uint8_t type);

/**
 * @brief Test whether a worker wall edit targets immutable perimeter/mirror walls.
 * @param col Source cell column.
 * @param direction Wall direction from the source cell.
 * @return True for east/west perimeter walls or the central mirror-axis walls.
 */
bool is_fixed_wall(int col, Direction direction);

/**
 * @brief Structure-of-Arrays robot storage with recycled dead slots.
 *
 * The simulator scans robots frequently for movement, combat, rewards, and
 * action serialization.  Storing each attribute in a contiguous array improves
 * cache predictability and makes BoardState copies cheap for Determinization.
 * Simulator-local indices are valid only within one concrete state; stable UIDs
 * are used when MCTS replays plans across sampled hidden states.
 */
struct RobotStore {
    std::array<std::array<char, UID_LEN>, MAX_ROBOTS> uid{};
    std::array<uint8_t, MAX_ROBOTS> alive{};
    std::array<uint8_t, MAX_ROBOTS> type{};
    std::array<uint8_t, MAX_ROBOTS> owner{};
    std::array<int16_t, MAX_ROBOTS> col{};
    std::array<int16_t, MAX_ROBOTS> row{};
    std::array<int32_t, MAX_ROBOTS> energy{};
    std::array<int16_t, MAX_ROBOTS> move_cd{};
    std::array<int16_t, MAX_ROBOTS> jump_cd{};
    std::array<int16_t, MAX_ROBOTS> build_cd{};
    int used = 0;

    /**
     * @brief Reset all robot slots and UID buffers.
     */
    void clear();

    /**
     * @brief Locate a live robot by its external UID.
     * @param value UID string from Kaggle or an MCTS plan edge.
     * @return Simulator-local slot index, or `-1` if absent.
     */
    [[nodiscard]] int find_uid(std::string_view value) const;

    /**
     * @brief Insert or recycle a robot slot with explicit observation data.
     * @param uid_value Stable external UID.
     * @param robot_type RobotType value.
     * @param robot_owner Owner player index.
     * @param robot_col Absolute column.
     * @param robot_row Absolute row.
     * @param robot_energy Current energy.
     * @param move_cooldown Remaining move cooldown.
     * @param jump_cooldown Remaining jump cooldown.
     * @param build_cooldown Remaining build cooldown.
     * @return Slot index, or `-1` if `MAX_ROBOTS` is exhausted.
     */
    [[nodiscard]] int add_robot(std::string_view uid_value, uint8_t robot_type, uint8_t robot_owner,
                                int robot_col, int robot_row, int robot_energy,
                                int move_cooldown, int jump_cooldown, int build_cooldown);

    /**
     * @brief Insert a simulator-generated robot with a deterministic synthetic UID.
     * @param serial Monotonic serial owned by the BoardState.
     * @param robot_type RobotType value for the spawned unit.
     * @param robot_owner Owner player index.
     * @param robot_col Absolute column.
     * @param robot_row Absolute row.
     * @param robot_energy Spawn energy.
     * @return Slot index, or `-1` if `MAX_ROBOTS` is exhausted.
     */
    [[nodiscard]] int add_generated_robot(uint32_t serial, uint8_t robot_type, uint8_t robot_owner,
                                          int robot_col, int robot_row, int robot_energy);

    /**
     * @brief Mark a robot slot dead so future insertions can recycle it.
     * @param index Simulator-local robot slot.
     */
    void remove(int index);
};

/**
 * @brief Full deterministic simulator state for one concrete world sample.
 *
 * BoardState is the object copied into every ISMCTS iteration.  Absolute arrays
 * retain map memory over scrolling rows, while derived Bitboards summarize the
 * current active window for O(1) tactical checks.  The type remains trivially
 * stack/arena friendly: no vectors, maps, or dynamic ownership appear in the
 * simulation hot path.
 */
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

    /**
     * @brief Reset all scalar state, map arrays, robot storage, and Bitboards.
     */
    void reset();

    /**
     * @brief Convert absolute coordinates into an absolute cell index.
     * @param c Column in `[0, WIDTH)`.
     * @param r Absolute row in `[0, MAX_ROWS)`.
     * @return `row * WIDTH + col`, or `-1` when out of range.
     */
    [[nodiscard]] int abs_index(int c, int r) const;

    /**
     * @brief Convert absolute coordinates into the active-window index.
     * @param c Column in `[0, WIDTH)`.
     * @param r Absolute row inside `[south_bound, north_bound]`.
     * @return Active-window index, or `-1` when outside the tactical window.
     */
    [[nodiscard]] int active_index(int c, int r) const;

    /**
     * @brief Test whether a coordinate is inside the current active window.
     * @param c Column to test.
     * @param r Absolute row to test.
     * @return True when the coordinate can be represented by active Bitboards.
     */
    [[nodiscard]] bool in_active(int c, int r) const;

    /**
     * @brief Read the wall bitfield at a coordinate.
     * @param c Column.
     * @param r Absolute row.
     * @return Stored wall bits, or all walls set when out of bounds.
     */
    [[nodiscard]] uint8_t wall_at(int c, int r) const;

    /**
     * @brief Test one-step passability from a source cell.
     * @param c Source column.
     * @param r Source absolute row.
     * @param direction Direction of travel.
     * @return True when no source wall blocks the step and the destination is within absolute bounds.
     */
    [[nodiscard]] bool can_move_through(int c, int r, Direction direction) const;

    /**
     * @brief Rebuild occupancy, visibility, and resource Bitboards from arrays.
     * @note Called after observation loads, Determinizations, and simulator steps.
     */
    void rebuild_active_bitboards();
};

/** @brief Decoded robot record before merge into belief or simulation state. */
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

/** @brief Sparse visible crystal record keyed by absolute cell coordinates. */
struct CellEnergyObservation {
    int col = 0;
    int row = 0;
    int energy = 0;
};

/** @brief Sparse mine record; mines are remembered after discovery. */
struct MineObservation {
    int col = 0;
    int row = 0;
    int energy = 0;
    int max_energy = MINE_MAX_ENERGY;
    int owner = -1;
};

/** @brief Sparse visible mining-node record. */
struct CellObservation {
    int col = 0;
    int row = 0;
};

/**
 * @brief Fixed-buffer decoded Python observation.
 *
 * pybind converts Kaggle dictionaries into this POD-like structure before the
 * engine updates belief.  `walls` is a flat active-window array of length
 * `ACTIVE_CELLS`; values are wall bitfields or `-1` for unknown cells.
 */
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

/**
 * @brief Primitive action buffer addressed by simulator-local robot index.
 */
struct PrimitiveActions {
    std::array<Action, MAX_ROBOTS> actions{};

    /** @brief Reset every robot action to `ACT_IDLE`. */
    void clear();
};

/**
 * @brief Fixed-buffer Kaggle action output before conversion to a Python dict.
 */
struct ActionResult {
    int count = 0;
    std::array<std::array<char, UID_LEN>, MAX_ROBOTS> uid{};
    std::array<Action, MAX_ROBOTS> action{};

    /** @brief Clear UID/action output slots. */
    void clear();

    /**
     * @brief Append one UID/action pair if capacity remains.
     * @param uid_value Robot UID.
     * @param primitive Selected primitive action.
     */
    void add(std::string_view uid_value, Action primitive);
};

/**
 * @brief Bounded macro list for one robot.
 *
 * Macro generation never allocates and never returns more than `MAX_MACROS`
 * entries, preserving deterministic expansion cost in ISMCTS.
 */
struct MacroList {
    int count = 0;
    std::array<MacroAction, MAX_MACROS> macros{};
};

/**
 * @brief Runtime-tunable ISMCTS and rollout hyperparameters.
 *
 * Defaults mirror the best Optuna study results embedded in `main.py`.  The
 * pybind layer validates and injects these values per Engine instance, allowing
 * tuning without recompiling the native extension.
 */
struct Hyperparameters {
    /** @brief Exploration strength in PUCT child selection. */
    float C_puct = 2.0884330868271443F;
    /** @brief Extra prior mass assigned to the deterministic all-robot baseline plan. */
    float baseline_prior_multiplier = 1.8863044112273712F;
    /** @brief Heuristic rollout horizon after tree traversal. */
    int rollout_depth = 80;
    /** @brief MacroAction-indexed prior weights; unused slots keep conservative defaults. */
    std::array<float, MAX_MACROS> macro_prior{};

    /** @brief Construct with tuned macro priors. */
    Hyperparameters();

    /** @brief Restore the embedded tuned macro prior vector. */
    void reset_macro_priors();

    /**
     * @brief Read the prior weight for a macro action.
     * @param macro MacroAction to score.
     * @return Positive finite prior weight, falling back to idle-safe default.
     */
    [[nodiscard]] float prior_for(MacroAction macro) const;
};

/**
 * @brief Fixed-arena ISMCTS tree node storing one UID-keyed joint macro plan.
 *
 * Children represent action-history information sets.  Plans are keyed by UID,
 * not simulator-local slot, because each Determinization may recreate hidden
 * robots and reorder slots while the externally observed controlled robots keep
 * stable identities.
 */
struct MCTSNode {
    int parent = -1;
    int first_child = -1;
    int next_sibling = -1;
    int child_count = 0;
    int visits = 0;
    int depth = 0;
    int plan_count = 0;
    float value_sum = 0.0F;
    float prior = 0.0F;
    uint8_t expanded = 0;
    std::array<std::array<char, UID_LEN>, MAX_MCTS_PLAN_ROBOTS> plan_uid{};
    std::array<MacroAction, MAX_MCTS_PLAN_ROBOTS> plan_macro{};
};

/**
 * @brief Zero-allocation node arena for one turn of ISMCTS.
 *
 * Resetting rewinds `used` only; the backing node array remains allocated in
 * the Engine object and is overwritten by subsequent searches.
 */
struct MCTSArena {
    std::array<MCTSNode, MAX_TREE_NODES> nodes{};
    int used = 0;

    /** @brief Rewind the arena for a new search without clearing storage. */
    void reset();

    /**
     * @brief Allocate and initialize one tree node from the fixed arena.
     * @param parent Parent node index, or `-1` for root.
     * @param depth Tree depth from root.
     * @param prior Normalized PUCT prior for this edge.
     * @return Node index, or `-1` when the arena is full.
     */
    [[nodiscard]] int create_node(int parent, int depth, float prior);
};

/**
 * @brief Player-centric Belief State for fog-of-war memory and Determinization.
 *
 * Visible facts overwrite remembered state.  Walls and mines persist because
 * the environment remembers them; crystals and enemy robots outside vision do
 * not.  Hidden enemies are represented as per-type probability fields that
 * diffuse through passable or unknown cells, then each ISMCTS iteration samples
 * a concrete BoardState from those distributions.
 */
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

    /** @brief Clear all remembered facts and enemy probability fields. */
    void reset();

    /**
     * @brief Merge a new observation into the player-centric belief.
     * @param obs Fixed-buffer observation decoded by pybind.
     */
    void update_from_observation(const ObservationInput& obs);

    /**
     * @brief Sample one concrete hidden world consistent with current belief.
     * @param seed Deterministic seed used for unknown rows and enemy sampling.
     * @return Concrete BoardState suitable for simulation and Rollouts.
     */
    [[nodiscard]] BoardState determinize(uint64_t seed) const;
};

/**
 * @brief Deterministic Maze Crawler rules simulator.
 *
 * `step()` is the hot path used by both real-turn debug stepping and ISMCTS
 * Rollouts.  It resolves the rulebook phase order using fixed-size scratch
 * buffers only, so repeated simulations do not allocate or depend on Python.
 */
class CrawlerSim {
public:
    BoardState state{};

    /** @brief Reset the concrete simulator state. */
    void reset();

    /**
     * @brief Load a concrete visible snapshot from observation and belief memory.
     * @param obs Latest fixed-buffer observation.
     * @param belief Player-centric remembered facts.
     */
    void load_from_observation(const ObservationInput& obs, const BeliefState& belief);

    /**
     * @brief Advance the simulator by one rule-faithful turn.
     * @param actions Primitive actions addressed by simulator-local robot slots.
     */
    void step(const PrimitiveActions& actions);

    /**
     * @brief Compute the deterministic baseline action for the current player.
     * @param robot_index Simulator-local robot slot.
     * @return Baseline primitive action.
     */
    [[nodiscard]] Action heuristic_action_for(int robot_index) const;

    /**
     * @brief Compute the deterministic baseline action for an arbitrary owner.
     * @param robot_index Simulator-local robot slot.
     * @param owner Player index whose policy should control the robot.
     * @return Baseline primitive action.
     */
    [[nodiscard]] Action heuristic_action_for_owner(int robot_index, int owner) const;

    /**
     * @brief Fill a full deterministic baseline plan for one owner.
     * @param owner Player index to control.
     * @param actions Mutable primitive action buffer to fill in-place.
     * @param baseline_macros Optional macro labels parallel to robot slots.
     */
    void fill_heuristic_plan_for_owner(int owner, PrimitiveActions& actions,
                                       std::array<MacroAction, MAX_ROBOTS>* baseline_macros = nullptr) const;

    /**
     * @brief Generate the bounded macro intent list for one robot.
     * @param robot_index Simulator-local robot slot.
     * @return Fixed-capacity MacroList for ISMCTS expansion.
     */
    [[nodiscard]] MacroList generate_macros_for(int robot_index) const;

    /**
     * @brief Translate one macro intent into a legal primitive action when possible.
     * @param robot_index Simulator-local robot slot.
     * @param macro Macro intent selected by ISMCTS.
     * @return Primitive action, or `ACT_IDLE` when the macro is not currently legal.
     */
    [[nodiscard]] Action primitive_for_macro(int robot_index, MacroAction macro) const;
};

/**
 * @brief High-level facade owned by the Python binding.
 *
 * Engine owns the Belief State, latest concrete simulator snapshot, search
 * arena, and tunable hyperparameters for one player.  Python calls update once
 * per Kaggle observation and then asks the engine to choose actions under a
 * millisecond budget.
 */
class Engine {
public:
    BeliefState belief{};
    CrawlerSim sim{};
    MCTSArena mcts{};
    Hyperparameters hyperparameters{};

    /** @brief Construct a player-zero engine with empty belief and simulator state. */
    Engine();

    /**
     * @brief Construct an engine for a specific player.
     * @param player Player index in `{0, 1}`.
     */
    explicit Engine(int player);

    /**
     * @brief Update belief and concrete simulator state from a new observation.
     * @param obs Fixed-buffer observation decoded by pybind.
     */
    void update_observation(const ObservationInput& obs);

    /**
     * @brief Apply primitive actions directly to the current simulator snapshot.
     * @param actions Primitive action buffer keyed by simulator-local robot slot.
     */
    void step_actions(const PrimitiveActions& actions);

    /**
     * @brief Sample one concrete hidden world while preserving observed live robots.
     * @param seed Deterministic seed for hidden rows and enemy locations.
     * @return BoardState suitable for one ISMCTS iteration.
     */
    [[nodiscard]] BoardState determinize(uint64_t seed) const;

    /**
     * @brief Choose Kaggle-compatible actions via fixed-arena ISMCTS.
     * @param time_budget_ms Per-turn budget in milliseconds.
     * @param seed Deterministic root seed for reproducible Determinizations.
     * @return UID/action result buffer for controlled live robots.
     */
    [[nodiscard]] ActionResult choose_actions(int time_budget_ms, uint64_t seed);

    /**
     * @brief Evaluate the current simulator snapshot with the rollout value model.
     * @param player Player perspective, or invalid value to use `sim.state.player`.
     * @return Value in approximately `[-1, 1]`.
     */
    [[nodiscard]] float debug_mcts_value(int player) const;
};

}  // namespace crawler
