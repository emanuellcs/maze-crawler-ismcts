#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "constants.hpp"
#include "enums.hpp"

namespace crawler {

/**
 * @brief 20x20 active-window Bitboard backed by 64-bit machine words.
 *
 * Bitboards encode tactical facts for the currently visible/active window:
 * occupancy, visibility, crystals, mines, and mining nodes.  Bitwise shifts and
 * masks replace array lookups in tight policy loops, giving O(1) collision and
 * goal membership tests during Rollouts and macro-policy generation.
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

}  // namespace crawler
