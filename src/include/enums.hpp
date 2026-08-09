#pragma once

#include <cstdint>

namespace crawler {

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

}  // namespace crawler
