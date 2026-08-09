#pragma once

#include <array>
#include <cstdint>

#include "constants.hpp"

namespace crawler {

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

}  // namespace crawler
