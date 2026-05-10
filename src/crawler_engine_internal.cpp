#include "crawler_engine_internal.hpp"

/**
 * @file crawler_engine_internal.cpp
 * @brief Shared deterministic primitives for Maze Crawler simulation and ISMCTS.
 *
 * The helpers in this file deliberately avoid policy and search decisions.
 * They provide the reproducible randomization, coordinate math, reciprocal wall
 * maintenance, scroll reconstruction, and optimistic row generation required by
 * both Belief State Determinization and the deterministic simulator.  Centralizing
 * these operations keeps the rule engine, Rollouts, and pybind debug paths aligned.
 */

#include <algorithm>
#include <cmath>
#include <cstring>

namespace crawler::detail {
namespace {

constexpr uint64_t RNG_MUL = 0xbf58476d1ce4e5b9ULL;

/**
 * @brief Draw a deterministic Bernoulli event from the engine RNG.
 * @param state Mutable 64-bit RNG state.
 * @param numerator Successful outcomes in the integer range.
 * @param denominator Total outcomes in the integer range.
 * @return True when the sampled residue falls below `numerator`.
 */
bool chance(uint64_t& state, int numerator, int denominator) {
    return static_cast<int>(next_u32(state) % static_cast<uint32_t>(denominator)) < numerator;
}

/**
 * @brief Convert a one-sided wall edit into the neighbor cell's reciprocal bit.
 * @param direction Direction of the wall from the source cell.
 * @return Opposite wall bit to write on the adjacent cell.
 */
uint8_t reciprocal_wall(Direction direction) {
    return direction_wall_bit(opposite_direction(direction));
}

/**
 * @brief Match Python's banker-style rounding for scroll interval ramping.
 * @param value Floating-point ramp value from the rulebook schedule.
 * @return Nearest integer using the same tie behavior as Python `round`.
 */
int py_round_interval(double value) {
    return static_cast<int>(std::nearbyint(value));
}

}  // namespace

/**
 * @brief Mix 64-bit seed material with splitmix-style avalanche operations.
 * @param x Input seed material.
 * @return High-dispersion deterministic 64-bit value.
 *
 * ISMCTS samples many hidden states under the same observed information set.
 * A cheap deterministic mixer keeps those samples reproducible while avoiding
 * correlated adjacent seeds for hidden enemy and future-row generation.
 */
uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

/**
 * @brief Advance the deterministic RNG and emit a 32-bit sample.
 * @param state Mutable RNG state.
 * @return Upper 32 bits of the mixed state.
 */
uint32_t next_u32(uint64_t& state) {
    state = state * RNG_MUL + 0x94d049bb133111ebULL;
    return static_cast<uint32_t>(mix64(state) >> 32U);
}

/**
 * @brief Copy a UID into a fixed-width NUL-terminated buffer.
 * @param dst Destination UID buffer.
 * @param src Source UID string.
 */
void copy_uid(std::array<char, UID_LEN>& dst, std::string_view src) {
    dst.fill('\0');
    const int n = std::min<int>(static_cast<int>(src.size()), UID_LEN - 1);
    if (n > 0) {
        std::memcpy(dst.data(), src.data(), static_cast<size_t>(n));
    }
}

/**
 * @brief Compare a stored fixed-width UID with a string view.
 * @param uid Stored NUL-terminated UID buffer.
 * @param value External UID to match.
 * @return True when the external UID fits and all bytes match.
 */
bool uid_equal(const std::array<char, UID_LEN>& uid, std::string_view value) {
    const int n = std::min<int>(static_cast<int>(value.size()), UID_LEN - 1);
    if (n == UID_LEN - 1 && static_cast<int>(value.size()) >= UID_LEN) {
        return false;
    }
    return std::strncmp(uid.data(), value.data(), static_cast<size_t>(n)) == 0 && uid[n] == '\0';
}

/**
 * @brief Map an active-window cell index to its 64-bit Bitboard word.
 * @param active_index Active-window index in `[0, ACTIVE_CELLS)`.
 * @return Word index, computed as integer division by 64.
 */
int active_word(int active_index) {
    /* Shifting by six is an O(1) division by 64 for Bitboard word selection. */
    return active_index >> 6;
}

/**
 * @brief Build the bit mask for an active-window cell within its word.
 * @param active_index Active-window index in `[0, ACTIVE_CELLS)`.
 * @return Single-bit mask for the cell.
 */
uint64_t active_mask(int active_index) {
    /* `active_index & 63` is modulo 64 without an integer division in policy loops. */
    return 1ULL << (active_index & 63);
}

/**
 * @brief Convert an absolute cell index into a row.
 * @param abs_index Absolute cell index.
 * @return Absolute row.
 */
int cell_row(int abs_index) {
    return abs_index / WIDTH;
}

/**
 * @brief Convert an absolute cell index into a column.
 * @param abs_index Absolute cell index.
 * @return Column in `[0, WIDTH)`.
 */
int cell_col(int abs_index) {
    return abs_index % WIDTH;
}

/**
 * @brief Compute the scroll cadence at a public environment step.
 * @param step Current step.
 * @return Rulebook scroll interval after applying the linear ramp.
 */
int scroll_interval(int step) {
    if (step >= SCROLL_RAMP_STEPS) {
        return SCROLL_END_INTERVAL;
    }
    const double progress = static_cast<double>(step) / static_cast<double>(SCROLL_RAMP_STEPS);
    const double value = static_cast<double>(SCROLL_START_INTERVAL) -
                         static_cast<double>(SCROLL_START_INTERVAL - SCROLL_END_INTERVAL) * progress;
    return std::max(SCROLL_END_INTERVAL, py_round_interval(value));
}

/**
 * @brief Reconstruct the scroll countdown for a freshly loaded observation.
 * @param step Public environment step.
 * @return Positive countdown value to the next boundary advance.
 *
 * Kaggle observations expose bounds and step, but not the hidden countdown.
 * Replaying the deterministic cadence keeps local simulation synchronized after
 * pybind reconstructs state from a sparse observation.
 */
int scroll_counter_at_step(int step) {
    int counter = SCROLL_START_INTERVAL;
    const int clamped_step = std::max(0, std::min(step, EPISODE_STEPS));
    for (int s = 0; s < clamped_step; ++s) {
        --counter;
        if (counter <= 0) {
            counter = scroll_interval(s);
        }
    }
    return std::max(1, counter);
}

/**
 * @brief Mutate a wall and its reciprocal neighbor bit.
 * @param state Concrete board state.
 * @param c Source column.
 * @param r Source row.
 * @param direction Wall direction from the source cell.
 * @param set_wall True to build the wall, false to remove it.
 *
 * Maze movement checks read only the source cell's directional bit.  Writing
 * both sides here preserves that O(1) lookup while keeping wall geometry
 * symmetric for later Rollouts and Determinizations.
 */
void set_or_clear_wall(BoardState& state, int c, int r, Direction direction, bool set_wall) {
    const int idx = state.abs_index(c, r);
    if (idx < 0) {
        return;
    }
    const uint8_t bit = direction_wall_bit(direction);
    if (set_wall) {
        /* OR sets exactly the requested directional bit without disturbing other walls. */
        state.walls[idx] = static_cast<uint8_t>(state.walls[idx] | bit);
    } else {
        /* AND with the inverted bit clears exactly one wall direction. */
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

/**
 * @brief Generate a symmetric plausible row for hidden future terrain.
 * @param state Concrete board state to mutate.
 * @param row Absolute row to populate.
 * @param seed Deterministic seed material.
 *
 * Determinization needs a concrete board even beyond the currently observed
 * active window.  The generator preserves east/west symmetry and approximate
 * resource densities so Rollouts can plan through unknown space without claiming
 * the generated cells are facts; later observations overwrite them exactly.
 */
void generate_optimistic_row(BoardState& state, int row, uint64_t seed) {
    if (row < 0 || row >= MAX_ROWS) {
        return;
    }

    uint64_t rng = mix64(seed ^ static_cast<uint64_t>(row + 1) * 0x9e3779b97f4a7c15ULL);
    const int half = WIDTH / 2;
    std::array<uint8_t, WIDTH> row_walls{};

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

    /* The central mirror wall is closed by default, with rare doors matching map generation. */
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
        const int idx = row * WIDTH + c;
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

}  // namespace crawler::detail
