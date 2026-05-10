#pragma once

/**
 * @file crawler_engine_internal.hpp
 * @brief Internal deterministic helper API shared by simulation, belief, policy, and ISMCTS.
 *
 * This header contains low-level primitives that are intentionally not exposed
 * through pybind.  The functions here support deterministic Determinization,
 * active-window Bitboard indexing, reciprocal wall maintenance, and synthetic
 * row generation.  Keeping these helpers centralized prevents each engine
 * module from inventing subtly different coordinate or randomization rules.
 */

#include "crawler_engine.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace crawler::detail {

/**
 * @brief Mix a 64-bit value into a high-dispersion deterministic seed.
 * @param x Input seed material.
 * @return Mixed 64-bit value.
 * @note Used for hidden-row generation and ISMCTS Determinizations so repeated
 *       runs with the same root seed remain reproducible.
 */
uint64_t mix64(uint64_t x);

/**
 * @brief Advance the deterministic engine RNG and return 32 random-looking bits.
 * @param state Mutable 64-bit RNG state.
 * @return Upper 32 bits of a mixed state transition.
 */
uint32_t next_u32(uint64_t& state);

/**
 * @brief Copy an external UID into a fixed-width NUL-terminated buffer.
 * @param dst Destination UID buffer.
 * @param src Source UID string.
 */
void copy_uid(std::array<char, UID_LEN>& dst, std::string_view src);

/**
 * @brief Compare a fixed-width UID buffer with an external UID string.
 * @param uid Stored NUL-terminated UID buffer.
 * @param value External UID string.
 * @return True when the values match exactly and fit in `UID_LEN`.
 */
bool uid_equal(const std::array<char, UID_LEN>& uid, std::string_view value);

/**
 * @brief Return the 64-bit word index for an active-window cell.
 * @param active_index Active-window index in `[0, ACTIVE_CELLS)`.
 * @return Word offset inside BitBoard::words.
 */
int active_word(int active_index);

/**
 * @brief Return the single-bit mask for an active-window cell.
 * @param active_index Active-window index in `[0, ACTIVE_CELLS)`.
 * @return `1ULL << (active_index % 64)`.
 */
uint64_t active_mask(int active_index);

/** @brief Convert an absolute cell index into its row. */
int cell_row(int abs_index);

/** @brief Convert an absolute cell index into its column. */
int cell_col(int abs_index);

/**
 * @brief Compute the rulebook scroll interval for a step.
 * @param step Current environment step.
 * @return Number of turns before the next scroll at that phase.
 */
int scroll_interval(int step);

/**
 * @brief Reconstruct the hidden scroll countdown from the public step number.
 * @param step Public environment step.
 * @return Countdown value that aligns a freshly loaded simulator with Kaggle.
 */
int scroll_counter_at_step(int step);

/**
 * @brief Set or clear a wall and the neighbor's reciprocal wall bit.
 * @param state Concrete board state to mutate.
 * @param c Source column.
 * @param r Source absolute row.
 * @param direction Wall direction from the source cell.
 * @param set_wall True to build the wall, false to remove it.
 */
void set_or_clear_wall(BoardState& state, int c, int r, Direction direction, bool set_wall);

/**
 * @brief Generate a plausible symmetric row for unobserved future terrain.
 * @param state Concrete board state to mutate.
 * @param row Absolute row to generate.
 * @param seed Deterministic seed material.
 * @note This is used only for Determinization and scroll lookahead; observed
 *       rows always overwrite generated guesses.
 */
void generate_optimistic_row(BoardState& state, int row, uint64_t seed);

}  // namespace crawler::detail
