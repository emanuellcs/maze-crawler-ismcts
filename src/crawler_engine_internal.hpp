#pragma once

#include "crawler_engine.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace crawler::detail {

// Deterministic mixer used for hidden-row sampling and determinization.
uint64_t mix64(uint64_t x);
uint32_t next_u32(uint64_t& state);

// Compact UID helpers keep RobotStore and ActionResult fixed-buffer.
void copy_uid(std::array<char, UID_LEN>& dst, std::string_view src);
bool uid_equal(const std::array<char, UID_LEN>& uid, std::string_view value);

// Active-window bitboard indexing helpers.
int active_word(int active_index);
uint64_t active_mask(int active_index);

// Absolute-cell coordinate helpers.
int cell_row(int abs_index);
int cell_col(int abs_index);

// Shared map/rule helpers used by the simulator and belief determinizer.
int scroll_interval(int step);
void set_or_clear_wall(BoardState& state, int c, int r, Direction direction, bool set_wall);
void generate_optimistic_row(BoardState& state, int row, uint64_t seed);

}  // namespace crawler::detail
