#include "crawler_engine_internal.hpp"

// Shared implementation utilities for deterministic simulation. Keep this file
// free of policy/search logic; it exists to support other engine modules.

#include <algorithm>
#include <cmath>
#include <cstring>

namespace crawler::detail {
namespace {

constexpr uint64_t RNG_MUL = 0xbf58476d1ce4e5b9ULL;

bool chance(uint64_t& state, int numerator, int denominator) {
    return static_cast<int>(next_u32(state) % static_cast<uint32_t>(denominator)) < numerator;
}

uint8_t reciprocal_wall(Direction direction) {
    return direction_wall_bit(opposite_direction(direction));
}

int py_round_interval(double value) {
    return static_cast<int>(std::nearbyint(value));
}

}  // namespace

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

int active_word(int active_index) {
    return active_index >> 6;
}

uint64_t active_mask(int active_index) {
    return 1ULL << (active_index & 63);
}

int cell_row(int abs_index) {
    return abs_index / WIDTH;
}

int cell_col(int abs_index) {
    return abs_index % WIDTH;
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

    // The central mirror wall is fixed except for rare generated doors.
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
