#include "crawler_engine_internal.hpp"

/**
 * @file crawler_engine_belief.cpp
 * @brief Player-centric fog-of-war memory and hidden-state Determinization.
 *
 * The Belief State is the information-set boundary of the engine.  It preserves
 * facts that the rules allow the player to remember, clears facts that vanish
 * outside vision, diffuses hidden enemy probability fields through plausible
 * motion, and samples concrete BoardState instances for ISMCTS Rollouts.  It
 * deliberately contains no turn-resolution mechanics; CrawlerSim owns the rules.
 */

#include <algorithm>
#include <array>
#include <cmath>

namespace crawler {
namespace {

constexpr float EPS = 1.0e-6F;

/**
 * @brief Test whether enemy probability mass may diffuse through an edge.
 * @param belief Current player-centric Belief State.
 * @param c Source column.
 * @param r Source row.
 * @param direction Candidate motion direction.
 * @return True when known walls do not forbid the move and the destination is active.
 *
 * Unknown walls are treated as potentially passable so the belief does not
 * over-prune hidden enemies.  Staying stationary is handled separately as an
 * additional diffusion choice.
 */
bool can_diffuse_through(const BeliefState& belief, int c, int r, Direction direction) {
    if (c < 0 || c >= WIDTH || r < 0 || r >= MAX_ROWS) {
        return false;
    }
    const int idx = r * WIDTH + c;
    if (belief.known_wall[idx] != 0 && (belief.wall[idx] & direction_wall_bit(direction)) != 0) {
        return false;
    }
    const int nc = c + direction_dc(direction);
    const int nr = r + direction_dr(direction);
    return nc >= 0 && nc < WIDTH && nr >= belief.south_bound && nr <= belief.north_bound && nr < MAX_ROWS;
}

/**
 * @brief Diffuse one enemy type's probability field over elapsed turns.
 * @param belief Belief State to mutate.
 * @param type RobotType whose hidden locations are being propagated.
 * @param elapsed Number of public steps since the previous observation.
 *
 * The model spreads probability uniformly over legal neighbor moves plus the
 * stationary option.  Scouts can move every turn; slower units use their move
 * period to cap the diffusion radius.  The result is not a perfect Bayesian
 * filter, but it is cheap, deterministic, and good enough to seed ISMCTS
 * Determinizations without allocating graph structures.
 */
void diffuse_enemy_type(BeliefState& belief, int type, int elapsed) {
    const int period = move_period(static_cast<uint8_t>(type));
    const int radius = (type == SCOUT) ? elapsed : ((elapsed + period - 1) / period);
    if (radius <= 0) {
        return;
    }

    std::array<float, MAX_CELLS> cur{};
    std::array<float, MAX_CELLS> next{};
    cur = belief.enemy_prob[static_cast<size_t>(type)];

    for (int step = 0; step < radius; ++step) {
        next.fill(0.0F);
        for (int idx = 0; idx < MAX_CELLS; ++idx) {
            const float p = cur[static_cast<size_t>(idx)];
            if (p <= EPS) {
                continue;
            }
            const int c = detail::cell_col(idx);
            const int r = detail::cell_row(idx);
            int choices = 1;
            std::array<Direction, 4> dirs{DIR_NORTH, DIR_SOUTH, DIR_EAST, DIR_WEST};
            for (Direction d : dirs) {
                if (can_diffuse_through(belief, c, r, d)) {
                    ++choices;
                }
            }
            const float share = p / static_cast<float>(choices);
            /* Keep one share at the current cell so hidden robots may wait or be blocked by cooldown. */
            next[static_cast<size_t>(idx)] += share;
            for (Direction d : dirs) {
                if (!can_diffuse_through(belief, c, r, d)) {
                    continue;
                }
                const int ni = (r + direction_dr(d)) * WIDTH + (c + direction_dc(d));
                next[static_cast<size_t>(ni)] += share;
            }
        }
        cur = next;
    }

    belief.enemy_prob[static_cast<size_t>(type)] = cur;
}

}  // namespace

/**
 * @brief Reset belief memory before any observations have been merged.
 */
void BeliefState::reset() {
    player = 0;
    turn = 0;
    south_bound = 0;
    north_bound = HEIGHT - 1;
    known_wall.fill(0);
    wall.fill(0);
    visible_crystal.fill(0);
    remembered_mine_energy.fill(0);
    remembered_mine_max.fill(0);
    remembered_mine_owner.fill(-1);
    remembered_node.fill(0);
    currently_visible.fill(0);
    for (auto& p : enemy_prob) {
        p.fill(0.0F);
    }
}

/**
 * @brief Merge one fixed-buffer observation into the Belief State.
 * @param obs Latest observation decoded by pybind.
 *
 * Visible own-robot ranges define cells where hidden enemies are impossible.
 * Walls and mines are durable memories; crystals are visible-only and are
 * cleared when a visible cell no longer reports one.  Observed enemies collapse
 * their per-type probability field into a delta distribution at the seen cell.
 */
void BeliefState::update_from_observation(const ObservationInput& obs) {
    const int new_turn = obs.step >= 0 ? obs.step : turn + 1;
    const int elapsed = std::max(1, new_turn - turn);
    player = obs.player;
    south_bound = obs.south_bound;
    north_bound = obs.north_bound;

    for (int type = FACTORY; type <= MINER; ++type) {
        diffuse_enemy_type(*this, type, elapsed);
    }

    currently_visible.fill(0);
    for (int i = 0; i < obs.robot_count; ++i) {
        const RobotObservation& robot = obs.robots[static_cast<size_t>(i)];
        if (robot.owner != player) {
            continue;
        }
        const int range = vision_range(static_cast<uint8_t>(robot.type));
        for (int dc = -range; dc <= range; ++dc) {
            const int rem = range - std::abs(dc);
            for (int dr = -rem; dr <= rem; ++dr) {
                const int c = robot.col + dc;
                const int r = robot.row + dr;
                if (c < 0 || c >= WIDTH || r < 0 || r >= MAX_ROWS) {
                    continue;
                }
                /* Vision is Manhattan-distance based, so nested loops use the remaining radius after |dc|. */
                currently_visible[static_cast<size_t>(r * WIDTH + c)] = 1;
            }
        }
    }

    for (int r = obs.south_bound; r <= obs.north_bound; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            const int local = (r - obs.south_bound) * WIDTH + c;
            const int abs = r * WIDTH + c;
            if (obs.walls[static_cast<size_t>(local)] >= 0) {
                known_wall[static_cast<size_t>(abs)] = 1;
                wall[static_cast<size_t>(abs)] = static_cast<uint8_t>(obs.walls[static_cast<size_t>(local)]);
            }
            if (currently_visible[static_cast<size_t>(abs)] != 0) {
                /* Facts not remembered by the environment must disappear once the cell is seen empty. */
                visible_crystal[static_cast<size_t>(abs)] = 0;
                remembered_mine_energy[static_cast<size_t>(abs)] = 0;
                remembered_mine_max[static_cast<size_t>(abs)] = 0;
                remembered_mine_owner[static_cast<size_t>(abs)] = -1;
                for (auto& prob : enemy_prob) {
                    prob[static_cast<size_t>(abs)] = 0.0F;
                }
            }
        }
    }

    for (int i = 0; i < obs.crystal_count; ++i) {
        const auto& crystal = obs.crystals[static_cast<size_t>(i)];
        if (crystal.col < 0 || crystal.col >= WIDTH || crystal.row < 0 || crystal.row >= MAX_ROWS) {
            continue;
        }
        visible_crystal[static_cast<size_t>(crystal.row * WIDTH + crystal.col)] =
            static_cast<int16_t>(crystal.energy);
    }

    for (int i = 0; i < obs.mine_count; ++i) {
        const auto& mine = obs.mines[static_cast<size_t>(i)];
        if (mine.col < 0 || mine.col >= WIDTH || mine.row < 0 || mine.row >= MAX_ROWS) {
            continue;
        }
        const int idx = mine.row * WIDTH + mine.col;
        remembered_mine_energy[static_cast<size_t>(idx)] = static_cast<int16_t>(mine.energy);
        remembered_mine_max[static_cast<size_t>(idx)] = static_cast<int16_t>(mine.max_energy);
        remembered_mine_owner[static_cast<size_t>(idx)] = static_cast<int8_t>(mine.owner);
    }

    for (int i = 0; i < obs.mining_node_count; ++i) {
        const auto& node = obs.mining_nodes[static_cast<size_t>(i)];
        if (node.col < 0 || node.col >= WIDTH || node.row < 0 || node.row >= MAX_ROWS) {
            continue;
        }
        remembered_node[static_cast<size_t>(node.row * WIDTH + node.col)] = 1;
    }

    for (int i = 0; i < obs.robot_count; ++i) {
        const RobotObservation& robot = obs.robots[static_cast<size_t>(i)];
        if (robot.owner == player || robot.col < 0 || robot.col >= WIDTH || robot.row < 0 || robot.row >= MAX_ROWS) {
            continue;
        }
        const int idx = robot.row * WIDTH + robot.col;
        for (auto& prob : enemy_prob) {
            prob[static_cast<size_t>(idx)] = 0.0F;
        }
        enemy_prob[static_cast<size_t>(robot.type)][static_cast<size_t>(idx)] = 1.0F;
    }

    turn = new_turn;
}

/**
 * @brief Sample one concrete BoardState from the current information set.
 * @param seed Deterministic seed used for generated rows and enemy sampling.
 * @return Concrete hidden-state sample for one ISMCTS iteration.
 *
 * Known facts are copied exactly.  Unknown rows near the frontier are filled by
 * the optimistic symmetric row generator so Rollouts can reason beyond current
 * vision.  Enemy probability fields are sampled independently by type into
 * lightweight generated robots owned by the opponent.
 */
BoardState BeliefState::determinize(uint64_t seed) const {
    BoardState result{};
    result.reset();
    result.player = player;
    result.step = turn;
    result.south_bound = south_bound;
    result.north_bound = north_bound;
    result.rng_state = detail::mix64(seed);

    result.wall_known = known_wall;
    result.walls = wall;
    result.crystal_energy = visible_crystal;
    result.mine_energy = remembered_mine_energy;
    result.mine_max = remembered_mine_max;
    result.mine_owner = remembered_mine_owner;
    result.mining_node = remembered_node;

    for (int r = south_bound; r <= std::min(MAX_ROWS - 1, north_bound + HEIGHT); ++r) {
        bool row_unknown = true;
        for (int c = 0; c < WIDTH; ++c) {
            if (known_wall[static_cast<size_t>(r * WIDTH + c)] != 0) {
                row_unknown = false;
                break;
            }
        }
        if (row_unknown) {
            /* Generated rows are hypotheses only; future observations overwrite wall_known facts. */
            detail::generate_optimistic_row(result, r, seed);
        }
    }

    const int enemy_owner = 1 - player;
    for (int type = FACTORY; type <= MINER; ++type) {
        float total = 0.0F;
        for (float p : enemy_prob[static_cast<size_t>(type)]) {
            total += p;
        }
        if (total <= 0.05F) {
            continue;
        }
        float draw = (static_cast<float>(detail::next_u32(result.rng_state) % 100000U) / 100000.0F) * total;
        int selected = -1;
        for (int idx = 0; idx < MAX_CELLS; ++idx) {
            draw -= enemy_prob[static_cast<size_t>(type)][static_cast<size_t>(idx)];
            if (draw <= 0.0F) {
                selected = idx;
                break;
            }
        }
        if (selected >= 0) {
            /* Synthetic UIDs are sufficient for hidden enemies because controlled plans are UID-keyed to observed robots. */
            const int slot = result.robots.add_generated_robot(
                result.next_generated_uid++, static_cast<uint8_t>(type), static_cast<uint8_t>(enemy_owner),
                detail::cell_col(selected), detail::cell_row(selected), max_energy(static_cast<uint8_t>(type)) / 2);
            (void)slot;
        }
    }

    result.rebuild_active_bitboards();
    return result;
}

}  // namespace crawler
