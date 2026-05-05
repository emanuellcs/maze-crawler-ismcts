#include "crawler_engine_internal.hpp"

// Exact deterministic game mechanics. This module owns the phase-ordered `step`
// implementation and avoids heap allocation in the hot simulation path.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace crawler {
namespace {

void compute_rewards(BoardState& state) {
    int factory_count[2] = {0, 0};
    int energy[2] = {0, 0};
    int units[2] = {0, 0};
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const int owner = state.robots.owner[static_cast<size_t>(i)];
        if (owner < 0 || owner > 1) {
            continue;
        }
        energy[owner] += state.robots.energy[static_cast<size_t>(i)];
        units[owner] += 1;
        if (state.robots.type[static_cast<size_t>(i)] == FACTORY) {
            factory_count[owner] += 1;
        }
    }

    const bool dead0 = factory_count[0] == 0;
    const bool dead1 = factory_count[1] == 0;
    if (!dead0 && !dead1 && state.step + 1 < EPISODE_STEPS) {
        state.reward0 = static_cast<float>(energy[0]);
        state.reward1 = static_cast<float>(energy[1]);
        return;
    }

    if (dead0 && !dead1) {
        state.done = true;
        state.winner = 1;
        state.reward0 = static_cast<float>(state.step - EPISODE_STEPS - 1);
        state.reward1 = static_cast<float>(energy[1]);
        return;
    }
    if (dead1 && !dead0) {
        state.done = true;
        state.winner = 0;
        state.reward0 = static_cast<float>(energy[0]);
        state.reward1 = static_cast<float>(state.step - EPISODE_STEPS - 1);
        return;
    }

    state.done = true;
    if (energy[0] > energy[1]) {
        state.winner = 0;
        state.reward0 = 1.0F;
        state.reward1 = 0.0F;
    } else if (energy[1] > energy[0]) {
        state.winner = 1;
        state.reward0 = 0.0F;
        state.reward1 = 1.0F;
    } else if (units[0] > units[1]) {
        state.winner = 0;
        state.reward0 = 1.0F;
        state.reward1 = 0.0F;
    } else if (units[1] > units[0]) {
        state.winner = 1;
        state.reward0 = 0.0F;
        state.reward1 = 1.0F;
    } else {
        state.winner = -1;
        state.reward0 = 0.5F;
        state.reward1 = 0.5F;
    }
}

}  // namespace

void CrawlerSim::reset() {
    state.reset();
}

void CrawlerSim::load_from_observation(const ObservationInput& obs, const BeliefState& belief) {
    const int prior_step = state.step;
    const int prior_scroll = state.scroll_counter;
    const uint32_t prior_uid = state.next_generated_uid;
    state.reset();
    state.player = obs.player;
    state.south_bound = obs.south_bound;
    state.north_bound = obs.north_bound;
    state.step = obs.step >= 0 ? obs.step : prior_step + 1;
    state.scroll_counter = prior_scroll > 0 ? prior_scroll : detail::scroll_interval(state.step);
    state.next_generated_uid = prior_uid == 0 ? 1 : prior_uid;
    state.wall_known = belief.known_wall;
    state.walls = belief.wall;
    state.crystal_energy = belief.visible_crystal;
    state.mine_energy = belief.remembered_mine_energy;
    state.mine_max = belief.remembered_mine_max;
    state.mine_owner = belief.remembered_mine_owner;
    state.mining_node = belief.remembered_node;

    for (int r = obs.south_bound; r <= obs.north_bound; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            const int local = (r - obs.south_bound) * WIDTH + c;
            const int abs = state.abs_index(c, r);
            if (abs < 0) {
                continue;
            }
            if (obs.walls[static_cast<size_t>(local)] >= 0) {
                state.wall_known[static_cast<size_t>(abs)] = 1;
                state.walls[static_cast<size_t>(abs)] = static_cast<uint8_t>(obs.walls[static_cast<size_t>(local)]);
            }
        }
    }

    for (int i = 0; i < obs.robot_count; ++i) {
        const auto& r = obs.robots[static_cast<size_t>(i)];
        const int slot = state.robots.add_robot(r.uid.data(), static_cast<uint8_t>(r.type),
                                                static_cast<uint8_t>(r.owner), r.col, r.row, r.energy,
                                                r.move_cd, r.jump_cd, r.build_cd);
        (void)slot;
    }

    state.rebuild_active_bitboards();
}

void CrawlerSim::step(const PrimitiveActions& input_actions) {
    if (state.done) {
        return;
    }

    std::array<Action, MAX_ROBOTS> actions = input_actions.actions;
    std::array<uint8_t, MAX_ROBOTS> destroyed{};
    std::array<uint8_t, MAX_ROBOTS> stationary{};
    std::array<uint8_t, MAX_ROBOTS> moved{};
    std::array<uint8_t, MAX_ROBOTS> offboard{};
    std::array<int16_t, MAX_ROBOTS> target_abs{};
    std::array<int16_t, MAX_CELLS> stationary_first{};
    std::array<int16_t, MAX_CELLS> mover_first{};
    std::array<int16_t, MAX_ROBOTS> stationary_next{};
    std::array<int16_t, MAX_ROBOTS> mover_next{};
    std::array<uint8_t, MAX_CELLS> stationary_count{};
    std::array<uint8_t, MAX_CELLS> mover_count{};
    std::array<uint8_t, MAX_CELLS> combat_cell{};

    target_abs.fill(-1);
    stationary_first.fill(-1);
    mover_first.fill(-1);
    stationary_next.fill(-1);
    mover_next.fill(-1);

    // Phase 1: cooldown tick.
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        if (state.robots.move_cd[static_cast<size_t>(i)] > 0) {
            --state.robots.move_cd[static_cast<size_t>(i)];
        }
        if (state.robots.jump_cd[static_cast<size_t>(i)] > 0) {
            --state.robots.jump_cd[static_cast<size_t>(i)];
        }
        if (state.robots.build_cd[static_cast<size_t>(i)] > 0) {
            --state.robots.build_cd[static_cast<size_t>(i)];
        }
    }

    // Phase 2: action type validation. Resource and wall legality is checked in the owning phase.
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const uint8_t type = state.robots.type[static_cast<size_t>(i)];
        const Action a = actions[static_cast<size_t>(i)];
        bool valid = a == ACT_IDLE || a == ACT_NORTH || a == ACT_SOUTH || a == ACT_EAST || a == ACT_WEST ||
                     (type == FACTORY && (a == ACT_BUILD_SCOUT || a == ACT_BUILD_WORKER || a == ACT_BUILD_MINER ||
                                          a == ACT_JUMP_NORTH || a == ACT_JUMP_SOUTH || a == ACT_JUMP_EAST ||
                                          a == ACT_JUMP_WEST)) ||
                     (type == WORKER && a >= ACT_BUILD_NORTH && a <= ACT_REMOVE_WEST) ||
                     (type == MINER && a == ACT_TRANSFORM) ||
                     (a >= ACT_TRANSFER_NORTH && a <= ACT_TRANSFER_WEST);
        if (!valid) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
        }
    }

    // Phase 3: per-turn energy drain; robots reaching zero cannot act this turn.
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        state.robots.energy[static_cast<size_t>(i)] =
            static_cast<int16_t>(std::max(0, state.robots.energy[static_cast<size_t>(i)] - ENERGY_PER_TURN));
        if (state.robots.energy[static_cast<size_t>(i)] == 0) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
        }
    }

    // Phase 4a: miner transform, before any movement or combat.
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0 || actions[static_cast<size_t>(i)] != ACT_TRANSFORM) {
            continue;
        }
        const int idx = state.abs_index(state.robots.col[static_cast<size_t>(i)],
                                        state.robots.row[static_cast<size_t>(i)]);
        if (idx < 0 || state.mining_node[static_cast<size_t>(idx)] == 0 ||
            state.robots.energy[static_cast<size_t>(i)] < TRANSFORM_COST) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
            continue;
        }
        state.mine_energy[static_cast<size_t>(idx)] =
            static_cast<int16_t>(std::min(MINE_MAX_ENERGY,
                                          state.robots.energy[static_cast<size_t>(i)] - TRANSFORM_COST));
        state.mine_max[static_cast<size_t>(idx)] = MINE_MAX_ENERGY;
        state.mine_owner[static_cast<size_t>(idx)] = static_cast<int8_t>(state.robots.owner[static_cast<size_t>(i)]);
        state.mining_node[static_cast<size_t>(idx)] = 0;
        destroyed[static_cast<size_t>(i)] = 1;
    }

    // Phase 4b: worker wall edits. Fixed-wall no-ops still consume energy by rule.
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0 || destroyed[static_cast<size_t>(i)] != 0) {
            continue;
        }
        const Action a = actions[static_cast<size_t>(i)];
        const bool build = a >= ACT_BUILD_NORTH && a <= ACT_BUILD_WEST;
        const bool remove = a >= ACT_REMOVE_NORTH && a <= ACT_REMOVE_WEST;
        if (!build && !remove) {
            continue;
        }
        const int cost = build ? WALL_BUILD_COST : WALL_REMOVE_COST;
        if (state.robots.energy[static_cast<size_t>(i)] < cost) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
            continue;
        }
        state.robots.energy[static_cast<size_t>(i)] =
            static_cast<int16_t>(state.robots.energy[static_cast<size_t>(i)] - cost);
        const Direction d = action_direction(a);
        const int c = state.robots.col[static_cast<size_t>(i)];
        const int r = state.robots.row[static_cast<size_t>(i)];
        if (!is_fixed_wall(c, d)) {
            detail::set_or_clear_wall(state, c, r, d, build);
        }
    }

    // Phase 4c: factory builds. Spawned units are stationary combat participants this same turn.
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0 || destroyed[static_cast<size_t>(i)] != 0) {
            continue;
        }
        const Action a = actions[static_cast<size_t>(i)];
        if (a != ACT_BUILD_SCOUT && a != ACT_BUILD_WORKER && a != ACT_BUILD_MINER) {
            continue;
        }
        int cost = SCOUT_COST;
        uint8_t new_type = SCOUT;
        if (a == ACT_BUILD_WORKER) {
            cost = WORKER_COST;
            new_type = WORKER;
        } else if (a == ACT_BUILD_MINER) {
            cost = MINER_COST;
            new_type = MINER;
        }
        if (state.robots.energy[static_cast<size_t>(i)] < cost ||
            state.robots.build_cd[static_cast<size_t>(i)] > 0) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
            continue;
        }
        const int c = state.robots.col[static_cast<size_t>(i)];
        const int r = state.robots.row[static_cast<size_t>(i)];
        const int sr = r + 1;
        if (sr > state.north_bound || !state.can_move_through(c, r, DIR_NORTH)) {
            actions[static_cast<size_t>(i)] = ACT_IDLE;
            continue;
        }
        state.robots.energy[static_cast<size_t>(i)] =
            static_cast<int16_t>(state.robots.energy[static_cast<size_t>(i)] - cost);
        state.robots.build_cd[static_cast<size_t>(i)] = FACTORY_BUILD_COOLDOWN;
        const int slot = state.robots.add_generated_robot(state.next_generated_uid++, new_type,
                                                          state.robots.owner[static_cast<size_t>(i)],
                                                          c, sr, cost);
        if (slot >= 0) {
            actions[static_cast<size_t>(slot)] = ACT_IDLE;
        }
    }

    // Phase 4d: transfers happen before movement. Overflow is discarded because the source sends all energy.
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0 || destroyed[static_cast<size_t>(i)] != 0) {
            continue;
        }
        const Action a = actions[static_cast<size_t>(i)];
        if (a < ACT_TRANSFER_NORTH || a > ACT_TRANSFER_WEST) {
            continue;
        }
        const Direction d = action_direction(a);
        const int c = state.robots.col[static_cast<size_t>(i)];
        const int r = state.robots.row[static_cast<size_t>(i)];
        if (!state.can_move_through(c, r, d)) {
            continue;
        }
        const int tc = c + direction_dc(d);
        const int tr = r + direction_dr(d);
        int target = -1;
        for (int j = 0; j < state.robots.used; ++j) {
            if (i == j || state.robots.alive[static_cast<size_t>(j)] == 0 || destroyed[static_cast<size_t>(j)] != 0) {
                continue;
            }
            if (state.robots.owner[static_cast<size_t>(j)] == state.robots.owner[static_cast<size_t>(i)] &&
                state.robots.col[static_cast<size_t>(j)] == tc && state.robots.row[static_cast<size_t>(j)] == tr) {
                target = j;
                break;
            }
        }
        if (target < 0) {
            continue;
        }
        const int cap = max_energy(state.robots.type[static_cast<size_t>(target)]);
        const int space = std::max(0, cap - state.robots.energy[static_cast<size_t>(target)]);
        const int source_energy = state.robots.energy[static_cast<size_t>(i)];
        const int amount = std::min<int>(source_energy, space);
        state.robots.energy[static_cast<size_t>(target)] =
            static_cast<int16_t>(state.robots.energy[static_cast<size_t>(target)] + amount);
        state.robots.energy[static_cast<size_t>(i)] = 0;
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (destroyed[static_cast<size_t>(i)] != 0) {
            state.robots.remove(i);
        }
    }

    // Phase 5a: collect simultaneous movement intentions without applying positions.
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const Action a = actions[static_cast<size_t>(i)];
        if (a >= ACT_NORTH && a <= ACT_WEST) {
            if (state.robots.move_cd[static_cast<size_t>(i)] > 0) {
                stationary[static_cast<size_t>(i)] = 1;
                continue;
            }
            const Direction d = action_direction(a);
            const int c = state.robots.col[static_cast<size_t>(i)];
            const int r = state.robots.row[static_cast<size_t>(i)];
            const int tc = c + direction_dc(d);
            const int tr = r + direction_dr(d);
            if (tc < 0 || tc >= WIDTH) {
                stationary[static_cast<size_t>(i)] = 1;
                continue;
            }
            if (tr < state.south_bound || tr > state.north_bound) {
                const uint8_t source_wall = state.wall_at(c, r);
                if ((source_wall & direction_wall_bit(d)) != 0) {
                    stationary[static_cast<size_t>(i)] = 1;
                } else {
                    offboard[static_cast<size_t>(i)] = 1;
                }
                continue;
            }
            if (state.can_move_through(c, r, d)) {
                target_abs[static_cast<size_t>(i)] = static_cast<int16_t>(state.abs_index(tc, tr));
            } else {
                stationary[static_cast<size_t>(i)] = 1;
            }
        } else if (a >= ACT_JUMP_NORTH && a <= ACT_JUMP_WEST) {
            if (state.robots.move_cd[static_cast<size_t>(i)] > 0 ||
                state.robots.jump_cd[static_cast<size_t>(i)] > 0) {
                stationary[static_cast<size_t>(i)] = 1;
                continue;
            }
            const Direction d = action_direction(a);
            const int tc = state.robots.col[static_cast<size_t>(i)] + direction_dc(d) * 2;
            const int tr = state.robots.row[static_cast<size_t>(i)] + direction_dr(d) * 2;
            state.robots.jump_cd[static_cast<size_t>(i)] = FACTORY_JUMP_COOLDOWN;
            state.robots.move_cd[static_cast<size_t>(i)] =
                static_cast<int16_t>(move_period(state.robots.type[static_cast<size_t>(i)]));
            if (tc >= 0 && tc < WIDTH && tr >= state.south_bound && tr <= state.north_bound) {
                target_abs[static_cast<size_t>(i)] = static_cast<int16_t>(state.abs_index(tc, tr));
            } else {
                offboard[static_cast<size_t>(i)] = 1;
            }
        } else {
            stationary[static_cast<size_t>(i)] = 1;
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (offboard[static_cast<size_t>(i)] != 0) {
            state.robots.remove(i);
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        if (stationary[static_cast<size_t>(i)] != 0) {
            const int idx = state.abs_index(state.robots.col[static_cast<size_t>(i)],
                                            state.robots.row[static_cast<size_t>(i)]);
            if (idx >= 0) {
                stationary_next[static_cast<size_t>(i)] = stationary_first[static_cast<size_t>(idx)];
                stationary_first[static_cast<size_t>(idx)] = static_cast<int16_t>(i);
                ++stationary_count[static_cast<size_t>(idx)];
            }
        }
        if (target_abs[static_cast<size_t>(i)] >= 0) {
            const int idx = target_abs[static_cast<size_t>(i)];
            mover_next[static_cast<size_t>(i)] = mover_first[static_cast<size_t>(idx)];
            mover_first[static_cast<size_t>(idx)] = static_cast<int16_t>(i);
            ++mover_count[static_cast<size_t>(idx)];
        }
    }

    // Phase 5b: resolve same-cell crush combat from the planned destinations.
    for (int r = state.south_bound; r <= state.north_bound; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            const int idx = state.abs_index(c, r);
            const int total = stationary_count[static_cast<size_t>(idx)] + mover_count[static_cast<size_t>(idx)];
            if (total <= 0) {
                continue;
            }
            if (total == 1) {
                const int mover = mover_first[static_cast<size_t>(idx)];
                if (mover >= 0) {
                    moved[static_cast<size_t>(mover)] = 1;
                }
                continue;
            }

            combat_cell[static_cast<size_t>(idx)] = 1;
            std::array<int16_t, MAX_ROBOTS> participants{};
            std::array<uint8_t, 4> type_count{};
            int count = 0;
            int factory_owner_mask = 0;

            for (int u = mover_first[static_cast<size_t>(idx)]; u >= 0; u = mover_next[static_cast<size_t>(u)]) {
                participants[static_cast<size_t>(count++)] = static_cast<int16_t>(u);
                const uint8_t t = state.robots.type[static_cast<size_t>(u)];
                ++type_count[static_cast<size_t>(t)];
                if (t == FACTORY) {
                    factory_owner_mask |= 1 << state.robots.owner[static_cast<size_t>(u)];
                }
            }
            for (int u = stationary_first[static_cast<size_t>(idx)]; u >= 0;
                 u = stationary_next[static_cast<size_t>(u)]) {
                participants[static_cast<size_t>(count++)] = static_cast<int16_t>(u);
                const uint8_t t = state.robots.type[static_cast<size_t>(u)];
                ++type_count[static_cast<size_t>(t)];
                if (t == FACTORY) {
                    factory_owner_mask |= 1 << state.robots.owner[static_cast<size_t>(u)];
                }
            }

            for (int pi = 0; pi < count; ++pi) {
                const int u = participants[static_cast<size_t>(pi)];
                const uint8_t t = state.robots.type[static_cast<size_t>(u)];
                bool dies = false;
                if (t == FACTORY) {
                    dies = type_count[static_cast<size_t>(FACTORY)] > 1 ||
                           (factory_owner_mask & 0b11) == 0b11;
                } else if (type_count[static_cast<size_t>(t)] > 1) {
                    dies = true;
                } else {
                    for (int stronger = static_cast<int>(t) + 1; stronger <= MINER; ++stronger) {
                        if (type_count[static_cast<size_t>(stronger)] > 0) {
                            dies = true;
                        }
                    }
                    if (type_count[static_cast<size_t>(FACTORY)] > 0) {
                        dies = true;
                    }
                }
                if (dies) {
                    destroyed[static_cast<size_t>(u)] = 1;
                }
            }

            for (int u = mover_first[static_cast<size_t>(idx)]; u >= 0; u = mover_next[static_cast<size_t>(u)]) {
                if (destroyed[static_cast<size_t>(u)] == 0) {
                    moved[static_cast<size_t>(u)] = 1;
                }
            }
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (moved[static_cast<size_t>(i)] == 0 || destroyed[static_cast<size_t>(i)] != 0 ||
            state.robots.alive[static_cast<size_t>(i)] == 0 || target_abs[static_cast<size_t>(i)] < 0) {
            continue;
        }
        const int idx = target_abs[static_cast<size_t>(i)];
        state.robots.col[static_cast<size_t>(i)] = static_cast<int16_t>(detail::cell_col(idx));
        state.robots.row[static_cast<size_t>(i)] = static_cast<int16_t>(detail::cell_row(idx));
        state.robots.move_cd[static_cast<size_t>(i)] =
            static_cast<int16_t>(move_period(state.robots.type[static_cast<size_t>(i)]));
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (destroyed[static_cast<size_t>(i)] != 0) {
            state.robots.remove(i);
        }
    }

    // Phase 6: crystals disappear on combat cells without survivors, otherwise the survivor collects.
    for (int idx = 0; idx < MAX_CELLS; ++idx) {
        if (combat_cell[static_cast<size_t>(idx)] == 0 || state.crystal_energy[static_cast<size_t>(idx)] <= 0) {
            continue;
        }
        bool survivor_here = false;
        for (int i = 0; i < state.robots.used; ++i) {
            if (state.robots.alive[static_cast<size_t>(i)] != 0 &&
                state.abs_index(state.robots.col[static_cast<size_t>(i)],
                                state.robots.row[static_cast<size_t>(i)]) == idx) {
                survivor_here = true;
                break;
            }
        }
        if (!survivor_here) {
            state.crystal_energy[static_cast<size_t>(idx)] = 0;
        }
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const int idx = state.abs_index(state.robots.col[static_cast<size_t>(i)],
                                        state.robots.row[static_cast<size_t>(i)]);
        if (idx >= 0 && state.crystal_energy[static_cast<size_t>(idx)] > 0) {
            const int cap = max_energy(state.robots.type[static_cast<size_t>(i)]);
            const int gain = std::min<int>(state.crystal_energy[static_cast<size_t>(idx)],
                                           std::max(0, cap - state.robots.energy[static_cast<size_t>(i)]));
            state.robots.energy[static_cast<size_t>(i)] =
                static_cast<int16_t>(state.robots.energy[static_cast<size_t>(i)] + gain);
            state.crystal_energy[static_cast<size_t>(idx)] = 0;
        }
    }

    // Phases 7-8: mine pickup happens before mine generation.
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const int idx = state.abs_index(state.robots.col[static_cast<size_t>(i)],
                                        state.robots.row[static_cast<size_t>(i)]);
        if (idx < 0 || state.mine_max[static_cast<size_t>(idx)] <= 0 ||
            state.mine_owner[static_cast<size_t>(idx)] != state.robots.owner[static_cast<size_t>(i)]) {
            continue;
        }
        const int cap = max_energy(state.robots.type[static_cast<size_t>(i)]);
        const int transfer = std::min<int>(state.mine_energy[static_cast<size_t>(idx)],
                                           std::max(0, cap - state.robots.energy[static_cast<size_t>(i)]));
        state.robots.energy[static_cast<size_t>(i)] =
            static_cast<int16_t>(state.robots.energy[static_cast<size_t>(i)] + transfer);
        state.mine_energy[static_cast<size_t>(idx)] =
            static_cast<int16_t>(state.mine_energy[static_cast<size_t>(idx)] - transfer);
    }

    for (int idx = 0; idx < MAX_CELLS; ++idx) {
        if (state.mine_max[static_cast<size_t>(idx)] > 0) {
            state.mine_energy[static_cast<size_t>(idx)] =
                static_cast<int16_t>(std::min<int>(state.mine_energy[static_cast<size_t>(idx)] + MINE_RATE,
                                                   state.mine_max[static_cast<size_t>(idx)]));
        }
    }

    // Phases 9-10: scroll and destroy everything below the new south bound.
    --state.scroll_counter;
    if (state.scroll_counter <= 0) {
        ++state.south_bound;
        ++state.north_bound;
        if (state.north_bound < MAX_ROWS) {
            detail::generate_optimistic_row(state, state.north_bound, state.rng_state ^ static_cast<uint64_t>(state.step));
        }
        state.scroll_counter = detail::scroll_interval(state.step);
    }

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] != 0 &&
            state.robots.row[static_cast<size_t>(i)] < state.south_bound) {
            state.robots.remove(i);
        }
    }

    for (int r = 0; r < state.south_bound && r < MAX_ROWS; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            const int idx = state.abs_index(c, r);
            state.crystal_energy[static_cast<size_t>(idx)] = 0;
            state.mine_energy[static_cast<size_t>(idx)] = 0;
            state.mine_max[static_cast<size_t>(idx)] = 0;
            state.mine_owner[static_cast<size_t>(idx)] = -1;
            state.mining_node[static_cast<size_t>(idx)] = 0;
        }
    }

    compute_rewards(state);
    ++state.step;
    state.rebuild_active_bitboards();
}

}  // namespace crawler
