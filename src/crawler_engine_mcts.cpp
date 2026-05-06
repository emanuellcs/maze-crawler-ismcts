#include "crawler_engine_internal.hpp"

// Fixed-arena Information Set MCTS over bounded joint macro actions. Nodes are
// action-history information sets; each iteration samples a concrete board from
// belief and replays the selected macro history through that determinization.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>

namespace crawler {
namespace {

constexpr float PUCT_C = 1.35F;
constexpr uint64_t ITERATION_SEED = 0x9e3779b97f4a7c15ULL;

struct PlanCandidate {
    int plan_count = 0;
    float prior = 1.0F;
    std::array<std::array<char, UID_LEN>, MAX_MCTS_PLAN_ROBOTS> uid{};
    std::array<MacroAction, MAX_MCTS_PLAN_ROBOTS> macro{};
};

struct EvalStats {
    std::array<int64_t, 2> energy{0, 0};
    std::array<int, 2> units{0, 0};
    std::array<int, 2> material{0, 0};
    std::array<int, 2> factories{0, 0};
    std::array<int, 2> best_factory_row{0, 0};
};

bool can_pay_after_drain_local(int energy, int cost) {
    return energy >= cost + ENERGY_PER_TURN;
}

bool ready_after_tick_local(int cooldown) {
    return cooldown <= 1;
}

int count_owner_type(const BoardState& state, int owner, RobotType type) {
    int count = 0;
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] != 0 &&
            state.robots.owner[static_cast<size_t>(i)] == owner &&
            state.robots.type[static_cast<size_t>(i)] == type) {
            ++count;
        }
    }
    return count;
}

MacroAction default_macro_for_robot(const BoardState& state, int robot_index, int owner) {
    const uint8_t type = state.robots.type[static_cast<size_t>(robot_index)];
    const int c = state.robots.col[static_cast<size_t>(robot_index)];
    const int r = state.robots.row[static_cast<size_t>(robot_index)];
    const int e = state.robots.energy[static_cast<size_t>(robot_index)];
    const uint8_t wall = state.wall_at(c, r);

    if (type == FACTORY) {
        if ((wall & WALL_N) != 0 &&
            ready_after_tick_local(state.robots.jump_cd[static_cast<size_t>(robot_index)]) &&
            ready_after_tick_local(state.robots.move_cd[static_cast<size_t>(robot_index)]) &&
            r + 2 <= state.north_bound) {
            return MACRO_FACTORY_JUMP_OBSTACLE;
        }
        if (ready_after_tick_local(state.robots.build_cd[static_cast<size_t>(robot_index)]) &&
            r + 1 <= state.north_bound && state.can_move_through(c, r, DIR_NORTH)) {
            if (count_owner_type(state, owner, WORKER) < 2 && can_pay_after_drain_local(e, WORKER_COST)) {
                return MACRO_FACTORY_BUILD_WORKER;
            }
            if (count_owner_type(state, owner, SCOUT) < 3 && can_pay_after_drain_local(e, SCOUT_COST)) {
                return MACRO_FACTORY_BUILD_SCOUT;
            }
            if (can_pay_after_drain_local(e, MINER_COST)) {
                return MACRO_FACTORY_BUILD_MINER;
            }
        }
        return MACRO_FACTORY_SAFE_ADVANCE;
    }

    if (type == WORKER) {
        if ((wall & WALL_N) != 0 && can_pay_after_drain_local(e, WALL_REMOVE_COST)) {
            return MACRO_WORKER_OPEN_NORTH_WALL;
        }
        return MACRO_WORKER_ADVANCE;
    }

    if (type == SCOUT) {
        return e >= (SCOUT_MAX_ENERGY * 4) / 5 ? MACRO_SCOUT_RETURN_ENERGY : MACRO_SCOUT_HUNT_CRYSTAL;
    }

    if (type == MINER) {
        const int idx = state.abs_index(c, r);
        if (idx >= 0 && state.mining_node[static_cast<size_t>(idx)] != 0 &&
            can_pay_after_drain_local(e, TRANSFORM_COST)) {
            return MACRO_MINER_TRANSFORM;
        }
        return MACRO_MINER_SEEK_NODE;
    }

    return MACRO_IDLE;
}

float macro_prior(MacroAction macro) {
    switch (macro) {
        case MACRO_FACTORY_BUILD_WORKER: return 1.25F;
        case MACRO_FACTORY_BUILD_SCOUT: return 1.10F;
        case MACRO_FACTORY_BUILD_MINER: return 0.85F;
        case MACRO_FACTORY_SAFE_ADVANCE: return 1.00F;
        case MACRO_FACTORY_JUMP_OBSTACLE: return 0.90F;
        case MACRO_WORKER_OPEN_NORTH_WALL: return 1.00F;
        case MACRO_WORKER_ADVANCE: return 0.95F;
        case MACRO_WORKER_ESCORT_FACTORY: return 0.70F;
        case MACRO_SCOUT_HUNT_CRYSTAL: return 1.00F;
        case MACRO_SCOUT_EXPLORE_NORTH: return 0.85F;
        case MACRO_SCOUT_RETURN_ENERGY: return 0.75F;
        case MACRO_MINER_SEEK_NODE: return 0.95F;
        case MACRO_MINER_TRANSFORM: return 1.15F;
        case MACRO_IDLE:
        default: return 0.20F;
    }
}

float candidate_prior(const PlanCandidate& candidate) {
    if (candidate.plan_count <= 0) {
        return 0.10F;
    }
    float sum = 0.0F;
    for (int i = 0; i < candidate.plan_count; ++i) {
        sum += macro_prior(candidate.macro[static_cast<size_t>(i)]);
    }
    return std::max(0.05F, sum / static_cast<float>(candidate.plan_count));
}

int collect_controlled_robots(const BoardState& state, int owner,
                              std::array<int, MAX_MCTS_PLAN_ROBOTS>& controlled) {
    int count = 0;
    for (int i = 0; i < state.robots.used && count < MAX_MCTS_PLAN_ROBOTS; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] != 0 &&
            state.robots.owner[static_cast<size_t>(i)] == owner) {
            controlled[static_cast<size_t>(count++)] = i;
        }
    }
    return count;
}

int generate_candidates(const CrawlerSim& sim, int root_player,
                        std::array<PlanCandidate, MAX_MCTS_CANDIDATES>& candidates) {
    std::array<int, MAX_MCTS_PLAN_ROBOTS> controlled{};
    const int controlled_count = collect_controlled_robots(sim.state, root_player, controlled);
    if (controlled_count <= 0) {
        return 0;
    }

    PlanCandidate baseline{};
    baseline.plan_count = controlled_count;
    for (int i = 0; i < controlled_count; ++i) {
        const int robot_index = controlled[static_cast<size_t>(i)];
        detail::copy_uid(baseline.uid[static_cast<size_t>(i)],
                         sim.state.robots.uid[static_cast<size_t>(robot_index)].data());
        baseline.macro[static_cast<size_t>(i)] = default_macro_for_robot(sim.state, robot_index, root_player);
    }
    baseline.prior = candidate_prior(baseline) * 1.35F;

    int count = 0;
    candidates[static_cast<size_t>(count++)] = baseline;

    for (int robot_ordinal = 0; robot_ordinal < controlled_count && count < MAX_MCTS_CANDIDATES; ++robot_ordinal) {
        const int robot_index = controlled[static_cast<size_t>(robot_ordinal)];
        const MacroList macros = sim.generate_macros_for(robot_index);
        for (int m = 0; m < macros.count && count < MAX_MCTS_CANDIDATES; ++m) {
            const MacroAction macro = macros.macros[static_cast<size_t>(m)];
            if (macro == baseline.macro[static_cast<size_t>(robot_ordinal)]) {
                continue;
            }
            PlanCandidate candidate = baseline;
            candidate.macro[static_cast<size_t>(robot_ordinal)] = macro;
            candidate.prior = candidate_prior(candidate);
            candidates[static_cast<size_t>(count++)] = candidate;
        }
    }

    return count;
}

void copy_candidate_to_node(const PlanCandidate& candidate, MCTSNode& node) {
    node.plan_count = candidate.plan_count;
    for (int i = 0; i < candidate.plan_count; ++i) {
        detail::copy_uid(node.plan_uid[static_cast<size_t>(i)], candidate.uid[static_cast<size_t>(i)].data());
        node.plan_macro[static_cast<size_t>(i)] = candidate.macro[static_cast<size_t>(i)];
    }
}

void expand_node(MCTSArena& arena, int node_index, const CrawlerSim& sim, int root_player) {
    MCTSNode& parent = arena.nodes[static_cast<size_t>(node_index)];
    if (parent.expanded != 0) {
        return;
    }
    parent.expanded = 1;
    if (parent.depth >= MCTS_TREE_DEPTH || arena.used >= MAX_TREE_NODES) {
        return;
    }

    std::array<PlanCandidate, MAX_MCTS_CANDIDATES> candidates;
    const int candidate_count = generate_candidates(sim, root_player, candidates);
    if (candidate_count <= 0) {
        return;
    }

    float prior_sum = 0.0F;
    for (int i = 0; i < candidate_count; ++i) {
        prior_sum += std::max(0.0F, candidates[static_cast<size_t>(i)].prior);
    }
    if (prior_sum <= 0.0F) {
        prior_sum = static_cast<float>(candidate_count);
    }

    for (int i = 0; i < candidate_count && arena.used < MAX_TREE_NODES; ++i) {
        const float normalized_prior = std::max(0.0F, candidates[static_cast<size_t>(i)].prior) / prior_sum;
        const int child = arena.create_node(node_index, parent.depth + 1, normalized_prior);
        if (child < 0) {
            break;
        }
        copy_candidate_to_node(candidates[static_cast<size_t>(i)], arena.nodes[static_cast<size_t>(child)]);
        arena.nodes[static_cast<size_t>(child)].next_sibling = parent.first_child;
        parent.first_child = child;
        ++parent.child_count;
    }
}

int select_child(const MCTSArena& arena, int node_index) {
    const MCTSNode& parent = arena.nodes[static_cast<size_t>(node_index)];
    const float parent_sqrt = std::sqrt(static_cast<float>(parent.visits + 1));
    int best = -1;
    float best_score = -std::numeric_limits<float>::infinity();

    for (int child = parent.first_child; child >= 0;
         child = arena.nodes[static_cast<size_t>(child)].next_sibling) {
        const MCTSNode& node = arena.nodes[static_cast<size_t>(child)];
        const float q = node.visits > 0 ? node.value_sum / static_cast<float>(node.visits) : 0.0F;
        const float u = PUCT_C * node.prior * parent_sqrt / static_cast<float>(node.visits + 1);
        const float score = q + u;
        if (score > best_score) {
            best_score = score;
            best = child;
        }
    }

    return best;
}

int find_plan_entry(const MCTSNode& node, std::string_view uid) {
    for (int i = 0; i < node.plan_count; ++i) {
        if (detail::uid_equal(node.plan_uid[static_cast<size_t>(i)], uid)) {
            return i;
        }
    }
    return -1;
}

void fill_heuristic_actions(const CrawlerSim& sim, PrimitiveActions& actions) {
    actions.clear();
    for (int i = 0; i < sim.state.robots.used; ++i) {
        if (sim.state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const int owner = sim.state.robots.owner[static_cast<size_t>(i)];
        if (owner == 0 || owner == 1) {
            actions.actions[static_cast<size_t>(i)] = sim.heuristic_action_for_owner(i, owner);
        }
    }
}

void apply_node_plan(CrawlerSim& sim, const MCTSNode& node, int root_player) {
    PrimitiveActions actions{};
    fill_heuristic_actions(sim, actions);

    for (int i = 0; i < node.plan_count; ++i) {
        const int robot_index = sim.state.robots.find_uid(node.plan_uid[static_cast<size_t>(i)].data());
        if (robot_index < 0 ||
            sim.state.robots.alive[static_cast<size_t>(robot_index)] == 0 ||
            sim.state.robots.owner[static_cast<size_t>(robot_index)] != root_player) {
            continue;
        }
        actions.actions[static_cast<size_t>(robot_index)] =
            sim.primitive_for_macro(robot_index, node.plan_macro[static_cast<size_t>(i)]);
    }

    sim.step(actions);
}

float evaluate_state(const BoardState& state, int root_player) {
    const int opponent = 1 - root_player;
    EvalStats stats{};
    stats.best_factory_row = {state.south_bound - 1, state.south_bound - 1};

    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const int owner = state.robots.owner[static_cast<size_t>(i)];
        if (owner < 0 || owner > 1) {
            continue;
        }
        const uint8_t type = state.robots.type[static_cast<size_t>(i)];
        stats.energy[static_cast<size_t>(owner)] += state.robots.energy[static_cast<size_t>(i)];
        ++stats.units[static_cast<size_t>(owner)];
        stats.material[static_cast<size_t>(owner)] += type == FACTORY ? 8 : static_cast<int>(type);
        if (type == FACTORY) {
            ++stats.factories[static_cast<size_t>(owner)];
            stats.best_factory_row[static_cast<size_t>(owner)] =
                std::max(stats.best_factory_row[static_cast<size_t>(owner)],
                         static_cast<int>(state.robots.row[static_cast<size_t>(i)]));
        }
    }

    const int64_t energy_diff = stats.energy[static_cast<size_t>(root_player)] -
                                stats.energy[static_cast<size_t>(opponent)];
    const int unit_diff = stats.units[static_cast<size_t>(root_player)] -
                          stats.units[static_cast<size_t>(opponent)];
    const bool root_dead = stats.factories[static_cast<size_t>(root_player)] == 0;
    const bool opponent_dead = stats.factories[static_cast<size_t>(opponent)] == 0;
    const bool tiebreak_terminal = (state.done && root_dead == opponent_dead) || state.step >= EPISODE_STEPS - 1;
    if (tiebreak_terminal) {
        if (energy_diff != 0) {
            return std::clamp(std::tanh(static_cast<float>(energy_diff) / 800.0F), -1.0F, 1.0F);
        }
        if (unit_diff != 0) {
            return std::clamp(std::tanh(static_cast<float>(unit_diff) / 4.0F), -1.0F, 1.0F);
        }
        return 0.0F;
    }

    if (root_dead) {
        return -1.0F;
    }
    if (opponent_dead) {
        return 1.0F;
    }

    const float energy_score =
        std::tanh(static_cast<float>(energy_diff) / 1000.0F);
    const float material_score =
        std::tanh(static_cast<float>(stats.material[static_cast<size_t>(root_player)] -
                                     stats.material[static_cast<size_t>(opponent)]) / 10.0F);
    const float unit_score = std::tanh(static_cast<float>(unit_diff) / 8.0F);
    const float progress_score =
        std::tanh(static_cast<float>(stats.best_factory_row[static_cast<size_t>(root_player)] -
                                     stats.best_factory_row[static_cast<size_t>(opponent)]) / 8.0F);
    const float margin_score =
        std::tanh(static_cast<float>((stats.best_factory_row[static_cast<size_t>(root_player)] - state.south_bound) -
                                     (stats.best_factory_row[static_cast<size_t>(opponent)] - state.south_bound)) / 8.0F);

    return std::clamp(0.55F * energy_score + 0.20F * material_score + 0.10F * unit_score +
                          0.10F * progress_score + 0.05F * margin_score,
                      -1.0F, 1.0F);
}

float rollout(CrawlerSim& sim, int root_player) {
    PrimitiveActions actions{};
    for (int depth = 0; depth < MCTS_ROLLOUT_DEPTH && !sim.state.done; ++depth) {
        fill_heuristic_actions(sim, actions);
        sim.step(actions);
    }
    return evaluate_state(sim.state, root_player);
}

void backpropagate(MCTSArena& arena, const std::array<int, MCTS_TREE_DEPTH + 2>& path,
                   int path_count, float value) {
    for (int i = 0; i < path_count; ++i) {
        MCTSNode& node = arena.nodes[static_cast<size_t>(path[static_cast<size_t>(i)])];
        ++node.visits;
        node.value_sum += value;
    }
}

int best_root_child(const MCTSArena& arena, int root) {
    const MCTSNode& root_node = arena.nodes[static_cast<size_t>(root)];
    int best = -1;
    int best_visits = 0;
    float best_value = -std::numeric_limits<float>::infinity();

    for (int child = root_node.first_child; child >= 0;
         child = arena.nodes[static_cast<size_t>(child)].next_sibling) {
        const MCTSNode& node = arena.nodes[static_cast<size_t>(child)];
        if (node.visits <= 0) {
            continue;
        }
        const float value = node.value_sum / static_cast<float>(node.visits);
        if (node.visits > best_visits || (node.visits == best_visits && value > best_value)) {
            best_visits = node.visits;
            best_value = value;
            best = child;
        }
    }

    return best;
}

ActionResult build_result_from_plan(const CrawlerSim& sim, const MCTSNode* node) {
    ActionResult result{};
    result.clear();
    for (int i = 0; i < sim.state.robots.used; ++i) {
        if (sim.state.robots.alive[static_cast<size_t>(i)] == 0 ||
            sim.state.robots.owner[static_cast<size_t>(i)] != sim.state.player) {
            continue;
        }

        Action action = sim.heuristic_action_for(i);
        if (node != nullptr) {
            const int plan_entry = find_plan_entry(*node, sim.state.robots.uid[static_cast<size_t>(i)].data());
            if (plan_entry >= 0) {
                action = sim.primitive_for_macro(i, node->plan_macro[static_cast<size_t>(plan_entry)]);
            }
        }
        result.add(sim.state.robots.uid[static_cast<size_t>(i)].data(), action);
    }
    return result;
}

}  // namespace

void MCTSArena::reset() {
    used = 0;
}

int MCTSArena::create_node(int parent, int depth, float prior) {
    if (used >= MAX_TREE_NODES) {
        return -1;
    }
    const int index = used++;
    MCTSNode& node = nodes[static_cast<size_t>(index)];
    node.parent = parent;
    node.first_child = -1;
    node.next_sibling = -1;
    node.child_count = 0;
    node.visits = 0;
    node.depth = depth;
    node.plan_count = 0;
    node.value_sum = 0.0F;
    node.prior = prior;
    node.expanded = 0;
    return index;
}

ActionResult Engine::choose_actions(int time_budget_ms, uint64_t seed) {
    bool has_controlled_robot = false;
    for (int i = 0; i < sim.state.robots.used; ++i) {
        if (sim.state.robots.alive[static_cast<size_t>(i)] != 0 &&
            sim.state.robots.owner[static_cast<size_t>(i)] == sim.state.player) {
            has_controlled_robot = true;
            break;
        }
    }
    if (!has_controlled_robot) {
        return build_result_from_plan(sim, nullptr);
    }

    if (time_budget_ms <= 0) {
        return build_result_from_plan(sim, nullptr);
    }

    mcts.reset();
    const int root = mcts.create_node(-1, 0, 1.0F);
    if (root < 0) {
        return build_result_from_plan(sim, nullptr);
    }

    const auto start = std::chrono::steady_clock::now();
    const int guard_ms = time_budget_ms >= 10 ? 2 : 0;
    const int run_ms = std::max(1, time_budget_ms - guard_ms);
    const auto deadline = start + std::chrono::milliseconds(run_ms);
    const int clock_check_interval = time_budget_ms <= 20 ? 1 : 8;

    int iterations = 0;
    while (mcts.used < MAX_TREE_NODES) {
        if ((iterations % clock_check_interval) == 0 && std::chrono::steady_clock::now() >= deadline) {
            break;
        }

        CrawlerSim search_sim{};
        search_sim.state = determinize(detail::mix64(seed ^ (static_cast<uint64_t>(iterations + 1) * ITERATION_SEED)));

        std::array<int, MCTS_TREE_DEPTH + 2> path{};
        int path_count = 0;
        int node = root;
        path[static_cast<size_t>(path_count++)] = root;

        while (!search_sim.state.done) {
            MCTSNode& current = mcts.nodes[static_cast<size_t>(node)];
            if (current.depth >= MCTS_TREE_DEPTH) {
                break;
            }
            if (current.expanded == 0) {
                expand_node(mcts, node, search_sim, sim.state.player);
            }
            if (current.child_count <= 0) {
                break;
            }

            const int child = select_child(mcts, node);
            if (child < 0) {
                break;
            }
            apply_node_plan(search_sim, mcts.nodes[static_cast<size_t>(child)], sim.state.player);
            node = child;
            path[static_cast<size_t>(path_count++)] = node;

            if (mcts.nodes[static_cast<size_t>(node)].visits == 0 ||
                path_count >= static_cast<int>(path.size())) {
                break;
            }
        }

        const float value = rollout(search_sim, sim.state.player);
        backpropagate(mcts, path, path_count, value);
        ++iterations;
    }

    const int best_child = best_root_child(mcts, root);
    if (best_child < 0) {
        return build_result_from_plan(sim, nullptr);
    }
    return build_result_from_plan(sim, &mcts.nodes[static_cast<size_t>(best_child)]);
}

float Engine::debug_mcts_value(int player) const {
    const int eval_player = (player == 0 || player == 1) ? player : sim.state.player;
    return evaluate_state(sim.state, eval_player);
}

}  // namespace crawler
