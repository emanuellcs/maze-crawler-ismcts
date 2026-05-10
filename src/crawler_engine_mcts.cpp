#include "crawler_engine_internal.hpp"

/**
 * @file crawler_engine_mcts.cpp
 * @brief Fixed-arena Information Set Monte Carlo Tree Search over joint macro plans.
 *
 * The tree is an information-set search: each iteration samples a concrete
 * BoardState from the Belief State, replays the selected UID-keyed macro history
 * through that Determinization, performs deterministic Rollouts, and
 * backpropagates a root-player value.  Expansion is bounded by a deterministic
 * baseline joint plan plus one-robot MacroAction deviations, avoiding the
 * primitive simultaneous-action Cartesian product.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>

namespace crawler {
namespace {

constexpr uint64_t ITERATION_SEED = 0x9e3779b97f4a7c15ULL;

/**
 * @brief Candidate joint macro plan considered during node expansion.
 *
 * Each plan stores controlled robot UIDs and macro intents rather than local
 * robot indices.  That keeps the action history replayable across sampled
 * Determinizations where hidden robots and slot ordering may differ.
 */
struct PlanCandidate {
    int plan_count = 0;
    float prior = 1.0F;
    std::array<std::array<char, UID_LEN>, MAX_MCTS_PLAN_ROBOTS> uid{};
    std::array<MacroAction, MAX_MCTS_PLAN_ROBOTS> macro{};
};

/**
 * @brief Aggregate material features used by the rollout evaluator.
 */
struct EvalStats {
    std::array<int64_t, 2> energy{0, 0};
    std::array<int, 2> units{0, 0};
    std::array<int, 2> material{0, 0};
    std::array<int, 2> factories{0, 0};
    std::array<int, 2> best_factory_row{0, 0};
};

/**
 * @brief Compute an unnormalized prior for a joint macro candidate.
 * @param hyperparameters Runtime search hyperparameters.
 * @param candidate Candidate joint macro plan.
 * @return Positive prior mass before expansion-level normalization.
 *
 * The prior is the arithmetic mean of participating per-macro priors.  Expansion
 * later normalizes all candidates so only relative scale matters; this lets
 * tuning adjust strategic preference without changing tree code.
 */
float candidate_prior(const Hyperparameters& hyperparameters, const PlanCandidate& candidate) {
    if (candidate.plan_count <= 0) {
        return 0.10F;
    }
    float sum = 0.0F;
    for (int i = 0; i < candidate.plan_count; ++i) {
        sum += hyperparameters.prior_for(candidate.macro[static_cast<size_t>(i)]);
    }
    return std::max(0.05F, sum / static_cast<float>(candidate.plan_count));
}

/**
 * @brief Collect controlled live robot slots up to the MCTS plan cap.
 * @param state Concrete board state.
 * @param owner Player whose robots should be controlled.
 * @param controlled Output slot list.
 * @return Number of collected robots.
 */
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

/**
 * @brief Generate bounded joint macro candidates for one expansion.
 * @param sim Concrete simulator state at the node.
 * @param root_player Player controlled by the search.
 * @param hyperparameters Runtime prior weights.
 * @param candidates Output candidate buffer.
 * @return Number of candidates written.
 *
 * A naive simultaneous-move tree branches as `product(|A_r|)` over controlled
 * robots.  This generator instead emits one deterministic all-robot baseline
 * and one-robot deviations from that baseline, capped by
 * `MAX_MCTS_CANDIDATES`.  The result is approximately linear branching while
 * preserving coordinated rollout behavior for robots not under deviation.
 */
int generate_candidates(const CrawlerSim& sim, int root_player, const Hyperparameters& hyperparameters,
                        std::array<PlanCandidate, MAX_MCTS_CANDIDATES>& candidates) {
    std::array<int, MAX_MCTS_PLAN_ROBOTS> controlled{};
    const int controlled_count = collect_controlled_robots(sim.state, root_player, controlled);
    if (controlled_count <= 0) {
        return 0;
    }

    PrimitiveActions baseline_actions{};
    baseline_actions.clear();
    std::array<MacroAction, MAX_ROBOTS> baseline_macros{};
    baseline_macros.fill(MACRO_IDLE);
    sim.fill_heuristic_plan_for_owner(root_player, baseline_actions, &baseline_macros);

    PlanCandidate baseline{};
    baseline.plan_count = controlled_count;
    for (int i = 0; i < controlled_count; ++i) {
        const int robot_index = controlled[static_cast<size_t>(i)];
        detail::copy_uid(baseline.uid[static_cast<size_t>(i)],
                         sim.state.robots.uid[static_cast<size_t>(robot_index)].data());
        baseline.macro[static_cast<size_t>(i)] = baseline_macros[static_cast<size_t>(robot_index)];
    }
    baseline.prior = candidate_prior(hyperparameters, baseline) * hyperparameters.baseline_prior_multiplier;

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
            candidate.prior = candidate_prior(hyperparameters, candidate);
            candidates[static_cast<size_t>(count++)] = candidate;
        }
    }

    float strongest_deviation = 0.0F;
    for (int i = 1; i < count; ++i) {
        strongest_deviation = std::max(strongest_deviation, candidates[static_cast<size_t>(i)].prior);
    }
    if (count > 1 && candidates[0].prior <= strongest_deviation) {
        /* Ensure the strong deterministic baseline remains selectable even when one tuned deviation has higher prior. */
        candidates[0].prior = strongest_deviation + std::max(0.001F, strongest_deviation * 0.001F);
    }

    return count;
}

/**
 * @brief Copy one candidate plan into a persistent MCTS node edge.
 * @param candidate Candidate generated during expansion.
 * @param node Destination tree node.
 */
void copy_candidate_to_node(const PlanCandidate& candidate, MCTSNode& node) {
    node.plan_count = candidate.plan_count;
    for (int i = 0; i < candidate.plan_count; ++i) {
        detail::copy_uid(node.plan_uid[static_cast<size_t>(i)], candidate.uid[static_cast<size_t>(i)].data());
        node.plan_macro[static_cast<size_t>(i)] = candidate.macro[static_cast<size_t>(i)];
    }
}

/**
 * @brief Expand a tree node with normalized joint macro children.
 * @param arena Fixed node arena.
 * @param node_index Node to expand.
 * @param sim Concrete simulator state corresponding to the node.
 * @param root_player Search player.
 * @param hyperparameters Runtime search parameters.
 *
 * Child priors are normalized over the candidate set and stored for PUCT.  The
 * sibling list is singly linked inside the arena, avoiding per-node dynamic
 * containers in the search hot path.
 */
void expand_node(MCTSArena& arena, int node_index, const CrawlerSim& sim, int root_player,
                 const Hyperparameters& hyperparameters) {
    MCTSNode& parent = arena.nodes[static_cast<size_t>(node_index)];
    if (parent.expanded != 0) {
        return;
    }
    parent.expanded = 1;
    if (parent.depth >= MCTS_TREE_DEPTH || arena.used >= MAX_TREE_NODES) {
        return;
    }

    std::array<PlanCandidate, MAX_MCTS_CANDIDATES> candidates;
    const int candidate_count = generate_candidates(sim, root_player, hyperparameters, candidates);
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

/**
 * @brief Select the highest-scoring child under the PUCT rule.
 * @param arena Fixed node arena.
 * @param node_index Parent node index.
 * @param hyperparameters Runtime search parameters containing `C_puct`.
 * @return Selected child index, or `-1` if no child exists.
 *
 * Selection score is `Q + U`, where `Q = value_sum / visits` and
 * `U = C_puct * prior * sqrt(parent_visits + 1) / (child_visits + 1)`.
 * The prior term accelerates promising macro intents early; the visit divisor
 * shifts pressure toward underexplored children as evidence accumulates.
 */
int select_child(const MCTSArena& arena, int node_index, const Hyperparameters& hyperparameters) {
    const MCTSNode& parent = arena.nodes[static_cast<size_t>(node_index)];
    const float parent_sqrt = std::sqrt(static_cast<float>(parent.visits + 1));
    int best = -1;
    float best_score = -std::numeric_limits<float>::infinity();

    for (int child = parent.first_child; child >= 0;
         child = arena.nodes[static_cast<size_t>(child)].next_sibling) {
        const MCTSNode& node = arena.nodes[static_cast<size_t>(child)];
        const float q = node.visits > 0 ? node.value_sum / static_cast<float>(node.visits) : 0.0F;
        const float u = hyperparameters.C_puct * node.prior * parent_sqrt / static_cast<float>(node.visits + 1);
        const float score = q + u;
        if (score > best_score) {
            best_score = score;
            best = child;
        }
    }

    return best;
}

/**
 * @brief Fill deterministic baseline actions for both players.
 * @param sim Concrete simulator state.
 * @param actions Output primitive action buffer.
 */
void fill_heuristic_actions(const CrawlerSim& sim, PrimitiveActions& actions) {
    actions.clear();
    sim.fill_heuristic_plan_for_owner(0, actions, nullptr);
    sim.fill_heuristic_plan_for_owner(1, actions, nullptr);
}

/**
 * @brief Apply one MCTS node's UID-keyed macro plan to a concrete simulator.
 * @param sim Concrete simulator to mutate by one step.
 * @param node Tree node whose edge plan should be played.
 * @param root_player Search-controlled player.
 *
 * Robots not explicitly changed by the node keep their deterministic heuristic
 * action.  This preserves coordinated baseline behavior while allowing the tree
 * to evaluate one-robot deviations.
 */
void apply_node_plan(CrawlerSim& sim, const MCTSNode& node, int root_player) {
    PrimitiveActions actions{};
    actions.clear();
    std::array<MacroAction, MAX_ROBOTS> baseline_macros{};
    baseline_macros.fill(MACRO_IDLE);
    if (root_player == 0) {
        sim.fill_heuristic_plan_for_owner(0, actions, &baseline_macros);
        sim.fill_heuristic_plan_for_owner(1, actions, nullptr);
    } else {
        sim.fill_heuristic_plan_for_owner(0, actions, nullptr);
        sim.fill_heuristic_plan_for_owner(1, actions, &baseline_macros);
    }

    for (int i = 0; i < node.plan_count; ++i) {
        const int robot_index = sim.state.robots.find_uid(node.plan_uid[static_cast<size_t>(i)].data());
        if (robot_index < 0 ||
            sim.state.robots.alive[static_cast<size_t>(robot_index)] == 0 ||
            sim.state.robots.owner[static_cast<size_t>(robot_index)] != root_player) {
            continue;
        }
        if (node.plan_macro[static_cast<size_t>(i)] == baseline_macros[static_cast<size_t>(robot_index)]) {
            continue;
        }
        actions.actions[static_cast<size_t>(robot_index)] =
            sim.primitive_for_macro(robot_index, node.plan_macro[static_cast<size_t>(i)]);
    }

    sim.step(actions);
}

/**
 * @brief Evaluate a concrete state from the root player's perspective.
 * @param state Concrete board state.
 * @param root_player Search-controlled player.
 * @return Smooth value in `[-1, 1]`.
 *
 * Terminal/tiebreak states follow the competition priorities: Factory survival,
 * energy, then unit count.  Non-terminal states blend energy, material, unit
 * count, Factory progress, and scroll-margin features to give Rollouts a
 * continuous gradient before the true terminal horizon.
 */
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

/**
 * @brief Roll a sampled state forward with deterministic heuristic policy.
 * @param sim Concrete simulator state to advance.
 * @param root_player Search-controlled player.
 * @param rollout_depth Maximum rollout horizon.
 * @return Evaluator value after rollout termination or horizon.
 *
 * Stochasticity enters through Determinization, not through the playout policy.
 * Keeping Rollouts deterministic reduces variance and makes visit statistics
 * reflect hidden-state samples rather than random action noise.
 */
float rollout(CrawlerSim& sim, int root_player, int rollout_depth) {
    PrimitiveActions actions{};
    for (int depth = 0; depth < rollout_depth && !sim.state.done; ++depth) {
        fill_heuristic_actions(sim, actions);
        sim.step(actions);
    }
    return evaluate_state(sim.state, root_player);
}

/**
 * @brief Backpropagate one rollout value over the selected tree path.
 * @param arena Fixed node arena.
 * @param path Node indices visited in this iteration.
 * @param path_count Number of valid path entries.
 * @param value Root-player rollout value.
 */
void backpropagate(MCTSArena& arena, const std::array<int, MCTS_TREE_DEPTH + 2>& path,
                   int path_count, float value) {
    for (int i = 0; i < path_count; ++i) {
        MCTSNode& node = arena.nodes[static_cast<size_t>(path[static_cast<size_t>(i)])];
        ++node.visits;
        node.value_sum += value;
    }
}

/**
 * @brief Choose the most robust root child after search.
 * @param arena Fixed node arena.
 * @param root Root node index.
 * @return Child with most visits, breaking ties by mean value.
 */
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

/**
 * @brief Build Python-facing primitive actions from a selected root plan.
 * @param sim Current concrete simulator snapshot.
 * @param node Selected root child, or nullptr for deterministic baseline.
 * @return Fixed-buffer UID/action result.
 */
ActionResult build_result_from_plan(const CrawlerSim& sim, const MCTSNode* node) {
    ActionResult result{};
    result.clear();
    PrimitiveActions actions{};
    actions.clear();
    std::array<MacroAction, MAX_ROBOTS> baseline_macros{};
    baseline_macros.fill(MACRO_IDLE);
    sim.fill_heuristic_plan_for_owner(sim.state.player, actions, &baseline_macros);

    if (node != nullptr) {
        for (int i = 0; i < node->plan_count; ++i) {
            const int robot_index = sim.state.robots.find_uid(node->plan_uid[static_cast<size_t>(i)].data());
            if (robot_index < 0 ||
                sim.state.robots.alive[static_cast<size_t>(robot_index)] == 0 ||
                sim.state.robots.owner[static_cast<size_t>(robot_index)] != sim.state.player ||
                node->plan_macro[static_cast<size_t>(i)] == baseline_macros[static_cast<size_t>(robot_index)]) {
                continue;
            }
            actions.actions[static_cast<size_t>(robot_index)] =
                sim.primitive_for_macro(robot_index, node->plan_macro[static_cast<size_t>(i)]);
        }
    }

    for (int i = 0; i < sim.state.robots.used; ++i) {
        if (sim.state.robots.alive[static_cast<size_t>(i)] == 0 ||
            sim.state.robots.owner[static_cast<size_t>(i)] != sim.state.player) {
            continue;
        }
        result.add(sim.state.robots.uid[static_cast<size_t>(i)].data(),
                   actions.actions[static_cast<size_t>(i)]);
    }
    return result;
}

}  // namespace

/**
 * @brief Rewind the fixed node arena for a new turn.
 */
void MCTSArena::reset() {
    used = 0;
}

/**
 * @brief Allocate and initialize a node from the fixed arena.
 * @param parent Parent node index, or `-1` for root.
 * @param depth Tree depth.
 * @param prior Normalized PUCT prior.
 * @return Node index, or `-1` if the arena is full.
 */
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

/**
 * @brief Select actions for the current player under a time budget.
 * @param time_budget_ms Milliseconds available to search.
 * @param seed Root seed for deterministic per-iteration samples.
 * @return UID/action buffer for all controlled live robots.
 *
 * The loop stops when either the deadline is reached or the fixed arena is full.
 * Each iteration samples a Determinization from belief, traverses/expands the
 * information-set tree with PUCT, rolls out deterministic policy, and
 * backpropagates the root-player value.
 */
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
        /*
         * The seed sequence combines the caller seed with an iteration constant
         * so consecutive samples explore different hidden worlds reproducibly.
         */
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
                expand_node(mcts, node, search_sim, sim.state.player, hyperparameters);
            }
            if (current.child_count <= 0) {
                break;
            }

            const int child = select_child(mcts, node, hyperparameters);
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

        const float value = rollout(search_sim, sim.state.player, hyperparameters.rollout_depth);
        backpropagate(mcts, path, path_count, value);
        ++iterations;
    }

    const int best_child = best_root_child(mcts, root);
    if (best_child < 0) {
        return build_result_from_plan(sim, nullptr);
    }
    return build_result_from_plan(sim, &mcts.nodes[static_cast<size_t>(best_child)]);
}

/**
 * @brief Return the rollout evaluator value for the current simulator snapshot.
 * @param player Perspective player, or invalid value to use the engine player.
 * @return Smooth evaluator value in `[-1, 1]`.
 */
float Engine::debug_mcts_value(int player) const {
    const int eval_player = (player == 0 || player == 1) ? player : sim.state.player;
    return evaluate_state(sim.state, eval_player);
}

}  // namespace crawler
