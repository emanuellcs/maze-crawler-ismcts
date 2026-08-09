#pragma once

#include <array>
#include <cstdint>

#include "board.hpp"
#include "constants.hpp"
#include "enums.hpp"

namespace crawler {

/**
 * @brief Runtime-tunable ISMCTS and rollout hyperparameters.
 *
 * Defaults mirror the best Optuna study results embedded in `main.py`.  The
 * pybind layer validates and injects these values per Engine instance, allowing
 * tuning without recompiling the native extension.
 */
struct Hyperparameters {
    /** @brief Number of parallel search threads to run. */
    int search_threads = 1;
    /** @brief Exploration strength in PUCT child selection. */
    float C_puct = 2.0884330868271443F;
    /** @brief Extra prior mass assigned to the deterministic all-robot baseline plan. */
    float baseline_prior_multiplier = 1.8863044112273712F;
    /** @brief Heuristic rollout horizon after tree traversal. */
    int rollout_depth = 80;
    /** @brief MacroAction-indexed prior weights; unused slots keep conservative defaults. */
    std::array<float, MAX_MACROS> macro_prior{};

    /** @brief Construct with tuned macro priors. */
    Hyperparameters();

    /** @brief Restore the embedded tuned macro prior vector. */
    void reset_macro_priors();

    /**
     * @brief Read the prior weight for a macro action.
     * @param macro MacroAction to score.
     * @return Positive finite prior weight, falling back to idle-safe default.
     */
    [[nodiscard]] float prior_for(MacroAction macro) const;
};

/**
 * @brief Fixed-arena ISMCTS tree node storing one UID-keyed joint macro plan.
 *
 * Children represent action-history information sets.  Plans are keyed by UID,
 * not simulator-local slot, because each Determinization may recreate hidden
 * robots and reorder slots while the externally observed controlled robots keep
 * stable identities.
 */
struct MCTSNode {
    int parent = -1;
    int first_child = -1;
    int next_sibling = -1;
    int child_count = 0;
    int visits = 0;
    int depth = 0;
    int plan_count = 0;
    float value_sum = 0.0F;
    float prior = 0.0F;
    uint8_t expanded = 0;
    std::array<std::array<char, UID_LEN>, MAX_MCTS_PLAN_ROBOTS> plan_uid{};
    std::array<MacroAction, MAX_MCTS_PLAN_ROBOTS> plan_macro{};
};

/**
 * @brief Zero-allocation node arena for one turn of ISMCTS.
 *
 * Resetting rewinds `used` only; the backing node array remains allocated in
 * the Engine object and is overwritten by subsequent searches.
 */
struct MCTSArena {
    std::array<MCTSNode, MAX_TREE_NODES> nodes{};
    int used = 0;

    /** @brief Rewind the arena for a new search without clearing storage. */
    void reset();

    /**
     * @brief Allocate and initialize one tree node from the fixed arena.
     * @param parent Parent node index, or `-1` for root.
     * @param depth Tree depth from root.
     * @param prior Normalized PUCT prior for this edge.
     * @return Node index, or `-1` when the arena is full.
     */
    [[nodiscard]] int create_node(int parent, int depth, float prior);
};

}  // namespace crawler
