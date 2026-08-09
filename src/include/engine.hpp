#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "board.hpp"
#include "constants.hpp"
#include "enums.hpp"
#include "observation.hpp"
#include "search.hpp"

namespace crawler {

/**
 * @brief Player-centric Belief State for fog-of-war memory and Determinization.
 *
 * Visible facts overwrite remembered state.  Walls and mines persist because
 * the environment remembers them; crystals and enemy robots outside vision do
 * not.  Hidden enemies are represented as per-type probability fields that
 * diffuse through passable or unknown cells, then each ISMCTS iteration samples
 * a concrete BoardState from those distributions.
 */
struct BeliefState {
    int player = 0;
    int turn = 0;
    int south_bound = 0;
    int north_bound = HEIGHT - 1;
    std::array<uint8_t, MAX_CELLS> known_wall{};
    std::array<uint8_t, MAX_CELLS> wall{};
    std::array<int16_t, MAX_CELLS> visible_crystal{};
    std::array<int16_t, MAX_CELLS> remembered_mine_energy{};
    std::array<int16_t, MAX_CELLS> remembered_mine_max{};
    std::array<int8_t, MAX_CELLS> remembered_mine_owner{};
    std::array<uint8_t, MAX_CELLS> remembered_node{};
    std::array<uint8_t, MAX_CELLS> currently_visible{};
    std::array<std::array<float, MAX_CELLS>, 4> enemy_prob{};

    /** @brief Clear all remembered facts and enemy probability fields. */
    void reset();

    /**
     * @brief Merge a new observation into the player-centric belief.
     * @param obs Fixed-buffer observation decoded by pybind.
     */
    void update_from_observation(const ObservationInput& obs);

    /**
     * @brief Sample one concrete hidden world consistent with current belief.
     * @param seed Deterministic seed used for unknown rows and enemy sampling.
     * @return Concrete BoardState suitable for simulation and Rollouts.
     */
    [[nodiscard]] BoardState determinize(uint64_t seed) const;
};

/**
 * @brief Deterministic Maze Crawler rules simulator.
 *
 * `step()` is the hot path used by both real-turn debug stepping and ISMCTS
 * Rollouts.  It resolves the rulebook phase order using fixed-size scratch
 * buffers only, so repeated simulations do not allocate or depend on Python.
 */
class CrawlerSim {
public:
    BoardState state{};

    /** @brief Reset the concrete simulator state. */
    void reset();

    /**
     * @brief Load a concrete visible snapshot from observation and belief memory.
     * @param obs Latest fixed-buffer observation.
     * @param belief Player-centric remembered facts.
     */
    void load_from_observation(const ObservationInput& obs, const BeliefState& belief);

    /**
     * @brief Advance the simulator by one rule-faithful turn.
     * @param actions Primitive actions addressed by simulator-local robot slots.
     */
    void step(const PrimitiveActions& actions);

    /**
     * @brief Fill a full deterministic baseline plan for one owner.
     * @param owner Player index to control.
     * @param actions Mutable primitive action buffer to fill in-place.
     * @param baseline_macros Optional macro labels parallel to robot slots.
     */
    void fill_heuristic_plan_for_owner(int owner, PrimitiveActions& actions,
                                       std::array<MacroAction, MAX_ROBOTS>* baseline_macros = nullptr) const;

    /**
     * @brief Generate the bounded macro intent list for one robot.
     * @param robot_index Simulator-local robot slot.
     * @return Fixed-capacity MacroList for ISMCTS expansion.
     */
    [[nodiscard]] MacroList generate_macros_for(int robot_index) const;

    /**
     * @brief Translate one macro intent into a legal primitive action when possible.
     * @param robot_index Simulator-local robot slot.
     * @param macro Macro intent selected by ISMCTS.
     * @return Primitive action, or `ACT_IDLE` when the macro is not currently legal.
     */
    [[nodiscard]] Action primitive_for_macro(int robot_index, MacroAction macro) const;
};

/**
 * @brief High-level facade owned by the Python binding.
 *
 * Engine owns the Belief State, latest concrete simulator snapshot, search
 * arena, and tunable hyperparameters for one player.  Python calls update once
 * per Kaggle observation and then asks the engine to choose actions under a
 * millisecond budget.
 */
class Engine {
public:
    BeliefState belief{};
    CrawlerSim sim{};
    MCTSArena mcts{};
    Hyperparameters hyperparameters{};

    /** @brief Construct a player-zero engine with empty belief and simulator state. */
    Engine();

    /**
     * @brief Construct an engine for a specific player.
     * @param player Player index in `{0, 1}`.
     */
    explicit Engine(int player);

    /**
     * @brief Update belief and concrete simulator state from a new observation.
     * @param obs Fixed-buffer observation decoded by pybind.
     */
    void update_observation(const ObservationInput& obs);

    /**
     * @brief Apply primitive actions directly to the current simulator snapshot.
     * @param actions Primitive action buffer keyed by simulator-local robot slot.
     */
    void step_actions(const PrimitiveActions& actions);

    /**
     * @brief Sample one concrete hidden world while preserving observed live robots.
     * @param seed Deterministic seed for hidden rows and enemy locations.
     * @return BoardState suitable for one ISMCTS iteration.
     */
    [[nodiscard]] BoardState determinize(uint64_t seed) const;

    /**
     * @brief Set the maximum number of search threads.
     * @param n Maximum number of threads (must be >= 1).
     */
    void set_search_thread_limit(int n);

    /**
     * @brief Choose Kaggle-compatible actions via fixed-arena ISMCTS.
     * @param time_budget_ms Per-turn budget in milliseconds.
     * @param seed Deterministic root seed for reproducible Determinizations.
     * @return UID/action result buffer for controlled live robots.
     */
    [[nodiscard]] ActionResult choose_actions(int time_budget_ms, uint64_t seed);

    /**
     * @brief Evaluate the current simulator snapshot with the rollout value model.
     * @param player Player perspective, or invalid value to use `sim.state.player`.
     * @return Value in approximately `[-1, 1]`.
     */
    [[nodiscard]] float debug_mcts_value(int player) const;
};

}  // namespace crawler
