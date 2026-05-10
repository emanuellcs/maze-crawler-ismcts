#include "crawler_engine.hpp"

/**
 * @file crawler_engine_engine.cpp
 * @brief High-level Engine facade wiring belief, simulation, and ISMCTS search.
 *
 * The Python bridge owns one Engine per player.  This module keeps that facade
 * intentionally thin: observations update the Belief State first, then rebuild
 * the latest concrete simulator snapshot, and action selection samples hidden
 * worlds from that belief through fixed-arena ISMCTS.
 */

namespace crawler {

/**
 * @brief Construct hyperparameters with tuned macro priors.
 */
Hyperparameters::Hyperparameters() {
    reset_macro_priors();
}

/**
 * @brief Restore embedded Optuna-tuned priors for every macro action.
 *
 * The values bias PUCT toward the deterministic strategic baseline while still
 * allowing Rollouts to overturn the prior through visit/value statistics.
 */
void Hyperparameters::reset_macro_priors() {
    macro_prior.fill(0.20F);
    macro_prior[static_cast<size_t>(MACRO_IDLE)] = 0.49504719444108913F;
    macro_prior[static_cast<size_t>(MACRO_FACTORY_SUPPORT_WORKER)] = 1.0390992283842135F;
    macro_prior[static_cast<size_t>(MACRO_FACTORY_SAFE_ADVANCE)] = 2.161864767112469F;
    macro_prior[static_cast<size_t>(MACRO_FACTORY_BUILD_WORKER)] = 0.997532683560502F;
    macro_prior[static_cast<size_t>(MACRO_FACTORY_BUILD_SCOUT)] = 1.8649154683704814F;
    macro_prior[static_cast<size_t>(MACRO_FACTORY_BUILD_MINER)] = 0.8269752415957998F;
    macro_prior[static_cast<size_t>(MACRO_FACTORY_JUMP_OBSTACLE)] = 1.4925105256980329F;
    macro_prior[static_cast<size_t>(MACRO_WORKER_OPEN_NORTH_WALL)] = 0.7351332485632837F;
    macro_prior[static_cast<size_t>(MACRO_WORKER_ESCORT_FACTORY)] = 1.583596636264722F;
    macro_prior[static_cast<size_t>(MACRO_WORKER_ADVANCE)] = 0.8874633406271473F;
    macro_prior[static_cast<size_t>(MACRO_SCOUT_HUNT_CRYSTAL)] = 1.083268581072123F;
    macro_prior[static_cast<size_t>(MACRO_SCOUT_EXPLORE_NORTH)] = 1.6851712172373043F;
    macro_prior[static_cast<size_t>(MACRO_SCOUT_RETURN_ENERGY)] = 1.1040824358243229F;
    macro_prior[static_cast<size_t>(MACRO_MINER_SEEK_NODE)] = 0.6983213636231111F;
    macro_prior[static_cast<size_t>(MACRO_MINER_TRANSFORM)] = 0.5309500821275892F;
}

/**
 * @brief Read a macro prior with a conservative fallback.
 * @param macro MacroAction index.
 * @return Positive prior used by joint-plan expansion.
 */
float Hyperparameters::prior_for(MacroAction macro) const {
    const size_t index = static_cast<size_t>(macro);
    if (index >= macro_prior.size()) {
        return 0.20F;
    }
    return macro_prior[index];
}

/**
 * @brief Construct an empty player-zero engine.
 */
Engine::Engine() {
    belief.reset();
    sim.reset();
}

/**
 * @brief Construct an engine and set the controlled player.
 * @param player Player index in `{0, 1}`.
 */
Engine::Engine(int player) : Engine() {
    belief.player = player;
    sim.state.player = player;
}

/**
 * @brief Merge a new observation and rebuild the concrete simulator snapshot.
 * @param obs Fixed-buffer observation decoded by pybind.
 *
 * Belief is updated first because CrawlerSim needs remembered walls, mines, and
 * fog-derived facts in addition to currently visible observation data.
 */
void Engine::update_observation(const ObservationInput& obs) {
    belief.update_from_observation(obs);
    sim.load_from_observation(obs, belief);
}

/**
 * @brief Apply primitive actions to the latest concrete simulator snapshot.
 * @param actions Simulator-indexed primitive action buffer.
 */
void Engine::step_actions(const PrimitiveActions& actions) {
    sim.step(actions);
}

/**
 * @brief Produce a concrete hidden-world sample for ISMCTS.
 * @param seed Deterministic sample seed.
 * @return BoardState containing sampled hidden facts and exact observed robots.
 *
 * Belief determinization creates hidden rows and enemies.  The current observed
 * live robots are then reinserted exactly by UID, owner, position, energy, and
 * cooldown so search plans are anchored to the real controllable units.
 */
BoardState Engine::determinize(uint64_t seed) const {
    BoardState sampled = belief.determinize(seed);
    sampled.scroll_counter = sim.state.scroll_counter;
    for (int i = 0; i < sim.state.robots.used; ++i) {
        if (sim.state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const int slot = sampled.robots.add_robot(sim.state.robots.uid[static_cast<size_t>(i)].data(),
                                                  sim.state.robots.type[static_cast<size_t>(i)],
                                                  sim.state.robots.owner[static_cast<size_t>(i)],
                                                  sim.state.robots.col[static_cast<size_t>(i)],
                                                  sim.state.robots.row[static_cast<size_t>(i)],
                                                  sim.state.robots.energy[static_cast<size_t>(i)],
                                                  sim.state.robots.move_cd[static_cast<size_t>(i)],
                                                  sim.state.robots.jump_cd[static_cast<size_t>(i)],
                                                  sim.state.robots.build_cd[static_cast<size_t>(i)]);
        (void)slot;
    }
    sampled.rebuild_active_bitboards();
    return sampled;
}

}  // namespace crawler
