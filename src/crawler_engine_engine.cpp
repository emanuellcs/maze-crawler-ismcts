#include "crawler_engine.hpp"

// High-level C++ facade used by the Python binding. It wires observation updates,
// belief determinization, and current action selection together.

namespace crawler {

Hyperparameters::Hyperparameters() {
    reset_macro_priors();
}

void Hyperparameters::reset_macro_priors() {
    // Macro priors use the best Optuna study results found during tuning.
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

float Hyperparameters::prior_for(MacroAction macro) const {
    const size_t index = static_cast<size_t>(macro);
    if (index >= macro_prior.size()) {
        return 0.20F;
    }
    return macro_prior[index];
}

// Construct an empty engine. The player-specific constructor sets ownership after
// both fixed-buffer stores have been reset.
Engine::Engine() {
    belief.reset();
    sim.reset();
}

Engine::Engine(int player) : Engine() {
    belief.player = player;
    sim.state.player = player;
}

// Observations update belief first; the concrete simulator snapshot is then
// rebuilt from visible facts plus remembered map state.
void Engine::update_observation(const ObservationInput& obs) {
    belief.update_from_observation(obs);
    sim.load_from_observation(obs, belief);
}

void Engine::step_actions(const PrimitiveActions& actions) {
    sim.step(actions);
}

// Produce a concrete world state for search rollouts while preserving all known
// live robots from the latest observation.
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
