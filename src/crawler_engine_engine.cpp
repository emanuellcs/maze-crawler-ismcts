#include "crawler_engine.hpp"

// High-level C++ facade used by the Python binding. It wires observation updates,
// belief determinization, and current action selection together.

#include <algorithm>

namespace crawler {

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

// Current policy placeholder touches macro generation to keep that path live,
// then falls back to deterministic primitive heuristics.
ActionResult Engine::choose_actions(int time_budget_ms, uint64_t seed) {
    (void)seed;
    const int rollout_budget = std::max(1, std::min(MAX_TREE_NODES, time_budget_ms * 2));
    int macro_touch_count = 0;
    for (int i = 0; i < sim.state.robots.used && macro_touch_count < rollout_budget; ++i) {
        if (sim.state.robots.alive[static_cast<size_t>(i)] == 0 ||
            sim.state.robots.owner[static_cast<size_t>(i)] != sim.state.player) {
            continue;
        }
        const MacroList macros = sim.generate_macros_for(i);
        macro_touch_count += macros.count;
    }

    ActionResult result{};
    result.clear();
    for (int i = 0; i < sim.state.robots.used; ++i) {
        if (sim.state.robots.alive[static_cast<size_t>(i)] == 0 ||
            sim.state.robots.owner[static_cast<size_t>(i)] != sim.state.player) {
            continue;
        }
        const Action action = sim.heuristic_action_for(i);
        result.add(sim.state.robots.uid[static_cast<size_t>(i)].data(), action);
    }
    return result;
}

}  // namespace crawler
