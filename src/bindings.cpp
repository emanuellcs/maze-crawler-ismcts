#include "crawler_engine.hpp"

// pybind11 bridge between Kaggle's Python observation/action dictionaries and
// the fixed-buffer C++ engine. Keep policy and simulator rules out of this file.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <string>

namespace py = pybind11;

namespace {

constexpr std::array<crawler::MacroAction, 15> HYPERPARAMETER_MACROS{
    crawler::MACRO_IDLE,
    crawler::MACRO_FACTORY_SUPPORT_WORKER,
    crawler::MACRO_FACTORY_SAFE_ADVANCE,
    crawler::MACRO_FACTORY_BUILD_WORKER,
    crawler::MACRO_FACTORY_BUILD_SCOUT,
    crawler::MACRO_FACTORY_BUILD_MINER,
    crawler::MACRO_FACTORY_JUMP_OBSTACLE,
    crawler::MACRO_WORKER_OPEN_NORTH_WALL,
    crawler::MACRO_WORKER_ESCORT_FACTORY,
    crawler::MACRO_WORKER_ADVANCE,
    crawler::MACRO_SCOUT_HUNT_CRYSTAL,
    crawler::MACRO_SCOUT_EXPLORE_NORTH,
    crawler::MACRO_SCOUT_RETURN_ENERGY,
    crawler::MACRO_MINER_SEEK_NODE,
    crawler::MACRO_MINER_TRANSFORM,
};

// Tunable keys intentionally mirror macro_action_name() without the MACRO_
// prefix, so Python search spaces can be human-readable and typo-checked.
void require_positive_finite(const std::string& key, double value) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw py::value_error(key + " must be a positive finite number");
    }
}

bool set_macro_prior(crawler::Hyperparameters& hyperparameters, const std::string& key, double value) {
    for (const crawler::MacroAction macro : HYPERPARAMETER_MACROS) {
        if (key == crawler::macro_action_name(macro)) {
            require_positive_finite(key, value);
            hyperparameters.macro_prior[static_cast<size_t>(macro)] = static_cast<float>(value);
            return true;
        }
    }
    return false;
}

void apply_hyperparameter(crawler::Hyperparameters& hyperparameters, const std::string& key,
                          const py::handle& value) {
    if (key == "rollout_depth") {
        const int depth = py::cast<int>(value);
        if (depth <= 0 || depth > 512) {
            throw py::value_error("rollout_depth must be an integer in [1, 512]");
        }
        hyperparameters.rollout_depth = depth;
        return;
    }

    const double numeric_value = py::cast<double>(value);
    if (key == "C_puct") {
        require_positive_finite(key, numeric_value);
        hyperparameters.C_puct = static_cast<float>(numeric_value);
        return;
    }
    if (key == "baseline_prior_multiplier") {
        require_positive_finite(key, numeric_value);
        hyperparameters.baseline_prior_multiplier = static_cast<float>(numeric_value);
        return;
    }
    if (set_macro_prior(hyperparameters, key, numeric_value)) {
        return;
    }

    throw py::key_error("unknown hyperparameter: " + key);
}

py::dict hyperparameters_to_dict(const crawler::Hyperparameters& hyperparameters) {
    py::dict out;
    out["C_puct"] = hyperparameters.C_puct;
    out["baseline_prior_multiplier"] = hyperparameters.baseline_prior_multiplier;
    out["rollout_depth"] = hyperparameters.rollout_depth;
    for (const crawler::MacroAction macro : HYPERPARAMETER_MACROS) {
        out[py::str(crawler::macro_action_name(macro))] = hyperparameters.prior_for(macro);
    }
    return out;
}

// Python observations key sparse cells as "col,row"; this helper decodes that
// form without allocating temporary structured objects in C++.
bool parse_cell_key(const std::string& key, int& col, int& row) {
    const std::size_t comma = key.find(',');
    if (comma == std::string::npos) {
        return false;
    }
    try {
        col = std::stoi(key.substr(0, comma));
        row = std::stoi(key.substr(comma + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// Convert Python dict/list observation fields into the fixed-buffer C++ input
// struct expected by BeliefState and CrawlerSim.
crawler::ObservationInput make_observation(int player, const py::object& walls_obj, const py::dict& crystals,
                                           const py::dict& robots, const py::dict& mines,
                                           const py::dict& mining_nodes, int south_bound,
                                           int north_bound, int step) {
    crawler::ObservationInput obs{};
    obs.player = player;
    obs.south_bound = south_bound;
    obs.north_bound = north_bound;
    obs.step = step;
    obs.walls.fill(-1);

    if (!walls_obj.is_none()) {
        py::sequence walls = py::reinterpret_borrow<py::sequence>(walls_obj);
        const int n = std::min<int>(static_cast<int>(py::len(walls)), crawler::ACTIVE_CELLS);
        for (int i = 0; i < n; ++i) {
            obs.walls[static_cast<size_t>(i)] = static_cast<int16_t>(py::cast<int>(walls[i]));
        }
    }

    for (auto item : robots) {
        if (obs.robot_count >= crawler::MAX_ROBOTS) {
            break;
        }
        const std::string uid = py::cast<std::string>(py::str(item.first));
        py::sequence data = py::reinterpret_borrow<py::sequence>(item.second);
        if (py::len(data) < 5) {
            continue;
        }
        crawler::RobotObservation& robot = obs.robots[static_cast<size_t>(obs.robot_count)];
        robot.uid.fill('\0');
        const int copy_n = std::min<int>(static_cast<int>(uid.size()), crawler::UID_LEN - 1);
        for (int i = 0; i < copy_n; ++i) {
            robot.uid[static_cast<size_t>(i)] = uid[static_cast<size_t>(i)];
        }
        robot.type = py::cast<int>(data[0]);
        robot.col = py::cast<int>(data[1]);
        robot.row = py::cast<int>(data[2]);
        robot.energy = py::cast<int>(data[3]);
        robot.owner = py::cast<int>(data[4]);
        robot.move_cd = py::len(data) > 5 ? py::cast<int>(data[5]) : 0;
        robot.jump_cd = py::len(data) > 6 ? py::cast<int>(data[6]) : 0;
        robot.build_cd = py::len(data) > 7 ? py::cast<int>(data[7]) : 0;
        ++obs.robot_count;
    }

    for (auto item : crystals) {
        if (obs.crystal_count >= crawler::MAX_OBS_CRYSTALS) {
            break;
        }
        int col = 0;
        int row = 0;
        if (!parse_cell_key(py::cast<std::string>(py::str(item.first)), col, row)) {
            continue;
        }
        crawler::CellEnergyObservation& crystal = obs.crystals[static_cast<size_t>(obs.crystal_count)];
        crystal.col = col;
        crystal.row = row;
        crystal.energy = py::cast<int>(item.second);
        ++obs.crystal_count;
    }

    for (auto item : mines) {
        if (obs.mine_count >= crawler::MAX_OBS_MINES) {
            break;
        }
        int col = 0;
        int row = 0;
        if (!parse_cell_key(py::cast<std::string>(py::str(item.first)), col, row)) {
            continue;
        }
        py::sequence data = py::reinterpret_borrow<py::sequence>(item.second);
        if (py::len(data) < 3) {
            continue;
        }
        crawler::MineObservation& mine = obs.mines[static_cast<size_t>(obs.mine_count)];
        mine.col = col;
        mine.row = row;
        mine.energy = py::cast<int>(data[0]);
        mine.max_energy = py::cast<int>(data[1]);
        mine.owner = py::cast<int>(data[2]);
        ++obs.mine_count;
    }

    for (auto item : mining_nodes) {
        if (obs.mining_node_count >= crawler::MAX_OBS_NODES) {
            break;
        }
        int col = 0;
        int row = 0;
        if (!parse_cell_key(py::cast<std::string>(py::str(item.first)), col, row)) {
            continue;
        }
        crawler::CellObservation& node = obs.mining_nodes[static_cast<size_t>(obs.mining_node_count)];
        node.col = col;
        node.row = row;
        ++obs.mining_node_count;
    }

    return obs;
}

// Convert fixed-buffer action results back to the Kaggle `{uid: action}` dict.
py::dict action_result_to_dict(const crawler::ActionResult& result) {
    py::dict out;
    for (int i = 0; i < result.count; ++i) {
        out[py::str(result.uid[static_cast<size_t>(i)].data())] =
            py::str(crawler::action_name(result.action[static_cast<size_t>(i)]));
    }
    return out;
}

// Small inspection surface for Python smoke tests and debugging.
py::dict board_summary(const crawler::BoardState& state) {
    py::dict out;
    out["player"] = state.player;
    out["step"] = state.step;
    out["southBound"] = state.south_bound;
    out["northBound"] = state.north_bound;
    out["done"] = state.done;
    out["winner"] = state.winner;
    out["reward0"] = state.reward0;
    out["reward1"] = state.reward1;
    int robots = 0;
    int own = 0;
    int enemy = 0;
    py::list robot_list;
    for (int i = 0; i < state.robots.used; ++i) {
        if (state.robots.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        ++robots;
        if (state.robots.owner[static_cast<size_t>(i)] == state.player) {
            ++own;
        } else {
            ++enemy;
        }
        py::dict robot;
        robot["uid"] = state.robots.uid[static_cast<size_t>(i)].data();
        robot["type"] = state.robots.type[static_cast<size_t>(i)];
        robot["col"] = state.robots.col[static_cast<size_t>(i)];
        robot["row"] = state.robots.row[static_cast<size_t>(i)];
        robot["energy"] = state.robots.energy[static_cast<size_t>(i)];
        robot["owner"] = state.robots.owner[static_cast<size_t>(i)];
        robot["move_cd"] = state.robots.move_cd[static_cast<size_t>(i)];
        robot["jump_cd"] = state.robots.jump_cd[static_cast<size_t>(i)];
        robot["build_cd"] = state.robots.build_cd[static_cast<size_t>(i)];
        robot_list.append(robot);
    }
    out["robots"] = robots;
    out["ownRobots"] = own;
    out["enemyRobots"] = enemy;
    out["robotList"] = robot_list;
    return out;
}

// Thin pybind facade. It owns one C++ Engine and translates Python dict actions
// into simulator-indexed PrimitiveActions.
class PyEngine {
public:
    explicit PyEngine(int player = 0) : engine(player) {}

    void update_observation(int player, const py::object& walls, const py::dict& crystals,
                            const py::dict& robots, const py::dict& mines,
                            const py::dict& mining_nodes, int south_bound, int north_bound,
                            int step = -1) {
        crawler::ObservationInput obs = make_observation(player, walls, crystals, robots, mines,
                                                         mining_nodes, south_bound, north_bound, step);
        engine.update_observation(obs);
    }

    py::dict choose_actions(int time_budget_ms = 2000, uint64_t seed = 0) {
        return action_result_to_dict(engine.choose_actions(time_budget_ms, seed));
    }

    void set_hyperparameters(const py::dict& params) {
        for (auto item : params) {
            apply_hyperparameter(engine.hyperparameters, py::cast<std::string>(py::str(item.first)), item.second);
        }
    }

    py::dict get_hyperparameters() const {
        return hyperparameters_to_dict(engine.hyperparameters);
    }

    void step(const py::dict& actions) {
        crawler::PrimitiveActions primitive{};
        primitive.clear();
        for (auto item : actions) {
            const std::string uid = py::cast<std::string>(py::str(item.first));
            const std::string action = py::cast<std::string>(py::str(item.second));
            const int index = engine.sim.state.robots.find_uid(uid);
            if (index >= 0) {
                primitive.actions[static_cast<size_t>(index)] = crawler::parse_action(action);
            }
        }
        engine.step_actions(primitive);
    }

    py::dict determinize(uint64_t seed = 0) const {
        return board_summary(engine.determinize(seed));
    }

    py::dict debug_state() const {
        return board_summary(engine.sim.state);
    }

    float debug_mcts_value(int player = -1) const {
        return engine.debug_mcts_value(player);
    }

private:
    crawler::Engine engine;
};

}  // namespace

PYBIND11_MODULE(crawler_engine, m) {
    m.doc() = "Fixed-buffer C++ engine scaffold for Kaggle Maze Crawler.";
    m.attr("__version__") = "0.1.0";

    py::class_<PyEngine>(m, "Engine")
        .def(py::init<int>(), py::arg("player") = 0)
        .def("update_observation", &PyEngine::update_observation,
             py::arg("player"), py::arg("walls"), py::arg("crystals"), py::arg("robots"),
             py::arg("mines"), py::arg("mining_nodes"), py::arg("southBound"),
             py::arg("northBound"), py::arg("step") = -1)
        .def("choose_actions", &PyEngine::choose_actions, py::arg("time_budget_ms") = 2000,
             py::arg("seed") = 0)
        .def("set_hyperparameters", &PyEngine::set_hyperparameters, py::arg("params"))
        .def("get_hyperparameters", &PyEngine::get_hyperparameters)
        .def("step", &PyEngine::step, py::arg("actions"))
        .def("determinize", &PyEngine::determinize, py::arg("seed") = 0)
        .def("debug_mcts_value", &PyEngine::debug_mcts_value, py::arg("player") = -1)
        .def("debug_state", &PyEngine::debug_state);

    m.def("action_name", [](int action) {
        return std::string(crawler::action_name(static_cast<crawler::Action>(action)));
    });
    m.def("macro_action_name", [](int macro) {
        return std::string(crawler::macro_action_name(static_cast<crawler::MacroAction>(macro)));
    });
}
