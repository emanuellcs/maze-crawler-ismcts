#pragma once

#include <cstdint>

namespace crawler {

/** @brief Fixed game limits and default Kaggle configuration values. */
constexpr int WIDTH = 20;
constexpr int HEIGHT = 20;
constexpr int ACTIVE_CELLS = WIDTH * HEIGHT;
constexpr int MAX_ROWS = 512;
constexpr int MAX_CELLS = WIDTH * MAX_ROWS;
constexpr int ACTIVE_WORDS = (ACTIVE_CELLS + 63) / 64;
constexpr int MAX_ROBOTS = 512;
constexpr int UID_LEN = 24;
constexpr int MAX_OBS_CRYSTALS = ACTIVE_CELLS;
constexpr int MAX_OBS_MINES = ACTIVE_CELLS;
constexpr int MAX_OBS_NODES = ACTIVE_CELLS;
constexpr int MAX_MACROS = 16;
constexpr int MAX_TREE_NODES = 16384;
constexpr int MAX_MCTS_PLAN_ROBOTS = 64;
constexpr int MAX_MCTS_CANDIDATES = 64;
constexpr int MCTS_TREE_DEPTH = 24;

constexpr int EPISODE_STEPS = 501;
constexpr int FACTORY_ENERGY = 1000;
constexpr int SCOUT_COST = 50;
constexpr int WORKER_COST = 200;
constexpr int MINER_COST = 300;
constexpr int SCOUT_MAX_ENERGY = 100;
constexpr int WORKER_MAX_ENERGY = 300;
constexpr int MINER_MAX_ENERGY = 500;
constexpr int WALL_BUILD_COST = 100;
constexpr int WALL_REMOVE_COST = 100;
constexpr int TRANSFORM_COST = 100;
constexpr int MINE_MAX_ENERGY = 1000;
constexpr int MINE_RATE = 50;
constexpr int ENERGY_PER_TURN = 1;
constexpr int FACTORY_BUILD_COOLDOWN = 10;
constexpr int FACTORY_JUMP_COOLDOWN = 20;
constexpr int FACTORY_MOVE_PERIOD = 2;
constexpr int SCOUT_MOVE_PERIOD = 1;
constexpr int WORKER_MOVE_PERIOD = 2;
constexpr int MINER_MOVE_PERIOD = 2;
constexpr int VISION_FACTORY = 4;
constexpr int VISION_SCOUT = 5;
constexpr int VISION_WORKER = 3;
constexpr int VISION_MINER = 3;
constexpr int SCROLL_START_INTERVAL = 10;
constexpr int SCROLL_END_INTERVAL = 2;
constexpr int SCROLL_RAMP_STEPS = 450;

}  // namespace crawler
