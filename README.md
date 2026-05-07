# Maze Crawler ISMCTS

This repository contains a C++20 agent core for Kaggle Maze Crawler, exposed to Kaggle through a thin Python `main.py` entrypoint. The engine is built around a deterministic fixed-buffer simulator, a player-centric belief model, and a zero-allocation Information Set Monte Carlo Tree Search (ISMCTS) over bounded joint macro actions.

It contains the complete native search loop, the rule-critical simulator fixes, and a Kaggle deployment path that ships source code plus vendored `pybind11` headers and JIT-compiles the extension inside the competition runtime.

## Current Status

Completed:

- C++20 deterministic simulator with fixed-capacity state and allocation-free hot turn stepping.
- Hybrid state model with absolute global arrays plus active 20x20 bitboards.
- Player-centric belief update and deterministic hidden-state sampling.
- Fixed-arena ISMCTS implementation with PUCT selection.
- Bounded macro-action expansion over joint plans instead of primitive action Cartesian products.
- Continuous rollout evaluation with terminal tiebreaker energy-margin shaping.
- Runtime hyperparameter injection from Python for PUCT, rollout depth, and macro priors.
- Process-parallel Optuna tuner for candidate-versus-`opponent.py` evaluation.
- `int32_t` robot energy storage and factory capacity handling.
- Spawn-combat, transfer, fixed-wall, jump/offboard, cooldown, mine, crystal, reward, and scrolling rule edge cases covered by tests.
- Kaggle source-bundle packaging with vendored `pybind11` headers.
- On-the-fly native compilation from `main.py` when no prebuilt extension is importable.

Remaining work is empirical rather than structural:

- Run large production Optuna studies over `C_puct`, macro priors, rollout depth, and time budgets.
- Improve rollout policy quality with self-play data and targeted heuristics.
- Run large-scale self-play and ablation studies across seed suites.
- Calibrate opening, mining, scout return, and factory advance policy weights.

## Repository Layout

```text
.
├── CMakeLists.txt              # Local C++20/pybind11 extension build
├── main.py                     # Kaggle agent entrypoint with JIT native build
├── package_submission.py       # Builds source bundle with vendored pybind11
├── submission.py               # Local alias for main.agent
├── opponent.py                 # Strong Python benchmark agent used by tune.py
├── requirements-dev.txt        # Build/test dependencies
├── test.py                     # Rule, MCTS, bridge, and package smoke tests
├── tune.py                     # Optuna opponent-backed hyperparameter tuner
├── rules/                      # Competition rule reference and notes
└── src/
    ├── bindings.cpp            # pybind11 observation/action bridge
    ├── crawler_engine.hpp      # Public API and fixed-buffer contracts
    ├── crawler_engine_internal.hpp
    ├── crawler_engine_internal.cpp
    ├── crawler_engine_state.cpp
    ├── crawler_engine_belief.cpp
    ├── crawler_engine_sim.cpp
    ├── crawler_engine_policy.cpp
    ├── crawler_engine_mcts.cpp
    └── crawler_engine_engine.cpp
```

## Architecture Overview

The engine is split by ownership boundary:

- `crawler_engine.hpp` defines constants, enums, state buffers, macro actions, MCTS node layout, and the public `Engine` facade.
- `crawler_engine_state.cpp` owns action parsing, geometry helpers, structure-of-arrays robot storage, active-window bitboards, and result buffers.
- `crawler_engine_internal.cpp` owns deterministic support functions: RNG mixing, active bit masks, scroll interval reconstruction, reciprocal wall edits, and optimistic hidden-row generation.
- `crawler_engine_belief.cpp` maintains player-centric map memory and hidden enemy probability fields, then samples determinizations for search.
- `crawler_engine_sim.cpp` is the rule engine. `CrawlerSim::step()` implements the phase-ordered turn transition with fixed-size scratch arrays only.
- `crawler_engine_policy.cpp` owns the deterministic fallback policy, opponent/rollout heuristics, macro generation, and macro-to-primitive translation.
- `crawler_engine_mcts.cpp` owns the fixed-arena ISMCTS loop, PUCT selection, rollout evaluation, and root action extraction.
- `crawler_engine_engine.cpp` wires observation updates, belief, current concrete state, and search together.
- `bindings.cpp` translates Kaggle-style Python dictionaries into fixed C++ buffers, exposes hyperparameter injection, and returns Kaggle-compatible `{uid: "ACTION"}` dictionaries.

## Fixed-Buffer State Model

Maze Crawler exposes a scrolling 20x20 active window, but strategic information persists beyond the current visible rows. The engine models that with two cooperating coordinate systems:

- Absolute arrays indexed by `row * WIDTH + col` across `MAX_ROWS = 512`.
- Active-window bitboards indexed by `(row - south_bound) * WIDTH + col`.

Core limits are compile-time constants:

```text
WIDTH = 20
HEIGHT = 20
ACTIVE_CELLS = 400
MAX_ROWS = 512
MAX_CELLS = 10240
MAX_ROBOTS = 512
MAX_MACROS = 16
MAX_TREE_NODES = 4096
MAX_MCTS_PLAN_ROBOTS = 64
MAX_MCTS_CANDIDATES = 64
MCTS_TREE_DEPTH = 24
MCTS_ROLLOUT_DEPTH = 48
```

`BoardState` stores wall knowledge, crystal energy, mine energy, mine caps, mine owners, and mining nodes in absolute arrays. This allows belief memory and rollout simulation to survive scroll movement without reindexing the whole world.

The current tactical window is rebuilt into compact `uint64_t` bitboards:

- `own_occupancy`
- `enemy_occupancy`
- `all_occupancy`
- `visibility`
- `crystals_active`
- `mines_active`
- `nodes_active`

`BitBoard` uses fixed `std::array<uint64_t, ACTIVE_WORDS>` storage. Low-level iteration support uses `std::countr_zero` in `pop_lsb`, which compilers lower to efficient bit-scan instructions on normal Kaggle CPUs.

Robots use a structure-of-arrays store:

```text
uid[512][24]
alive[512]
type[512]
owner[512]
col[512]
row[512]
energy[512]
move_cd[512]
jump_cd[512]
build_cd[512]
```

Energy is stored as `int32_t`, not `int16_t`. This is required for factories: transfers, collection, and reward accounting can push factory energy far beyond unit caps. Factory capacity calculations use `int32_t` space, while non-factory units remain capped by their rule-defined maximums. Reward accumulation uses `int64_t` margins to avoid narrowing during terminal comparisons.

## ISMCTS Core

`Engine::choose_actions(time_budget_ms, seed)` runs Information Set MCTS over the current belief state. Each iteration samples one concrete board from the belief, replays the selected macro history through that determinization, rolls out with deterministic heuristics, and backpropagates a root-player value.

The search tree is backed by `MCTSArena`:

```cpp
struct MCTSArena {
    std::array<MCTSNode, MAX_TREE_NODES> nodes{};
    int used = 0;
};
```

Resetting a turn rewinds `used`; node storage remains allocated in place and is overwritten. Node creation is a bump-pointer operation. No heap allocations are performed in the tree hot loop.

Each `MCTSNode` stores:

- Parent, first-child, and next-sibling indices.
- Visit count and accumulated value.
- PUCT prior.
- Depth and expansion flag.
- One joint macro plan keyed by robot UID.

Plans are UID-keyed so the same action history remains meaningful across sampled determinizations where simulator-local robot slots may differ.

### Selection

Selection uses PUCT:

```text
score = Q + C_puct * prior * sqrt(parent_visits + 1) / (child_visits + 1)
```

The current code uses:

```text
C_puct = 1.35
```

Root action choice is visit-count first, value second. This keeps the final move robust under short Kaggle budgets while still using rollout value to break ties.

`C_puct` is a runtime hyperparameter. Python can override it per `Engine` instance without affecting other engines in the same process.

### Expansion

The search does not branch over every primitive action for every robot. It branches over bounded macro plans.

For the controlled side, candidate generation builds:

- A baseline joint plan using the deterministic opponent-style policy for every controlled robot.
- One-robot deviations from that baseline, drawn from each robot's bounded `MacroList`.

This keeps branching capped by `MAX_MCTS_CANDIDATES = 64`, even when many units are alive. The controlled robot list is capped at `MAX_MCTS_PLAN_ROBOTS = 64`.

Supported macro intents include:

- `FACTORY_SAFE_ADVANCE`
- `FACTORY_SUPPORT_WORKER`
- `FACTORY_BUILD_WORKER`
- `FACTORY_BUILD_SCOUT`
- `FACTORY_BUILD_MINER`
- `FACTORY_JUMP_OBSTACLE`
- `WORKER_OPEN_NORTH_WALL`
- `WORKER_ESCORT_FACTORY`
- `WORKER_ADVANCE`
- `SCOUT_HUNT_CRYSTAL`
- `SCOUT_EXPLORE_NORTH`
- `SCOUT_RETURN_ENERGY`
- `MINER_SEEK_NODE`
- `MINER_TRANSFORM`
- `IDLE`

Macro priors encode the current tactical bias. Examples:

```text
FACTORY_SUPPORT_WORKER = 1.40
FACTORY_BUILD_WORKER = 1.25
MINER_TRANSFORM = 1.15
FACTORY_BUILD_SCOUT = 1.10
SCOUT_HUNT_CRYSTAL = 1.00
WORKER_ADVANCE = 0.95
SCOUT_RETURN_ENERGY = 0.75
IDLE = 0.20
```

Child priors are normalized across generated candidates.

The baseline joint plan receives an additional `baseline_prior_multiplier` before normalization. This gives Optuna a way to tune how strongly the engine trusts its deterministic default plan relative to one-robot deviations.

### Simulation and Rollout

When a selected node is applied:

- Opponent and non-overridden friendly actions are filled by `heuristic_action_for_owner`.
- Planned root-player macro entries are translated into primitive actions with `primitive_for_macro`.
- The deterministic simulator executes one full turn.

Rollouts continue for up to `MCTS_ROLLOUT_DEPTH = 48` turns by default or until terminal state. They use the same deterministic heuristic policy for both owners. This keeps rollout evaluation cheap, reproducible, and rule-compliant. The effective rollout cap is exposed as the `rollout_depth` hyperparameter.

### Value Function

Terminal death states return exact win/loss values. Tiebreaker states use a continuous energy margin rather than a binary threshold:

```text
tanh(energy_diff / 800.0)
```

If terminal energy is tied, unit count is shaped with:

```text
tanh(unit_diff / 4.0)
```

Non-terminal rollout leaves use a continuous blend:

```text
0.55 * tanh(energy_diff / 1000.0)
+ 0.20 * tanh(material_diff / 10.0)
+ 0.10 * tanh(unit_diff / 8.0)
+ 0.10 * tanh(best_factory_row_diff / 8.0)
+ 0.05 * tanh(factory_margin_diff / 8.0)
```

This gives the search a usable gradient during mid-game while preserving the competition tiebreaker logic near episode end.

## Rule Fidelity

`CrawlerSim::step()` implements the turn order as a deterministic phase machine:

1. Cooldown tick.
2. Primitive action type validation.
3. Per-turn energy drain.
4. Miner transform.
5. Worker wall edits.
6. Factory builds.
7. Energy transfers.
8. Simultaneous movement intent collection.
9. Same-cell crush combat.
10. Position application for surviving movers.
11. Crystal combat cleanup and collection.
12. Mine pickup.
13. Mine generation.
14. Scroll advancement and hidden-row generation.
15. Boundary destruction and old-row cleanup.
16. Reward/winner update.
17. Active tactical bitboard rebuild.

The simulator tests and code cover the rule details that matter most for search quality:

- Cooldowns tick before action execution, so cooldown `1` is effectively ready.
- Energy drains before special actions; macro legality checks account for that.
- Miner transform runs before movement and destroys the miner.
- Worker wall edits spend energy even on fixed walls, but fixed walls do not change.
- Reciprocal wall bits are maintained when walls are built or removed.
- Factory builds happen before movement/combat.
- Spawned robots are stationary combat participants on the same turn.
- Transfers happen before movement.
- Transfers drain the source to zero and discard overflow beyond target capacity.
- Factories use `int32_t` energy capacity for transfer/collection space.
- Movement is simultaneous.
- Same-type collisions annihilate all units of that type in the cell.
- Crush hierarchy is `Factory > Miner > Worker > Scout`.
- Enemy factories mutually annihilate.
- Friendly fire is active through the same crush rules.
- Crystals disappear on combat cells with no survivor; otherwise the survivor collects subject to energy space.
- Mine pickup happens before mine generation.
- Scroll cadence is reconstructed from the observed step, including the ramp from interval 4 to interval 1 over the first 400 steps.
- New rows are generated deterministically for sampled future state.
- Units below the new `south_bound` are destroyed.
- Resources below the active boundary are cleared.
- Jumps set both movement and jump cooldowns and destroy the factory if the landing cell is off board.

## Belief and Determinization

`BeliefState` is player-centric. It tracks:

- Known walls.
- Remembered wall layout.
- Visible crystal energy.
- Remembered mine energy, maximum, and owner.
- Remembered mining nodes.
- Current visibility mask.
- Per-type enemy probability fields for factories, scouts, workers, and miners.

On observation update:

- Enemy probability diffuses according to elapsed turns and unit movement periods.
- Known walls constrain diffusion.
- Friendly vision clears impossible enemy locations.
- Observed walls overwrite map memory.
- Visible crystals are treated as current facts.
- Mines and mining nodes are remembered.
- Observed enemies collapse their type probability to the observed cell.

`BeliefState::determinize(seed)` copies known facts into a concrete `BoardState`, generates plausible unknown rows, and samples hidden enemies from their type-specific probability fields. `Engine::determinize(seed)` then re-inserts all currently observed live robots with their exact UID, owner, position, energy, and cooldowns. That concrete state is what the MCTS iteration uses.

## Kaggle JIT Deployment

Kaggle submissions normally enter through Python, but this repository ships a native C++ engine without requiring a prebuilt wheel.

`package_submission.py` creates `submission.tar.gz` containing:

- `main.py`
- every `src/*.cpp`
- every `src/*.hpp`
- the complete `pybind11` include tree under `vendor/pybind11/include`

At runtime, `main.py` first tries:

```python
import crawler_engine
```

If no compatible extension is present, `_ensure_native_engine()` compiles one in place with `g++`:

```text
g++ -std=c++20 -O3 -DNDEBUG -fPIC -shared -ffast-math -march=native \
    -Isrc -Ivendor/pybind11/include -I<python include dirs> \
    src/*.cpp -o crawler_engine<EXT_SUFFIX>
```

After compilation, it invalidates import caches and imports the newly built extension. The extension suffix comes from `sysconfig`, so the produced filename matches the running Python interpreter.

If compilation or import fails, `main.py` falls back to a small pure-Python policy. That fallback exists for diagnostics and graceful failure only; the competitive path is the native JIT engine.

The package smoke test extracts `submission.tar.gz` into a clean temporary directory with no `PYTHONPATH`, runs `python main.py`, verifies the JIT import, then calls `main.agent` against a minimal Kaggle-style observation.

## Python API

Local development builds a module named `crawler_engine`.

```python
import crawler_engine

engine = crawler_engine.Engine(player=0)

engine.update_observation(
    player,
    walls,          # flat length-400 sequence, -1 for unknown
    crystals,       # {"col,row": energy}
    robots,         # {"uid": [type, col, row, energy, owner, move_cd, jump_cd, build_cd]}
    mines,          # {"col,row": [energy, maxEnergy, owner]}
    mining_nodes,   # {"col,row": 1}
    southBound,
    northBound,
    step,
)

actions = engine.choose_actions(time_budget_ms=2000, seed=123)
```

Returned actions are Kaggle-compatible:

```python
{"f0": "BUILD_WORKER", "s0": "NORTH"}
```

Search hyperparameters are per engine instance:

```python
engine.set_hyperparameters({
    "C_puct": 2.0,
    "baseline_prior_multiplier": 1.1,
    "rollout_depth": 32,
    "FACTORY_BUILD_WORKER": 1.5,
    "SCOUT_RETURN_ENERGY": 0.9,
})

params = engine.get_hyperparameters()
```

`set_hyperparameters` accepts partial dictionaries and rejects unknown keys. Numeric values must be positive and finite; `rollout_depth` must be a positive integer. Macro prior keys use the readable macro names returned by `crawler_engine.macro_action_name`, without the internal `MACRO_` prefix.

Debug/test helpers:

- `engine.step(actions)` applies primitive actions to the current simulator snapshot.
- `engine.determinize(seed=0)` returns a summary of a sampled rollout state.
- `engine.debug_state()` returns a summary of the current concrete simulator snapshot.
- `engine.debug_mcts_value(player=-1)` returns the current evaluator value.
- `crawler_engine.action_name(int_action)` returns a primitive action string.
- `crawler_engine.macro_action_name(int_macro)` returns a macro action string.

## Hyperparameter Tuning

`tune.py` runs Bayesian optimization with Optuna against the strong root-level `opponent.py` benchmark. Each trial:

1. Samples `C_puct`, `baseline_prior_multiplier`, `rollout_depth`, and macro priors.
2. Builds fresh Kaggle-compatible candidate wrappers around `crawler_engine.Engine`.
3. Injects the sampled parameters with `engine.set_hyperparameters(params)`.
4. Runs the candidate against `opponent.py` on each configured seed.
5. Swaps player order for every seed.
6. Returns the average final energy margin from the candidate perspective.

The score is:

```text
candidate_total_energy - opponent_total_energy
```

Final energy is read from each owner player's final observation because enemy robots can be hidden by fog of war. A positive score means the candidate ended ahead of `opponent.py` on average.

The tuner uses `ProcessPoolExecutor` with Optuna `ask`/`tell` instead of thread parallelism. Each worker process creates its own Python agents and C++ `Engine` instances, so mutable search state and pybind objects are not shared between workers.

### Tuning Host Profile

The default tuning settings target this local workstation:

```text
CPU: Intel Core i7-12700
Core layout: 12 total cores = 8 Performance-cores + 4 Efficient-cores
Threads: 20 total with Intel Hyper-Threading
Max turbo: 4.90 GHz
Intel Turbo Boost Max Technology 3.0 frequency: 4.90 GHz
Performance-core max/base: 4.80 GHz / 2.10 GHz
Efficient-core max/base: 3.60 GHz / 1.60 GHz
Cache: 25 MB Intel Smart Cache
Total L2 cache: 12 MB
Memory: 16 GB DDR4
Instruction set: 64-bit
Instruction extensions: SSE4.1, SSE4.2, AVX2
Scheduling/power features: Intel Thread Director, Speed Shift, Enhanced SpeedStep, idle states
Other CPU features: Intel DL Boost, Gaussian & Neural Accelerator 3.0, Optane Memory support
```

The default `--n-jobs 16` is intentionally below the 20 hardware threads. It keeps the 8 P-cores and most logical threads busy while leaving headroom for the OS, SQLite writes, Python/Kaggle environment overhead, and the 16 GB RAM limit. If the machine starts swapping or SQLite latency rises, reduce to `--n-jobs 12`; if CPU utilization is low and memory is stable, test `--n-jobs 18`.

The study persists to SQLite by default:

```text
sqlite:///tune.db
```

That makes long runs pause/resume safe: rerun the same command with the same `--storage` and `--study-name`.

If the native extension cannot be imported, `tune.py` tries the same JIT path used by `main.py`. A trial-level import, compile, or agent error is logged and marked failed instead of terminating the whole study.

Smoke run:

```bash
PYTHONPATH=build /tmp/maze-crawler-venv/bin/python tune.py \
  --trials 1 \
  --n-jobs 1 \
  --seeds 1 \
  --time-budget 10 \
  --storage sqlite:////tmp/maze-crawler-opponent-smoke.db \
  --study-name opponent-smoke
```

Production-style run:

```bash
PYTHONPATH=build /tmp/maze-crawler-venv/bin/python tune.py \
  --trials 1000 \
  --n-jobs 16 \
  --seeds 5 \
  --time-budget 300 \
  --storage sqlite:///tune.db \
  --study-name crawl-vs-opponent
```

Important CLI defaults:

```text
--trials 100
--seeds 3
--time-budget 300
--n-jobs 16
--storage sqlite:///tune.db
--study-name crawl-vs-opponent
```

Default search ranges:

```text
C_puct: 0.5 to 3.0
baseline_prior_multiplier: 0.75 to 2.0
rollout_depth: 16 to 96, step 8
IDLE prior: 0.05 to 0.75
all other macro priors: 0.25 to 2.5
```

## Local Build

Requirements:

- CMake 3.20 or newer.
- C++20 compiler.
- Python 3 with development headers.
- `pybind11`, `numpy`, `pytest`, `kaggle-environments`, and `optuna` from `requirements-dev.txt`.

Create a development environment:

```bash
python3 -m venv /tmp/maze-crawler-venv
/tmp/maze-crawler-venv/bin/python -m pip install -r requirements-dev.txt
```

Configure and build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/tmp/maze-crawler-venv/bin/python

cmake --build build -j
```

Release builds use `-O3`, `-march=native`, `-ffast-math`, and `NDEBUG` for GCC/Clang.

## Test

Run the smoke script:

```bash
PYTHONPATH=build /tmp/maze-crawler-venv/bin/python test.py
```

Run through pytest:

```bash
PYTHONPATH=build /tmp/maze-crawler-venv/bin/python -m pytest -q test.py
```

Current coverage includes:

- pybind bridge import and action generation.
- Factory build and spawn-before-combat behavior.
- Spawned unit participation in same-turn combat.
- Same-type annihilation.
- Enemy factory mutual destruction.
- Stronger-type crush resolution.
- Crystal collection and combat consumption.
- Transfer source drain, target cap, and factory `int32_t` overflow protection.
- Scroll counter reconstruction for high-step observations.
- Continuous MCTS tiebreaker value from energy margin.
- Period-two movement cooldown behavior.
- Factory jump cooldown behavior.
- Fixed center-wall edit no-op with energy cost.
- Off-board jump destruction.
- MCTS small-time-budget behavior and action validity.
- Hyperparameter API defaults, partial updates, validation errors, and action generation after injection.
- Packaged submission JIT compilation in an extracted clean directory.

## Kaggle Submission Bundle

Build the source bundle:

```bash
/tmp/maze-crawler-venv/bin/python package_submission.py
```

This writes:

```text
submission.tar.gz
```

The archive is intentionally source-first. It does not rely on the local build directory or a platform-specific `.so`; Kaggle compiles the native extension in the extracted submission directory.

## Kaggle Entrypoint

`main.py` defines:

```python
def agent(obs, config):
    ...
```

The entrypoint keeps one C++ `Engine` per player in `_ENGINES`. Each call:

1. Ensures the native engine is importable, JIT-compiling it if needed.
2. Extracts Kaggle observation fields.
3. Updates the C++ belief and concrete simulator snapshot.
4. Calls `engine.choose_actions(2000, seed=...)`.
5. Returns `{uid: "ACTION"}`.

A minimal local smoke call after building:

```bash
PYTHONPATH=build /tmp/maze-crawler-venv/bin/python - <<'PY'
from types import SimpleNamespace
from main import agent

obs = SimpleNamespace(
    player=0,
    walls=[0] * 400,
    crystals={},
    robots={"f0": [0, 5, 2, 1000, 0, 0, 0, 0]},
    mines={},
    miningNodes={},
    southBound=0,
    northBound=19,
    step=0,
)
config = SimpleNamespace(width=20, workerCost=200, wallRemoveCost=100)
print(agent(obs, config))
PY
```

## CI/CD

`.github/workflows/ci.yml` builds and tests the extension on Ubuntu with Python 3.11 and 3.12. The workflow:

- Installs `requirements-dev.txt`.
- Configures CMake against the active Python interpreter.
- Builds the release extension.
- Runs `test.py`.
- Runs `pytest`.
- Verifies `main.agent` returns a Kaggle-compatible dict.
- Uploads compiled extension artifacts and `compile_commands.json`.
- Packages source artifacts.
- Publishes bundled artifacts on version tags.

## Development Rules

- Keep `CrawlerSim::step()` allocation-free. Use fixed-size buffers for per-turn scratch.
- Keep exact rules in `crawler_engine_sim.cpp`.
- Keep belief and hidden-information updates in `crawler_engine_belief.cpp`.
- Keep search in `crawler_engine_mcts.cpp`.
- Keep policy, rollout heuristics, and macro translation in `crawler_engine_policy.cpp`.
- Keep Python conversion in `bindings.cpp`; do not put rules or search logic in the bridge.
- Preserve fixed-capacity data layouts unless the code is explicitly outside the rollout/search hot path.
- Add tests before changing any rule-sensitive behavior.

## Roadmap

The remaining roadmap is data science and empirical optimization:

- Hyperparameter tuning:
  - `C_puct`
  - rollout depth
  - macro prior weights
  - baseline prior multiplier
  - dynamic time budgets
- Dynamic time control:
  - opening versus mid-game budgets
  - unit-count-aware iteration budgets
  - deadline guard tuning for Kaggle variance
- Rollout policy refinement:
  - better scout crystal routing
  - miner-node conversion timing
  - worker wall opening versus escort tradeoffs
  - factory build mix by phase and visible economy
- Evaluation:
  - deterministic seed suites
  - large-scale self-play
  - candidate-versus-`opponent.py` production studies
  - macro-prior ablations
  - rollout-depth ablations
  - best-parameter confidence intervals
  - opponent-policy robustness tests

Core engine implementation, rule fidelity, fixed-arena search, Kaggle JIT deployment, and the opponent-backed Optuna tuning loop are complete.

## License

See [`LICENSE`](./LICENSE).
