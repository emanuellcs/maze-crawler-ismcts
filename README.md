# Maze Crawler ISMCTS

This repository contains an ahead-of-time C++ inference engine for Kaggle's
Maze Crawler competition, exposed to Python through `pybind11`. The project is
structured to keep the Kaggle `main.py` entrypoint thin while moving simulator
state, deterministic rollouts, belief updates, and future ISMCTS search into
compiled code.

The current implementation is a working engine scaffold. It builds as a Python
extension module named `crawler_engine`, ingests Kaggle-style observations,
executes rule-focused simulator steps, and returns Kaggle-compatible primitive
actions.

## Goals

- Provide a deterministic C++ simulator for Maze Crawler turn mechanics.
- Keep hot simulation state fixed-capacity and allocation-free in `CrawlerSim::step`.
- Maintain a Python bridge that can be called directly from Kaggle's agent API.
- Represent hidden information in a lightweight belief state suitable for
  determinization and search rollouts.
- Search over bounded macro-action intents rather than the full primitive action
  Cartesian product.

## Current Status

Implemented:

- Fixed-buffer C++ state model for robots, cells, observations, actions, and macros.
- Python extension module using `pybind11`.
- Kaggle-compatible `main.py` agent entrypoint with a Python fallback policy.
- Deterministic turn stepping for cooldowns, validation, energy drain, special
  actions, movement, crush combat, resource collection, mine generation,
  scrolling, boundary destruction, and rewards.
- Macro-action generation and a conservative macro-to-primitive translation stub.
- Rule and bridge smoke tests in `test.py`.

Not yet complete:

- Full ISMCTS tree search and tuned rollout policy.
- Large-scale simulator parity testing against the official Kaggle environment.
- Submission packaging automation for compiled multi-file artifacts.

## Repository Layout

```text
.
├── CMakeLists.txt              # C++20/pybind11 build configuration
├── main.py                     # Kaggle agent entrypoint
├── submission.py               # Local alias for main.agent
├── requirements-dev.txt        # Python build/test dependencies
├── test.py                     # Bridge and simulator smoke tests
├── rules/
│   ├── README.md               # Competition rule reference
│   └── AGENTS.md               # Local competition setup and usage notes
└── src/
    ├── bindings.cpp            # pybind11 bridge
    ├── crawler_engine.hpp      # Public C++ API and fixed-buffer contracts
    ├── crawler_engine_internal.hpp
    ├── crawler_engine_internal.cpp
    ├── crawler_engine_state.cpp
    ├── crawler_engine_belief.cpp
    ├── crawler_engine_sim.cpp
    ├── crawler_engine_policy.cpp
    └── crawler_engine_engine.cpp
```

## C++ Architecture

The engine is split by responsibility:

- `crawler_engine.hpp` defines the public API, constants, enums, and fixed-size
  state structures shared by all modules.
- `crawler_engine_state.cpp` implements fixed-buffer storage, action parsing,
  board queries, robot storage, and tactical bitboards.
- `crawler_engine_belief.cpp` implements player-centric belief updates, enemy
  probability diffusion, memory of discovered mines/layout, and hidden-state
  determinization.
- `crawler_engine_sim.cpp` implements exact deterministic game mechanics and the
  phase-ordered `CrawlerSim::step()` function.
- `crawler_engine_policy.cpp` implements the current heuristic policy,
  macro-action generation, and macro-to-primitive translation.
- `crawler_engine_engine.cpp` implements the high-level `Engine` facade used by
  Python.
- `crawler_engine_internal.cpp` implements shared deterministic helpers such as
  RNG mixing, scroll interval calculation, wall mutation, and hidden row
  generation.
- `bindings.cpp` converts Python observations/actions to and from the fixed C++
  buffers.

## Core Data Model

The engine uses fixed-capacity data structures to support fast rollouts:

- Board width is fixed at `20`.
- Active tactical window is `20 x 20`.
- Absolute row storage is allocated for `20 * 512` cells.
- Robots are stored in a Structure of Arrays layout with `MAX_ROBOTS = 512`.
- UIDs are stored in fixed `char[24]` buffers.
- Active-window occupancy, visibility, crystal, mine, and node masks use compact
  bitboards backed by `uint64_t`.
- `PrimitiveActions` and `ActionResult` are fixed-size buffers addressed by
  simulator robot index.

Cell arrays in `BoardState` are indexed by absolute cell index:

```text
abs_index = row * WIDTH + col
```

Observation wall arrays are indexed by active-window local index:

```text
local_index = (row - southBound) * WIDTH + col
```

## Simulator Semantics

`CrawlerSim::step()` follows the documented Maze Crawler turn order:

1. Cooldown tick.
2. Action validation.
3. Per-robot energy consumption.
4. Special actions:
   - Miner transform.
   - Worker wall build/remove.
   - Factory build.
   - Energy transfer.
5. Simultaneous movement and crush combat.
6. Crystal collection.
7. Mine pickup.
8. Mine generation.
9. Scroll advancement and row generation.
10. Boundary destruction.
11. Reward and win-condition update.
12. Tactical bitboard rebuild.

Important mechanical details implemented in the simulator:

- Crush hierarchy is `Factory > Miner > Worker > Scout`.
- Same-type collisions annihilate all units of that type.
- Friendly fire is active.
- Factory builds spawn the new robot before movement/combat, so the spawned unit
  is a stationary combat participant on that turn.
- Transfers send all source energy; target energy is capped and overflow is
  discarded.
- Worker edits on fixed walls still spend energy but do not change the wall.
- North/south off-board movement without a blocking wall destroys the unit.
- Factory jumps ignore walls and destroy the factory if the landing cell is off
  board.

`CrawlerSim::step()` uses fixed-size scratch arrays and does not allocate heap
memory in the hot path.

## Belief and Determinization

`BeliefState` is player-centric. It tracks:

- Known walls and remembered layout.
- Currently visible crystals.
- Remembered mines.
- Remembered mining nodes.
- Currently visible cells.
- Per-type enemy probability fields.

On each observation update:

- Known wall facts are copied into permanent memory.
- Visible cells clear impossible hidden enemy probability.
- Observed enemies overwrite probability at their current cells.
- Hidden enemy probabilities diffuse by movement capability and known walls.
- Crystals are treated as visible-only facts.
- Mines are remembered once observed.

`BeliefState::determinize(seed)` produces a concrete `BoardState` for future
search rollouts by copying known facts, sampling plausible hidden rows, and
placing hidden enemies from probability fields.

## Macro Actions

The engine includes a bounded macro-action layer to keep future search branching
manageable. Macro actions represent intent-level decisions such as:

- `FACTORY_SAFE_ADVANCE`
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

`CrawlerSim::generate_macros_for(robot_index)` returns a bounded `MacroList`.
`CrawlerSim::primitive_for_macro(robot_index, macro)` translates one macro into
a primitive Kaggle action when possible, otherwise `IDLE`.

The current `Engine::choose_actions()` still uses a deterministic heuristic
fallback. The macro path is present and kept live for later ISMCTS integration.

## Python API

Build output creates a Python module named `crawler_engine`.

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
{"robot_uid": "NORTH", "factory_uid": "BUILD_WORKER"}
```

Additional debug/test APIs:

- `engine.step(actions)` applies a Python dict of primitive actions to the
  current simulator state.
- `engine.determinize(seed=0)` returns a summary of a sampled rollout state.
- `engine.debug_state()` returns a summary of the current concrete simulator state.
- `crawler_engine.action_name(int_action)` returns the primitive action string.
- `crawler_engine.macro_action_name(int_macro)` returns the macro action string.

## Kaggle Entrypoint

`main.py` defines the competition entrypoint:

```python
def agent(obs, config):
    ...
```

The entrypoint keeps one C++ engine per player in process-local memory:

```python
_ENGINES = {}
```

For each call it:

1. Extracts Kaggle observation fields.
2. Updates the C++ engine observation.
3. Calls `choose_actions`.
4. Returns `{uid: "ACTION"}`.

If the compiled extension cannot be imported, `main.py` falls back to a small
pure-Python policy. This is intended to make import failures visible without
crashing immediately; it is not intended to be competitive.

## Requirements

System requirements:

- CMake 3.20 or newer.
- C++20 compiler.
- Python 3 with development module support.
- `pybind11` installed in the Python interpreter used by CMake.

Python development dependencies are listed in `requirements-dev.txt`:

- `pybind11`
- `numpy`
- `pytest`
- `kaggle-environments`

A virtual environment is recommended because many hosted development containers
use externally managed system Python installations.

```bash
python3 -m venv /tmp/maze-crawler-venv
/tmp/maze-crawler-venv/bin/python -m pip install -r requirements-dev.txt
```

## Build

Configure CMake with the same Python interpreter that has `pybind11` installed:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/tmp/maze-crawler-venv/bin/python
```

Build the extension:

```bash
cmake --build build -j
```

The compiled module is written under `build/`, for example:

```text
build/crawler_engine.cpython-312-x86_64-linux-gnu.so
```

Release builds use:

- `-O3`
- `-march=native`
- `-ffast-math`
- `NDEBUG`

These flags are selected in `CMakeLists.txt` for GCC/Clang release builds.

## Test

Run the smoke script:

```bash
PYTHONPATH=build /tmp/maze-crawler-venv/bin/python test.py
```

Run the same tests through pytest:

```bash
PYTHONPATH=build /tmp/maze-crawler-venv/bin/python -m pytest -q test.py
```

Current test coverage includes:

- Python bridge import and action generation.
- Factory build and spawn-before-combat behavior.
- Same-type annihilation.
- Factory/factory mutual destruction.
- Stronger-type crush resolution.
- Crystal collection and combat consumption.
- Transfer source drain and target cap behavior.
- Period-two movement cooldown behavior.
- Factory jump cooldown and movement cooldown behavior.
- Fixed center-wall edit no-op behavior.
- Off-board jump destruction.

## Local Agent Smoke Run

After building the extension, the Python entrypoint can be exercised directly:

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

## Development Notes

- Read `rules/README.md` before changing simulator mechanics.
- Keep `CrawlerSim::step()` allocation-free. Use fixed-size `std::array`
  scratch buffers for per-turn working state.
- Keep pybind conversion logic in `bindings.cpp`; do not put search or rules
  logic there.
- Keep hidden-information update logic in `crawler_engine_belief.cpp`.
- Keep exact game mechanics in `crawler_engine_sim.cpp`.
- Keep heuristics and macro-action translation in `crawler_engine_policy.cpp`.
- Prefer preserving fixed-capacity containers unless a change is explicitly
  outside the rollout hot path.
- Add tests for every rule edge case before tuning policy behavior.

## Troubleshooting

### CMake cannot find pybind11

Configure with the Python interpreter that has `pybind11` installed:

```bash
cmake -S . -B build \
  -DPython3_EXECUTABLE=/tmp/maze-crawler-venv/bin/python
```

If the error persists, verify:

```bash
/tmp/maze-crawler-venv/bin/python -m pybind11 --cmakedir
```

### Python cannot import crawler_engine

Ensure the build directory is on `PYTHONPATH`:

```bash
PYTHONPATH=build /tmp/maze-crawler-venv/bin/python -c "import crawler_engine; print(crawler_engine.__version__)"
```

### Tests use the wrong Python interpreter

Run tests through the same virtual environment used by CMake:

```bash
PYTHONPATH=build /tmp/maze-crawler-venv/bin/python -m pytest -q test.py
```

### Build artifacts are stale after source-list changes

Reconfigure CMake:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/tmp/maze-crawler-venv/bin/python
cmake --build build -j
```

## Roadmap

Near-term engineering priorities:

- Expand simulator parity tests against official environment traces.
- Add more targeted edge cases for scrolling, mines, factory elimination, and
  simultaneous special-action interactions.
- Replace heuristic action selection with macro-ISMCTS.
- Add rollout evaluation functions and macro priors.
- Add profiling benchmarks for `CrawlerSim::step()`.
- Prepare repeatable Kaggle submission packaging for compiled artifacts.

## License

See `LICENSE`.
