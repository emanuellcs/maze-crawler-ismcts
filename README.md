# Maze Crawler ISMCTS

Ahead-of-time C++ engine scaffold for Kaggle's [**Maze Crawler**](https://www.kaggle.com/competitions/maze-crawler) competition.

The goal of this repository is to keep the Python submission layer thin and move the expensive work into a deterministic C++ engine exposed through `pybind11`. The current implementation is a working scaffold: it builds, imports from Python, runs rule-focused smoke tests, and contains the fixed-buffer state layout needed for later search tuning.

## Architecture

- `src/crawler_engine.hpp` defines the fixed-size state model:
  - absolute global grid arrays: `20 * 512` cells for walls, crystals, mines, nodes, and belief fields.
  - active 20x20 tactical bitboards backed by `uint64_t[7]`.
  - robot Structure of Arrays storage with fixed capacity and UID slots.
- `src/` is split by engine responsibility:
  - `crawler_engine_state.cpp` implements fixed-buffer storage, action parsing, board queries, and tactical bitboards.
  - `crawler_engine_belief.cpp` implements observation merging, enemy probability diffusion, and determinization.
  - `crawler_engine_sim.cpp` implements deterministic turn stepping, combat, resources, and scrolling.
  - `crawler_engine_policy.cpp` implements heuristic actions, macro-action generation, and macro-to-primitive translation.
  - `crawler_engine_internal.cpp` contains shared deterministic helpers for RNG, row generation, scrolling, and wall mutation.
- `src/bindings.cpp` exposes the engine as a Python module named `crawler_engine`.
- `main.py` is the Kaggle entrypoint. It maintains one C++ engine per player and returns `{uid: "PRIMITIVE_ACTION"}`.
- `submission.py` is a local alias for `main.agent`.
- `test.py` contains bridge and core rule smoke tests.

## Requirements

The system Python in many dev containers is externally managed, so a virtual environment is recommended.

```bash
python3 -m venv /tmp/maze-crawler-venv
/tmp/maze-crawler-venv/bin/python -m pip install -r requirements-dev.txt
```

Dependencies:

- `pybind11`
- `numpy`
- `pytest`
- `kaggle-environments`

## Build

Configure CMake with the same Python interpreter that has `pybind11` installed:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/tmp/maze-crawler-venv/bin/python
cmake --build build -j
```

Release builds use C++20 and, for GCC/Clang, `-O3 -march=native -ffast-math`.

The compiled extension is written under `build/`, for example:

```text
build/crawler_engine.cpython-313-x86_64-linux-gnu.so
```

## Test

Run the smoke script:

```bash
PYTHONPATH=build /tmp/maze-crawler-venv/bin/python test.py
```

Run the pytest form:

```bash
PYTHONPATH=build /tmp/maze-crawler-venv/bin/python -m pytest -q test.py
```

Current smoke coverage includes:

- Python bridge import and action generation.
- Factory build/spawn behavior.
- Same-type annihilation.
- Stronger-type crush resolution.
- Crystal collection and combat consumption.
- Transfer cap behavior.
- Off-board jump destruction.

## Python API

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

`actions` is a Kaggle-compatible dictionary:

```python
{"0-0": "BUILD_WORKER", "3-4": "NORTH"}
```

The bound API also exposes:

- `engine.step(actions)`
- `engine.determinize(seed=0)`
- `engine.debug_state()`

## Kaggle Entrypoint

`main.py` defines:

```python
def agent(obs, config):
    ...
```

For local use with the compiled extension:

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

If `crawler_engine` cannot be imported, `main.py` falls back to a simple Python policy so import failures are visible without crashing immediately.

## Implementation Notes

- Hot simulation state is fixed-capacity. `CrawlerSim::step()` uses fixed scratch arrays and avoids heap allocation.
- Board truth and belief use absolute row indexing to avoid repeated active-window coordinate arithmetic in longer rollouts.
- Active-window bitboards are rebuilt for tactical masks and iterate with C++20 bit operations.
- Enemy belief update explicitly clears probability from visible empty cells, then diffuses hidden probability by unit speed and known walls.
- Hidden future rows are sampled with optimistic east/west symmetry and center-door priors rather than random wall noise.

## Current Status

This is a scaffold, not a finished leaderboard bot. The deterministic engine and bridge are in place, but the Macro-ISMCTS layer currently uses conservative fixed-buffer placeholders and heuristic fallback actions. The next major work item is validating simulator parity against the official environment on a larger set of generated edge cases, then replacing the placeholder search policy with tuned rollouts and macro priors.
