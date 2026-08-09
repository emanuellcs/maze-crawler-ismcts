# Contributing

This guide covers the development workflow for the Maze Crawler engine: environment setup, building, testing, benchmarking, code style, and how to add a hyperparameter or an opponent archetype.

## Development Workflow

1. Create a feature branch from `main`.
2. Make the change, including regression tests for any rule-sensitive behavior.
3. Run `make test` and `make diffcheck`.
4. Run `make bench` to measure the change against the opponent league.
5. If the change targets search strength, run a tuning study and promote the champion.
6. Open a pull request with a concise description and the benchmark result.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Building

```bash
make build
```

The build produces the `crawler_engine` extension under `build/`. Import it with `PYTHONPATH=build`.

The CMake build is the development path. The `main.py` JIT path compiles the same sources with the equivalent flags and is exercised by the packaging tests.

## Testing

```bash
make test
```

The pytest suite lives in `tests/` and is split by concern:

| Module | Covers |
| --- | --- |
| `test_bridge.py` | pybind observation ingestion, hyperparameters, determinization |
| `test_sim.py` | Rule-phase simulator behavior: combat, transfers, scroll, jumps |
| `test_policy.py` | Deterministic baseline policy and macro library |
| `test_search.py` | ISMCTS search behavior and time-budget handling |
| `test_packaging.py` | Hot-path allocation guard, Kaggle JIT packaging |
| `test_opponents.py` | Opponent archetype registration |
| `test_differential.py` | Scroll cadence and terminal-step rules fidelity |

Run a single module with `PYTHONPATH=build python -m pytest -q tests/test_sim.py`.

## Rules Fidelity

```bash
make diffcheck
```

`scripts/diffcheck.py` plays short full games with the native agent and verifies the scroll cadence and termination match the environment. Run it before changing any simulator behavior.

## Benchmarking

```bash
make bench
```

The benchmark harness plays the champion against every opponent in `config/league.json` plus the hand-crafted archetypes on held-out seeds with seat rotation. It reports win rates with a Wilson confidence interval. The promotion gate and Elo ladder are described in [docs/BENCHMARKING.md](./docs/BENCHMARKING.md).

## Tuning

```bash
make tune
```

The tuning system is documented in [docs/TUNING.md](./docs/TUNING.md). In summary:

- The search space and champion values live in `config/hyperparameters.json`.
- Trials run in spawned worker processes with fixed seeds and seat rotation.
- The champion is promoted back to `config/hyperparameters.json` and registered in `config/league.json`.

## Code Style and Comment Policy

The repository follows a concise-API, minimal-internals policy.

### C++

- No `@file`, `@author`, or `@date` boilerplate. At most one `//` purpose line at the top of a large translation unit.
- Headers carry a concise `/// @brief` and only essential `@param` and `@return` annotations. Struct fields get inline `///<` only when the semantics are non-obvious.
- Internal and static functions get no doc block. Add a single `//` comment only where the "why" is non-obvious, such as rules ordering, collision precedence, aliasing, or invariants.
- Comments explain why, not what. Delete stale and dead-code comments and keep comments in sync with the code.
- Prefer a named `constexpr` over a magic-number comment.
- Aim for roughly 5 to 10 percent comment lines in hot files.

### Python

- Module docstrings are one to three sentences describing purpose.
- Concise docstrings appear only on public entrypoints such as `main.agent`, `set_hyperparameters`, CLI functions, and public `tuning/` functions.
- Internal helpers get no docstring. Inline `#` comments appear only for non-obvious reasoning.
- No shouting comments and no redundant `Args`, `Returns`, or `Notes` blocks on internal code.

### Configuration

`config/hyperparameters.json` is the single source of truth for tuned parameters and search-space bounds. Do not hardcode tuned values in `main.py` or in C++ defaults. Regenerate derived tables from the config file.

## Adding a New Hyperparameter

1. Add the parameter to the native config struct. For Maze Crawler this is `Hyperparameters` in `src/include/search.hpp`.
2. Expose it through the pybind layer in `src/crawler_engine_bindings.cpp` so `set_hyperparameters` and `get_hyperparameters` handle it.
3. Add the parameter name, type, bounds, and default to `config/hyperparameters.json` under `schema`.
4. If the parameter is tuned, add it to the champion block.
5. Add a round-trip assertion in `tests/test_bridge.py`.
6. Run `make test`.

## Adding a New Opponent Archetype

1. Create `opponents/<name>.py` with an `agent(obs, config)` function and an `act` fallback wrapper.
2. Make the policy deterministic for a given seed so benchmark results are reproducible.
3. Register the archetype in the `HANDCRAFTED_OPPONENTS` tuple in `scripts/tune.py` and `scripts/bench.py`.
4. Add the name to the parameterized test in `tests/test_opponents.py`.
5. Run `make test` and `make bench`.

## CI/CD

The GitHub Actions workflow builds and tests the project on Ubuntu for Python 3.11 and 3.12. The pipeline runs the pytest suite, the differential rules-fidelity check, a benchmark smoke, a profiling perf gate, and an ASan/UBSan sanitizer build. Tagged releases publish workflow artifacts through `gh release`.

## Troubleshooting

- `ModuleNotFoundError: crawler_engine`: build the extension or set `PYTHONPATH=build`.
- JIT packaging failures: ensure pybind11 is installed, since `scripts/package_submission.py` vendors its headers into the bundle.
- Failing differential checks: the simulator no longer matches the environment scroll cadence or terminal step. Check `tests/test_differential.py` for the locked expectations.
