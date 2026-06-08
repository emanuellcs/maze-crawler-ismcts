"""Kaggle entrypoint for the fixed-buffer Maze Crawler ISMCTS engine.

The module owns the Python side of the runtime contract: import or JIT-compile
the native pybind11 extension, cache one C++ ``Engine`` per player, translate
sparse Kaggle observations into dense native buffers, and execute ISMCTS
within the rule-defined time budget.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# The engine player expects these types to be available for type hints and docstrings.
# During JIT or Kaggle submission, we mock or fallback if the extension is absent.
try:
    import crawler_engine
except ImportError:
    crawler_engine = None

_ENGINES = {}
_JIT_ATTEMPTED = False
_ROOT = Path(__file__).resolve().parent
BEST_PARAMS = {
    "C_puct": 2.0884330868271443,
    "baseline_prior_multiplier": 1.8863044112273712,
    "rollout_depth": 80,
    "IDLE": 0.49504719444108913,
    "FACTORY_SUPPORT_WORKER": 1.0390992283842135,
    "FACTORY_SAFE_ADVANCE": 2.161864767112469,
    "FACTORY_BUILD_WORKER": 0.997532683560502,
    "FACTORY_BUILD_SCOUT": 1.8649154683704814,
    "FACTORY_BUILD_MINER": 0.8269752415957998,
    "FACTORY_JUMP_OBSTACLE": 1.4925105256980329,
    "WORKER_OPEN_NORTH_WALL": 0.7351332485632837,
    "WORKER_ESCORT_FACTORY": 1.583596636264722,
    "WORKER_ADVANCE": 0.8874633406271473,
    "SCOUT_HUNT_CRYSTAL": 1.083268581072123,
    "SCOUT_EXPLORE_NORTH": 1.6851712172373043,
    "SCOUT_RETURN_ENERGY": 1.1040824358243229,
    "MINER_SEEK_NODE": 0.6983213636231111,
    "MINER_TRANSFORM": 0.5309500821275892,
}
_CURRENT_PARAMS = BEST_PARAMS.copy()


def set_hyperparameters(**kwargs):
    """Update global hyperparameters and clear the engine cache.

    This function is used by tuning harnesses to ensure fresh engines are
    created with the requested parameters.
    """

    global _CURRENT_PARAMS, _ENGINES
    _CURRENT_PARAMS.update(kwargs)
    _ENGINES.clear()


def _jit_log(message):
    """Emit native-build diagnostics to stderr without polluting actions."""
    print(f"[JIT] {message}", file=sys.stderr)


def _ensure_native_engine():
    """Import the native extension, attempting JIT compilation if needed.

    In development, the extension is usually built via CMake.  In Kaggle, the
    source is provided in the submission, and we compile it on the first turn
    using a temporary directory for build artifacts.
    """

    global crawler_engine, _JIT_ATTEMPTED
    if crawler_engine is not None:
        return True
    if _JIT_ATTEMPTED:
        return False
    _JIT_ATTEMPTED = True

    # Search for pre-built .so in common locations.
    for p in [_ROOT, _ROOT / "build", _ROOT / "lib"]:
        sos = list(p.glob("crawler_engine*.so"))
        if sos:
            sys.path.insert(0, str(p))
            try:
                import crawler_engine
                _jit_log(f"Loaded pre-built extension from {p}")
                return True
            except ImportError:
                sys.path.pop(0)

    # Attempt JIT if pybind11 and source are available.
    src_dir = _ROOT / "src"
    if not src_dir.exists():
        return False

    try:
        import pybind11
    except ImportError:
        _jit_log("pybind11 not found; skipping JIT")
        return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        build_dir = Path(tmp_dir)
        _jit_log(f"Compiling native engine in {build_dir}")
        
        # In a real Kaggle environment, we'd use g++ directly to avoid CMake overhead.
        # This mirrors the logic in package_submission.py.
        sources = list(src_dir.glob("*.cpp"))
        cmd = [
            "g++", "-O3", "-shared", "-std=c++20", "-fPIC",
            f"-I{src_dir}",
            *subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"]).decode().split(),
            *[str(s) for s in sources],
            "-o", str(build_dir / "crawler_engine.so"),
            "-march=native", "-ffast-math"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            sys.path.insert(0, str(build_dir))
            # We must copy the .so out of the temp dir if we want it to persist for the session,
            # or keep build_dir in sys.path and accept it's transient.
            # For simplicity in this scaffold, we just import it.
            import crawler_engine
            _jit_log("JIT compilation successful")
            return True
        except subprocess.CalledProcessError as e:
            _jit_log(f"JIT compilation failed: {e.stderr.decode()}")
            return False


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _fallback_agent(obs, config):
    """Return empty actions if the native engine is unavailable."""
    return {}


def agent(obs, config):
    """Kaggle-compatible agent entrypoint."""

    if not _ensure_native_engine():
        return _fallback_agent(obs, config)

    if crawler_engine is None:
        return _fallback_agent(obs, config)

    player = int(_get(obs, "player", 0))
    engine = _ENGINES.get(player)
    if engine is None:
        engine = crawler_engine.Engine(player)
        engine.set_hyperparameters(_CURRENT_PARAMS)
        _ENGINES[player] = engine

    step = int(_get(obs, "step", -1))
    engine.update_observation(
        player,
        _get(obs, "walls", []),
        _get(obs, "crystals", {}) or {},
        _get(obs, "robots", {}) or {},
        _get(obs, "mines", {}) or {},
        _get(obs, "miningNodes", {}) or {},
        int(_get(obs, "southBound", 0)),
        int(_get(obs, "northBound", 19)),
        step,
    )
    return engine.choose_actions(2000, seed=(step + 1) * 1315423911 + player)


if __name__ == "__main__":
    print(f"crawler_engine native available: {_ensure_native_engine()}")
