"""Kaggle entrypoint for the fixed-buffer Maze Crawler ISMCTS engine.

The module owns the Python side of the runtime contract: import or JIT-compile
the native pybind11 extension, cache one C++ ``Engine`` per player, translate
sparse Kaggle observations into dense native buffers, and execute ISMCTS
within the rule-defined time budget.
"""

import json
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


def _load_champion():
    """Load the tuned champion parameters from the single source of truth.

    ``config/hyperparameters.json`` is authoritative.  If it is missing (for
    example during a bare checkout or in tests), an empty set is returned and
    the native engine falls back to its compiled-in defaults.
    """
    try:
        data = json.loads((_ROOT / "config" / "hyperparameters.json").read_text())
    except Exception:
        return {}
    params = dict(data.get("champion", {}))
    params.pop("search_time_ms", None)
    return params


BEST_PARAMS = _load_champion()
_CURRENT_PARAMS = BEST_PARAMS.copy()
_SEARCH_TIME_MS = 2000


def set_hyperparameters(**kwargs):
    """Update global hyperparameters and clear the engine cache.

    ``search_time_ms`` is a runtime knob (not an engine hyperparameter) that
    caps per-turn native search time. It lets tuning harnesses and local smoke
    tests trade strength for wall-clock speed; the Kaggle submission keeps the
    full budget.
    """

    global _CURRENT_PARAMS, _ENGINES, _SEARCH_TIME_MS
    if "search_time_ms" in kwargs:
        _SEARCH_TIME_MS = int(kwargs.pop("search_time_ms"))
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
            f"-I{src_dir}/include",
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
    return engine.choose_actions(_SEARCH_TIME_MS, seed=(step + 1) * 1315423911 + player)


if __name__ == "__main__":
    print(f"crawler_engine native available: {_ensure_native_engine()}")
