"""Kaggle entrypoint for the Maze Crawler C++ engine."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from random import choice
import subprocess
import sys
import sysconfig
import traceback

try:
    import crawler_engine

    _ENGINE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - fallback is for submission diagnostics.
    crawler_engine = None
    _ENGINE_IMPORT_ERROR = exc


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


def _jit_log(message):
    """Emit native-build diagnostics to stderr without polluting action output."""

    print(f"[crawler_engine jit] {message}", file=sys.stderr, flush=True)


def _pybind11_include_dir():
    """Locate pybind11 headers from the submission vendor tree or dev install."""

    candidates = []
    vendor = _ROOT / "vendor" / "pybind11" / "include"
    candidates.append(vendor)
    try:
        import pybind11

        candidates.append(Path(pybind11.get_include()))
    except Exception:
        pass

    for path in candidates:
        if (path / "pybind11" / "pybind11.h").exists():
            return path
    return None


def _python_include_dirs():
    """Return Python include directories for the active Kaggle interpreter."""

    paths = sysconfig.get_paths()
    include_dirs = []
    for key in ("include", "platinclude"):
        value = paths.get(key)
        if value and Path(value).exists() and value not in include_dirs:
            include_dirs.append(value)
    return include_dirs


def _compile_native_engine():
    """JIT-compile crawler_engine when Kaggle has no compatible prebuilt module."""

    sources = sorted((_ROOT / "src").glob("*.cpp"))
    if not sources:
        _jit_log("no C++ sources found under src/")
        return False

    pybind_include = _pybind11_include_dir()
    if pybind_include is None:
        _jit_log(
            "pybind11 headers not found; expected vendor/pybind11/include or installed pybind11"
        )
        return False

    python_includes = _python_include_dirs()
    if not python_includes:
        _jit_log("Python development headers not found via sysconfig")
        return False

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    output = _ROOT / f"crawler_engine{ext_suffix}"
    compiler = os.environ.get("CXX", "g++")
    command = [
        compiler,
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        "-shared",
        "-ffast-math",
        "-march=native",
        "-Isrc",
        f"-I{pybind_include}",
    ]
    command.extend(f"-I{path}" for path in python_includes)
    command.extend(str(path.relative_to(_ROOT)) for path in sources)
    command.extend(["-o", str(output)])

    _jit_log("compiling native engine")
    _jit_log("command: " + " ".join(command))
    try:
        result = subprocess.run(
            command, cwd=_ROOT, capture_output=True, text=True, timeout=180
        )
    except Exception:
        _jit_log("compiler invocation failed")
        _jit_log(traceback.format_exc())
        return False

    if result.stdout:
        _jit_log("compiler stdout:\n" + result.stdout)
    if result.stderr:
        _jit_log("compiler stderr:\n" + result.stderr)
    if result.returncode != 0:
        _jit_log(f"compiler exited with status {result.returncode}")
        return False

    _jit_log(f"native engine built at {output.name}")
    return True


def _ensure_native_engine():
    """Import or build the native extension once per Python process."""

    global crawler_engine, _JIT_ATTEMPTED
    if crawler_engine is not None:
        return True
    if _JIT_ATTEMPTED:
        return False

    _JIT_ATTEMPTED = True
    if _ENGINE_IMPORT_ERROR is not None:
        _jit_log(f"initial import failed: {_ENGINE_IMPORT_ERROR}")
    if not _compile_native_engine():
        return False

    try:
        importlib.invalidate_caches()
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        crawler_engine = importlib.import_module("crawler_engine")
        _jit_log("native engine import succeeded after JIT compile")
        return True
    except Exception:
        _jit_log("native engine import failed after JIT compile")
        _jit_log(traceback.format_exc())
        crawler_engine = None
        return False


def _get(obj, name, default=None):
    """Read fields from either Kaggle dict observations or SimpleNamespace tests."""

    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _cfg(config, name, default):
    """Read configuration values with a default for local smoke objects."""

    return _get(config, name, default)


def _fallback_agent(obs, config):
    """Minimal legal policy used only when native compilation/import fails."""

    actions = {}
    width = _cfg(config, "width", 20)
    player = _get(obs, "player", 0)
    robots = _get(obs, "robots", {}) or {}
    walls = _get(obs, "walls", []) or []
    south_bound = _get(obs, "southBound", 0)

    my_robots = {uid: data for uid, data in robots.items() if data[4] == player}
    for uid, data in my_robots.items():
        rtype, col, row, energy = data[0], data[1], data[2], data[3]
        build_cd = data[7] if len(data) > 7 else 0
        idx = (row - south_bound) * width + col
        w = walls[idx] if 0 <= idx < len(walls) and walls[idx] != -1 else 0

        if rtype == 0:
            if w & 1:
                actions[uid] = "JUMP_NORTH"
            elif energy >= _cfg(config, "workerCost", 200) and build_cd == 0:
                actions[uid] = "BUILD_WORKER"
            else:
                actions[uid] = "NORTH"
        elif rtype == 2 and (w & 1) and energy >= _cfg(config, "wallRemoveCost", 100):
            actions[uid] = "REMOVE_NORTH"
        else:
            passable = []
            if not (w & 1):
                passable.append("NORTH")
            if not (w & 2):
                passable.append("EAST")
            if not (w & 4):
                passable.append("SOUTH")
            if not (w & 8):
                passable.append("WEST")
            actions[uid] = (
                "NORTH"
                if "NORTH" in passable
                else (choice(passable) if passable else "IDLE")
            )
    return actions


def agent(obs, config):
    """Kaggle entrypoint: update one persistent C++ engine and return actions."""

    _ensure_native_engine()
    if crawler_engine is None:
        return _fallback_agent(obs, config)

    player = int(_get(obs, "player", 0))
    engine = _ENGINES.get(player)
    if engine is None:
        engine = crawler_engine.Engine(player)
        engine.set_hyperparameters(BEST_PARAMS)
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
