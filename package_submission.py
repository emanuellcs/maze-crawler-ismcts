"""Create the Kaggle source bundle for the Maze Crawler ISMCTS engine.

The bundle contains ``main.py``, every C++ source/header needed by the
fixed-buffer engine, and vendored pybind11 headers.  Kaggle can then JIT-compile
the same native simulator, Belief State, Bitboard, Rollout, and ISMCTS code
without relying on a prebuilt platform-specific extension.
"""

from __future__ import annotations

from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission.tar.gz"


def _pybind11_include_dir() -> Path:
    """Find installed pybind11 headers to vendor into the source bundle.

    Returns
    -------
    pathlib.Path
        Include directory containing ``pybind11/pybind11.h``.

    Raises
    ------
    RuntimeError
        If pybind11 is unavailable or the header layout is unexpected.
    """

    try:
        import pybind11
    except Exception as exc:  # pragma: no cover - packaging requires dev deps.
        raise RuntimeError(
            "pybind11 is required to vendor headers into the submission"
        ) from exc

    include = Path(pybind11.get_include())
    if not (include / "pybind11" / "pybind11.h").exists():
        raise RuntimeError(f"pybind11 headers not found under {include}")
    return include


def _add_file(tar: tarfile.TarFile, path: Path, arcname: Path) -> None:
    """Add one file with a stable archive name and no recursive traversal.

    Parameters
    ----------
    tar:
        Open output archive.
    path:
        Source file on disk.
    arcname:
        Archive-relative path used by Kaggle after extraction.
    """

    tar.add(path, arcname=str(arcname), recursive=False)


def build_package() -> Path:
    """Create ``submission.tar.gz`` with the entrypoint, engine, and pybind11.

    Returns
    -------
    pathlib.Path
        Path to the generated archive.
    """

    pybind_include = _pybind11_include_dir()
    with tarfile.open(OUT, "w:gz") as tar:
        _add_file(tar, ROOT / "main.py", Path("main.py"))

        for path in sorted((ROOT / "src").glob("*.cpp")) + sorted(
            (ROOT / "src").glob("*.hpp")
        ):
            _add_file(tar, path, Path("src") / path.name)

        for path in sorted(pybind_include.rglob("*")):
            if path.is_file():
                _add_file(
                    tar,
                    path,
                    Path("vendor")
                    / "pybind11"
                    / "include"
                    / path.relative_to(pybind_include),
                )

    return OUT


if __name__ == "__main__":
    package = build_package()
    print(f"wrote {package}")
