"""Pytest bootstrap: make the repository root importable from the tests/ tree."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for path in (_ROOT, _ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
