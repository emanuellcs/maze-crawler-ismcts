"""Local alias for the Kaggle ``main.py`` Maze Crawler ISMCTS entrypoint.

Some local tools look for ``submission.agent`` while Kaggle requires
``main.agent`` at archive root.  Re-exporting the same callable avoids a second
Python entrypoint and preserves one documented observation/action contract.
"""

from main import agent

__all__ = ["agent"]
