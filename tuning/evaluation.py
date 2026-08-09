"""Wall-clock-bounded match execution helpers.

Both tuning and benchmarking need matches that cannot hang the process.  This
module provides a single threaded wrapper used by every game adapter so timeout
handling is identical everywhere.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Callable, Optional

_LOG = logging.getLogger("tuning.evaluation")

FAILURE_SCORE = -1.0e6


def run_with_timeout(run: Callable[[], Any], timeout: float) -> Any:
    """Execute ``run`` on a helper thread and enforce a wall-clock cap.

    Returns the callable's result or ``None`` if it did not finish in time or
    raised.  ``None`` is treated by adapters as a failed match.
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run)
            try:
                return future.result(timeout=max(1.0, timeout))
            except concurrent.futures.TimeoutError:
                _LOG.warning("match timed out after %.1fs", timeout)
                return None
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("match runner crashed: %s", exc)
        return None


def state_failed(status: Any) -> bool:
    """Return True when a Kaggle final state marks a failure."""
    text = str(getattr(status, "upper", lambda: str(status))())
    return any(token in text for token in ("ERROR", "INVALID", "TIMEOUT"))
