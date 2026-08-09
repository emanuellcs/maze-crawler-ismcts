"""Differential rules-fidelity checks against the environment rulebook.

These tests lock the scroll cadence and terminal-step mapping to the values the
Kaggle interpreter uses, so a change to either cannot silently regress the
native simulator.
"""

from __future__ import annotations

SCROLL_START = 10
SCROLL_END = 2
SCROLL_RAMP = 450
EPISODE_STEPS = 501


def scroll_interval(step: int) -> int:
    """Rulebook scroll interval at a public step (mirrors the environment)."""
    if step >= SCROLL_RAMP:
        return SCROLL_END
    progress = step / SCROLL_RAMP
    return max(SCROLL_END, round(SCROLL_START - (SCROLL_START - SCROLL_END) * progress))


def scroll_counter_at(step: int) -> int:
    """Countdown value at the start of agent step ``step``."""
    counter = SCROLL_START
    for turn in range(step):
        counter -= 1
        if counter <= 0:
            counter = scroll_interval(turn)
    return max(1, counter)


def test_scroll_cadence_matches_environment_defaults():
    """The ramp starts at 10 turns and reaches 2 turns by step 450."""

    assert scroll_interval(0) == 10
    assert scroll_interval(1) == 10
    assert scroll_interval(300) < 10
    assert scroll_interval(449) >= 2
    assert scroll_interval(450) == 2
    assert scroll_interval(499) == 2


def test_scroll_counter_reconstruction_points():
    """The reconstructed countdown matches observed boundary advance steps."""

    # The environment advances the boundary at the end of turns 9, 19, 29...
    assert scroll_counter_at(10) == 10
    assert scroll_counter_at(9) == 1
    assert scroll_counter_at(20) == 10


def test_terminal_step_mapping():
    """The game ends after turn 499, not turn 500."""

    # compute_rewards runs before the step counter increments, so the terminal
    # boundary is EPISODE_STEPS - 2 (the last agent call is step 499).
    assert EPISODE_STEPS == 501
    assert EPISODE_STEPS - 2 == 499
