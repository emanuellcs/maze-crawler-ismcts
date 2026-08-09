"""Packaging and hot-path allocation-guard tests."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile

import package_submission

from _fixtures import REPO_ROOT


def test_policy_source_avoids_dynamic_allocation_primitives():
    """Ensure the policy hot path remains free of dynamic allocation primitives."""

    with open(
        os.path.join(REPO_ROOT, "src", "crawler_engine_policy.cpp"),
        encoding="utf-8",
    ) as policy_file:
        policy_source = policy_file.read()

    for forbidden in ("std::vector", "std::deque", "std::set", "new"):
        assert forbidden not in policy_source


def test_packaged_submission_jit_compiles_in_extracted_directory():
    """Verify the Kaggle source bundle can JIT-compile in a clean directory."""

    package_path = package_submission.build_package()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(package_path, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        smoke = subprocess.run(
            [sys.executable, "main.py"],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert smoke.returncode == 0, smoke.stdout + smoke.stderr
        assert "crawler_engine native available: True" in smoke.stdout

        call = subprocess.run(
            [
                sys.executable,
                "-c",
                "from types import SimpleNamespace\n"
                "from main import agent\n"
                "obs = SimpleNamespace(player=0, walls=[0] * 400, crystals={}, "
                "robots={'f0': [0, 5, 2, 1000, 0, 0, 0, 0]}, mines={}, "
                "miningNodes={}, southBound=0, northBound=19, step=0)\n"
                "config = SimpleNamespace(width=20, workerCost=200, wallRemoveCost=100)\n"
                "actions = agent(obs, config)\n"
                "assert 'f0' in actions\n"
                "print(actions)\n",
            ],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert call.returncode == 0, call.stdout + call.stderr
