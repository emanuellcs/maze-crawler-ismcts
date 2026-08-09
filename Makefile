# Maze Crawler ISMCTS — development convenience targets.
#
# All commands assume the shared virtualenv at the workspace root:
#     source ../.venv/bin/activate
# Build artifacts are written to ./build and imported via PYTHONPATH=build.

PYTHON ?= python

.PHONY: build test bench tune diffcheck profile package clean

build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
	  -DPython3_EXECUTABLE="$(shell $(PYTHON) -c 'import sys; print(sys.executable)')"
	cmake --build build -j"$$(nproc)"

test: build
	PYTHONPATH=build $(PYTHON) -m pytest -q tests

bench: build
	PYTHONPATH=build $(PYTHON) scripts/bench.py

tune: build
	PYTHONPATH=build $(PYTHON) scripts/tune.py

diffcheck: build
	PYTHONPATH=build $(PYTHON) scripts/diffcheck.py

profile: build
	PYTHONPATH=build $(PYTHON) scripts/profiling.py

package: build
	$(PYTHON) scripts/package_submission.py

clean:
	rm -rf build dist submission.tar.gz .pytest_cache
