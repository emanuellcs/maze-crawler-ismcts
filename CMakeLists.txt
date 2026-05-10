cmake_minimum_required(VERSION 3.20)

# Maze Crawler native extension build.
#
# CMake is the development and CI build path for the same C++20 engine that
# `main.py` can JIT-compile inside Kaggle.  The produced `crawler_engine` module
# contains the fixed-buffer simulator, Belief State Determinization, Bitboards,
# rollout policy, and fixed-arena ISMCTS search exposed through pybind11.
project(maze_crawler_ismcts LANGUAGES CXX)

# C++20 is required for `std::countr_zero`, aggregate fixed-buffer state, and
# clean pybind11 integration.  Compiler extensions are disabled so the source
# package remains portable across local, CI, and Kaggle compilers.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for editor tooling and release artifacts.  The file is
# not part of the Kaggle submission path, but it is useful for code review and CI.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Bind the extension to the active Python interpreter and module ABI.  CI passes
# `Python3_EXECUTABLE` explicitly so Python 3.11 and 3.12 builds do not cross
# wires with any system interpreter.
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Ask the installed pybind11 package where its CMake config lives.  This avoids
# vendoring pybind11 for local development while `package_submission.py` still
# vendors headers for Kaggle's source-only JIT path.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
)

if(NOT PYBIND11_CMAKE_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR "pybind11 is not available. Install requirements-dev.txt first.")
endif()

# Make the discovered pybind11 config visible to `find_package(pybind11)`.
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# Build one Python extension module containing every native engine component.
# The source split mirrors runtime ownership: belief, simulation, policy, MCTS,
# fixed-buffer state, internal helpers, facade, and pybind conversion.
pybind11_add_module(crawler_engine
  src/crawler_engine_belief.cpp
  src/crawler_engine_engine.cpp
  src/crawler_engine_internal.cpp
  src/crawler_engine_mcts.cpp
  src/crawler_engine_policy.cpp
  src/crawler_engine_sim.cpp
  src/crawler_engine_state.cpp
  src/bindings.cpp
)

target_include_directories(crawler_engine PRIVATE src)

# Warnings stay enabled in every config to catch narrowing or rule-drift bugs.
# Release builds use `-O3` plus `-march=native` so local/CI binaries can exploit
# CPU features such as AVX2 when available.  The code does not require AVX2
# intrinsics; `-march=native` lets the compiler auto-vectorize fixed-array scans
# in Bitboard, policy, and rollout loops.  `-ffast-math` is acceptable here
# because search/evaluation scores are heuristic and not persisted as exact
# protocol values.
target_compile_options(crawler_engine PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3 -march=native -ffast-math>
)

# Match the JIT path's Release behavior by defining `NDEBUG` outside debug
# builds, eliminating assertion overhead in the mission-critical search path.
target_compile_definitions(crawler_engine PRIVATE
  $<$<CONFIG:Release>:NDEBUG>
)
