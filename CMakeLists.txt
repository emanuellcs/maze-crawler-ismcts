cmake_minimum_required(VERSION 3.20)

project(maze_crawler_ismcts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
)

if(NOT PYBIND11_CMAKE_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR "pybind11 is not available. Install requirements-dev.txt first.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(crawler_engine
  src/crawler_engine_belief.cpp
  src/crawler_engine_engine.cpp
  src/crawler_engine_internal.cpp
  src/crawler_engine_policy.cpp
  src/crawler_engine_sim.cpp
  src/crawler_engine_state.cpp
  src/bindings.cpp
)

target_include_directories(crawler_engine PRIVATE src)

target_compile_options(crawler_engine PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3 -march=native -ffast-math>
)

target_compile_definitions(crawler_engine PRIVATE
  $<$<CONFIG:Release>:NDEBUG>
)
