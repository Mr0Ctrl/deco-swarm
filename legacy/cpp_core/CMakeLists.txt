cmake_minimum_required(VERSION 3.16)
project(deco_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# Find Eigen3 (Header-only high-performance linear algebra library)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(include)

# 1. Standalone Core C++ static library
add_library(deco_core_lib STATIC
    src/geometry.cpp
    src/collision.cpp
)
target_link_libraries(deco_core_lib PUBLIC Eigen3::Eigen)

# 2. Pybind11 Python Extension Module
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Use pybind11 from active Python interpreter
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(deco_core_cpp
    src/bindings.cpp
)

target_link_libraries(deco_core_cpp PRIVATE deco_core_lib Eigen3::Eigen)
