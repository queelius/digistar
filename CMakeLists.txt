cmake_minimum_required(VERSION 3.16)
project(DigiStar 
    VERSION 0.1.0
    DESCRIPTION "High-performance particle simulation with big atoms"
    LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -march=native")

# Options
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_BENCHMARKS "Build benchmark programs" ON)
option(USE_OPENMP "Use OpenMP for parallelization" ON)
option(USE_CUDA "Build CUDA backend" OFF)

# Find packages
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# FFTW3
pkg_check_modules(FFTW3 REQUIRED fftw3f)

# OpenMP
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# CUDA
if(USE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    else()
        message(WARNING "CUDA not found. CUDA backend will not be built.")
        set(USE_CUDA OFF)
    endif()
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${FFTW3_INCLUDE_DIRS}
)

# Link directories
link_directories(
    ${FFTW3_LIBRARY_DIRS}
)

# Source files
set(CORE_SOURCES
    src/physics/pools.h
    src/physics/types.h
    src/physics/spatial_index.h
    src/physics/particle_mesh.h
)

set(DSL_SOURCES
    src/dsl/sexpr.cpp
    src/dsl/evaluator.cpp
    src/dsl/command.cpp
)

set(BACKEND_SOURCES
    src/backend/cpu_backend_reference.cpp
    src/backend/cpu_backend_openmp.cpp
    src/backend/BackendFactory.cpp
)

set(SIMULATION_SOURCES
    src/simulation/simulation.cpp
    src/core/simulation.cpp
)

set(RENDERING_SOURCES
    src/visualization/ascii_renderer.cpp
)

# Libraries
add_library(digistar_core INTERFACE)
target_include_directories(digistar_core INTERFACE src)

add_library(digistar_dsl STATIC ${DSL_SOURCES})
target_link_libraries(digistar_dsl PUBLIC digistar_core)

add_library(digistar_backend STATIC ${BACKEND_SOURCES})
target_link_libraries(digistar_backend PUBLIC 
    digistar_core 
    ${FFTW3_LIBRARIES}
    Threads::Threads
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(digistar_backend PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(digistar_simulation STATIC ${SIMULATION_SOURCES})
target_link_libraries(digistar_simulation PUBLIC 
    digistar_core
    digistar_backend
    digistar_dsl
)

add_library(digistar_rendering STATIC ${RENDERING_SOURCES})
target_link_libraries(digistar_rendering PUBLIC digistar_core)

# Main executable
add_executable(digistar src/main.cpp)
target_link_libraries(digistar 
    digistar_simulation
    digistar_backend
    digistar_rendering
    digistar_dsl
    ${FFTW3_LIBRARIES}
    Threads::Threads
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation
install(TARGETS digistar
    RUNTIME DESTINATION bin
)

install(DIRECTORY src/
    DESTINATION include/digistar
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration
message(STATUS "")
message(STATUS "DigiStar Configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:      ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ flags:         ${CMAKE_CXX_FLAGS}")
message(STATUS "  Build tests:       ${BUILD_TESTS}")
message(STATUS "  Build examples:    ${BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks:  ${BUILD_BENCHMARKS}")
message(STATUS "  Use OpenMP:        ${USE_OPENMP}")
message(STATUS "  Use CUDA:          ${USE_CUDA}")
message(STATUS "  FFTW3:             ${FFTW3_LIBRARIES}")
message(STATUS "")