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

# SDL2 for graphics viewer
find_package(SDL2 QUIET)
if(NOT SDL2_FOUND)
    pkg_check_modules(SDL2 QUIET sdl2)
endif()

if(SDL2_FOUND)
    message(STATUS "Found SDL2, graphics viewer will be built")
    set(BUILD_GRAPHICS_VIEWER ON)
else()
    message(WARNING "SDL2 not found, graphics viewer will be disabled")
    set(BUILD_GRAPHICS_VIEWER OFF)
endif()

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
    src/dsl/pattern_matcher.cpp
    src/dsl/procedural_generator.cpp
    src/dsl/event_bridge.cpp
    src/dsl/script_manager.cpp
    src/dsl/thread_pool.cpp
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

# Graphics viewer sources (only if SDL2 is available)
if(BUILD_GRAPHICS_VIEWER)
    set(GRAPHICS_VIEWER_SOURCES
        src/viewer/graphics_viewer.cpp
        src/viewer/viewer_event_bridge.cpp
    )
endif()

set(EVENT_SOURCES
    src/events/event_system.cpp
    src/events/event_producer.cpp
    src/events/event_consumer.cpp
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

# Graphics viewer library (optional)
if(BUILD_GRAPHICS_VIEWER)
    add_library(digistar_viewer STATIC ${GRAPHICS_VIEWER_SOURCES})
    target_link_libraries(digistar_viewer PUBLIC 
        digistar_core
        digistar_events
        ${SDL2_LIBRARIES}
    )
    target_include_directories(digistar_viewer PUBLIC ${SDL2_INCLUDE_DIRS})
    target_compile_definitions(digistar_viewer PUBLIC DIGISTAR_HAS_GRAPHICS_VIEWER)
endif()

add_library(digistar_events STATIC ${EVENT_SOURCES})
target_link_libraries(digistar_events PUBLIC 
    digistar_core
    Threads::Threads
)

# Main executable
add_executable(digistar src/main.cpp)
set(DIGISTAR_LIBRARIES
    digistar_simulation
    digistar_backend
    digistar_rendering
    digistar_events
    digistar_dsl
    ${FFTW3_LIBRARIES}
    Threads::Threads
)

# Add graphics viewer library if available
if(BUILD_GRAPHICS_VIEWER)
    list(APPEND DIGISTAR_LIBRARIES digistar_viewer)
endif()

target_link_libraries(digistar ${DIGISTAR_LIBRARIES})

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
    
    # Graphics viewer examples (only if SDL2 available)
    if(BUILD_GRAPHICS_VIEWER)
        add_executable(simple_graphics_test examples/simple_graphics_test.cpp)
        target_link_libraries(simple_graphics_test ${DIGISTAR_LIBRARIES})
        
        add_executable(graphics_viewer_demo examples/graphics_viewer_demo.cpp)
        target_link_libraries(graphics_viewer_demo ${DIGISTAR_LIBRARIES})
    endif()
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
message(STATUS "  Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:        ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ flags:           ${CMAKE_CXX_FLAGS}")
message(STATUS "  Build tests:         ${BUILD_TESTS}")
message(STATUS "  Build examples:      ${BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks:    ${BUILD_BENCHMARKS}")
message(STATUS "  Graphics viewer:     ${BUILD_GRAPHICS_VIEWER}")
message(STATUS "  Use OpenMP:          ${USE_OPENMP}")
message(STATUS "  Use CUDA:            ${USE_CUDA}")
message(STATUS "  FFTW3:               ${FFTW3_LIBRARIES}")
if(BUILD_GRAPHICS_VIEWER)
    message(STATUS "  SDL2:                ${SDL2_LIBRARIES}")
endif()
message(STATUS "")