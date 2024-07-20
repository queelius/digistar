cmake_minimum_required(VERSION 3.17)
project(SimulationProject)

# Set the CUDA and C++ standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/simulation.cu
    src/simulation_api.cpp
    main.cpp
)

# Add the executable
add_executable(simulation ${SOURCES})

# Specify the CUDA architectures
set_target_properties(simulation PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_SEPARABLE_COMPILATION ON
)

# If you're using CMake 3.17 or newer, this might not be necessary
find_package(CUDA)

# Link against the CUDA runtime library, adjusting for CMake version and setup
# For CMake 3.18 and newer, you might directly link to CUDA libraries like this
target_link_libraries(simulation cudart)