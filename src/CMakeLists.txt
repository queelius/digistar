cmake_minimum_required(VERSION 3.18)
project(digistar LANGUAGES CXX)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(HAS_CUDA TRUE)
else()
    set(HAS_CUDA FALSE)
    message(STATUS "CUDA not found - GPU backend will not be available")
endif()

# Find OpenMP
find_package(OpenMP)

# Find FFTW (for CPU PM solver)
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(FFTW fftw3f)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
if(HAS_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
endif()

# Backend library - simplified version for now
add_library(backends STATIC
    backend/backends_simple.cpp
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(backends PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link FFTW if available
if(FFTW_FOUND)
    target_include_directories(backends PUBLIC ${FFTW_INCLUDE_DIRS})
    target_link_libraries(backends PUBLIC ${FFTW_LIBRARIES})
    target_compile_definitions(backends PUBLIC HAS_FFTW)
endif()

# Include directories
target_include_directories(backends PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Benchmark executable
add_executable(benchmark_backends benchmark_backends.cpp)
target_link_libraries(benchmark_backends backends)

# Original gravity baseline (if CUDA available)
if(HAS_CUDA)
    add_executable(gravity_baseline gravity_baseline.cu)
    target_link_libraries(gravity_baseline 
        CUDA::cudart
        CUDA::cufft
    )
    set_target_properties(gravity_baseline PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Optional: Add visualization later
# find_package(OpenGL)
# find_package(GLFW3)
# if(OpenGL_FOUND AND GLFW3_FOUND)
#     add_executable(gravity_viz gravity_viz.cpp)
#     target_link_libraries(gravity_viz OpenGL::GL glfw)
# endif()

# Build instructions
message(STATUS "")
message(STATUS "Build with: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build")
message(STATUS "Run with: ./build/gravity_baseline")
message(STATUS "")