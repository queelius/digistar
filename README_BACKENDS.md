# Backend System Overview

## Available Backends

### 1. CUDA Backend (GPU)
- **Location**: `src/backend/CUDABackend.cu`
- **Status**: ✅ Fully implemented
- **Features**:
  - PM solver using cuFFT for O(n) gravity
  - Handles 20-50M particles on RTX 3060
  - Full periodic boundary conditions
  - Optimized memory coalescing
- **Requirements**: NVIDIA GPU with CUDA

### 2. Simple CPU Backend  
- **Location**: `src/backend/backends_simple.cpp`
- **Status**: ✅ Working
- **Features**:
  - O(n²) direct gravity calculation
  - OpenMP parallelization
  - Good for < 10K particles
- **Requirements**: Any x86-64 CPU

### 3. AVX2 Backend (TODO)
- **Location**: `src/backend/AVX2OpenMPBackend.cpp`
- **Status**: 🚧 In progress
- **Features**:
  - PM solver with FFTW
  - AVX2 SIMD (8 floats per instruction)
  - Target: 100K-1M particles on modern CPUs

## Building

### On GPU Machine (with CUDA):
```bash
mkdir build && cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release
make -j
./benchmark_backends cuda  # Force CUDA
./benchmark_backends auto   # Auto-detect (will use CUDA)
```

### On CPU-only Machine:
```bash
mkdir build && cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release
make -j
./benchmark_backends auto   # Will use CPU backend
```

## Performance Expectations

| Backend | Hardware | Particles @ 60 FPS |
|---------|----------|-------------------|
| CUDA | RTX 3060 | 20-50M |
| CUDA | RTX 4090 | 100M+ |
| Simple CPU | 12-core | 5-10K |
| AVX2 (planned) | 12-core | 100K-500K |

## Code Structure

```
ISimulationBackend (interface)
├── CUDABackend (GPU implementation)
│   └── CUDABackendImpl (CUDA kernels)
├── SimpleBackend (CPU O(n²))
└── AVX2Backend (CPU with SIMD) [TODO]
```

## Key Files

- `ISimulationBackend.h` - Abstract interface all backends implement
- `BackendFactory.cpp` - Runtime backend selection
- `benchmark_backends.cpp` - Performance testing tool

## Moving Between Machines

The same code works on both your:
- **Desktop (RTX 3060)**: Automatically uses CUDA backend
- **Laptop (no GPU)**: Automatically uses CPU backend

Just copy the repo and rebuild - the factory auto-detects capabilities!