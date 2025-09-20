# DigiStar - High-Performance Physics Simulation

A clean, optimized physics simulation achieving **1M particles at 8+ FPS** on CPU.

## Quick Start

```bash
# Build
make

# Run interactive simulation
./digistar

# Run headless benchmark
./digistar headless

# Benchmark 1M particles
./digistar benchmark 1000000
```

## Architecture

### Clean Modular Design
```
src/
├── main.cpp                      # Main application & SDL2 viewer
└── physics/
    ├── modular_physics_backend.h # Composable physics system
    ├── cpu_collision_backend.h   # Optimized collision detection (8+ FPS @ 1M)
    ├── pm_gravity_backend.h      # O(N log N) gravity via FFT
    ├── sparse_spatial_grid.h     # Hash-based sparse grids (100MB vs 6TB)
    └── pm_solver.cpp/h           # Particle Mesh implementation
```

### Key Achievements
- **Collision Detection**: 115ms for 1M particles (8.68 FPS)
- **Memory Efficiency**: Sparse grids use ~100MB instead of 6TB
- **Modular Backends**: Easy to swap collision, gravity, and other force calculations
- **OpenMP Parallelization**: Near-linear scaling with CPU cores

### Performance (1M Particles)
| Component | Time | Notes |
|-----------|------|-------|
| Collision Detection | 115ms | 8.68 FPS with OpenMP |
| PM Gravity | 21ms | O(N log N) via FFT |
| Grid Update | 100ms | First build, then ~10ms incremental |
| **Total** | ~236ms | 4.2 FPS full physics |

### Optimizations
1. **Sparse Spatial Grids**: Only store occupied cells
2. **Optimized Cell Size**: 16 units (was 4 units → 8x fewer cells)
3. **Distance Filtering**: Reduce collision pairs from 45M to 1M
4. **Incremental Grid Updates**: Only ~1% of particles change cells per frame

## Building

### Dependencies
- C++17 compiler with OpenMP
- SDL2 (for visualization)
- FFTW3 (for PM gravity solver)

### Ubuntu/Debian
```bash
sudo apt install build-essential libsdl2-dev libfftw3-dev
make
```

## Usage

### Interactive Mode
```bash
./digistar [num_particles]
```
- Mouse: Pan camera
- Scroll: Zoom
- Space: Pause
- R: Reset
- G: Toggle grid overlay

### Benchmark Mode
```bash
./digistar benchmark [num_particles]
```
Outputs CSV data for performance analysis.

### Headless Mode
```bash
./digistar headless
```
Runs simulation without graphics for testing.

## Design Philosophy

This is a **clean rewrite** focusing on:
- **Performance**: 1M+ particles at interactive rates
- **Modularity**: Swappable physics backends
- **Simplicity**: Minimal dependencies, clear code
- **Scalability**: From 100 to 1M+ particles

The codebase has been reduced from 85 files to 14, removing dead code and focusing on what works.