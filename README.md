# DigiStar - High-Performance N-Body Particle Simulation

A massively parallel particle simulation system capable of simulating millions of particles in real-time using multiple force calculation algorithms.

## Features

- **Multiple Force Algorithms**
  - Brute Force: O(n²) - accurate baseline for small systems
  - Barnes-Hut: O(n log n) - tree-based approximation for medium systems
  - Particle Mesh: O(n) - FFT-based for large-scale simulations
  
- **Optimized Backends**
  - CPU with OpenMP parallelization
  - SSE2/AVX2 SIMD optimization
  - CUDA GPU acceleration (optional)
  
- **Real-Time Visualization**
  - ASCII terminal visualization for up to 2M particles
  - Interactive controls for panning and zooming
  - Energy conservation monitoring
  
- **Validated Physics**
  - Accurate solar system simulation with real units
  - Energy conservation tests
  - Orbital mechanics validation

## Quick Start

```bash
# Build everything
make all

# Run solar system example
./build/bin/solar_system

# Run million particle demo
./build/bin/million_particles

# Compare algorithm accuracy
./build/bin/backend_comparison
```

## Project Structure

```
digistar/
├── src/                    # Source code
│   ├── backend/           # Backend implementations
│   ├── algorithms/        # Force calculation algorithms
│   ├── spatial/           # Spatial data structures
│   ├── dsl/               # Domain-specific language
│   └── core/              # Core functionality
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── validation/       # Physics validation
├── benchmarks/           # Performance benchmarks
├── examples/             # Example applications
├── tools/                # Utility tools
└── docs/                 # Design documentation
    ├── DSL_DESIGN.md                  # DSL overview
    ├── DSL_LANGUAGE_SPEC.md           # Language specification
    ├── DSL_ADVANCED_FEATURES.md       # Advanced DSL features
    ├── EVENT_SYSTEM_ARCHITECTURE.md   # Event system design
    └── EMERGENT_COMPOSITE_SYSTEM.md   # Composite body system
```

## Building

### Requirements
- C++17 compiler (GCC 8+ or Clang 10+)
- OpenMP support
- FFTW3 library (optional, for PM algorithm)
- CUDA toolkit (optional, for GPU backend)

### Build Commands
```bash
make all        # Build everything
make backends   # Build backend libraries
make tests      # Build test suite
make benchmarks # Build benchmarks
make examples   # Build example applications
make tools      # Build utility tools
make cuda       # Build CUDA backend (requires NVCC)
make clean      # Clean build artifacts
```

## Performance

Achieved performance on typical hardware:
- **1M particles**: 24 FPS with Barnes-Hut on 8-core CPU
- **2M particles**: 12 FPS with Particle Mesh on 8-core CPU
- **10M particles**: Target with GPU acceleration

## Algorithms

### Barnes-Hut Tree
- Quadtree spatial subdivision
- Opening angle θ = 0.5 for accuracy/speed balance
- ~1% energy drift over 1000 steps

### Particle Mesh
- Custom FFT implementation (zero dependencies)
- Cloud-In-Cell (CIC) interpolation
- Periodic boundary conditions

## Testing

```bash
# Run unit tests
./build/bin/test_algorithms

# Test convergence
./build/bin/test_convergence

# Validate accuracy
./build/bin/test_accuracy
```

## Examples

### Solar System Simulation
Accurate simulation of the solar system with real gravitational constant and units:
```bash
./build/bin/solar_system
```

### Million Particle Cloud
Interactive visualization of 1-2 million particles:
```bash
./build/bin/million_particles
```

## Original Vision

The long-term vision is to create a highly interactive and scalable sandbox space simulation game that can simulate vast numbers of "big atoms" interacting through various forces. Key goals include:

- Simulating 10+ million particles with complex multi-star systems
- Supporting concurrent players and AI bots for multiplayer experience
- Providing a DSL for celestial mechanics
- Enabling novel physics including relativistic effects and exotic phenomena
- GPU acceleration with CUDA for maximum performance
- Efficient spatial data structures (octrees, quadtrees)
- RESTful and binary UDP interfaces for networking
- Python integration for scripting and AI control
- Shared memory IPC for high-frequency communication

## Documentation

- [Backend Architecture](docs/BACKENDS.md)
- [Algorithm Details](docs/ALGORITHMS.md)
- [Original Design Document](docs/design_document.md)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
