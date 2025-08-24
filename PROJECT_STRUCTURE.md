# DigiStar Project Structure

## Directory Layout

```
digistar/
├── src/                    # Source code
│   ├── backend/           # Backend implementations
│   │   ├── ISimulationBackend.h    # Main backend interface
│   │   ├── SimpleBackend.cpp       # CPU backend (Brute, Barnes-Hut, PM)
│   │   ├── CUDABackend.cu         # GPU backend
│   │   ├── AVX2Backend.cpp        # SIMD optimized backend
│   │   └── SSE2Backend.cpp        # SSE2 optimized backend
│   │
│   ├── algorithms/        # Force calculation algorithms
│   │   ├── BarnesHut.h           # Barnes-Hut tree algorithm
│   │   ├── ParticleMesh.h        # PM with FFT
│   │   └── FastFFT2D.h           # Custom FFT implementation
│   │
│   ├── spatial/           # Spatial data structures
│   │   └── QuadTree.h            # Quadtree for Barnes-Hut
│   │
│   ├── core/             # Core functionality
│   │   ├── Particle.h           # Particle structure
│   │   └── SimulationParams.h   # Simulation parameters
│   │
│   └── monitoring/        # Monitoring and visualization
│       └── TerminalMonitor.h    # ASCII visualization
│
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   │   ├── test_quadtree.cpp
│   │   ├── test_fft.cpp
│   │   └── test_algorithms.cpp
│   │
│   ├── integration/      # Integration tests
│   │   ├── test_backends.cpp
│   │   └── test_convergence.cpp
│   │
│   └── validation/       # Physics validation
│       ├── test_energy_conservation.cpp
│       ├── test_orbits.cpp
│       └── test_accuracy.cpp
│
├── benchmarks/           # Performance benchmarks
│   ├── benchmark_algorithms.cpp
│   ├── benchmark_scaling.cpp
│   └── benchmark_backends.cpp
│
├── experiments/          # Physics experiments and prototypes
│   ├── spatial/         # Spatial indexing experiments
│   ├── integration/     # Numerical integration methods
│   ├── collisions/      # Collision detection and response
│   ├── composites/      # Composite body behavior
│   ├── thermal/         # Temperature and radiation
│   ├── materials/       # Spring and material properties
│   ├── celestial/       # Black holes, stars, orbits
│   ├── Makefile         # Build system for experiments
│   └── README.md        # Experiment documentation
│
├── examples/             # Example applications
│   ├── solar_system.cpp        # Accurate solar system
│   ├── galaxy_collision.cpp    # Galaxy merger simulation
│   ├── million_particles.cpp   # Large scale demo
│   └── interactive_viewer.cpp  # Interactive ASCII viewer
│
├── tools/                # Utility tools
│   ├── backend_comparison.cpp  # Compare algorithm accuracy
│   └── convergence_test.cpp    # Test numerical convergence
│
├── docs/                 # Documentation
│   ├── README.md
│   ├── BACKENDS.md
│   └── ALGORITHMS.md
│
├── build/                # Build output (gitignored)
│   ├── bin/             # Executables
│   ├── lib/             # Libraries
│   └── obj/             # Object files
│
├── CMakeLists.txt       # CMake build configuration
├── Makefile            # Alternative make build
└── README.md           # Project overview
```

## Cleanup Plan

### Phase 1: Consolidate Versions
- Keep only the latest, best version of each component
- ISimulationBackend_v2.h → ISimulationBackend.h
- SimpleBackend_v4.cpp → SimpleBackend.cpp
- Remove all v1, v2, v3 versions

### Phase 2: Organize Files
- Move all test_*.cpp files to tests/ directory
- Move benchmark_*.cpp to benchmarks/
- Move example apps to examples/
- Clean up root directory

### Phase 3: Update Includes
- Fix all #include paths
- Use consistent relative paths
- Update build system

### Phase 4: Documentation
- Update README with new structure
- Document each component
- Add build instructions