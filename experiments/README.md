# DigiStar Physics Experiments

This directory contains experimental implementations and tests for DigiStar physics concepts.

## Directory Structure

```
experiments/
├── spatial/           # Spatial indexing and grid experiments
├── integration/       # Numerical integration methods
├── collisions/        # Collision detection and response
├── composites/        # Composite body formation and behavior
├── thermal/           # Temperature and radiation
├── materials/         # Springs and material properties
├── celestial/         # Black holes, star formation, orbits
└── build/            # Build outputs (gitignored)
```

## Categories

### Spatial Indexing (`spatial/`)
- `sparse_grid.cpp` - Sparse grid implementation for 2M+ particles
- `spatial_index.cpp` - Hierarchical spatial indexing with toroidal wraparound
- `spatial_scale.cpp` - Scalability analysis for different grid approaches

### Integration Methods (`integration/`)
- `integration_methods.cpp` - Comparison of different integrators (Euler, Verlet, RK4)
- `adaptive_timestep.cpp` - Adaptive timestep experiments (TODO)

### Collision Systems (`collisions/`)
- `composite_collision.cpp` - Localized contact forces with spring propagation
- `composite_collision_v2.cpp` - Improved with velocity broad phase
- `soft_contact.cpp` - Soft repulsion forces and deformation

### Composite Bodies (`composites/`)
- `composite_bodies.cpp` - Basic composite detection and properties
- `composite_effects.cpp` - Resonance, tidal forces, structural failure
- `virtual_springs.cpp` - Material-based spring networks

### Thermal & Radiation (`thermal/`)
- `thermal_radiation.cpp` - Heat transfer and Stefan-Boltzmann radiation
- `solar_sail.cpp` - Radiation pressure and momentum transfer

### Celestial Mechanics (`celestial/`)
- `black_holes.cpp` - Accretion, jets, and gravitational effects
- `black_holes_v2.cpp` - Improved with Hawking radiation
- `star_formation.cpp` - Gravitational collapse and fusion ignition
- `particle_merging.cpp` - Conservation-preserving merger system

## Building

### Build All
```bash
./build_all.sh
```

### Build Individual
```bash
g++ -std=c++17 -O3 -fopenmp -o build/experiment_name experiment_name.cpp -lm
```

## Key Experiments

### 1. Spatial Indexing Scalability
**File**: `spatial/sparse_grid.cpp`
**Finding**: Sparse grids with incremental updates handle 2M particles at 36 FPS using only 260MB RAM.

### 2. Integration Method Comparison
**File**: `integration/integration_methods.cpp`
**Finding**: Velocity Verlet best for orbits (0.001% drift), Semi-implicit Euler best for general use.

### 3. Composite Collision
**File**: `collisions/composite_collision_v2.cpp`
**Finding**: Localized contact forces with spring propagation create realistic deformation.

### 4. Black Hole Dynamics
**File**: `celestial/black_holes_v2.cpp`
**Finding**: Accretion disks form naturally from angular momentum conservation.

## Running Experiments

Most experiments are self-contained and output results to console:

```bash
cd experiments
./build_all.sh
./build/sparse_grid          # Test 2M particle handling
./build/integration_methods   # Compare integrators
./build/composite_collision   # Test collision system
```

## Performance Benchmarks

| Experiment | Particles | FPS | Memory | Key Metric |
|-----------|-----------|-----|--------|------------|
| Sparse Grid | 2,000,000 | 36 | 260MB | Incremental updates |
| Composite Collision | 100 | 600 | 10MB | Contact detection |
| Black Holes | 10,000 | 120 | 50MB | Accretion rate |
| Integration | 10 | 10,000 | 1MB | Energy conservation |

## Future Experiments

- [ ] GPU vs CPU comparison
- [ ] Network synchronization
- [ ] Particle budget management
- [ ] Implicit integration for very stiff springs
- [ ] Continuous collision detection
- [ ] Multi-resolution timesteps
- [ ] XPBD for rigid constraints

## Notes

- All experiments use single-precision floats for GPU compatibility
- Toroidal space assumed (wraparound at boundaries)
- Conservation laws verified in merger/fission experiments
- Integration timestep critical for stability (dt < 2/sqrt(k/m))