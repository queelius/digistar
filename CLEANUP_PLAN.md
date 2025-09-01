# Code Cleanup Plan for DigiStar

## Current State Analysis

### Backend Architecture Confusion
We have **two competing backend interfaces**:

1. **`ISimulationBackend`** (in `ISimulationBackend.h`)
   - Simpler interface focused on force algorithms
   - Used by: SimpleBackend, SSE2Backend, AVX2Backend
   - Status: Obsolete, should be removed

2. **`IBackend`** (in `backend_interface.h`) 
   - Comprehensive interface with pools, physics systems, spatial indices
   - Used by: main simulation.h, cpu_backend_reference, cpu_backend_openmp
   - Status: **This is the correct interface to keep**

### Missing Files
- `SimpleBackend_v3.cpp` is referenced but doesn't exist
- Examples are broken due to this missing file

## Cleanup Actions

### Phase 1: Remove Obsolete Code

#### Obsolete Backend Files (DELETE)
- `src/backend/ISimulationBackend.h` - obsolete interface
- `src/backend/SimpleBackend.cpp` - uses wrong interface
- `src/backend/SSE2Backend.cpp` - uses wrong interface  
- `src/backend/AVX2Backend.cpp` - uses wrong interface
- `src/benchmark_backends.cpp` - uses obsolete interface

#### Experimental Code to Remove (DELETE)
- `experiments/interactive/` - all compiled binaries (no extension files)
- `experiments/collisions/` - test implementations
- `experiments/composites/` - test implementations
- `experiments/materials/` - test implementations
- `experiments/thermal/` - test implementations
- `experiments/spatial/` - test implementations
- `experiments/integration/` - test implementations

Keep only:
- `experiments/celestial/` - has useful reference implementations
- `experiments/README.md` - documentation

#### Test/Benchmark Files to Remove
- `benchmarks/benchmark_million.cpp` - references missing files
- `benchmarks/test_million_viz.cpp` - references missing files
- `tests/integration/test_backends.cpp` - uses obsolete interface
- `tests/integration/test_backends_fast.cpp` - uses obsolete interface
- `tests/integration/test_barnes_hut_backend.cpp` - uses obsolete interface

### Phase 2: Fix and Consolidate

#### Backend Structure (KEEP & FIX)
Primary backend interface and implementations to keep:
- `src/backend/backend_interface.h` ✓ (main interface)
- `src/backend/cpu_backend_reference.h/cpp` ✓ (reference implementation)
- `src/backend/cpu_backend_openmp.h/cpp` ✓ (parallel CPU)
- `src/backend/CUDABackend.h/cu` ✓ (needs implementation)
- `src/backend/BackendFactory.cpp` - needs updating to use correct interface

#### Core Systems (KEEP)
- `src/physics/` - all files (types, pools, spatial_index)
- `src/core/` - simulation files
- `src/simulation/` - main simulation logic
- `src/dsl/` - command system and DSL
- `src/algorithms/` - force calculation algorithms
- `src/spatial/` - spatial data structures
- `src/visualization/` - ASCII renderer

#### Examples to Fix
- `examples/solar_system.cpp` - update to use correct backend
- `examples/solar_system_simple.cpp` - update to use correct backend
- `examples/million_particles.cpp` - update to use correct backend
- Keep DSL examples and JSON configs as-is

### Phase 3: Create Missing Implementation

Create a new `SimpleBackend` that implements the correct `IBackend` interface:
- Should be a simplified version of `cpu_backend_reference.cpp`
- No threading for simplicity
- Support basic gravity and contact forces
- Can be used for examples and testing

## File Organization After Cleanup

```
src/
├── backend/
│   ├── backend_interface.h         # Main interface (IBackend)
│   ├── cpu_backend_reference.h/cpp # Reference implementation
│   ├── cpu_backend_openmp.h/cpp    # OpenMP parallel
│   ├── cpu_backend_simple.h/cpp    # NEW: Simple backend for examples
│   ├── CUDABackend.h/cu           # GPU (needs implementation)
│   └── BackendFactory.cpp         # Factory (needs update)
├── physics/                       # Keep all
├── core/                          # Keep all
├── simulation/                    # Keep all
├── dsl/                          # Keep all
├── algorithms/                   # Keep all
├── spatial/                      # Keep all
└── visualization/                # Keep all

examples/                         # Fix to use new SimpleBackend
experiments/
├── celestial/                    # Keep for reference
└── README.md                     # Keep

benchmarks/                       # Remove broken ones, keep working
tests/                           # Update to use correct interface
```

## Implementation Priority

1. **First**: Delete obsolete files (Phase 1)
2. **Second**: Create new `cpu_backend_simple.h/cpp` implementing `IBackend`
3. **Third**: Update `BackendFactory.cpp` to use correct interface
4. **Fourth**: Fix examples to use new backend
5. **Fifth**: Update tests and benchmarks

## Benefits After Cleanup

- Single, consistent backend interface
- Working examples
- Clear architecture
- No missing file references
- Reduced confusion for future development