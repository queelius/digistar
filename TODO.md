# DigiStar TODO List

## 🚨 Blocking Issues (Must Fix for Basic Functionality)

### Backend Compilation Errors
- [ ] **cpu_backend_openmp.cpp field mismatches**
  - Line 311-312: `config.contact_stiffness/damping` → `physics_config.contact_stiffness/damping`
  - Fix `springs.is_broken` → `!springs.active`
  - Fix `springs.particle1/2` → `springs.particle1_id/particle2_id`
  - Fix `particles.temp_internal` → `particles.temperature`
  - Fix `springs.current_strain` → `springs.strain`
  - Fix `springs.break_strain` → hardcoded value or new field

- [ ] **cpu_backend_reference.cpp remaining issues**
  - Fix `SpatialIndex::query_radius` → `SpatialIndex::query` (✅ Done)
  - Fix CompositePool field names (partially done)
  - Add missing spring creation methods
  - Fix UnionFind references for composite detection

- [ ] **Missing Dependencies**
  - Add UnionFind implementation for composite body detection
  - Complete GridSpatialIndex implementation

### DSL Test System Issues
- [ ] **test_eval_let hanging** - Likely infinite loop in let binding evaluation
- [ ] **test_eval_particle_creation segfault** - Memory access issues with new pool system
- [ ] **test_eval_springs segfault** - Depends on particle creation working
- [ ] **test_eval_queries segfault** - Query system not compatible with new pools
- [ ] **test_eval_control segfault** - Control flow issues

## 🔧 Core System Implementation

### Pool System Completion
- [ ] **Implement missing ParticlePool methods**
  - `clear_forces()` - Zero out force arrays for active particles
  - `apply_boundaries()` - Handle toroidal wrapping
  - `grow()` method for dynamic resizing (future)

- [ ] **Implement missing SpringPool methods**  
  - `add_spring()` or equivalent functionality
  - `deactivate_and_compact()` method
  - Damage/breaking system fields if needed

- [ ] **Implement missing ContactPool methods**
  - Contact pair generation
  - Integration with spatial indexing

### Spatial Indexing System
- [ ] **Complete GridSpatialIndex implementation**
  - `insert()` method for adding particles
  - `query()` method for radius queries  
  - `findPairs()` method for collision detection
  - `clear()` method for rebuilding

- [ ] **Multi-resolution spatial grids**
  - Contact grid (2-4 units)
  - Spring grid (10-20 units)
  - Thermal grid (50-100 units)  
  - Radiation grid (200-500 units)

### Physics Integration
- [ ] **Wire up ParticleMesh (PM) gravity solver**
  - Connect to FFTW3 for FFT operations
  - Implement grid-to-particle force interpolation
  - Handle toroidal boundary conditions

- [ ] **Contact force computation**
  - Soft contact model with stiffness/damping
  - Heat generation from compression
  - Integration with spatial indexing

- [ ] **Spring network dynamics**
  - Hooke's law force computation
  - Damping forces
  - Breaking/formation mechanics

## 🎯 Working Simulation Goals

### Basic Functionality
- [ ] **Create simple N-body simulation**
  - Use existing PM gravity infrastructure
  - Basic integration (Velocity Verlet or Semi-implicit)
  - ASCII visualization output

- [ ] **Two-body collision test**
  - Demonstrate contact forces
  - Verify energy conservation
  - Test toroidal boundaries

- [ ] **Spring network test**
  - Create connected particle mesh
  - Test spring formation/breaking
  - Demonstrate composite body behavior

### Example Programs
- [ ] **Fix existing examples**
  - `solar_system.cpp` - Basic planetary motion
  - `solar_system_simple.cpp` - Minimal test case
  - `million_particles.cpp` - Performance test

- [ ] **Create new demos**
  - Basic collision demo
  - Spring network demo
  - Composite breaking demo

## 🚀 Performance & Architecture

### Memory and Performance
- [ ] **SIMD optimization verification**
  - Ensure SoA layout works with compiler vectorization  
  - Benchmark against old AoS design
  - Add alignment verification

- [ ] **Memory pool optimization**
  - Implement grow() methods for dynamic sizing
  - Add memory usage tracking
  - Optimize allocation patterns

### Threading and Parallelization
- [ ] **OpenMP integration**
  - Parallel force computation loops
  - Thread-safe spatial indexing  
  - Load balancing strategies

- [ ] **GPU preparation**
  - Verify data layouts work for CUDA
  - Design GPU-friendly algorithms
  - Memory transfer optimization

## 📚 Documentation & Quality

### Code Documentation
- [ ] **Update CLAUDE.md**
  - Reflect new CMake build system
  - Update architectural decisions
  - Add current status

- [ ] **API Documentation**
  - Document new pool interfaces
  - Explain stable ID system
  - Usage examples for each pool type

### Testing Infrastructure  
- [ ] **Complete test coverage**
  - Unit tests for all pool operations
  - Integration tests for physics systems
  - Performance regression tests

- [ ] **Continuous Integration**
  - CMake build verification
  - Cross-platform testing
  - Performance benchmarking

## 🔮 Future Enhancements

### Advanced Physics
- [ ] **Thermal radiation system**
  - Temperature-dependent emission
  - Radiative heat transfer
  - Stellar evolution effects

- [ ] **Magnetic field interactions**
  - Dipole-dipole forces
  - Magnetic field visualization
  - Plasma physics effects

### Scalability
- [ ] **Distributed computing**
  - Multi-node parallelization
  - Network communication optimization
  - Load balancing across nodes

- [ ] **Advanced spatial structures**
  - Octree for very large systems
  - Adaptive mesh refinement
  - Multi-scale simulation techniques

---

## Priority Guidelines

1. **🚨 Blocking Issues** - Must fix before any simulation can work
2. **🔧 Core System** - Essential for basic simulation functionality  
3. **🎯 Working Simulation** - Get demonstrations working
4. **🚀 Performance** - Optimize for scale and speed
5. **📚 Documentation** - Improve maintainability
6. **🔮 Future** - Advanced features and research

## Current Status Summary

✅ **Completed Recently**
- Modern CMake build system
- Pool architecture redesign with stable IDs
- Basic unit tests (16 passing)
- Code cleanup and modernization

⚠️ **In Progress**  
- Backend compilation fixes
- DSL test debugging
- Basic simulation demos

❌ **Major Blockers**
- Backend field name mismatches preventing compilation
- DSL evaluation system crashes
- Missing spatial indexing implementation