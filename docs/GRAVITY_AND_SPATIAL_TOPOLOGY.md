# Gravity and Spatial Topology Design

## Core Decision: Toroidal Space with PM Gravity

### The Topology Choice: Toroidal (Periodic) Space

We use a 2D toroidal topology - space wraps around at the boundaries like the surface of a torus.

**Motivation:**
- **No boundaries** - No special edge cases, particles can't "escape"
- **Conservation** - Mass, energy, momentum naturally conserved
- **Mathematically clean** - Simple modulo arithmetic for positions
- **Physically plausible** - Universe might actually work this way
- **Computationally elegant** - No boundary forces or deletions needed

**Implementation:**
```cpp
// Position wrapping is trivial
position.x = fmod(position.x + WORLD_SIZE, WORLD_SIZE);
position.y = fmod(position.y + WORLD_SIZE, WORLD_SIZE);

// Distance considers shortest path
float wrapped_distance(float2 a, float2 b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    
    // Take shortest path (might wrap)
    if (abs(dx) > WORLD_SIZE/2) dx = WORLD_SIZE - abs(dx);
    if (abs(dy) > WORLD_SIZE/2) dy = WORLD_SIZE - abs(dy);
    
    return sqrt(dx*dx + dy*dy);
}
```

### The Gravity Method: Particle-Mesh (PM)

PM is the ideal gravity solver for our toroidal space.

**Why PM over Barnes-Hut:**
1. **Natural periodicity** - FFT assumes periodic boundaries by default
2. **No tree rebuilding** - Grid is static, only density changes
3. **Automatic shortest path** - FFT handles toroidal topology correctly
4. **Superior performance** - O(N log N) with small constant
5. **Smooth forces** - No artifacts from tree approximations

**How PM Works:**
```
1. Particles → Grid: Assign mass to grid cells
2. Grid → Fourier: FFT the density field  
3. Solve Poisson: ∇²φ = 4πGρ in Fourier space
4. Fourier → Grid: Inverse FFT to get potential
5. Grid → Forces: F = -∇φ at particle positions
```

**The FFT naturally handles periodicity:**
- Computing potential at (0,0) automatically includes particles at (999,999)
- No special wraparound logic needed
- Mathematically correct for toroidal topology

## Why Different Forces Need Different Methods

### Scale Separation in Forces

Our simulation has forces at vastly different scales:

```
Force Type        Range           Update Frequency    Method
-----------------------------------------------------------------
Gravity          Global          Every frame         PM (global)
Radiation        ~1000 units     Every frame         Field/PM(?)
Springs          ~10 units       Every frame         Spatial grid
Heat             ~10 units       Every 10 frames     Spatial grid
Collisions       ~1 unit         Every frame         Fine grid
```

### Why PM Only Works for Long-Range Forces

PM excels at smooth, long-range forces but fails for short-range:

**PM Strengths:**
- Global forces (gravity, radiation fields)
- Smooth variations
- Periodic boundaries
- Fixed computational cost

**PM Weaknesses:**
- **Resolution limited by grid** - Can't resolve contact forces
- **Smooths over details** - Misses particle-particle collisions
- **Fixed grid** - Wastes resolution in empty space
- **No cutoffs** - Always computes globally

### Contact Forces: Need Different Approach

For collision detection between particles with ~1 unit radius:
- PM grid would need millions of cells for accuracy
- Most cells would be empty (space is sparse)
- Overkill when we only need local neighbors

Better: **Uniform spatial grid** with appropriate cell size

### Spring Forces: Medium Range

Springs connect particles within ~10 units:
- PM too coarse
- Don't need global calculation
- Natural cutoff at spring formation distance

Better: **Coarser spatial grid** or **distance-limited tree**

## The Hybrid Architecture

```cpp
class ForceCalculator {
    // Global forces (gravity)
    PMSolver pm_gravity;         // Handles ALL particles globally
    
    // Medium-range forces  
    SpatialGrid spring_grid;     // ~20 unit cells
    
    // Short-range forces
    CollisionGrid contact_grid;  // ~2 unit cells
    
    void calculate_all_forces() {
        // Gravity - PM handles toroidal space naturally
        pm_gravity.calculate_forces(all_particles);
        
        // Springs - local grid lookup
        spring_grid.apply_spring_forces(particles, springs);
        
        // Collisions - fine grid for contacts
        contact_grid.detect_collisions(particles);
    }
};
```

## PM for Other Fields?

### Radiation Pressure

PM could work well for radiation:
- Radiation is long-range (though falls off as 1/r in 2D)
- Smooth field
- Benefits from periodicity

```cpp
class RadiationField : public PMSolver {
    // Use PM to solve radiation pressure field
    // Similar to gravity but different equation
};
```

### Temperature/Heat

PM is **not** appropriate for temperature because:
- Heat conducts locally (through contacts/springs)
- Not a long-range force
- Needs high resolution at boundaries
- Discontinuous (hot star next to cold space)

Temperature needs: **Nearest-neighbor diffusion** through spring network

## Integration Considerations for Gravity

### Symplectic Integration is Critical
Our experiments show orbital mechanics require symplectic integrators:

```cpp
// Energy drift over 100 time units:
// Forward Euler: 26% drift - UNUSABLE
// Symplectic Euler: 0.003% drift - Good
// Velocity Verlet: 0.001% drift - BEST
// Leapfrog: 0.003% drift - Good
// RK4: 162% drift - Surprisingly bad!
```

### Recommended Gravity Integrators

1. **Velocity Verlet** (Primary choice)
   ```cpp
   // Position update with old acceleration
   pos += vel * dt + 0.5 * accel_old * dt * dt;
   
   // Compute new gravitational forces
   accel_new = compute_PM_gravity(pos);
   
   // Velocity update with average acceleration
   vel += 0.5 * (accel_old + accel_new) * dt;
   ```
   - Conserves energy to machine precision
   - Time-reversible
   - Works perfectly with PM method

2. **Leapfrog** (Alternative)
   ```cpp
   vel += compute_PM_gravity(pos) * dt;
   pos += vel * dt;
   ```
   - Simpler than Verlet
   - Also symplectic
   - Velocities offset by half timestep

### Multi-Timescale Integration
Gravity changes slowly compared to collisions:

```cpp
void update_physics(float dt_frame) {
    // Gravity: Large timestep (full frame)
    compute_PM_gravity_field();
    integrate_verlet_gravity(dt_frame);
    
    // Contacts: Small timestep (subcycle 100x)
    float dt_contact = dt_frame / 100;
    for (int i = 0; i < 100; i++) {
        update_contact_forces(dt_contact);
        integrate_contacts(dt_contact);
    }
}
```

### Key Insight: Don't Mix Integrators Carelessly
- Use **Verlet** for gravity (energy conservation)
- Use **Semi-implicit** for contacts (stability)
- Never use **Forward Euler** for orbits
- Surprisingly, **RK4 fails** for Hamiltonian systems

## Summary

**Toroidal space + PM gravity** is the natural choice because:
1. PM assumes periodicity - perfect match
2. No boundary handling needed
3. Efficient global force calculation
4. Leaves other methods for short-range forces
5. **Works perfectly with symplectic integrators**

**Keep Barnes-Hut available** for:
- Possible future uses
- Alternative if PM has issues
- Parameterizable for different forces
- Comparison/validation

**Integration strategy**:
- Velocity Verlet for gravity (energy conservation)
- Different timesteps for different forces
- Never compromise on symplectic integration for orbits

**Other forces need different methods** based on their scale and nature. The key insight: use the right tool for each force's natural scale and the right integrator for each force's requirements.