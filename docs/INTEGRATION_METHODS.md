# Numerical Integration Methods for DigiStar Physics

## Executive Summary

DigiStar simulates diverse physics ranging from orbital mechanics to stiff spring networks. Each physics regime requires different numerical integration methods for stability and accuracy. This document explores integration strategies based on our experiments and theoretical analysis.

## The Integration Challenge

### Core Problem
Different physics have vastly different timescales:
- **Gravity**: Changes over minutes to years
- **Orbits**: Require long-term energy conservation
- **Springs**: Can be very stiff (submarines, buildings)
- **Collisions**: Generate huge impulsive forces
- **Thermal**: Slow diffusion processes

Using a single integrator and timestep for all physics leads to either:
1. **Instability**: Timestep too large for stiff forces
2. **Inefficiency**: Timestep too small for slow physics

### Stability Criteria

For explicit methods, stability requires:
```
dt < 2 / ω_max
```
Where ω_max is the maximum frequency in the system:
- Spring: ω = sqrt(k/m)
- Collision: ω ~ sqrt(k_contact/m)
- Gravity: ω ~ sqrt(G*M/r³)

## Integration Methods Overview

### 1. Explicit Methods
Simple, fast per step, but conditionally stable.

#### Forward Euler (First Order)
```cpp
v_new = v_old + a * dt
x_new = x_old + v_old * dt
```
- **Pros**: Simplest possible
- **Cons**: Energy drift, requires tiny timesteps
- **Use**: Never for production

#### Symplectic Euler (First Order)
```cpp
v_new = v_old + a * dt
x_new = x_old + v_new * dt  // Uses NEW velocity
```
- **Pros**: Better stability than Forward Euler
- **Cons**: Still first-order accuracy
- **Use**: Quick prototypes, non-critical physics

#### Velocity Verlet (Second Order, Symplectic)
```cpp
x_new = x_old + v_old * dt + 0.5 * a_old * dt²
// Compute a_new from x_new
v_new = v_old + 0.5 * (a_old + a_new) * dt
```
- **Pros**: Energy-conserving, time-reversible
- **Cons**: Requires force evaluation at new position
- **Use**: **Orbital mechanics, gravity**

#### Leapfrog (Second Order, Symplectic)
```cpp
v_half = v_old + 0.5 * a_old * dt
x_new = x_old + v_half * dt
// Compute a_new from x_new
v_new = v_half + 0.5 * a_new * dt
```
- **Pros**: Symplectic, simple
- **Cons**: Velocities offset by half timestep
- **Use**: N-body simulations

#### RK4 (Fourth Order)
```cpp
k1 = dt * f(x, t)
k2 = dt * f(x + k1/2, t + dt/2)
k3 = dt * f(x + k2/2, t + dt/2)
k4 = dt * f(x + k3, t + dt)
x_new = x + (k1 + 2*k2 + 2*k3 + k4) / 6
```
- **Pros**: High accuracy
- **Cons**: 4 force evaluations, not symplectic
- **Use**: When accuracy matters more than speed

### 2. Implicit Methods
Stable for stiff systems but require solving equations.

#### Backward Euler (First Order)
```cpp
// Solve: x_new = x_old + dt * f(x_new)
// For linear springs: (I - dt²*K/M) * v_new = v_old + dt*F_external/M
```
- **Pros**: Unconditionally stable
- **Cons**: Requires linear solve, introduces damping
- **Use**: Very stiff systems where stability > accuracy

#### Implicit Midpoint (Second Order)
```cpp
// Solve: x_new = x_old + dt * f((x_old + x_new)/2)
```
- **Pros**: Symplectic, stable
- **Cons**: Requires iterative solve
- **Use**: Stiff Hamiltonian systems

### 3. Semi-Implicit Methods
Balance between explicit and implicit.

#### Semi-Implicit Euler
```cpp
v_new = v_old + a * dt
x_new = x_old + v_new * dt  // Uses NEW velocity
```
- **Pros**: Simple, more stable than explicit
- **Cons**: First-order accuracy
- **Use**: **Moderate stiffness springs**

### 4. Position-Based Methods
Modern game physics approach.

#### XPBD (Extended Position-Based Dynamics)
```cpp
// Predict positions
x_predicted = x + v * dt + a * dt²

// Solve constraints iteratively
for (int i = 0; i < iterations; i++) {
    solve_distance_constraints(x_predicted);
    solve_collision_constraints(x_predicted);
}

// Update velocity from position change
v = (x_predicted - x) / dt
x = x_predicted
```
- **Pros**: Unconditionally stable, handles contacts well
- **Cons**: Not energy-conserving, requires iterations
- **Use**: **Game physics, rigid structures**

## Adaptive Integration Strategies

### 1. Adaptive Timestep
```cpp
float compute_adaptive_dt(const Particle& p) {
    // Based on maximum acceleration
    float a_mag = p.force.length() / p.mass;
    float dt_accel = sqrt(2 * MAX_DISPLACEMENT / a_mag);
    
    // Based on velocity (CFL condition)
    float dt_vel = MAX_DISPLACEMENT / p.vel.length();
    
    // Based on spring stiffness
    float dt_spring = 0.1 * sqrt(p.mass / max_spring_stiffness);
    
    return min(dt_accel, dt_vel, dt_spring);
}
```

### 2. Multiple Timesteps (Subcycling)
```cpp
void update_physics(float dt_frame) {
    // Slow physics
    update_gravity(dt_frame);
    update_radiation(dt_frame);
    
    // Medium physics (10x subcycles)
    float dt_medium = dt_frame / 10;
    for (int i = 0; i < 10; i++) {
        update_soft_springs(dt_medium);
        update_thermal(dt_medium);
    }
    
    // Fast physics (100x subcycles)
    float dt_fast = dt_frame / 100;
    for (int i = 0; i < 100; i++) {
        update_collisions(dt_fast);
        update_stiff_springs(dt_fast);
    }
}
```

### 3. Mixed Integration (IMEX)
```cpp
void integrate_IMEX(Particle& p, float dt) {
    // Implicit for stiff forces
    float2 v_implicit = solve_implicit_springs(p, dt);
    
    // Explicit for non-stiff forces
    float2 v_explicit = p.vel + (p.gravity_force / p.mass) * dt;
    
    // Combine
    p.vel = v_implicit + (v_explicit - p.vel);
    p.pos += p.vel * dt;
}
```

## Practical Recommendations for DigiStar

### 1. Particle Classification
Tag each particle with its dominant physics:

```cpp
enum IntegratorType {
    ORBIT,          // Use Velocity Verlet
    FREE_SPACE,     // Use Leapfrog
    SOFT_BODY,      // Use Semi-implicit Euler
    RIGID_BODY,     // Use XPBD or Backward Euler
    COLLISION,      // Use small timestep + Semi-implicit
    HIGH_ENERGY     // Use adaptive timestep
};
```

### 2. Hierarchical Integration
```cpp
class HierarchicalIntegrator {
    // Different integrators for different particle groups
    void integrate(float dt) {
        // Group 1: Orbital particles (symplectic)
        #pragma omp parallel for
        for (auto& p : orbital_particles) {
            integrate_velocity_verlet(p, dt);
        }
        
        // Group 2: Spring networks (semi-implicit)
        #pragma omp parallel for
        for (auto& p : spring_particles) {
            integrate_semi_implicit(p, dt * 0.1);  // Smaller timestep
        }
        
        // Group 3: Colliding particles (adaptive)
        #pragma omp parallel for
        for (auto& p : collision_particles) {
            float adaptive_dt = compute_safe_dt(p);
            integrate_semi_implicit(p, min(dt, adaptive_dt));
        }
    }
};
```

### 3. Stability Monitoring
```cpp
struct StabilityMonitor {
    float max_velocity = 1000.0f;  // m/s
    float max_acceleration = 10000.0f;  // m/s²
    float energy_tolerance = 0.01f;  // 1% drift allowed
    
    bool check_stability(const Particle& p) {
        if (p.vel.length() > max_velocity) return false;
        if (p.force.length() / p.mass > max_acceleration) return false;
        if (isnan(p.pos.x) || isinf(p.pos.x)) return false;
        return true;
    }
    
    void handle_instability(Particle& p) {
        // Option 1: Clamp values
        float v_mag = p.vel.length();
        if (v_mag > max_velocity) {
            p.vel = p.vel * (max_velocity / v_mag);
        }
        
        // Option 2: Rollback and retry with smaller timestep
        // Option 3: Switch to more stable integrator
    }
};
```

## Performance Considerations

### GPU Optimization
- **Parallel-friendly**: Leapfrog, Semi-implicit Euler
- **Requires synchronization**: Velocity Verlet (needs new forces)
- **Difficult on GPU**: Implicit methods (need linear solvers)

### Cache Efficiency
- **Good**: Methods that update position and velocity together
- **Bad**: Methods requiring multiple passes over data

### Accuracy vs Speed Tradeoffs
| Method | Order | Stability | Speed | Energy | Best For |
|--------|-------|-----------|-------|--------|----------|
| Forward Euler | 1 | Poor | Fastest | Drifts | Never use |
| Symplectic Euler | 1 | OK | Fast | OK | Prototypes |
| Velocity Verlet | 2 | Good | Medium | Conserves | Orbits |
| RK4 | 4 | OK | Slow | Good | High accuracy |
| Backward Euler | 1 | Excellent | Slow | Dissipates | Stiff springs |
| Semi-implicit | 1 | Good | Fast | OK | **General use** |
| XPBD | 1 | Excellent | Medium | Poor | Rigid bodies |

## Experimental Validation Plan

### Test Cases

1. **Orbital Stability**: Two-body orbit for 1000 periods
   - Measure: Energy drift, angular momentum conservation
   - Compare: Verlet vs Euler vs RK4

2. **Stiff Spring**: k = 10000, mass = 1
   - Measure: Stability at different timesteps
   - Compare: Explicit vs Semi-implicit vs Backward Euler

3. **Collision Cascade**: 100 particles in box
   - Measure: Energy conservation, momentum conservation
   - Compare: Fixed vs Adaptive timestep

4. **Mixed System**: Orbit + Springs + Collisions
   - Measure: Overall stability, performance
   - Compare: Single vs Multiple integrators

### Success Metrics

- Energy drift < 0.01% per orbit
- No explosions with dt = 0.01
- 60 FPS with 1M particles
- Momentum conserved to machine precision

## Conclusion

DigiStar requires a sophisticated integration strategy:

1. **Use multiple integrators**: Match method to physics
2. **Implement subcycling**: Different timesteps for different forces
3. **Monitor stability**: Detect and handle problems early
4. **Optimize for GPU**: Choose parallel-friendly methods

The recommended baseline:
- **Velocity Verlet** for gravity/orbits (energy conservation)
- **Semi-implicit Euler** for general dynamics (good balance)
- **XPBD** for rigid structures (unconditional stability)
- **Adaptive timestep** for high-energy events (safety)

This hybrid approach provides stability, accuracy, and performance across DigiStar's diverse physics regimes.