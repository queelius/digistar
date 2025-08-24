# Black Hole Dynamics Design

## Core Philosophy

Black holes are **simple objects with extreme effects**. They emerge naturally from extreme density or mass, create spectacular dynamics, and serve as natural endpoints for stellar evolution.

## Formation Conditions

Black holes form through multiple pathways:

### 1. Density Collapse
When matter becomes sufficiently dense, gravitational collapse is inevitable:

```cpp
bool densityCollapse(const Particle& p) {
    float volume = M_PI * p.radius * p.radius;  // 2D area
    float density = p.mass / volume;
    
    // Critical density (simplified Schwarzschild in 2D)
    float critical_density = c * c / (8.0f * M_PI * G * p.radius * p.radius);
    
    return density > critical_density;
}
```

### 2. Stellar Death
Massive stars that can no longer sustain fusion collapse:

```cpp
bool stellarCollapse(const Particle& p) {
    // Chandrasekhar limit (simplified)
    const float CHANDRASEKHAR_MASS = 1e6f;  // Simulation units
    
    return p.mass > CHANDRASEKHAR_MASS && 
           p.temp_internal < FUSION_THRESHOLD &&
           !p.is_fusing;
}
```

### 3. Runaway Accretion
Objects that grow too massive too quickly:

```cpp
bool runawayAccretion(const Particle& p, float mass_gain_rate) {
    // If gaining mass faster than radiation can push away
    float eddington_limit = 4.0f * M_PI * G * p.mass * c / THOMSON_CROSS_SECTION;
    return mass_gain_rate > eddington_limit;
}
```

## Black Hole Properties

```cpp
struct BlackHoleParticle : public ThermalParticle {
    // Core properties
    bool is_black_hole = false;
    float event_horizon = 0;          // Point of no return
    float schwarzschild_radius = 0;   // Theoretical event horizon
    
    // Accretion dynamics
    float accretion_rate = 0;         // Mass/second being absorbed
    float accretion_temp = 0;         // Temperature of accretion disk
    float last_meal_mass = 0;         // Mass of last absorbed particle
    
    // Spin (optional complexity)
    float angular_momentum = 0;       // From absorbed particles
    float spin_parameter = 0;         // -1 to 1 (fraction of maximum)
    
    // Statistics
    int particles_absorbed = 0;
    float total_mass_absorbed = 0;
    
    void updateBlackHole() {
        // Event horizon scales with mass
        schwarzschild_radius = 2.0f * G * mass / (c * c);
        
        // Effective event horizon (slightly larger for gameplay)
        event_horizon = schwarzschild_radius * 1.2f;
        
        // Use event horizon as effective radius
        radius = event_horizon;
        
        // Black holes don't radiate (ignoring Hawking)
        temp_internal = 0;
        luminosity = 0;
        emissivity = 0;
        
        // Accretion disk temperature (if actively feeding)
        if (accretion_rate > 0) {
            // Virial temperature from gravitational energy
            accretion_temp = G * mass * PROTON_MASS / (3.0f * BOLTZMANN * event_horizon);
            
            // Disk luminosity (fraction of absorbed mass-energy)
            float efficiency = 0.1f;  // 10% mass->energy conversion
            luminosity = efficiency * accretion_rate * c * c;
        }
    }
};
```

## Interaction Mechanics

### 1. Particle Absorption

```cpp
bool checkAbsorption(BlackHoleParticle& bh, Particle& p) {
    float2 r = p.pos - bh.pos;
    float dist = length(r);
    
    // Inside event horizon = doomed
    if (dist < bh.event_horizon + p.radius) {
        // Add mass and momentum
        float2 momentum_before = bh.vel * bh.mass;
        float2 momentum_added = p.vel * p.mass;
        
        bh.mass += p.mass;
        bh.vel = (momentum_before + momentum_added) / bh.mass;
        
        // Add angular momentum (creates spin)
        float2 r_cross_v = {r.x * p.vel.y - r.y * p.vel.x};
        bh.angular_momentum += length(r_cross_v) * p.mass;
        
        // Track statistics
        bh.particles_absorbed++;
        bh.total_mass_absorbed += p.mass;
        bh.last_meal_mass = p.mass;
        bh.accretion_rate = p.mass / dt;  // Instantaneous rate
        
        // Update black hole properties
        bh.updateBlackHole();
        
        return true;  // Particle should be deleted
    }
    return false;
}
```

### 2. Tidal Forces (Spaghettification)

```cpp
struct TidalEffect {
    bool should_break;
    float stress_level;  // 0-1, 1 = breaking point
    float2 stretch_direction;
};

TidalEffect calculateTidalForces(const BlackHoleParticle& bh, const Particle& p) {
    float2 r = p.pos - bh.pos;
    float dist = length(r);
    
    TidalEffect effect;
    effect.should_break = false;
    effect.stress_level = 0;
    
    // Tidal force scales as M*R/r³ where R is particle radius
    float tidal_force = 2.0f * G * bh.mass * p.radius / (dist * dist * dist);
    
    // Compare to material strength
    effect.stress_level = tidal_force / p.material.tensile_strength;
    
    // Direction of stretching (radial)
    effect.stretch_direction = normalize(r);
    
    // Break if stress exceeds strength
    if (effect.stress_level > 1.0f) {
        effect.should_break = true;
    }
    
    // Start feeling effects at 5x event horizon
    float danger_zone = bh.event_horizon * 5.0f;
    if (dist > danger_zone) {
        effect.stress_level *= (danger_zone / dist);  // Reduce effect
    }
    
    return effect;
}

// Apply tidal disruption
void applyTidalDisruption(Particle& p, const TidalEffect& tidal) {
    if (tidal.should_break && p.mass > MIN_FISSION_MASS) {
        // Rip into smaller pieces
        int fragments = 2 + (int)(tidal.stress_level);  // More stress = more pieces
        
        vector<Particle> pieces = createFragments(p, fragments);
        
        // Pieces fly apart perpendicular to stretch direction
        for (auto& piece : pieces) {
            float2 perpendicular = {-tidal.stretch_direction.y, 
                                    tidal.stretch_direction.x};
            piece.vel += perpendicular * (random() - 0.5f) * DISRUPTION_VELOCITY;
        }
    }
}
```

### 3. Accretion Disk Formation

```cpp
struct AccretionDisk {
    float inner_radius;   // Event horizon
    float outer_radius;   // Extent of disk
    float peak_temp;      // Hottest part of disk
    float total_mass;     // Mass in disk
    
    // Disk creates drag and heating
    void applyDiskEffects(Particle& p, const BlackHoleParticle& bh) {
        float2 r = p.pos - bh.pos;
        float dist = length(r);
        
        // Check if in disk region
        if (dist < inner_radius || dist > outer_radius) return;
        
        // Orbital velocity at this radius
        float v_orbital = sqrt(G * bh.mass / dist);
        float2 orbital_direction = {-r.y / dist, r.x / dist};
        float2 v_target = orbital_direction * v_orbital;
        
        // Drag toward orbital velocity (circularization)
        float2 v_diff = v_target - p.vel;
        float drag_force = DISK_DRAG_COEFFICIENT * length(v_diff);
        p.vel += v_diff * drag_force * dt / p.mass;
        
        // Friction heating
        float friction_heat = drag_force * length(v_diff);
        p.temp_internal += friction_heat * dt / (p.mass * p.specific_heat);
        
        // Temperature profile (hotter closer to BH)
        float disk_temp = peak_temp * pow(inner_radius / dist, 0.75f);
        
        // Radiative heating/cooling toward disk temperature
        float temp_diff = disk_temp - p.temp_internal;
        p.temp_internal += temp_diff * THERMAL_EXCHANGE_RATE * dt;
        
        // Gradual inspiral
        float inspiral_rate = ACCRETION_RATE_CONSTANT / (dist * dist);
        float2 inspiral = normalize(bh.pos - p.pos) * inspiral_rate;
        p.vel += inspiral;
    }
};

// Create/update accretion disk
AccretionDisk updateAccretionDisk(const BlackHoleParticle& bh, 
                                  const vector<Particle>& nearby_particles) {
    AccretionDisk disk;
    disk.inner_radius = bh.event_horizon * 1.5f;  // Innermost stable orbit
    disk.outer_radius = bh.event_horizon * 20.0f;  // Typical extent
    disk.total_mass = 0;
    
    // Calculate disk properties from particles
    for (const auto& p : nearby_particles) {
        float dist = distance(p.pos, bh.pos);
        if (dist > disk.inner_radius && dist < disk.outer_radius) {
            disk.total_mass += p.mass;
        }
    }
    
    // Peak temperature from virial theorem
    disk.peak_temp = G * bh.mass * PROTON_MASS / 
                     (3.0f * BOLTZMANN * disk.inner_radius);
    
    return disk;
}
```

### 4. Black Hole Mergers

```cpp
BlackHoleParticle mergeBlackHoles(const BlackHoleParticle& bh1, 
                                  const BlackHoleParticle& bh2) {
    BlackHoleParticle merged;
    merged.is_black_hole = true;
    
    // Conservation of mass
    merged.mass = bh1.mass + bh2.mass;
    
    // Conservation of momentum
    float2 momentum = bh1.vel * bh1.mass + bh2.vel * bh2.mass;
    merged.vel = momentum / merged.mass;
    
    // Position weighted by mass
    merged.pos = (bh1.pos * bh1.mass + bh2.pos * bh2.mass) / merged.mass;
    
    // Combine angular momentum
    merged.angular_momentum = bh1.angular_momentum + bh2.angular_momentum;
    
    // Some mass-energy radiated as gravitational waves (optional)
    float chirp_mass = pow(bh1.mass * bh2.mass, 0.6f) / 
                      pow(bh1.mass + bh2.mass, 0.2f);
    float energy_radiated = 0.05f * chirp_mass * c * c;  // ~5% of mass
    merged.mass -= energy_radiated / (c * c);
    
    // Combined statistics
    merged.particles_absorbed = bh1.particles_absorbed + bh2.particles_absorbed;
    merged.total_mass_absorbed = bh1.total_mass_absorbed + bh2.total_mass_absorbed;
    
    merged.updateBlackHole();
    return merged;
}
```

## Hawking Radiation (Optional)

For very small black holes or long timescales:

```cpp
void applyHawkingRadiation(BlackHoleParticle& bh, float dt) {
    // Temperature inversely proportional to mass
    float hawking_temp = HAWKING_CONSTANT / (8.0f * M_PI * G * bh.mass);
    
    // Power radiated (Stefan-Boltzmann for black hole)
    float power = HAWKING_CONSTANT / (bh.mass * bh.mass);
    
    // Mass loss
    float mass_loss = power * dt / (c * c);
    bh.mass -= mass_loss;
    
    // Catastrophic evaporation for tiny black holes
    if (bh.mass < PLANCK_MASS * 1e6f) {
        // Explode! Convert remaining mass to energy
        createGammaRayBurst(bh.pos, bh.mass * c * c);
        bh.is_deleted = true;
    }
}
```

## Visual Representation

```cpp
void renderBlackHole(const BlackHoleParticle& bh) {
    // The event horizon (pure black)
    drawCircle(bh.pos, bh.event_horizon, BLACK);
    
    // Photon sphere (light bending creates bright ring)
    float photon_radius = 1.5f * bh.event_horizon;
    drawRing(bh.pos, photon_radius, BRIGHT_WHITE, 0.5f);
    
    // Accretion disk (if feeding)
    if (bh.accretion_rate > 0) {
        for (float r = bh.event_horizon * 2; r < bh.event_horizon * 10; r *= 1.2f) {
            // Temperature falls off with radius
            float temp = bh.accretion_temp * pow(bh.event_horizon * 2 / r, 0.75f);
            Color disk_color = temperatureToColor(temp);
            drawRing(bh.pos, r, disk_color, 0.3f);
        }
    }
    
    // Gravitational lensing (optional visual effect)
    // Distort background stars near black hole
    applyGravitationalLensing(bh.pos, bh.mass);
}
```

## Performance Optimization

```cpp
class BlackHoleManager {
    vector<int> black_hole_indices;
    
    void update(vector<Particle>& particles) {
        // Track which particles are black holes
        black_hole_indices.clear();
        for (int i = 0; i < particles.size(); i++) {
            if (particles[i].is_black_hole) {
                black_hole_indices.push_back(i);
            }
        }
        
        // Check for new black hole formation
        for (auto& p : particles) {
            if (!p.is_black_hole && shouldFormBlackHole(p)) {
                convertToBlackHole(p);
            }
        }
        
        // Black hole interactions (expensive, so optimize)
        for (int bh_idx : black_hole_indices) {
            auto& bh = particles[bh_idx];
            
            // Only check particles within influence radius
            float influence = bh.event_horizon * 50.0f;
            
            for (int i = 0; i < particles.size(); i++) {
                if (i == bh_idx) continue;
                
                float dist = distance(particles[i].pos, bh.pos);
                if (dist < influence) {
                    // Check absorption
                    if (dist < bh.event_horizon) {
                        absorbParticle(bh, particles[i]);
                        particles[i].is_deleted = true;
                    }
                    // Check tidal effects
                    else if (dist < bh.event_horizon * 5.0f) {
                        applyTidalForces(bh, particles[i]);
                    }
                }
            }
        }
    }
};
```

## Emergent Phenomena

This system creates realistic black hole dynamics:

1. **Binary Black Holes** - Orbit and eventually merge with gravitational waves
2. **Active Galactic Nuclei** - Supermassive black holes with bright accretion disks
3. **Tidal Disruption Events** - Stars torn apart create bright flares
4. **Jets** - Particles accelerated along spin axis (optional)
5. **Stellar Mass Black Holes** - Form from collapsing stars
6. **Primordial Black Holes** - Form from extreme density fluctuations

## Implementation Phases

1. **Phase 1**: Basic absorption and event horizon
2. **Phase 2**: Tidal disruption and spaghettification  
3. **Phase 3**: Accretion disk dynamics
4. **Phase 4**: Black hole mergers
5. **Phase 5**: Visual effects (lensing, jets)

## Key Parameters

```cpp
// Fundamental constants (scaled for simulation)
const float G = 1.0f;                  // Gravitational constant
const float c = 100.0f;                // Speed of light (scaled)
const float CRITICAL_DENSITY = 1e6f;   // Forms black hole
const float CHANDRASEKHAR_MASS = 1e5f; // Maximum non-BH mass

// Gameplay parameters
const float EVENT_HORIZON_SCALE = 1.2f;  // Slightly larger for gameplay
const float TIDAL_RANGE = 5.0f;          // Tidal effects at 5x horizon
const float ACCRETION_EFFICIENCY = 0.1f; // 10% mass->energy in disk
const float DISK_DRAG = 0.1f;            // Circularization rate
```

## Summary

Black holes in DigiStar are:
- **Simple**: Just particles with extreme gravity
- **Emergent**: Form naturally from density/mass
- **Dramatic**: Create spectacular tidal and accretion effects
- **Permanent**: No escape once captured
- **Efficient**: Remove particles, improving performance

The beauty is that complex phenomena emerge from simple rules!