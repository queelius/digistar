# Temperature and Radiation Dynamics Design

## Core Philosophy

Temperature creates a **repulsive force field** that opposes gravity, enabling rich dynamics like stellar winds, solar sails, and thermal equilibrium. Hot bodies radiate energy, cool down over time, and push on other objects based on their cross-sectional area.

## First Principles in 2D

### 1. Radiation Field

Each hot particle creates a radiation field that spreads radially:

```
Intensity(r) = Power / (2πr)  // 2D: energy spreads over circumference
```

Where Power = εσT⁴A (Stefan-Boltzmann law adapted for 2D)
- ε = emissivity (0-1)
- σ = Stefan-Boltzmann constant
- T = temperature
- A = surface "area" (circumference in 2D = 2πR)

### 2. Radiation Pressure

When radiation hits a particle, it exerts pressure:

```
Force = Intensity × Cross_section × Absorption
      = (Power / 2πr) × (2R_target) × α
```

Where:
- R_target = radius of receiving particle
- α = absorption coefficient (0 = transparent, 1 = black body)
- **Key insight**: Larger particles feel more force!

### 3. Energy Transfer

The absorbed energy heats the particle:

```
Energy_absorbed = Intensity × Cross_section × α × dt
ΔT = Energy_absorbed / (mass × specific_heat)
   = Energy_absorbed / (ρ × πR² × c_p)  // 2D area
```

**Key insight**: Smaller particles heat up faster (less thermal mass)!

### 4. Radiative Cooling

Hot particles lose energy through radiation:

```
Power_radiated = εσT⁴ × (2πR)  // Circumference in 2D
Energy_lost = Power_radiated × dt
T_new = T_old - Energy_lost / (mass × specific_heat)
```

## Implementation Design

### Particle Properties

```cpp
struct ThermalParticle : Particle {
    // Existing
    float temp_internal;     // Kelvin
    
    // New thermal properties
    float emissivity = 0.9;       // How well it radiates (0-1)
    float absorptivity = 0.9;     // How well it absorbs (0-1)
    float specific_heat = 1000.0; // J/(kg·K)
    float thermal_mass;            // Cached: mass * specific_heat
    
    // Radiation state
    float luminosity;              // Current power output (W)
    float incident_radiation = 0;  // Radiation hitting us this frame
};
```

### Radiation Field Calculation

```cpp
class RadiationField {
    // Efficient field calculation using spatial hashing
    void computeRadiation(vector<ThermalParticle>& particles) {
        // Step 1: Calculate luminosity for each particle
        for (auto& p : particles) {
            // Stefan-Boltzmann radiation
            float circumference = 2.0f * M_PI * p.radius;
            p.luminosity = p.emissivity * STEFAN_BOLTZMANN * 
                          pow(p.temp_internal, 4) * circumference;
        }
        
        // Step 2: Calculate radiation pressure on each particle
        for (size_t i = 0; i < particles.size(); i++) {
            float2 total_force = {0, 0};
            float total_energy = 0;
            
            for (size_t j = 0; j < particles.size(); j++) {
                if (i == j) continue;
                if (particles[j].luminosity < EPSILON) continue;
                
                float2 r = particles[i].pos - particles[j].pos;
                float dist = length(r);
                
                // Radiation intensity at this distance
                float intensity = particles[j].luminosity / (2.0f * M_PI * dist);
                
                // Cross-section (accounting for orientation)
                float cross_section = calculateCrossSection(
                    particles[i], normalize(r)
                );
                
                // Radiation pressure force
                float pressure = intensity / SPEED_OF_LIGHT;
                float2 force = pressure * cross_section * normalize(r);
                
                total_force += force * particles[i].absorptivity;
                total_energy += intensity * cross_section * 
                               particles[i].absorptivity;
            }
            
            // Apply radiation pressure
            particles[i].vel += total_force * dt / particles[i].mass;
            
            // Apply heating
            particles[i].incident_radiation = total_energy;
            float energy_absorbed = total_energy * dt;
            particles[i].temp_internal += energy_absorbed / 
                                         particles[i].thermal_mass;
        }
        
        // Step 3: Radiative cooling
        for (auto& p : particles) {
            float energy_radiated = p.luminosity * dt;
            p.temp_internal -= energy_radiated / p.thermal_mass;
            
            // Minimum temperature (cosmic background)
            p.temp_internal = max(p.temp_internal, 2.7f);
        }
    }
    
    // Calculate cross-section for radiation absorption
    float calculateCrossSection(const ThermalParticle& p, float2 ray_dir) {
        // All particles are circular
        // Cross-section in 2D is diameter
        return 2.0f * p.radius;
    }
};
```

## Solar Sail Dynamics

A solar sail is made of many circular particles connected by springs in a line or grid:

```cpp
struct SolarSail {
    vector<int> particle_ids;    // Particles forming the sail
    vector<Spring> springs;       // Structural connections
    
    // Create a line of particles (acts like a sail)
    static SolarSail createLine(int num_particles, float spacing) {
        SolarSail sail;
        // Create particles in a line
        // Connect with springs to maintain structure
        // Line orientation affects total cross-section to sun
        return sail;
    }
    
    float getEffectiveCrossSection(float2 sun_direction) {
        // A line of particles has different cross-section
        // depending on orientation to sun:
        // - Perpendicular: sum of all diameters
        // - Parallel: overlapping, minimal cross-section
        // - Angled: intermediate
        
        float2 line_direction = getLineDirection();
        float alignment = abs(dot(line_direction, sun_direction));
        
        // More particles exposed when perpendicular
        float exposed_fraction = 1.0f - 0.8f * alignment;
        return num_particles * particle_diameter * exposed_fraction;
    }
};
```

## Temperature Effects

### 1. Thermal Pressure (Close Range)

When particles are very close, temperature creates direct pressure:

```cpp
float2 thermalPressure(const Particle& p1, const Particle& p2) {
    float2 r = p2.pos - p1.pos;
    float dist = length(r);
    
    if (dist > THERMAL_CUTOFF) return {0, 0};
    
    // Ideal gas pressure: P = nkT
    float pressure1 = BOLTZMANN * p1.temp_internal * p1.density;
    float pressure2 = BOLTZMANN * p2.temp_internal * p2.density;
    
    // Average pressure at interface
    float P_interface = (pressure1 + pressure2) / 2.0f;
    
    // Force = Pressure × Contact_area
    float contact_length = calculateContactLength(p1, p2, dist);
    float2 force = P_interface * contact_length * normalize(r);
    
    return force;
}
```

### 2. Phase Transitions

Temperature drives phase changes:

```cpp
enum Phase { SOLID, LIQUID, GAS, PLASMA };

Phase getPhase(float temperature, float pressure) {
    if (temperature < MELTING_POINT) return SOLID;
    if (temperature < BOILING_POINT) return LIQUID;
    if (temperature < IONIZATION_TEMP) return GAS;
    return PLASMA;
}

// Phase affects material properties
void updateMaterialProperties(Particle& p) {
    Phase phase = getPhase(p.temp_internal, p.pressure);
    
    switch(phase) {
        case SOLID:
            p.spring_stiffness = 1000.0f;  // Rigid
            p.viscosity = 0.0f;
            break;
        case LIQUID:
            p.spring_stiffness = 10.0f;    // Soft springs
            p.viscosity = 0.1f;            // Flows
            break;
        case GAS:
            p.spring_stiffness = 0.0f;     // No springs
            p.viscosity = 0.01f;           // Low viscosity
            break;
        case PLASMA:
            p.spring_stiffness = 0.0f;
            p.viscosity = 0.001f;
            p.charge = 1.0f;               // Ionized!
            break;
    }
}
```

## Energy Conservation

Total energy must be conserved:

```cpp
struct EnergyTracker {
    float total_kinetic = 0;
    float total_thermal = 0;
    float total_potential = 0;
    float total_radiated = 0;  // Energy left the system
    
    void validate(const vector<Particle>& particles) {
        float current_total = calculateTotalEnergy(particles);
        float initial_total = total_kinetic + total_thermal + total_potential;
        
        float error = abs(current_total + total_radiated - initial_total);
        assert(error < ENERGY_TOLERANCE);
    }
};
```

## Performance Optimization

### Spatial Hashing for Radiation

Only calculate radiation from nearby hot sources:

```cpp
class RadiationGrid {
    // Grid cells store hot particles only
    unordered_map<int, vector<int>> hot_particles;
    
    void update(const vector<Particle>& particles) {
        hot_particles.clear();
        
        for (int i = 0; i < particles.size(); i++) {
            if (particles[i].temp_internal > MIN_RADIATING_TEMP) {
                int cell = getGridCell(particles[i].pos);
                hot_particles[cell].push_back(i);
            }
        }
    }
    
    vector<int> getNearbyRadiators(float2 pos, float radius) {
        // Return hot particles within influence radius
        vector<int> result;
        int search_cells = ceil(radius / CELL_SIZE);
        
        for (int dx = -search_cells; dx <= search_cells; dx++) {
            for (int dy = -search_cells; dy <= search_cells; dy++) {
                int cell = getGridCell(pos + float2{dx*CELL_SIZE, dy*CELL_SIZE});
                if (hot_particles.count(cell)) {
                    result.insert(result.end(), 
                                hot_particles[cell].begin(),
                                hot_particles[cell].end());
                }
            }
        }
        return result;
    }
};
```

## Example Scenarios

### 1. Star with Solar Wind

```cpp
// Hot star pushes away nearby particles
Particle star;
star.mass = 1e6;
star.radius = 100;
star.temp_internal = 1e7;  // 10 million K
star.emissivity = 1.0;     // Perfect black body

// Nearby dust particles are blown away
for (auto& dust : dust_particles) {
    float2 r = dust.pos - star.pos;
    float dist = length(r);
    
    // Radiation pressure overcomes gravity at certain distance
    float F_radiation = star.luminosity / (2*PI*dist) * dust.cross_section / c;
    float F_gravity = G * star.mass * dust.mass / (dist * dist);
    
    if (F_radiation > F_gravity) {
        // Particle is blown away!
    }
}
```

### 2. Solar Sail Navigation

```cpp
// Sail adjusts orientation to navigate
SolarSail sail;
sail.setOrientation(perpendicular_to_sun);  // Maximum thrust
sail.setOrientation(parallel_to_sun);        // Minimum thrust
sail.setOrientation(45_degrees);             // Directional control
```

### 3. Thermal Equilibrium

```cpp
// Objects reach equilibrium temperature
// Energy in = Energy out
// Incident_radiation = Emitted_radiation
// T_equilibrium = pow(Incident/(ε*σ*A), 0.25)
```

## Implementation Phases

1. **Phase 1**: Basic radiation pressure
   - Hot particles push others away
   - Simple 1/r intensity falloff in 2D
   
2. **Phase 2**: Energy transfer
   - Heating and cooling
   - Temperature equilibrium
   
3. **Phase 3**: Composite objects
   - Solar sails with orientation
   - Cross-section calculation
   
4. **Phase 4**: Advanced effects
   - Phase transitions
   - Thermal conduction
   - Anisotropic radiation

## Key Design Decisions

1. **Temperature as internal energy density** - Scales properly with particle size
2. **Cross-section based absorption** - Big particles intercept more radiation
3. **Radiative cooling** - Hot objects naturally cool down
4. **Composite objects matter** - Lines of particles create solar sails
5. **Energy conservation** - Total energy tracked and preserved

This system creates emergent behaviors:
- Stars naturally create "habitable zones"
- Solar sails can navigate using light
- Gas clouds heat up near stars
- Thermal pressure prevents gravitational collapse
- Objects reach thermal equilibrium

The beauty is that complex phenomena emerge from simple physical laws!