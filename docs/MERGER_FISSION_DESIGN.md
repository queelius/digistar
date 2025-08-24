# Particle Merging, Fission, and Star Formation Design

## Core Insight

Particle count should be **dynamic and responsive** to physical conditions. Dense regions naturally merge particles (accretion), while violent events split them (fragmentation). This gives us computational efficiency AND physical realism.

## Merging (Fusion/Accretion)

### When Particles Merge

Particles merge when they:
1. **Overlap significantly** (distance < 0.5 * sum of radii)
2. **Have similar velocities** (relative velocity < threshold)
3. **Are cool enough** (not bouncing off from thermal motion)
4. **Pass probability check** (stochastic process)

### Conservation Laws

When particles A and B merge into C:
```
mass_C = mass_A + mass_B
momentum_C = momentum_A + momentum_B
position_C = (mass_A * pos_A + mass_B * pos_B) / mass_C
velocity_C = momentum_C / mass_C
radius_C = cbrt(radius_A³ + radius_B³)  // Volume conservation
temp_C = (mass_A * temp_A + mass_B * temp_B) / mass_C
```

### Energy Handling

The kinetic energy "lost" in inelastic merger:
```
KE_lost = 0.5 * (mass_A * mass_B) / mass_C * |vel_A - vel_B|²
```
This becomes **internal heat**:
```
temp_C += KE_lost / (specific_heat * mass_C)
```

### Benefits
- **Computational**: Fewer particles in dense regions
- **Physical**: Natural accretion disks, planetary formation
- **Emergent**: Black holes form when enough mass concentrates

## Fission (Fragmentation)

### When Particles Split

Particles split when:
1. **High-energy impact** (collision energy > binding energy)
2. **Tidal forces** (stretched beyond limits)
3. **Thermal stress** (internal temp > vaporization point)
4. **Rotational breakup** (spinning too fast)

### Split Mechanics

One particle becomes N fragments:
```
For each fragment i:
  mass_i = mass_parent * fraction_i  // Random or impact-based
  velocity_i = vel_parent + escape_velocity * random_direction
  temp_i = temp_parent * energy_fraction
```

### Energy Requirements

Splitting requires energy input:
```
E_required = binding_energy * mass
```
This energy comes from:
- Impact kinetic energy
- Internal temperature
- Tidal work

## Temperature at Mesoscale

### The Challenge

Temperature is molecular kinetic energy, but our particles ARE bulk matter. How do we handle this?

### Two-Temperature Model

Each particle has:
1. **Kinetic Temperature**: From particle motion relative to local mean
2. **Internal Temperature**: Represents molecular motion within the chunk

```
T_kinetic = avg(|v_particle - v_local_mean|²)
T_internal = stored thermal energy / (mass * specific_heat)
```

### Temperature Evolution

Internal temperature changes from:
- **Compression heating**: When particles merge or compress
- **Friction heating**: From damped spring oscillations
- **Radiation cooling**: Hot particles lose energy
- **Conduction**: Heat flows between touching particles

## Star Formation

### Natural Emergence

Stars should form WITHOUT explicit rules, purely from physics:

1. **Gravitational Collapse**
   - Cloud of particles attracts under gravity
   - As they fall inward, kinetic energy increases
   
2. **Compression Heating**
   - Particles merge as density increases
   - Merger energy becomes internal heat
   - Temperature rises with compression
   
3. **Pressure Resistance**
   - Hot particles create radiation pressure
   - This opposes further collapse
   - Balance creates stable configuration
   
4. **Stellar Ignition**
   - Above threshold temperature (~10⁷ K equivalent)
   - Particle gains "fusion" flag
   - Begins converting mass to energy
   
5. **Stellar Properties**
   ```
   if (temp > FUSION_THRESHOLD && density > FUSION_DENSITY) {
       particle.is_fusing = true;
       particle.luminosity = mass * FUSION_RATE;
       particle.radiation_pressure = luminosity / (4π * r²);
   }
   ```

### Avoiding Micro-Management

We DON'T simulate:
- Individual fusion reactions
- Stellar nucleosynthesis  
- Convection zones
- Magnetic dynamos

We DO simulate:
- Mass-energy conversion (E=mc²)
- Radiation pressure
- Temperature-based properties
- Gravitational effects

## Black Hole Formation

### Natural Process

When density exceeds threshold:
```
schwarzschild_radius = 2 * G * mass / c²
if (radius < schwarzschild_radius * SAFETY_FACTOR) {
    particle.is_black_hole = true;
    particle.event_horizon = schwarzschild_radius;
    // Merges become inescapable
    // No radiation escapes
}
```

### Computational Benefits

Black holes are SIMPLE:
- One particle (very efficient!)
- Only gravitational interaction
- Permanent merging (no fission)
- No internal dynamics

## Implementation Strategy

### Phase 1: Basic Merging
- Simple overlap detection
- Mass/momentum conservation
- Fixed merge probability

### Phase 2: Temperature Model
- Internal vs kinetic temperature
- Compression heating
- Radiation cooling

### Phase 3: Fission
- Impact fragmentation
- Thermal breakup
- Fragment distribution

### Phase 4: Star Formation
- Fusion threshold
- Radiation pressure
- Mass-luminosity relation

### Phase 5: Exotic Objects
- Black holes
- Neutron stars (high density, no fusion)
- White dwarfs (cooling remnants)

## Performance Impact

### Pros
- **Fewer particles in dense regions** (cores, black holes)
- **More particles where needed** (collisions, explosions)
- **Adaptive detail** (computational resources follow interesting events)

### Cons
- **Merge checks** (but only for close particles)
- **Split calculations** (but rare events)
- **Temperature tracking** (one float per particle)

### Net Result
Should be **faster** for realistic simulations because particle count adapts to need.

## Example: Stellar Nursery

Starting with 100,000 cold gas particles:

1. **Initial**: Uniform cloud, 100k particles
2. **Collapse begins**: Gravity pulls inward
3. **Clumping**: Local overdensities, merging starts (90k particles)
4. **Protostars**: Hot cores form, 50k particles
5. **Ignition**: Several stars ignite, 30k particles  
6. **Stellar wind**: Radiation pressure clears gas
7. **Final**: 5-10 stars, 10k particles in planets/debris

The simulation naturally:
- Formed stars of different sizes
- Created planetary systems
- Left debris and gas
- Self-optimized particle count

## Design Philosophy

**Let physics decide**, not rules:
- Don't spawn a "star object" - let particles become stellar through compression
- Don't delete particles arbitrarily - let them merge when physics says so
- Don't add particles randomly - let them split from energetic events

This creates a living, breathing universe where complexity emerges from simple physical laws.