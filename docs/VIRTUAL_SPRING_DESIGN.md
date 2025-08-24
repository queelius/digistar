# Virtual Spring Network Design

## Core Philosophy

Springs form dynamically between particles based on their **material properties** and **physical state**, not abstract interaction vectors. This makes spring behavior intuitive and predictable.

## Material-Based Spring Generation

### Particle Material Properties

```cpp
enum MaterialType {
    ROCK,      // Rigid, brittle
    METAL,     // Strong, ductile
    ICE,       // Rigid but melts
    ORGANIC,   // Soft, flexible
    PLASMA,    // No springs (too hot)
    DUST,      // Weak cohesion
};

struct MaterialProperties {
    MaterialType type;
    
    // Mechanical properties
    float stiffness;        // Young's modulus analog
    float tensile_strength; // Breaking force
    float shear_strength;   // Sideways breaking force
    float plasticity;       // Permanent deformation threshold
    float damping_ratio;    // Energy dissipation
    
    // Bonding properties  
    float cohesion;         // How easily springs form (0-1)
    float adhesion;         // Stickiness to different materials (0-1)
    float melting_point;    // Temperature where springs weaken
    float vaporization;     // Temperature where springs break
};
```

### Spring Generation Rules

Springs form when particles meet these conditions:

```cpp
bool shouldFormSpring(const Particle& p1, const Particle& p2) {
    // 1. Distance check
    float dist = distance(p1.pos, p2.pos);
    float contact_dist = (p1.radius + p2.radius) * 1.2f;
    if (dist > contact_dist) return false;
    
    // 2. Relative velocity (must be slow enough to "stick")
    float v_rel = length(p1.vel - p2.vel);
    float v_escape = sqrt(2 * G * (p1.mass + p2.mass) / dist);
    if (v_rel > v_escape * 0.5f) return false;  // Too fast, would bounce
    
    // 3. Temperature check (not too hot)
    float avg_temp = (p1.temp + p2.temp) / 2.0f;
    float min_melt = min(p1.material.melting_point, p2.material.melting_point);
    if (avg_temp > min_melt * 0.8f) return false;  // Too hot to bond
    
    // 4. Material compatibility
    float bond_probability = calculateBondProbability(p1.material, p2.material);
    return random() < bond_probability;
}

float calculateBondProbability(const Material& m1, const Material& m2) {
    if (m1.type == m2.type) {
        // Same material - use cohesion
        return m1.cohesion;
    } else {
        // Different materials - use adhesion
        return (m1.adhesion + m2.adhesion) / 2.0f;
    }
}
```

## Spring Properties from Materials

When a spring forms, its properties are derived from the particles it connects:

```cpp
Spring generateSpring(const Particle& p1, const Particle& p2) {
    Spring spring;
    spring.particle1 = p1.id;
    spring.particle2 = p2.id;
    
    // Rest length is current distance (springs form at equilibrium)
    spring.rest_length = distance(p1.pos, p2.pos);
    
    // Stiffness: average of materials, scaled by contact area
    float contact_area = min(p1.radius, p2.radius) * 2.0f;  // 2D
    spring.stiffness = (p1.material.stiffness + p2.material.stiffness) / 2.0f 
                      * contact_area;
    
    // Breaking force: minimum of the two materials (weakest link)
    spring.break_force = min(p1.material.tensile_strength * contact_area,
                             p2.material.tensile_strength * contact_area);
    
    // Damping: average of materials
    spring.damping = (p1.material.damping_ratio + p2.material.damping_ratio) / 2.0f;
    
    // Plasticity threshold
    spring.plastic_threshold = min(p1.material.plasticity, p2.material.plasticity) 
                              * spring.rest_length;
    
    // Temperature limits
    spring.max_temp = min(p1.material.melting_point, p2.material.melting_point);
    
    return spring;
}
```

## Dynamic Spring Behavior

### Temperature Effects

Springs weaken as temperature approaches melting point:

```cpp
float getTemperatureModifier(const Spring& spring, float temp) {
    if (temp >= spring.max_temp) return 0.0f;  // Spring melts/breaks
    
    // Linear weakening near melting point
    float melt_fraction = temp / spring.max_temp;
    if (melt_fraction > 0.7f) {
        // Starts weakening at 70% of melting point
        return 1.0f - (melt_fraction - 0.7f) / 0.3f;
    }
    return 1.0f;
}
```

### Plastic Deformation

Springs can permanently stretch/compress:

```cpp
void updateSpringPlasticity(Spring& spring, float current_length) {
    float strain = abs(current_length - spring.rest_length);
    
    if (strain > spring.plastic_threshold) {
        // Permanent deformation
        float plastic_strain = strain - spring.plastic_threshold;
        float deformation = plastic_strain * 0.1f;  // 10% becomes permanent
        
        if (current_length > spring.rest_length) {
            spring.rest_length += deformation;  // Stretched
        } else {
            spring.rest_length -= deformation;  // Compressed
        }
        
        // Weakening from plastic deformation
        spring.break_force *= 0.95f;  // 5% weaker after deformation
    }
}
```

### Breaking Conditions

Springs break under various conditions:

```cpp
bool shouldBreak(const Spring& spring, const Particle& p1, const Particle& p2) {
    float current_length = distance(p1.pos, p2.pos);
    float force = spring.stiffness * abs(current_length - spring.rest_length);
    
    // 1. Tensile failure
    if (force > spring.break_force) return true;
    
    // 2. Temperature failure
    float avg_temp = (p1.temp + p2.temp) / 2.0f;
    if (avg_temp > spring.max_temp) return true;
    
    // 3. Excessive stretching (ductile failure)
    if (current_length > spring.rest_length * 2.5f) return true;
    
    // 4. Impact failure (high relative velocity)
    float v_rel = length(p1.vel - p2.vel);
    float impact_energy = 0.5f * (p1.mass * p2.mass / (p1.mass + p2.mass)) * v_rel * v_rel;
    if (impact_energy > spring.break_force * spring.rest_length) return true;
    
    return false;
}
```

## Material Examples

### Rock
```cpp
MaterialProperties rock = {
    .type = ROCK,
    .stiffness = 10000.0f,      // Very stiff
    .tensile_strength = 100.0f,  // Moderate strength
    .shear_strength = 50.0f,     // Brittle in shear
    .plasticity = 0.01f,         // Almost no plastic deformation
    .damping_ratio = 0.1f,       // Low damping
    .cohesion = 0.7f,            // Forms springs easily with other rock
    .adhesion = 0.2f,            // Doesn't stick well to other materials
    .melting_point = 1500.0f,    // High melting point
    .vaporization = 3000.0f
};
```

### Metal
```cpp
MaterialProperties metal = {
    .type = METAL,
    .stiffness = 20000.0f,       // Very stiff
    .tensile_strength = 500.0f,  // Very strong
    .shear_strength = 300.0f,    // Good shear strength
    .plasticity = 0.2f,          // Ductile - bends before breaking
    .damping_ratio = 0.05f,      // Rings like a bell (low damping)
    .cohesion = 0.8f,            // Welds easily to other metal
    .adhesion = 0.3f,            // Moderate adhesion
    .melting_point = 1000.0f,    // Moderate melting point
    .vaporization = 2500.0f
};
```

### Ice
```cpp
MaterialProperties ice = {
    .type = ICE,
    .stiffness = 5000.0f,        // Fairly stiff when cold
    .tensile_strength = 50.0f,   // Weak
    .shear_strength = 30.0f,     // Very brittle
    .plasticity = 0.02f,         // Almost no plasticity
    .damping_ratio = 0.2f,       // Some damping
    .cohesion = 0.9f,            // Ice sticks to ice very well
    .adhesion = 0.4f,            // Moderate adhesion when cold
    .melting_point = 273.0f,     // Melts at 0°C
    .vaporization = 373.0f       // Becomes steam at 100°C
};
```

### Organic (Soft tissue, wood, etc.)
```cpp
MaterialProperties organic = {
    .type = ORGANIC,
    .stiffness = 100.0f,         // Soft
    .tensile_strength = 20.0f,   // Weak
    .shear_strength = 10.0f,     // Tears easily
    .plasticity = 0.5f,          // Very plastic/flexible
    .damping_ratio = 0.5f,       // High damping (absorbs energy)
    .cohesion = 0.6f,            // Moderate self-adhesion
    .adhesion = 0.5f,            // Sticky
    .melting_point = 400.0f,     // Burns/denatures
    .vaporization = 500.0f       // Completely destroyed
};
```

## Composite Object Examples

### Spaceship Hull (Metal + Metal)
- Springs have high stiffness (20000)
- High breaking force (500 × contact_area)
- Low damping (rings when hit)
- Can plastically deform under extreme stress
- Springs weaken but don't break until 1000K

### Asteroid (Rock + Rock)
- Springs have high stiffness (10000)
- Moderate breaking force (100 × contact_area)
- Brittle - breaks suddenly with little warning
- Almost no plastic deformation
- Springs stable up to 1500K

### Comet (Ice + Dust)
- Mixed material springs are weaker
- Ice-ice bonds strong when cold (cohesion 0.9)
- Ice-dust bonds weak (adhesion ~0.3)
- Springs fail at 273K (ice melts)
- Creates realistic "dirty snowball" behavior

### Space Station (Metal + Organic)
- Metal frame with organic components
- Metal-metal joints very strong
- Metal-organic joints weaker (adhesion ~0.4)
- Organic parts add damping (vibration absorption)
- Different failure modes for different connections

## Emergent Behaviors

This system creates realistic behaviors:

1. **Fracturing**: Brittle materials (rock) break suddenly along weak points
2. **Ductile Failure**: Metals bend and stretch before breaking
3. **Melting**: Ice structures weaken and fail as temperature rises
4. **Welding**: Hot metal particles can form new springs when they cool
5. **Composite Strength**: Mixed materials create structures with varied properties
6. **Fatigue**: Repeated stress weakens springs through plastic deformation
7. **Shock Absorption**: Soft materials dampen vibrations in structures

## Implementation Priority

1. **Phase 1**: Basic material types and spring generation
2. **Phase 2**: Temperature effects and melting
3. **Phase 3**: Plastic deformation and fatigue
4. **Phase 4**: Advanced materials and welding
5. **Phase 5**: Optimization with spatial hashing

## Integration and Stability Considerations

### Stiffness Hierarchy
Different materials require different integration approaches:

```cpp
enum MaterialIntegration {
    SOFT,      // k < 100, any integrator works
    MEDIUM,    // k = 100-1000, semi-implicit recommended  
    STIFF,     // k = 1000-10000, implicit or tiny timestep
    RIGID      // k > 10000, must use implicit methods
};

MaterialIntegration getIntegrationRequirement(const Material& mat) {
    float max_k = mat.young_modulus / characteristic_length;
    if (max_k < 100) return SOFT;
    if (max_k < 1000) return MEDIUM;
    if (max_k < 10000) return STIFF;
    return RIGID;
}
```

### Timestep Adaptation
Virtual springs with varying stiffness need adaptive timesteps:

```cpp
float computeVirtualSpringTimestep(const VirtualSpring& spring) {
    // Account for temperature weakening
    float effective_k = spring.stiffness * getTemperatureModifier(spring, temp);
    
    // Stability criterion with safety factor
    float dt_max = 0.5f * 2.0f / sqrt(effective_k / reduced_mass);
    
    // Further reduce for plastic deformation
    if (spring.is_yielding) {
        dt_max *= 0.5f;  // Extra safety during plastic flow
    }
    
    return dt_max;
}
```

### Mixed Material Challenges
When different materials connect:
- **Steel-rubber joint**: 100x stiffness difference
- **Solution**: Use smallest required timestep
- **Optimization**: Subcycle stiff components

## Key Advantages

- **Intuitive**: Materials behave like real-world counterparts
- **Predictable**: Designers can reason about composite behavior
- **Emergent**: Complex structures arise from simple rules
- **Efficient**: Properties computed once when spring forms
- **Realistic**: Based on actual material science principles
- **Stable**: Integration requirements known from material properties

This approach makes spring networks grounded in physical reality rather than abstract math, making it much easier to design and understand composite objects!