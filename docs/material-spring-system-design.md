# Material-Based Spring Formation System Design

## Overview

This document describes an enhanced spring formation system that creates rich emergent structures beyond simple spherical formations. The system uses material properties, directional bonding, and temperature-dependent mechanics to enable diverse structural formation while maintaining blazing performance for millions of particles.

## Core Architecture

### 1. Material Properties Extension

Add material properties to particles (minimal memory overhead):

```cpp
struct ParticleMaterial {
    uint8_t material_type;      // 256 material types
    uint8_t bonding_pattern;     // Directional bonding preference
    float temperature;           // Thermal state affects bonding
    float bonding_affinity;      // Base bonding strength modifier
};
```

### 2. Dual Force System

**Contact Forces** (Always Active):
- Apply to ALL particles regardless of material
- Soft Hertzian contact model
- Provides fluid-like behavior for liquids
- Prevents interpenetration

**Virtual Springs** (Material-Dependent):
- Form only between compatible materials
- Use directional bonding patterns for non-spherical structures
- Temperature-modulated formation probability

### 3. Fast Compatibility Matrix

```cpp
// 256x256 lookup table (64KB - fits in L2 cache)
struct BondingMatrix {
    struct BondingRule {
        uint8_t can_bond : 1;           // Can form springs?
        uint8_t bond_strength : 4;      // 0-15 strength levels
        uint8_t directional : 1;        // Use directional bonding?
        uint8_t temp_sensitive : 1;     // Temperature affects bonding?
        uint8_t phase_transition : 1;   // Can change material type?
    };
    BondingRule rules[256][256];
};
```

## Material Type System

### Base Material Categories

```cpp
enum MaterialCategory : uint8_t {
    // Fluids (0-31) - No springs, contact forces only
    WATER = 0,
    OIL = 1,
    MAGMA = 2,
    PLASMA = 3,

    // Gases (32-63) - No springs, weak contact
    AIR = 32,
    HYDROGEN = 33,
    HELIUM = 34,

    // Metals (64-95) - Strong springs, crystalline patterns
    IRON = 64,
    COPPER = 65,
    ALUMINUM = 66,
    TITANIUM = 67,

    // Crystals (96-127) - Directional bonding, lattice formation
    QUARTZ = 96,
    DIAMOND = 97,
    ICE = 98,
    SALT = 99,

    // Organics (128-159) - Complex bonding, branching structures
    CARBON = 128,
    POLYMER = 129,
    PROTEIN = 130,
    DNA = 131,

    // Ceramics (160-191) - Rigid sheets, high temperature
    SILICON = 160,
    GRAPHENE = 161,
    CLAY = 162,

    // Exotic (192-255) - Special behaviors
    DARK_MATTER = 192,
    STRANGE_MATTER = 193,
    NEUTRONIUM = 194,
};
```

## Directional Bonding Patterns

### Pattern Encoding (8-bit)

```cpp
enum BondingPattern : uint8_t {
    ISOTROPIC = 0,      // Uniform in all directions (default)
    LINEAR = 1,         // Prefers 180° bonds (chains)
    TRIANGULAR = 2,     // 120° angles (graphene sheets)
    TETRAHEDRAL = 3,    // 109.5° angles (diamond lattice)
    SQUARE = 4,         // 90° angles (salt crystals)
    HEXAGONAL = 5,      // 60° angles (ice, quartz)
    OCTAHEDRAL = 6,     // 8 preferred directions
    BRANCHING = 7,      // Tree-like structures
    HELICAL = 8,        // Spiral formations
    PLANAR = 9,         // Sheet formation
};
```

### Directional Scoring Function

```cpp
float getDirectionalScore(const Particle& p1, const Particle& p2,
                         BondingPattern pattern) {
    // Calculate angle from p1 to p2
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float angle = atan2f(dy, dx);

    // Score based on pattern alignment
    switch(pattern) {
        case LINEAR:
            // Prefer 0° or 180° bonds
            return 1.0f - fabsf(cosf(angle));

        case TRIANGULAR:
            // Prefer 120° separation
            return 0.5f * (cosf(3.0f * angle) + 1.0f);

        case TETRAHEDRAL:
            // 3D tetrahedral angles (simplified to 2D)
            return 0.5f * (cosf(2.0f * angle - M_PI/3) + 1.0f);

        case SQUARE:
            // Prefer 90° angles
            return fabsf(sinf(2.0f * angle));

        case HEXAGONAL:
            // Prefer 60° angles
            return 0.5f * (cosf(6.0f * angle) + 1.0f);

        default:
            return 1.0f; // No directional preference
    }
}
```

## Spring Formation Rules

### Enhanced Formation Criteria

```cpp
bool shouldFormSpring(const Particle& p1, const Particle& p2,
                     const ParticleMaterial& m1, const ParticleMaterial& m2,
                     const BondingMatrix& matrix, float distance, float rel_velocity) {

    // 1. Check material compatibility
    auto rule = matrix.rules[m1.material_type][m2.material_type];
    if (!rule.can_bond) return false;

    // 2. Distance check (material-dependent)
    float max_dist = BASE_BOND_DISTANCE * (1.0f + rule.bond_strength * 0.1f);
    if (distance > max_dist) return false;

    // 3. Velocity check (bond only if slow enough)
    float max_vel = BASE_BOND_VELOCITY / (1.0f + rule.bond_strength * 0.2f);
    if (rel_velocity > max_vel) return false;

    // 4. Temperature check (if temperature-sensitive)
    if (rule.temp_sensitive) {
        float avg_temp = (m1.temperature + m2.temperature) * 0.5f;
        float bond_temp_range = getBondingTempRange(m1.material_type);
        if (avg_temp < bond_temp_range.min || avg_temp > bond_temp_range.max) {
            return false;
        }
    }

    // 5. Directional check (if directional bonding)
    if (rule.directional) {
        float dir_score = getDirectionalScore(p1, p2, m1.bonding_pattern);
        if (dir_score < DIRECTIONAL_THRESHOLD) return false;
    }

    // 6. Saturation check (limit bonds per particle by material)
    uint32_t max_bonds = getMaxBonds(m1.material_type);
    if (particle_springs[p1.id].size() >= max_bonds) return false;

    return true;
}
```

### Material-Specific Bond Limits

```cpp
uint32_t getMaxBonds(uint8_t material_type) {
    switch(material_type) {
        // Linear polymers - 2 bonds (chains)
        case POLYMER: return 2;

        // Triangular sheets - 3 bonds
        case GRAPHENE: return 3;

        // Tetrahedral crystals - 4 bonds
        case DIAMOND:
        case ICE: return 4;

        // Square lattices - 4 bonds
        case SALT: return 4;

        // Hexagonal crystals - 6 bonds
        case QUARTZ: return 6;

        // Metals - 8-12 bonds (FCC/BCC lattice)
        case IRON:
        case COPPER: return 12;

        // Organics - variable (4-6)
        case PROTEIN:
        case DNA: return 6;

        // Default
        default: return 8;
    }
}
```

## Temperature and Phase Transitions

### Temperature Effects

```cpp
struct ThermalProperties {
    float melting_point;
    float boiling_point;
    float bond_weakening_rate;  // How much bonds weaken with temperature
    float thermal_conductivity;
};

// Update spring strength based on temperature
float getTemperatureModifiedStrength(float base_strength, float temperature,
                                    const ThermalProperties& props) {
    if (temperature > props.melting_point) {
        // Bonds weaken near melting point
        float melt_factor = (temperature - props.melting_point) /
                           (props.boiling_point - props.melting_point);
        return base_strength * (1.0f - melt_factor * props.bond_weakening_rate);
    }
    return base_strength;
}
```

### Phase Transition System

```cpp
void checkPhaseTransition(ParticleMaterial& material, float temperature) {
    switch(material.material_type) {
        case ICE:
            if (temperature > 273.0f) {
                material.material_type = WATER;
                material.bonding_pattern = ISOTROPIC;
                // Existing ice springs will break due to incompatibility
            }
            break;

        case WATER:
            if (temperature < 273.0f) {
                material.material_type = ICE;
                material.bonding_pattern = HEXAGONAL;
            } else if (temperature > 373.0f) {
                material.material_type = AIR;  // Steam
            }
            break;

        case IRON:
            if (temperature > 1538.0f) {
                material.material_type = MAGMA;  // Molten iron
                material.bonding_pattern = ISOTROPIC;
            }
            break;
    }
}
```

## Emergent Structure Examples

### 1. Crystalline Lattices (Metals, Crystals)
- Directional bonding creates regular lattices
- FCC/BCC for metals, hexagonal for ice
- Strong, rigid structures

### 2. Polymer Chains (Organics)
- Linear bonding pattern
- 2 bonds per particle maximum
- Flexible, can entangle

### 3. Graphene Sheets (Carbon)
- Triangular bonding pattern
- 3 bonds per particle
- Forms 2D sheets that can stack

### 4. Branching Structures (Organics)
- Variable bonding angles
- 3-4 bonds per particle
- Creates tree-like or coral-like structures

### 5. Fluid Behavior (Water, Oil)
- No springs, only contact forces
- Particles flow freely
- Surface tension emerges from contact forces

### 6. Composite Materials
- Mixed material types can form interfaces
- Some materials bond across types (adhesion)
- Creates complex multi-material structures

## Performance Optimizations

### 1. Bit-Packed Material Data
```cpp
struct PackedMaterial {
    uint32_t data;  // All material properties in 32 bits
    // Bits 0-7:   material_type
    // Bits 8-11:  bonding_pattern (16 patterns)
    // Bits 12-19: temperature (8-bit fixed point)
    // Bits 20-23: bond_count (current bonds)
    // Bits 24-31: flags and reserved
};
```

### 2. SIMD-Friendly Bonding Check
```cpp
// Check 4 particle pairs simultaneously with AVX2
__m256 checkBondingBatch(const float* dist2, const uint32_t* materials,
                         const BondingMatrix& matrix) {
    // Load 4 distances
    __m256 distances = _mm256_load_ps(dist2);

    // Check distance threshold (vectorized)
    __m256 max_dist = _mm256_set1_ps(MAX_BOND_DIST * MAX_BOND_DIST);
    __m256 dist_mask = _mm256_cmp_ps(distances, max_dist, _CMP_LT_OS);

    // Material compatibility would need scalar checks
    // but can be pipelined effectively
    return dist_mask;
}
```

### 3. Temporal Coherence
- Cache spring candidates from previous frame
- Only recheck if particles moved significantly
- Reduces redundant compatibility checks

### 4. Hierarchical Material Grouping
```cpp
// Group particles by material for better cache locality
struct MaterialGroup {
    uint8_t material_type;
    std::vector<uint32_t> particle_indices;
};

// Process compatible material pairs together
for (auto& group1 : material_groups) {
    for (auto& group2 : material_groups) {
        if (!matrix.rules[group1.type][group2.type].can_bond) continue;
        // Process all particles in these groups together
        processGroupPair(group1, group2);
    }
}
```

## Implementation Phases

### Phase 1: Basic Material System
1. Add material type to particles (uint8_t)
2. Implement bonding compatibility matrix
3. Modify spring formation to check compatibility
4. Test with 3-4 material types (water, metal, crystal)

### Phase 2: Directional Bonding
1. Add bonding pattern field
2. Implement directional scoring
3. Test lattice formation with crystals
4. Verify chain formation with polymers

### Phase 3: Temperature System
1. Add temperature field to particles
2. Implement thermal conductivity
3. Add temperature-dependent bonding
4. Implement phase transitions

### Phase 4: Advanced Materials
1. Expand to full material palette
2. Add exotic materials with special rules
3. Implement multi-material bonding
4. Fine-tune emergence parameters

## Expected Performance

For 1 million particles on 12-core CPU:
- Material lookup: <0.1ms (L1/L2 cache hits)
- Directional scoring: ~2ms (vectorizable)
- Spring formation: ~5ms (with all checks)
- Total overhead: <10ms per frame
- Target: 30+ FPS with full material system

Memory overhead per particle:
- Material properties: 4 bytes
- Temperature: 4 bytes (can be packed)
- Total: 8 bytes per particle (8MB for 1M particles)

## Validation Tests

1. **Crystallization Test**: Cool liquid metal, verify lattice formation
2. **Polymer Test**: Create polymer chains, test entanglement
3. **Phase Transition**: Heat ice to water to steam
4. **Multi-Material**: Metal-ceramic composite formation
5. **Performance Benchmark**: 1M particles, mixed materials, maintain 30 FPS

## Conclusion

This material-based spring system enables rich emergent structures while maintaining high performance. The key innovations are:
1. Dual force system (contact + springs)
2. Fast material compatibility matrix
3. Directional bonding patterns
4. Temperature-dependent mechanics
5. Efficient bit-packed representation

The system scales to millions of particles while creating diverse, physically-plausible structures that go far beyond simple spherical aggregates.