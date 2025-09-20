/**
 * Material System for DigiStar Physics Engine
 *
 * Implements material properties, bonding rules, and directional patterns
 * for rich emergent structure formation. Designed for blazing fast performance
 * with millions of particles.
 *
 * Key Features:
 * - 256 material types with unique properties
 * - Directional bonding for non-spherical structures
 * - Temperature-dependent phase transitions
 * - Fast 64KB bonding compatibility matrix
 * - SIMD-optimized bonding checks
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <array>
#include <vector>
#include <cstring>
#include <immintrin.h>  // For SIMD

namespace digistar {

// ============================================================================
// Material Type Definitions
// ============================================================================

enum MaterialCategory : uint8_t {
    // Fluids (0-31) - No springs, contact forces only
    WATER = 0,
    OIL = 1,
    MAGMA = 2,
    PLASMA = 3,
    ACID = 4,
    MERCURY = 5,

    // Gases (32-63) - No springs, very weak contact
    AIR = 32,
    HYDROGEN = 33,
    HELIUM = 34,
    OXYGEN = 35,
    NITROGEN = 36,
    STEAM = 37,

    // Metals (64-95) - Strong springs, metallic bonding
    IRON = 64,
    COPPER = 65,
    ALUMINUM = 66,
    TITANIUM = 67,
    GOLD = 68,
    SILVER = 69,
    STEEL = 70,

    // Crystals (96-127) - Directional bonding, regular lattices
    QUARTZ = 96,
    DIAMOND = 97,
    ICE = 98,
    SALT = 99,
    SILICON_CRYSTAL = 100,
    RUBY = 101,

    // Organics (128-159) - Complex bonding, flexible structures
    CARBON = 128,
    POLYMER = 129,
    PROTEIN = 130,
    DNA = 131,
    CELLULOSE = 132,
    RUBBER = 133,

    // Ceramics (160-191) - Rigid, high temperature resistance
    SILICON_CARBIDE = 160,
    GRAPHENE = 161,
    CLAY = 162,
    GLASS = 163,
    CONCRETE = 164,

    // Exotic (192-255) - Special physics behaviors
    DARK_MATTER = 192,
    STRANGE_MATTER = 193,
    NEUTRONIUM = 194,
    ANTIMATTER = 195,
    ZERO_POINT = 196,
};

// ============================================================================
// Bonding Patterns for Directional Structure Formation
// ============================================================================

enum BondingPattern : uint8_t {
    ISOTROPIC = 0,      // Uniform in all directions (spherical)
    LINEAR = 1,         // 180° bonds (chains, polymers)
    TRIANGULAR = 2,     // 120° angles (graphene sheets)
    TETRAHEDRAL = 3,    // 109.5° angles (diamond, ice)
    SQUARE = 4,         // 90° angles (salt crystals)
    HEXAGONAL = 5,      // 60° angles (quartz, ice sheets)
    OCTAHEDRAL = 6,     // 6 directions (FCC metals)
    DODECAHEDRAL = 7,   // 12 directions (BCC metals)
    BRANCHING = 8,      // Tree-like (organics)
    HELICAL = 9,        // Spiral (DNA, proteins)
    PLANAR = 10,        // 2D sheets (graphene)
    RANDOM = 11,        // Amorphous (glass)
};

// ============================================================================
// Bonding Rules and Compatibility Matrix
// ============================================================================

struct BondingRule {
    uint8_t can_bond : 1;        // Can form springs?
    uint8_t strength : 4;        // Bond strength (0-15)
    uint8_t directional : 1;     // Use directional bonding?
    uint8_t temp_sensitive : 1;  // Temperature affects bonding?
    uint8_t breaks_easily : 1;   // Low breaking threshold?
};

// 256x256 bonding compatibility matrix (64KB - fits in L2 cache)
class BondingMatrix {
public:
    BondingMatrix() { initializeDefaults(); }

    // Get bonding rule between two materials
    inline BondingRule getRule(uint8_t mat1, uint8_t mat2) const {
        return rules_[mat1][mat2];
    }

    // Set custom bonding rule
    void setRule(uint8_t mat1, uint8_t mat2, BondingRule rule) {
        rules_[mat1][mat2] = rule;
        rules_[mat2][mat1] = rule;  // Symmetric
    }

private:
    std::array<std::array<BondingRule, 256>, 256> rules_;

    void initializeDefaults();
};

// ============================================================================
// Particle Material Properties
// ============================================================================

struct ParticleMaterial {
    uint8_t type;           // Material type (0-255)
    uint8_t pattern;        // Bonding pattern
    uint16_t temperature;   // Temperature (K) as fixed-point

    // Pack into 32-bit word for cache efficiency
    uint32_t pack() const {
        return (uint32_t(type) << 0) |
               (uint32_t(pattern) << 8) |
               (uint32_t(temperature) << 16);
    }

    void unpack(uint32_t packed) {
        type = (packed >> 0) & 0xFF;
        pattern = (packed >> 8) & 0xFF;
        temperature = (packed >> 16) & 0xFFFF;
    }
};

// ============================================================================
// Material Properties Database
// ============================================================================

struct MaterialProperties {
    float density;              // kg/m³
    float melting_point;        // Kelvin
    float boiling_point;        // Kelvin
    float specific_heat;        // J/(kg·K)
    float thermal_conductivity; // W/(m·K)
    float young_modulus;        // Pa (stiffness)
    float bonding_distance;     // Preferred bond length
    float bonding_energy;       // eV (bond strength)
    uint8_t max_bonds;          // Maximum bonds per particle
    uint8_t bonding_pattern;    // Default pattern
    uint32_t color;             // RGBA visualization color
};

// Get properties for a material type
const MaterialProperties& getMaterialProperties(uint8_t type);

// ============================================================================
// Directional Bonding Functions
// ============================================================================

/**
 * Calculate directional bonding score based on angle between particles
 * Returns 0.0 (worst alignment) to 1.0 (perfect alignment)
 */
inline float getDirectionalScore(float dx, float dy, BondingPattern pattern,
                                 float existing_bond_angle = NAN) {
    float angle = atan2f(dy, dx);

    switch(pattern) {
        case LINEAR:
            // Prefer 0° or 180° bonds (straight lines)
            return 0.5f * (1.0f + fabsf(cosf(angle)));

        case TRIANGULAR:
            // Prefer 120° separation (triangular lattice)
            return 0.5f * (cosf(3.0f * angle) + 1.0f);

        case TETRAHEDRAL: {
            // 109.5° angles (simplified to 2D projection)
            float tetra_angle = 109.5f * M_PI / 180.0f;
            return expf(-2.0f * (angle - tetra_angle) * (angle - tetra_angle));
        }

        case SQUARE:
            // Prefer 90° angles (cubic lattice)
            return 0.5f * (1.0f + fabsf(sinf(2.0f * angle)));

        case HEXAGONAL:
            // Prefer 60° angles (hexagonal lattice)
            return 0.5f * (cosf(6.0f * angle) + 1.0f);

        case OCTAHEDRAL: {
            // 6 preferred directions
            float score = 0.0f;
            for (int i = 0; i < 6; i++) {
                float pref_angle = i * M_PI / 3.0f;
                score = fmaxf(score, expf(-4.0f * (angle - pref_angle) * (angle - pref_angle)));
            }
            return score;
        }

        case BRANCHING: {
            // Tree-like branching at ~30-45° angles
            if (!std::isnan(existing_bond_angle)) {
                float branch_angle = fabsf(angle - existing_bond_angle);
                if (branch_angle > M_PI) branch_angle = 2*M_PI - branch_angle;
                float ideal = 40.0f * M_PI / 180.0f;
                return expf(-2.0f * (branch_angle - ideal) * (branch_angle - ideal));
            }
            return 1.0f;  // First bond can be any direction
        }

        case PLANAR:
            // Prefer bonds in XY plane (for 2D this is always true)
            return 1.0f;

        case ISOTROPIC:
        case RANDOM:
        default:
            return 1.0f;  // No directional preference
    }
}

// ============================================================================
// Temperature and Phase Transitions
// ============================================================================

/**
 * Get temperature-modified bond strength
 */
inline float getTemperatureModifiedStrength(float base_strength, float temperature,
                                           float melting_point, float boiling_point) {
    if (temperature < melting_point * 0.8f) {
        return base_strength;  // Full strength when cold
    } else if (temperature < melting_point) {
        // Gradual weakening near melting point
        float factor = (temperature - 0.8f * melting_point) / (0.2f * melting_point);
        return base_strength * (1.0f - 0.5f * factor);
    } else if (temperature < boiling_point) {
        // Very weak bonds in liquid phase
        return base_strength * 0.1f;
    } else {
        // No bonds in gas phase
        return 0.0f;
    }
}

/**
 * Check and perform phase transitions
 */
inline void checkPhaseTransition(ParticleMaterial& material, float temperature) {
    // Water/Ice/Steam transitions
    if (material.type == ICE && temperature > 273.0f) {
        material.type = WATER;
        material.pattern = ISOTROPIC;
    } else if (material.type == WATER) {
        if (temperature < 273.0f) {
            material.type = ICE;
            material.pattern = HEXAGONAL;
        } else if (temperature > 373.0f) {
            material.type = STEAM;
            material.pattern = ISOTROPIC;
        }
    } else if (material.type == STEAM && temperature < 373.0f) {
        material.type = WATER;
    }

    // Metal melting
    else if (material.type == IRON && temperature > 1811.0f) {
        material.type = MAGMA;
        material.pattern = ISOTROPIC;
    } else if (material.type == MAGMA && temperature < 1811.0f) {
        material.type = IRON;
        material.pattern = DODECAHEDRAL;
    }
}

// ============================================================================
// SIMD-Optimized Bonding Checks
// ============================================================================

/**
 * Check multiple particle pairs for bonding eligibility using AVX2
 * Processes 8 pairs simultaneously
 */
inline __m256i checkBondingBatchAVX2(
    const float* distances,      // 8 distances
    const uint8_t* materials1,   // 8 material types
    const uint8_t* materials2,   // 8 material types
    const BondingMatrix& matrix,
    float max_distance) {

    // Load and check distances
    __m256 dist_vec = _mm256_loadu_ps(distances);
    __m256 max_dist_vec = _mm256_set1_ps(max_distance);
    __m256 dist_mask = _mm256_cmp_ps(dist_vec, max_dist_vec, _CMP_LT_OS);

    // Convert float mask to integer mask
    __m256i result = _mm256_castps_si256(dist_mask);

    // Material compatibility must be checked individually
    // but can be pipelined effectively
    alignas(32) uint32_t mask_array[8];
    _mm256_store_si256((__m256i*)mask_array, result);

    for (int i = 0; i < 8; i++) {
        if (mask_array[i]) {
            auto rule = matrix.getRule(materials1[i], materials2[i]);
            mask_array[i] = rule.can_bond ? 0xFFFFFFFF : 0;
        }
    }

    return _mm256_load_si256((__m256i*)mask_array);
}

// ============================================================================
// Material Group Management for Cache Efficiency
// ============================================================================

/**
 * Group particles by material type for better cache locality
 */
struct MaterialGroup {
    uint8_t material_type;
    std::vector<uint32_t> particle_indices;

    void clear() { particle_indices.clear(); }
    void reserve(size_t n) { particle_indices.reserve(n); }
    void add(uint32_t idx) { particle_indices.push_back(idx); }
};

/**
 * Organize particles into material groups for efficient processing
 */
class MaterialGroupManager {
public:
    void rebuild(const ParticleMaterial* materials, size_t count) {
        // Clear existing groups
        for (auto& group : groups_) {
            group.clear();
        }

        // Sort particles into groups
        for (size_t i = 0; i < count; i++) {
            uint8_t type = materials[i].type;
            groups_[type].add(i);
        }
    }

    // Iterate over non-empty groups
    template<typename Func>
    void forEachGroup(Func&& func) {
        for (auto& group : groups_) {
            if (!group.particle_indices.empty()) {
                func(group);
            }
        }
    }

    // Process compatible material pairs
    template<typename Func>
    void forEachCompatiblePair(const BondingMatrix& matrix, Func&& func) {
        for (size_t i = 0; i < 256; i++) {
            if (groups_[i].particle_indices.empty()) continue;

            for (size_t j = i; j < 256; j++) {
                if (groups_[j].particle_indices.empty()) continue;

                auto rule = matrix.getRule(i, j);
                if (rule.can_bond) {
                    func(groups_[i], groups_[j], rule);
                }
            }
        }
    }

private:
    std::array<MaterialGroup, 256> groups_;
};

// ============================================================================
// Spring Formation Decision Function
// ============================================================================

struct SpringFormationContext {
    float max_distance = 3.0f;
    float max_velocity = 1.0f;
    float directional_threshold = 0.3f;
    float temperature_penalty = 0.5f;
};

/**
 * Comprehensive spring formation decision
 */
inline bool shouldFormSpring(
    float distance,
    float relative_velocity,
    const ParticleMaterial& mat1,
    const ParticleMaterial& mat2,
    const BondingMatrix& matrix,
    const SpringFormationContext& ctx,
    float dx, float dy,
    uint8_t current_bonds1,
    uint8_t current_bonds2) {

    // 1. Get bonding rule
    auto rule = matrix.getRule(mat1.type, mat2.type);
    if (!rule.can_bond) return false;

    // 2. Check current bond saturation
    auto props1 = getMaterialProperties(mat1.type);
    auto props2 = getMaterialProperties(mat2.type);
    if (current_bonds1 >= props1.max_bonds || current_bonds2 >= props2.max_bonds) {
        return false;
    }

    // 3. Distance check (material-specific)
    float max_dist = ctx.max_distance * (1.0f + rule.strength * 0.1f);
    if (distance > max_dist) return false;

    // 4. Velocity check (stronger bonds need lower velocity)
    float max_vel = ctx.max_velocity / (1.0f + rule.strength * 0.2f);
    if (relative_velocity > max_vel) return false;

    // 5. Temperature check
    if (rule.temp_sensitive) {
        float avg_temp = (mat1.temperature + mat2.temperature) * 0.5f;
        float avg_melt = (props1.melting_point + props2.melting_point) * 0.5f;
        if (avg_temp > avg_melt * 0.9f) {
            // Near melting point, bonds unlikely to form
            return false;
        }
    }

    // 6. Directional bonding check
    if (rule.directional) {
        BondingPattern pattern = static_cast<BondingPattern>(mat1.pattern);
        float score = getDirectionalScore(dx, dy, pattern);
        if (score < ctx.directional_threshold) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// BondingMatrix Implementation
// ============================================================================

inline void BondingMatrix::initializeDefaults() {
    // Initialize all to no bonding
    for (auto& row : rules_) {
        for (auto& rule : row) {
            rule = {0, 0, 0, 0, 0};
        }
    }

    // === FLUIDS - No springs ===
    // Water with itself (no springs, fluid behavior)
    setRule(WATER, WATER, {0, 0, 0, 0, 0});

    // === METALS - Strong metallic bonding ===
    setRule(IRON, IRON, {1, 12, 0, 1, 0});      // Strong, temp-sensitive
    setRule(COPPER, COPPER, {1, 10, 0, 1, 0});
    setRule(GOLD, GOLD, {1, 8, 0, 1, 0});

    // Metal alloys (can bond across types)
    setRule(IRON, COPPER, {1, 9, 0, 1, 0});
    setRule(IRON, TITANIUM, {1, 14, 0, 1, 0});

    // === CRYSTALS - Directional bonding ===
    setRule(ICE, ICE, {1, 6, 1, 1, 0});         // Hexagonal, temp-sensitive
    setRule(SALT, SALT, {1, 8, 1, 0, 0});       // Square lattice
    setRule(DIAMOND, DIAMOND, {1, 15, 1, 0, 0}); // Tetrahedral, very strong
    setRule(QUARTZ, QUARTZ, {1, 10, 1, 0, 0});  // Hexagonal

    // === ORGANICS - Complex bonding ===
    setRule(CARBON, CARBON, {1, 12, 1, 0, 0});  // Various forms
    setRule(POLYMER, POLYMER, {1, 4, 1, 0, 1}); // Linear chains, breaks easily
    setRule(PROTEIN, PROTEIN, {1, 6, 1, 1, 1}); // Complex, temp-sensitive
    setRule(DNA, DNA, {1, 5, 1, 1, 1});         // Helical

    // === CERAMICS - High temperature ===
    setRule(GRAPHENE, GRAPHENE, {1, 14, 1, 0, 0}); // Triangular, very strong
    setRule(SILICON_CARBIDE, SILICON_CARBIDE, {1, 13, 1, 0, 0});
    setRule(GLASS, GLASS, {1, 7, 0, 0, 1});     // Amorphous, breaks easily

    // === CROSS-MATERIAL BONDING ===
    // Adhesion between different materials
    setRule(WATER, ICE, {1, 4, 0, 1, 0});       // Water can bond to ice
    setRule(POLYMER, RUBBER, {1, 3, 0, 0, 1});  // Weak polymer bonds
    setRule(IRON, SILICON_CARBIDE, {1, 8, 0, 0, 0}); // Metal-ceramic interface

    // === EXOTIC MATERIALS ===
    setRule(DARK_MATTER, DARK_MATTER, {1, 2, 0, 0, 0}); // Weak, mysterious
    setRule(NEUTRONIUM, NEUTRONIUM, {1, 15, 0, 0, 0});  // Extremely strong
    setRule(ANTIMATTER, ANTIMATTER, {0, 0, 0, 0, 0});   // No bonds (annihilates)
}

} // namespace digistar