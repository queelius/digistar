/**
 * Material System Implementation
 *
 * Provides material properties database and helper functions for
 * the DigiStar material-based spring formation system.
 */

#include "material_system.h"
#include <unordered_map>

namespace digistar {

// Static material properties database
static std::unordered_map<uint8_t, MaterialProperties> g_material_database;
static bool g_database_initialized = false;

// Initialize the material properties database
static void initializeMaterialDatabase() {
    if (g_database_initialized) return;

    // === FLUIDS ===
    g_material_database[WATER] = {
        1000.0f,      // density (kg/m³)
        273.15f,      // melting point (K)
        373.15f,      // boiling point (K)
        4186.0f,      // specific heat (J/(kg·K))
        0.6f,         // thermal conductivity (W/(m·K))
        2.2e9f,       // young modulus (Pa) - bulk modulus for water
        0.0f,         // bonding distance - no bonds
        0.0f,         // bonding energy - no bonds
        0,            // max bonds
        ISOTROPIC,    // bonding pattern
        0xFF2060C0    // color (blue)
    };

    g_material_database[OIL] = {
        920.0f,       // density
        233.0f,       // melting point
        573.0f,       // boiling point
        2000.0f,      // specific heat
        0.15f,        // thermal conductivity
        1.5e9f,       // young modulus
        0.0f,         // bonding distance
        0.0f,         // bonding energy
        0,            // max bonds
        ISOTROPIC,    // pattern
        0xFF404020    // color (dark brown)
    };

    g_material_database[MAGMA] = {
        2700.0f,      // density (molten rock)
        1473.0f,      // melting point
        3000.0f,      // boiling point
        1000.0f,      // specific heat
        2.0f,         // thermal conductivity
        0.0f,         // young modulus (liquid)
        0.0f,         // bonding distance
        0.0f,         // bonding energy
        0,            // max bonds
        ISOTROPIC,    // pattern
        0xFFFF4000    // color (orange-red)
    };

    // === GASES ===
    g_material_database[AIR] = {
        1.225f,       // density at STP
        55.0f,        // melting point (N2/O2 mix)
        80.0f,        // boiling point
        1005.0f,      // specific heat
        0.025f,       // thermal conductivity
        0.0f,         // young modulus
        0.0f,         // bonding distance
        0.0f,         // bonding energy
        0,            // max bonds
        ISOTROPIC,    // pattern
        0x40FFFFFF    // color (transparent white)
    };

    g_material_database[HYDROGEN] = {
        0.09f,        // density
        14.0f,        // melting point
        20.0f,        // boiling point
        14300.0f,     // specific heat
        0.18f,        // thermal conductivity
        0.0f,         // young modulus
        0.0f,         // bonding distance
        0.0f,         // bonding energy
        0,            // max bonds
        ISOTROPIC,    // pattern
        0x40FFE0E0    // color (light cyan)
    };

    g_material_database[STEAM] = {
        0.6f,         // density at 100°C
        273.15f,      // melting point (becomes water)
        373.15f,      // boiling point
        2010.0f,      // specific heat
        0.025f,       // thermal conductivity
        0.0f,         // young modulus
        0.0f,         // bonding distance
        0.0f,         // bonding energy
        0,            // max bonds
        ISOTROPIC,    // pattern
        0x80FFFFFF    // color (translucent white)
    };

    // === METALS ===
    g_material_database[IRON] = {
        7874.0f,      // density
        1811.0f,      // melting point
        3134.0f,      // boiling point
        449.0f,       // specific heat
        80.4f,        // thermal conductivity
        211e9f,       // young modulus
        2.5f,         // bonding distance
        4.3f,         // bonding energy (eV)
        12,           // max bonds (BCC/FCC)
        DODECAHEDRAL, // pattern
        0xFF808080    // color (gray)
    };

    g_material_database[COPPER] = {
        8960.0f,      // density
        1357.0f,      // melting point
        2835.0f,      // boiling point
        385.0f,       // specific heat
        401.0f,       // thermal conductivity
        130e9f,       // young modulus
        2.6f,         // bonding distance
        3.5f,         // bonding energy
        12,           // max bonds (FCC)
        DODECAHEDRAL, // pattern
        0xFFB87333    // color (copper)
    };

    g_material_database[GOLD] = {
        19300.0f,     // density
        1337.0f,      // melting point
        3129.0f,      // boiling point
        129.0f,       // specific heat
        318.0f,       // thermal conductivity
        78e9f,        // young modulus
        2.9f,         // bonding distance
        3.8f,         // bonding energy
        12,           // max bonds (FCC)
        DODECAHEDRAL, // pattern
        0xFFFFD700    // color (gold)
    };

    g_material_database[ALUMINUM] = {
        2700.0f,      // density
        933.0f,       // melting point
        2792.0f,      // boiling point
        897.0f,       // specific heat
        237.0f,       // thermal conductivity
        70e9f,        // young modulus
        2.9f,         // bonding distance
        3.4f,         // bonding energy
        12,           // max bonds
        DODECAHEDRAL, // pattern
        0xFFC0C0C0    // color (silver)
    };

    g_material_database[TITANIUM] = {
        4540.0f,      // density
        1941.0f,      // melting point
        3560.0f,      // boiling point
        523.0f,       // specific heat
        21.9f,        // thermal conductivity
        116e9f,       // young modulus
        2.9f,         // bonding distance
        4.9f,         // bonding energy
        12,           // max bonds
        HEXAGONAL,    // pattern (HCP)
        0xFFE0E0E0    // color (light gray)
    };

    // === CRYSTALS ===
    g_material_database[ICE] = {
        917.0f,       // density
        273.15f,      // melting point
        273.15f,      // boiling point (sublimation)
        2090.0f,      // specific heat
        2.2f,         // thermal conductivity
        9.5e9f,       // young modulus
        2.76f,        // bonding distance (H-bond)
        0.5f,         // bonding energy (hydrogen bond)
        4,            // max bonds (tetrahedral)
        HEXAGONAL,    // pattern
        0xFFE0F8FF    // color (light blue)
    };

    g_material_database[SALT] = {
        2165.0f,      // density (NaCl)
        1074.0f,      // melting point
        1686.0f,      // boiling point
        880.0f,       // specific heat
        6.5f,         // thermal conductivity
        40e9f,        // young modulus
        2.8f,         // bonding distance
        8.2f,         // bonding energy (ionic)
        6,            // max bonds (cubic)
        SQUARE,       // pattern
        0xFFF0F0F0    // color (white)
    };

    g_material_database[DIAMOND] = {
        3515.0f,      // density
        4000.0f,      // melting point (at high pressure)
        4300.0f,      // boiling point
        630.0f,       // specific heat
        2000.0f,      // thermal conductivity
        1220e9f,      // young modulus (hardest material)
        1.54f,        // bonding distance (C-C)
        7.4f,         // bonding energy
        4,            // max bonds (tetrahedral)
        TETRAHEDRAL,  // pattern
        0xFFB0E0FF    // color (light cyan)
    };

    g_material_database[QUARTZ] = {
        2650.0f,      // density (SiO2)
        1983.0f,      // melting point
        2503.0f,      // boiling point
        740.0f,       // specific heat
        1.3f,         // thermal conductivity
        87e9f,        // young modulus
        1.61f,        // bonding distance (Si-O)
        5.0f,         // bonding energy
        6,            // max bonds
        HEXAGONAL,    // pattern
        0xFFF0D0FF    // color (light purple)
    };

    // === ORGANICS ===
    g_material_database[CARBON] = {
        2267.0f,      // density (graphite)
        3915.0f,      // melting point
        4300.0f,      // boiling point
        710.0f,       // specific heat
        120.0f,       // thermal conductivity
        1000e9f,      // young modulus (varies widely)
        1.42f,        // bonding distance (C-C)
        6.0f,         // bonding energy
        4,            // max bonds (can vary)
        TETRAHEDRAL,  // pattern (can vary)
        0xFF202020    // color (black)
    };

    g_material_database[POLYMER] = {
        950.0f,       // density (polyethylene)
        400.0f,       // melting point
        600.0f,       // boiling point (decomposition)
        2300.0f,      // specific heat
        0.5f,         // thermal conductivity
        1e9f,         // young modulus
        1.54f,        // bonding distance
        3.5f,         // bonding energy
        2,            // max bonds (chain)
        LINEAR,       // pattern
        0xFF40A040    // color (green)
    };

    g_material_database[PROTEIN] = {
        1350.0f,      // density
        500.0f,       // denaturation temperature
        600.0f,       // decomposition
        1900.0f,      // specific heat
        0.2f,         // thermal conductivity
        2e9f,         // young modulus
        3.8f,         // bonding distance (peptide)
        3.0f,         // bonding energy
        4,            // max bonds
        HELICAL,      // pattern
        0xFFFF60A0    // color (pink)
    };

    g_material_database[DNA] = {
        1700.0f,      // density
        350.0f,       // melting temperature
        400.0f,       // decomposition
        1500.0f,      // specific heat
        0.3f,         // thermal conductivity
        1e9f,         // young modulus
        3.4f,         // bonding distance
        2.0f,         // bonding energy (H-bonds)
        3,            // max bonds
        HELICAL,      // pattern
        0xFF8080FF    // color (light blue)
    };

    g_material_database[RUBBER] = {
        920.0f,       // density
        200.0f,       // glass transition
        500.0f,       // decomposition
        2000.0f,      // specific heat
        0.13f,        // thermal conductivity
        0.01e9f,      // young modulus (very soft)
        3.0f,         // bonding distance
        2.5f,         // bonding energy
        3,            // max bonds
        BRANCHING,    // pattern
        0xFF404040    // color (dark gray)
    };

    // === CERAMICS ===
    g_material_database[GRAPHENE] = {
        2267.0f,      // density
        4000.0f,      // melting point
        4300.0f,      // boiling point
        710.0f,       // specific heat
        5000.0f,      // thermal conductivity
        1000e9f,      // young modulus
        1.42f,        // bonding distance
        7.4f,         // bonding energy
        3,            // max bonds (planar)
        TRIANGULAR,   // pattern
        0xFF101010    // color (very dark)
    };

    g_material_database[SILICON_CARBIDE] = {
        3210.0f,      // density
        3003.0f,      // melting point
        3500.0f,      // boiling point
        750.0f,       // specific heat
        120.0f,       // thermal conductivity
        450e9f,       // young modulus
        1.89f,        // bonding distance
        5.5f,         // bonding energy
        4,            // max bonds
        TETRAHEDRAL,  // pattern
        0xFF606060    // color (dark gray)
    };

    g_material_database[GLASS] = {
        2500.0f,      // density
        1700.0f,      // melting point
        2230.0f,      // boiling point
        840.0f,       // specific heat
        1.0f,         // thermal conductivity
        70e9f,        // young modulus
        2.3f,         // bonding distance
        4.0f,         // bonding energy
        4,            // max bonds
        RANDOM,       // pattern (amorphous)
        0x80A0C0E0    // color (transparent blue)
    };

    g_material_database[CONCRETE] = {
        2400.0f,      // density
        1500.0f,      // decomposition temp
        2000.0f,      // melting (decomposes)
        880.0f,       // specific heat
        0.8f,         // thermal conductivity
        30e9f,        // young modulus
        3.0f,         // bonding distance
        3.5f,         // bonding energy
        6,            // max bonds
        RANDOM,       // pattern
        0xFF808080    // color (gray)
    };

    // === EXOTIC MATERIALS ===
    g_material_database[DARK_MATTER] = {
        100.0f,       // density (unknown)
        0.0f,         // melting point (unknown)
        1e6f,         // boiling point (unknown)
        1000.0f,      // specific heat (unknown)
        0.0f,         // thermal conductivity (none)
        0.0f,         // young modulus
        5.0f,         // bonding distance
        0.1f,         // bonding energy (very weak)
        2,            // max bonds
        ISOTROPIC,    // pattern
        0x80400080    // color (translucent purple)
    };

    g_material_database[NEUTRONIUM] = {
        4e17f,        // density (neutron star material)
        1e9f,         // melting point
        1e10f,        // boiling point
        1000.0f,      // specific heat
        1e10f,        // thermal conductivity
        1e20f,        // young modulus
        1e-15f,       // bonding distance (nuclear)
        1000.0f,      // bonding energy (MeV scale)
        20,           // max bonds
        ISOTROPIC,    // pattern
        0xFFFFFFFF    // color (white)
    };

    g_material_database[ANTIMATTER] = {
        1000.0f,      // density (like normal matter)
        273.0f,       // melting point
        373.0f,       // boiling point
        4186.0f,      // specific heat
        0.6f,         // thermal conductivity
        2.2e9f,       // young modulus
        0.0f,         // bonding distance (annihilates)
        0.0f,         // bonding energy
        0,            // max bonds (none)
        ISOTROPIC,    // pattern
        0xFFFF00FF    // color (magenta)
    };

    g_material_database[ZERO_POINT] = {
        0.0f,         // density (vacuum energy)
        0.0f,         // melting point
        1e10f,        // boiling point
        1e6f,         // specific heat
        1e10f,        // thermal conductivity
        0.0f,         // young modulus
        10.0f,        // bonding distance (quantum foam)
        -1.0f,        // bonding energy (negative - repulsive)
        1,            // max bonds
        RANDOM,       // pattern
        0x40FFFF00    // color (translucent yellow)
    };

    g_database_initialized = true;
}

// Default material properties for undefined materials
static const MaterialProperties DEFAULT_MATERIAL = {
    1000.0f,      // density
    1000.0f,      // melting point
    2000.0f,      // boiling point
    1000.0f,      // specific heat
    1.0f,         // thermal conductivity
    1e9f,         // young modulus
    2.0f,         // bonding distance
    3.0f,         // bonding energy
    6,            // max bonds
    ISOTROPIC,    // pattern
    0xFF808080    // color (gray)
};

// Get material properties for a given type
const MaterialProperties& getMaterialProperties(uint8_t type) {
    initializeMaterialDatabase();

    auto it = g_material_database.find(type);
    if (it != g_material_database.end()) {
        return it->second;
    }
    return DEFAULT_MATERIAL;
}

} // namespace digistar