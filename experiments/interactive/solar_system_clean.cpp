/**
 * @file solar_system_clean.cpp
 * @brief Clean, parameterized multi-star system simulation
 * 
 * This is a reference implementation of a large-scale celestial mechanics
 * simulation featuring multiple star systems, planets, moons, asteroids,
 * comets, and Kuiper belt objects.
 * 
 * Features:
 * - Fully parameterized configuration
 * - Clean separation of concerns
 * - Proper physics abstractions
 * - Entity tracking system
 * - Interactive visualization
 * 
 * @author DigiStar Project
 * @date 2024
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <cstring>
#include <algorithm>
#include <random>
#include <memory>
#include <string>
#include <map>
#include <array>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>

// ============================================================================
// Configuration Section - All magic numbers go here
// ============================================================================

namespace Config {
    // Simulation parameters
    namespace Simulation {
        constexpr double G = 4.0 * M_PI * M_PI;  // G in AU³/M☉·year²
        constexpr double TIME_STEP = 0.00001;    // years (~0.0876 hours)
        constexpr double SOFTENING = 1e-6;       // Softening parameter for gravity
        constexpr int OMP_THREADS = 4;           // OpenMP thread count
    }
    
    // System generation parameters
    namespace Generation {
        // Particle counts
        constexpr int SATURN_RING_PARTICLES = 5000;
        constexpr int RANDOM_ASTEROIDS = 100000;
        constexpr int RANDOM_KBOS = 50000;
        constexpr int ALPHA_CEN_ASTEROIDS = 3000;
        
        // Distances
        constexpr float ALPHA_CEN_DISTANCE = 500.0f;  // AU from Sol
        constexpr float ASTEROID_BELT_INNER = 2.2f;   // AU
        constexpr float ASTEROID_BELT_OUTER = 3.3f;   // AU
        constexpr float KUIPER_BELT_INNER = 30.0f;    // AU
        constexpr float KUIPER_BELT_OUTER = 50.0f;    // AU
        
        // Masses
        constexpr float DEFAULT_ASTEROID_MASS = 1e-12f;  // Solar masses
        constexpr float DEFAULT_KBO_MASS = 1e-14f;       // Solar masses
        constexpr float RING_PARTICLE_MASS = 1e-20f;     // Solar masses
    }
    
    // Visualization parameters
    namespace Display {
        constexpr int SCREEN_WIDTH = 80;
        constexpr int SCREEN_HEIGHT = 40;
        constexpr int TRAIL_LENGTH = 200;
        constexpr int FRAME_SKIP = 5;           // Display every Nth frame
        constexpr int TARGET_FPS = 20;
        constexpr int FRAME_TIME_MS = 1000 / TARGET_FPS;
        
        // Zoom levels
        constexpr float ZOOM_FACTOR = 1.5f;
        constexpr float DEFAULT_ZOOM = 0.5f;
        constexpr float TRACKING_ZOOM = 10.0f;
    }
    
    // Random seed for reproducibility
    constexpr int RANDOM_SEED = 42;
}

// ============================================================================
// Core Data Structures
// ============================================================================

/**
 * @brief 2D vector class for positions, velocities, and forces
 */
struct Vector2 {
    float x, y;
    
    Vector2() : x(0), y(0) {}
    Vector2(float x_, float y_) : x(x_), y(y_) {}
    
    Vector2 operator+(const Vector2& o) const { return {x + o.x, y + o.y}; }
    Vector2 operator-(const Vector2& o) const { return {x - o.x, y - o.y}; }
    Vector2 operator*(float s) const { return {x * s, y * s}; }
    Vector2 operator/(float s) const { return {x / s, y / s}; }
    Vector2& operator+=(const Vector2& o) { x += o.x; y += o.y; return *this; }
    Vector2& operator-=(const Vector2& o) { x -= o.x; y -= o.y; return *this; }
    
    float length() const { return std::sqrt(x * x + y * y); }
    float length_squared() const { return x * x + y * y; }
    Vector2 normalized() const { 
        float len = length();
        return len > 0 ? Vector2(x/len, y/len) : Vector2(0, 0);
    }
    
    static float dot(const Vector2& a, const Vector2& b) {
        return a.x * b.x + a.y * b.y;
    }
};

/**
 * @brief Types of celestial bodies
 */
enum class BodyType : uint8_t {
    STAR = 0,
    PLANET = 1,
    MOON = 2,
    ASTEROID = 3,
    KBO = 4,
    RING_PARTICLE = 5,
    COMET = 6
};

/**
 * @brief Star system identifier
 */
enum class StarSystem : uint8_t {
    SOL = 0,
    ALPHA_CENTAURI = 1
};

/**
 * @brief Celestial body representation
 */
struct CelestialBody {
    Vector2 position;
    Vector2 velocity;
    Vector2 force;
    float mass;
    BodyType type;
    StarSystem system;
    std::string name;
    
    // Display properties
    char symbol;
    int priority;
    
    CelestialBody() 
        : mass(0), type(BodyType::ASTEROID), system(StarSystem::SOL), 
          symbol('.'), priority(1) {}
    
    CelestialBody(const Vector2& pos, const Vector2& vel, float m, 
                  BodyType t, const std::string& n = "", 
                  StarSystem sys = StarSystem::SOL)
        : position(pos), velocity(vel), force(0, 0), mass(m), 
          type(t), system(sys), name(n) {
        
        // Set display properties based on type
        switch(type) {
            case BodyType::STAR:         symbol = '@'; priority = 10; break;
            case BodyType::PLANET:       symbol = 'O'; priority = 8; break;
            case BodyType::MOON:         symbol = 'o'; priority = 6; break;
            case BodyType::ASTEROID:     symbol = '.'; priority = 2; break;
            case BodyType::KBO:          symbol = ','; priority = 1; break;
            case BodyType::RING_PARTICLE:symbol = '*'; priority = 3; break;
            case BodyType::COMET:        symbol = '!'; priority = 7; break;
        }
    }
};

// ============================================================================
// Celestial Data
// ============================================================================

namespace CelestialData {
    
    /**
     * @brief Planet data structure
     */
    struct PlanetData {
        const char* name;
        float semi_major_au;
        float mass_solar;
        int id;
    };
    
    /**
     * @brief Moon data structure
     */
    struct MoonData {
        const char* name;
        float distance_km;
        float mass_kg;
    };
    
    /**
     * @brief Asteroid data structure
     */
    struct AsteroidData {
        const char* name;
        float semi_major_au;
        float mass_kg;
    };
    
    /**
     * @brief Comet data structure
     */
    struct CometData {
        const char* name;
        float perihelion_au;
        float aphelion_au;
        float mass_kg;
    };
    
    // Solar system planets
    constexpr PlanetData PLANETS[] = {
        {"Mercury", 0.387f, 1.66e-7f, 0},
        {"Venus", 0.723f, 2.45e-6f, 1},
        {"Earth", 1.000f, 3.00e-6f, 2},
        {"Mars", 1.524f, 3.23e-7f, 3},
        {"Jupiter", 5.203f, 9.55e-4f, 4},
        {"Saturn", 9.537f, 2.86e-4f, 5},
        {"Uranus", 19.191f, 4.37e-5f, 6},
        {"Neptune", 30.069f, 5.15e-5f, 7}
    };
    
    // Earth's moon
    constexpr MoonData EARTH_MOON = {"Moon", 384400, 7.34e22};
    
    // Mars moons
    constexpr MoonData MARS_MOONS[] = {
        {"Phobos", 9377, 1.06e16},
        {"Deimos", 23460, 1.48e15}
    };
    
    // Jupiter's major moons
    constexpr MoonData JUPITER_MOONS[] = {
        {"Io", 421800, 8.93e22},
        {"Europa", 671100, 4.80e22},
        {"Ganymede", 1070400, 1.48e23},
        {"Callisto", 1882700, 1.08e23},
        {"Amalthea", 181400, 2.08e18},
        {"Himalia", 11460000, 6.70e18}
    };
    
    // Saturn's major moons
    constexpr MoonData SATURN_MOONS[] = {
        {"Titan", 1221865, 1.35e23},
        {"Rhea", 527068, 2.31e21},
        {"Iapetus", 3560854, 1.81e21},
        {"Dione", 377415, 1.10e21},
        {"Tethys", 294672, 6.18e20},
        {"Enceladus", 238040, 1.08e20},
        {"Mimas", 185540, 3.75e19}
    };
    
    // Saturn's rings (km)
    struct RingData {
        float inner_radius_km;
        float outer_radius_km;
        float density_factor;  // Relative density
    };
    
    constexpr RingData SATURN_RINGS[] = {
        {74500, 91980, 0.2f},   // C ring
        {91980, 117580, 0.4f},  // B ring
        {122000, 136780, 0.4f}  // A ring
    };
    
    // Uranus moons
    constexpr MoonData URANUS_MOONS[] = {
        {"Miranda", 129900, 6.59e19},
        {"Ariel", 190900, 1.35e21},
        {"Umbriel", 266000, 1.17e21},
        {"Titania", 436300, 3.53e21},
        {"Oberon", 583500, 3.01e21}
    };
    
    // Neptune moons
    constexpr MoonData NEPTUNE_MOONS[] = {
        {"Triton", 354800, 2.14e22},
        {"Nereid", 5513818, 3.10e19},
        {"Proteus", 117647, 4.40e19}
    };
    
    // Named asteroids
    constexpr AsteroidData NAMED_ASTEROIDS[] = {
        {"Ceres", 2.77f, 9.39e20f},
        {"Vesta", 2.36f, 2.59e20f},
        {"Pallas", 2.77f, 2.04e20f},
        {"Hygiea", 3.14f, 8.67e19f},
        {"Juno", 2.67f, 2.67e19f},
        {"Psyche", 2.92f, 2.27e19f},
        {"Davida", 3.17f, 3.84e19f},
        {"Iris", 2.39f, 1.36e19f},
        {"Eunomia", 2.64f, 3.12e19f},
        {"Eros", 1.46f, 6.69e15f},
        {"Itokawa", 1.32f, 3.51e10f},
        {"Bennu", 1.13f, 7.33e10f},
        {"Ryugu", 1.19f, 4.50e11f}
    };
    
    // Named Kuiper Belt Objects
    constexpr AsteroidData NAMED_KBOS[] = {
        {"Pluto", 39.48f, 1.31e22f},
        {"Eris", 67.78f, 1.66e22f},
        {"Makemake", 45.79f, 3.1e21f},
        {"Haumea", 43.34f, 4.01e21f},
        {"Gonggong", 67.38f, 1.75e21f},
        {"Quaoar", 43.69f, 1.4e21f},
        {"Sedna", 525.86f, 1e21f},
        {"Orcus", 39.42f, 6.32e20f},
        {"Salacia", 42.24f, 4.92e20f},
        {"Varuna", 42.92f, 3.7e20f}
    };
    
    // Famous comets
    constexpr CometData COMETS[] = {
        {"Halley", 0.586f, 35.08f, 2.2e14f},
        {"Hale-Bopp", 0.914f, 370.8f, 1e16f},
        {"Hyakutake", 0.230f, 3410.0f, 1e13f},
        {"ISON", 0.012f, 1000.0f, 1e12f},
        {"Encke", 0.33f, 4.1f, 1e13f},
        {"Swift-Tuttle", 0.96f, 51.23f, 1e16f},
        {"Tempel-Tuttle", 0.98f, 19.69f, 1e15f},
        {"NEOWISE", 0.306f, 715.0f, 1e14f}
    };
    
    // Alpha Centauri system data
    namespace AlphaCentauri {
        constexpr float A_MASS = 1.1f;        // Solar masses
        constexpr float B_MASS = 0.91f;       // Solar masses
        constexpr float PROXIMA_MASS = 0.12f; // Solar masses
        constexpr float SEPARATION = 23.0f;   // AU between A and B
        
        struct ExoplanetData {
            const char* name;
            float distance_au;
            float mass_solar;
        };
        
        constexpr ExoplanetData EXOPLANETS[] = {
            {"Proxima b", 0.05f, 3.0e-6f},  // Real exoplanet
            {"Aurora", 0.5f, 5e-6f},         // Fictional
            {"Pandora", 1.2f, 5e-6f},        // Avatar reference
            {"Polyphemus", 2.0f, 1e-3f},    // Gas giant (Pandora's host)
            {"Minerva", 3.5f, 5e-6f},
            {"Chiron", 5.0f, 1e-3f}
        };
    }
}

// ============================================================================
// Physics Engine
// ============================================================================

/**
 * @brief Physics engine for N-body gravitational simulation
 */
class PhysicsEngine {
private:
    const float softening_sq;
    
public:
    PhysicsEngine() : softening_sq(Config::Simulation::SOFTENING * 
                                   Config::Simulation::SOFTENING) {}
    
    /**
     * @brief Compute gravitational forces between all bodies
     */
    void compute_forces(std::vector<CelestialBody>& bodies) {
        // Clear forces
        #pragma omp parallel for
        for (size_t i = 0; i < bodies.size(); i++) {
            bodies[i].force = Vector2(0, 0);
        }
        
        // Direct N-body for major bodies (stars, planets, moons, named objects)
        for (size_t i = 0; i < bodies.size(); i++) {
            if (should_use_direct_nbody(bodies[i])) {
                compute_direct_forces(bodies, i);
            }
        }
        
        // Simplified central force for small bodies
        #pragma omp parallel for
        for (size_t i = 0; i < bodies.size(); i++) {
            if (!should_use_direct_nbody(bodies[i])) {
                compute_central_force(bodies, i);
            }
        }
    }
    
    /**
     * @brief Perform Velocity Verlet integration
     */
    void integrate(std::vector<CelestialBody>& bodies, float dt) {
        // Update positions
        #pragma omp parallel for
        for (size_t i = 0; i < bodies.size(); i++) {
            auto& b = bodies[i];
            if (b.mass > 0) {
                Vector2 acc = b.force / b.mass;
                b.position += b.velocity * dt + acc * (0.5f * dt * dt);
            }
        }
        
        // Save old forces
        std::vector<Vector2> old_forces(bodies.size());
        #pragma omp parallel for
        for (size_t i = 0; i < bodies.size(); i++) {
            old_forces[i] = bodies[i].force;
        }
        
        // Compute new forces
        compute_forces(bodies);
        
        // Update velocities
        #pragma omp parallel for
        for (size_t i = 0; i < bodies.size(); i++) {
            auto& b = bodies[i];
            if (b.mass > 0) {
                Vector2 avg_force = (old_forces[i] + b.force) * 0.5f;
                Vector2 acc = avg_force / b.mass;
                b.velocity += acc * dt;
            }
        }
    }
    
private:
    bool should_use_direct_nbody(const CelestialBody& body) {
        return body.type <= BodyType::MOON || !body.name.empty();
    }
    
    void compute_direct_forces(std::vector<CelestialBody>& bodies, size_t i) {
        for (size_t j = 0; j < bodies.size(); j++) {
            if (i == j) continue;
            
            // Only compute forces with significant bodies
            if (bodies[j].type > BodyType::MOON && bodies[j].mass < 1e-9) continue;
            
            // Don't compute inter-system forces except for stars
            if (bodies[i].system != bodies[j].system &&
                bodies[i].type != BodyType::STAR && 
                bodies[j].type != BodyType::STAR) continue;
            
            Vector2 delta = bodies[j].position - bodies[i].position;
            float dist_sq = delta.length_squared();
            dist_sq = std::max(dist_sq, softening_sq);
            
            float dist = std::sqrt(dist_sq);
            float force_mag = Config::Simulation::G * bodies[i].mass * 
                            bodies[j].mass / dist_sq;
            
            bodies[i].force += delta.normalized() * force_mag;
        }
    }
    
    void compute_central_force(std::vector<CelestialBody>& bodies, size_t i) {
        // Find the primary star in this system
        for (size_t j = 0; j < bodies.size(); j++) {
            if (bodies[j].type == BodyType::STAR && 
                bodies[j].system == bodies[i].system) {
                
                Vector2 to_star = bodies[j].position - bodies[i].position;
                float dist_sq = to_star.length_squared();
                
                if (dist_sq > softening_sq) {
                    float force_mag = Config::Simulation::G * bodies[i].mass * 
                                    bodies[j].mass / dist_sq;
                    bodies[i].force += to_star.normalized() * force_mag;
                }
                
                // For ring particles, also add force from parent planet
                if (bodies[i].type == BodyType::RING_PARTICLE) {
                    add_planet_force_to_ring_particle(bodies, i);
                }
                
                break;  // Found the star
            }
        }
    }
    
    void add_planet_force_to_ring_particle(std::vector<CelestialBody>& bodies, 
                                          size_t ring_idx) {
        // Find Saturn (hacky, but works for now)
        for (size_t j = 0; j < bodies.size(); j++) {
            if (bodies[j].name == "Saturn") {
                Vector2 to_planet = bodies[j].position - bodies[ring_idx].position;
                float dist_sq = to_planet.length_squared();
                
                if (dist_sq > 1e-8f) {
                    float force_mag = Config::Simulation::G * bodies[ring_idx].mass * 
                                    bodies[j].mass / dist_sq;
                    bodies[ring_idx].force += to_planet.normalized() * force_mag;
                }
                break;
            }
        }
    }
};

// ============================================================================
// System Builder
// ============================================================================

/**
 * @brief Builder class for constructing celestial systems
 */
class SystemBuilder {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniform;
    
public:
    SystemBuilder() : rng(Config::RANDOM_SEED), uniform(0.0f, 1.0f) {}
    
    /**
     * @brief Build the complete multi-star system
     */
    void build_complete_system(std::vector<CelestialBody>& bodies) {
        bodies.clear();
        bodies.reserve(100000);
        
        std::cout << "Building solar system...\n";
        build_sol_system(bodies);
        
        std::cout << "Building Alpha Centauri system...\n";
        build_alpha_centauri_system(bodies);
        
        std::cout << "Complete system built: " << bodies.size() << " bodies\n\n";
    }
    
private:
    void build_sol_system(std::vector<CelestialBody>& bodies) {
        // Add Sun
        bodies.emplace_back(Vector2(0, 0), Vector2(0, 0), 1.0f, 
                          BodyType::STAR, "Sol", StarSystem::SOL);
        
        // Add planets and their moons
        add_planets_and_moons(bodies);
        
        // Add named asteroids
        add_named_asteroids(bodies);
        
        // Add named KBOs
        add_named_kbos(bodies);
        
        // Add comets
        add_comets(bodies);
        
        // Add random asteroids
        add_random_asteroids(bodies, Config::Generation::RANDOM_ASTEROIDS);
        
        // Add random KBOs
        add_random_kbos(bodies, Config::Generation::RANDOM_KBOS);
    }
    
    void add_planets_and_moons(std::vector<CelestialBody>& bodies) {
        using namespace CelestialData;
        
        for (const auto& planet : PLANETS) {
            float v = std::sqrt(Config::Simulation::G / planet.semi_major_au);
            bodies.emplace_back(
                Vector2(planet.semi_major_au, 0),
                Vector2(0, v),
                planet.mass_solar,
                BodyType::PLANET,
                planet.name,
                StarSystem::SOL
            );
            
            // Add moons based on planet ID
            Vector2 planet_pos(planet.semi_major_au, 0);
            Vector2 planet_vel(0, v);
            
            switch(planet.id) {
                case 2: // Earth
                    add_moon(bodies, planet_pos, planet_vel, planet.mass_solar, 
                           EARTH_MOON);
                    break;
                case 3: // Mars
                    for (const auto& moon : MARS_MOONS) {
                        add_moon(bodies, planet_pos, planet_vel, planet.mass_solar, moon);
                    }
                    break;
                case 4: // Jupiter
                    for (const auto& moon : JUPITER_MOONS) {
                        add_moon(bodies, planet_pos, planet_vel, planet.mass_solar, moon);
                    }
                    break;
                case 5: // Saturn
                    for (const auto& moon : SATURN_MOONS) {
                        add_moon(bodies, planet_pos, planet_vel, planet.mass_solar, moon);
                    }
                    add_saturn_rings(bodies, planet_pos, planet_vel, planet.mass_solar);
                    break;
                case 6: // Uranus
                    for (const auto& moon : URANUS_MOONS) {
                        add_moon(bodies, planet_pos, planet_vel, planet.mass_solar, moon);
                    }
                    break;
                case 7: // Neptune
                    for (const auto& moon : NEPTUNE_MOONS) {
                        add_moon(bodies, planet_pos, planet_vel, planet.mass_solar, moon);
                    }
                    break;
            }
        }
    }
    
    void add_moon(std::vector<CelestialBody>& bodies,
                  const Vector2& planet_pos, const Vector2& planet_vel,
                  float planet_mass, const CelestialData::MoonData& moon) {
        float moon_dist = moon.distance_km / 1.496e11f;  // km to AU
        float moon_mass = moon.mass_kg / 1.989e30f;      // kg to solar masses
        float moon_v = std::sqrt(Config::Simulation::G * planet_mass / moon_dist);
        float angle = uniform(rng) * 2 * M_PI;
        
        bodies.emplace_back(
            planet_pos + Vector2(moon_dist * cos(angle), moon_dist * sin(angle)),
            planet_vel + Vector2(-moon_v * sin(angle), moon_v * cos(angle)),
            moon_mass,
            BodyType::MOON,
            moon.name,
            StarSystem::SOL
        );
    }
    
    void add_saturn_rings(std::vector<CelestialBody>& bodies,
                         const Vector2& saturn_pos, const Vector2& saturn_vel,
                         float saturn_mass) {
        using namespace CelestialData;
        
        for (int i = 0; i < Config::Generation::SATURN_RING_PARTICLES; i++) {
            // Choose ring based on density
            float ring_choice = uniform(rng);
            float cumulative = 0;
            float total_density = 0;
            
            for (const auto& ring : SATURN_RINGS) {
                total_density += ring.density_factor;
            }
            
            float r_km = 0;
            for (const auto& ring : SATURN_RINGS) {
                cumulative += ring.density_factor / total_density;
                if (ring_choice <= cumulative) {
                    r_km = ring.inner_radius_km + 
                          (ring.outer_radius_km - ring.inner_radius_km) * uniform(rng);
                    break;
                }
            }
            
            float r_au = r_km / 1.496e11f;
            float theta = uniform(rng) * 2 * M_PI;
            float v_ring = std::sqrt(Config::Simulation::G * saturn_mass / r_au);
            
            bodies.emplace_back(
                saturn_pos + Vector2(r_au * cos(theta), r_au * sin(theta)),
                saturn_vel + Vector2(-v_ring * sin(theta), v_ring * cos(theta)),
                Config::Generation::RING_PARTICLE_MASS,
                BodyType::RING_PARTICLE,
                "",
                StarSystem::SOL
            );
        }
    }
    
    void add_named_asteroids(std::vector<CelestialBody>& bodies) {
        for (const auto& ast : CelestialData::NAMED_ASTEROIDS) {
            float v = std::sqrt(Config::Simulation::G / ast.semi_major_au);
            float theta = uniform(rng) * 2 * M_PI;
            
            bodies.emplace_back(
                Vector2(ast.semi_major_au * cos(theta), ast.semi_major_au * sin(theta)),
                Vector2(-v * sin(theta), v * cos(theta)),
                ast.mass_kg / 1.989e30f,
                BodyType::ASTEROID,
                ast.name,
                StarSystem::SOL
            );
        }
    }
    
    void add_named_kbos(std::vector<CelestialBody>& bodies) {
        for (const auto& kbo : CelestialData::NAMED_KBOS) {
            float v = std::sqrt(Config::Simulation::G / kbo.semi_major_au);
            float theta = uniform(rng) * 2 * M_PI;
            
            bodies.emplace_back(
                Vector2(kbo.semi_major_au * cos(theta), kbo.semi_major_au * sin(theta)),
                Vector2(-v * sin(theta), v * cos(theta)),
                kbo.mass_kg / 1.989e30f,
                BodyType::KBO,
                kbo.name,
                StarSystem::SOL
            );
        }
    }
    
    void add_comets(std::vector<CelestialBody>& bodies) {
        for (const auto& comet : CelestialData::COMETS) {
            // Start at perihelion
            float r = comet.perihelion_au;
            float angle = uniform(rng) * 2 * M_PI;
            
            // Vis-viva equation for velocity
            float semi_major = (comet.perihelion_au + comet.aphelion_au) / 2.0f;
            float v_sq = Config::Simulation::G * (2.0f/r - 1.0f/semi_major);
            float v = std::sqrt(std::max(0.0f, v_sq));
            
            bodies.emplace_back(
                Vector2(r * cos(angle), r * sin(angle)),
                Vector2(-v * sin(angle), v * cos(angle)),
                comet.mass_kg / 1.989e30f,
                BodyType::COMET,
                comet.name,
                StarSystem::SOL
            );
        }
    }
    
    void add_random_asteroids(std::vector<CelestialBody>& bodies, int count) {
        for (int i = 0; i < count; i++) {
            float r = Config::Generation::ASTEROID_BELT_INNER + 
                     (Config::Generation::ASTEROID_BELT_OUTER - 
                      Config::Generation::ASTEROID_BELT_INNER) * uniform(rng);
            float theta = uniform(rng) * 2 * M_PI;
            float v = std::sqrt(Config::Simulation::G / r);
            
            bodies.emplace_back(
                Vector2(r * cos(theta), r * sin(theta)),
                Vector2(-v * sin(theta), v * cos(theta)),
                Config::Generation::DEFAULT_ASTEROID_MASS,
                BodyType::ASTEROID,
                "",
                StarSystem::SOL
            );
        }
    }
    
    void add_random_kbos(std::vector<CelestialBody>& bodies, int count) {
        for (int i = 0; i < count; i++) {
            float r = Config::Generation::KUIPER_BELT_INNER + 
                     (Config::Generation::KUIPER_BELT_OUTER - 
                      Config::Generation::KUIPER_BELT_INNER) * uniform(rng);
            float theta = uniform(rng) * 2 * M_PI;
            float v = std::sqrt(Config::Simulation::G / r);
            
            bodies.emplace_back(
                Vector2(r * cos(theta), r * sin(theta)),
                Vector2(-v * sin(theta), v * cos(theta)),
                Config::Generation::DEFAULT_KBO_MASS,
                BodyType::KBO,
                "",
                StarSystem::SOL
            );
        }
    }
    
    void build_alpha_centauri_system(std::vector<CelestialBody>& bodies) {
        using namespace CelestialData::AlphaCentauri;
        
        Vector2 center(Config::Generation::ALPHA_CEN_DISTANCE, 0);
        
        // Binary star positions and velocities
        float total_mass = A_MASS + B_MASS;
        float r_a = SEPARATION * B_MASS / total_mass;
        float r_b = SEPARATION * A_MASS / total_mass;
        float v_binary = std::sqrt(Config::Simulation::G * total_mass / SEPARATION) * 0.5f;
        
        // Alpha Centauri A
        bodies.emplace_back(
            center + Vector2(r_a, 0),
            Vector2(0, v_binary * B_MASS / total_mass),
            A_MASS,
            BodyType::STAR,
            "Alpha Cen A",
            StarSystem::ALPHA_CENTAURI
        );
        
        // Alpha Centauri B
        bodies.emplace_back(
            center + Vector2(-r_b, 0),
            Vector2(0, -v_binary * A_MASS / total_mass),
            B_MASS,
            BodyType::STAR,
            "Alpha Cen B",
            StarSystem::ALPHA_CENTAURI
        );
        
        // Proxima Centauri
        bodies.emplace_back(
            center + Vector2(0, -30),
            Vector2(0.1f, 0),
            PROXIMA_MASS,
            BodyType::STAR,
            "Proxima Cen",
            StarSystem::ALPHA_CENTAURI
        );
        
        // Add exoplanets
        for (const auto& planet : EXOPLANETS) {
            float parent_mass = (planet.distance_au < 0.1f) ? PROXIMA_MASS : A_MASS;
            Vector2 parent_pos = (planet.distance_au < 0.1f) ? 
                               center + Vector2(0, -30) : center + Vector2(r_a, 0);
            
            float v = std::sqrt(Config::Simulation::G * parent_mass / planet.distance_au);
            float angle = uniform(rng) * 2 * M_PI;
            
            bodies.emplace_back(
                parent_pos + Vector2(planet.distance_au * cos(angle), 
                                   planet.distance_au * sin(angle)),
                Vector2(-v * sin(angle), v * cos(angle)),
                planet.mass_solar,
                BodyType::PLANET,
                planet.name,
                StarSystem::ALPHA_CENTAURI
            );
        }
        
        // Add asteroid belt
        for (int i = 0; i < Config::Generation::ALPHA_CEN_ASTEROIDS; i++) {
            float r = 7.0f + 3.0f * uniform(rng);
            float theta = uniform(rng) * 2 * M_PI;
            float v = std::sqrt(Config::Simulation::G * total_mass / r);
            
            bodies.emplace_back(
                center + Vector2(r * cos(theta), r * sin(theta)),
                Vector2(-v * sin(theta), v * cos(theta)),
                Config::Generation::DEFAULT_ASTEROID_MASS,
                BodyType::ASTEROID,
                "",
                StarSystem::ALPHA_CENTAURI
            );
        }
    }
};

// ============================================================================
// Visualization Components
// ============================================================================

/**
 * @brief Non-blocking keyboard input handler
 */
class InputHandler {
private:
    struct termios old_tio;
    
public:
    InputHandler() {
        // Set terminal to non-blocking mode
        struct termios new_tio;
        tcgetattr(STDIN_FILENO, &old_tio);
        new_tio = old_tio;
        new_tio.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);
        
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    }
    
    ~InputHandler() {
        // Restore terminal settings
        tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
    }
    
    char get_key() {
        char c = 0;
        if (read(STDIN_FILENO, &c, 1) > 0) {
            return c;
        }
        return 0;
    }
};

/**
 * @brief Entity tracking system for following celestial bodies
 */
class EntityTracker {
private:
    std::vector<int> named_indices;
    int current_target;
    bool tracking_enabled;
    
public:
    EntityTracker() : current_target(-1), tracking_enabled(false) {}
    
    void initialize(const std::vector<CelestialBody>& bodies) {
        named_indices.clear();
        for (size_t i = 0; i < bodies.size(); i++) {
            if (!bodies[i].name.empty()) {
                named_indices.push_back(i);
            }
        }
    }
    
    void cycle_target() {
        if (named_indices.empty()) return;
        
        if (current_target < 0) {
            current_target = 0;
        } else {
            current_target = (current_target + 1) % named_indices.size();
        }
    }
    
    void enable_tracking() { tracking_enabled = true; }
    void disable_tracking() { tracking_enabled = false; }
    bool is_tracking() const { return tracking_enabled && current_target >= 0; }
    
    int get_target_index() const {
        if (current_target >= 0 && current_target < named_indices.size()) {
            return named_indices[current_target];
        }
        return -1;
    }
    
    std::string get_target_name(const std::vector<CelestialBody>& bodies) const {
        int idx = get_target_index();
        if (idx >= 0 && idx < bodies.size()) {
            return bodies[idx].name;
        }
        return "";
    }
};

/**
 * @brief ASCII visualization system
 */
class Visualizer {
private:
    std::vector<std::vector<char>> display;
    std::vector<std::pair<Vector2, char>> trails;
    Vector2 camera_pos;
    float zoom;
    
public:
    Visualizer() : zoom(Config::Display::DEFAULT_ZOOM) {
        display.resize(Config::Display::SCREEN_HEIGHT, 
                      std::vector<char>(Config::Display::SCREEN_WIDTH, ' '));
        camera_pos = {0, 0};
    }
    
    void update_camera(const Vector2& target_pos, bool smooth = true) {
        if (smooth) {
            camera_pos.x = camera_pos.x * 0.9f + target_pos.x * 0.1f;
            camera_pos.y = camera_pos.y * 0.9f + target_pos.y * 0.1f;
        } else {
            camera_pos = target_pos;
        }
    }
    
    void set_zoom(float new_zoom) {
        zoom = std::max(0.01f, std::min(100.0f, new_zoom));
    }
    
    void adjust_zoom(float factor) {
        set_zoom(zoom * factor);
    }
    
    void render(const std::vector<CelestialBody>& bodies, 
                const EntityTracker& tracker) {
        // Clear display
        for (auto& row : display) {
            std::fill(row.begin(), row.end(), ' ');
        }
        
        // Update trails
        trails.push_back({bodies[0].position, '*'});  // Sun trail
        if (trails.size() > Config::Display::TRAIL_LENGTH) {
            trails.erase(trails.begin());
        }
        
        // Draw trails
        for (const auto& [pos, symbol] : trails) {
            int sx = (pos.x - camera_pos.x) * zoom + Config::Display::SCREEN_WIDTH / 2;
            int sy = (pos.y - camera_pos.y) * zoom + Config::Display::SCREEN_HEIGHT / 2;
            
            if (sx >= 0 && sx < Config::Display::SCREEN_WIDTH && 
                sy >= 0 && sy < Config::Display::SCREEN_HEIGHT) {
                display[sy][sx] = ':';
            }
        }
        
        // Draw bodies
        for (const auto& body : bodies) {
            int sx = (body.position.x - camera_pos.x) * zoom + Config::Display::SCREEN_WIDTH / 2;
            int sy = (body.position.y - camera_pos.y) * zoom + Config::Display::SCREEN_HEIGHT / 2;
            
            if (sx >= 0 && sx < Config::Display::SCREEN_WIDTH && 
                sy >= 0 && sy < Config::Display::SCREEN_HEIGHT) {
                display[sy][sx] = body.symbol;
            }
        }
        
        // Draw to terminal
        std::cout << "\033[H";  // Home cursor
        
        // Header
        std::cout << "+" << std::string(Config::Display::SCREEN_WIDTH, '-') << "+\n";
        
        // Display grid
        for (const auto& row : display) {
            std::cout << "|";
            for (char c : row) {
                std::cout << c;
            }
            std::cout << "|\n";
        }
        
        // Footer
        std::cout << "+" << std::string(Config::Display::SCREEN_WIDTH, '-') << "+\n";
        
        // Status line
        if (tracker.is_tracking()) {
            std::cout << "Tracking: " << std::setw(20) << tracker.get_target_name(bodies);
        } else {
            std::cout << "Free Camera" << std::string(15, ' ');
        }
        std::cout << " | Zoom: " << std::fixed << std::setprecision(2) << zoom;
        std::cout << " | Bodies: " << bodies.size();
        std::cout << " | Pos: (" << std::fixed << std::setprecision(1) 
                  << camera_pos.x << ", " << camera_pos.y << ")";
        std::cout << std::string(20, ' ') << "\n";
    }
    
    const Vector2& get_camera_pos() const { return camera_pos; }
    float get_zoom() const { return zoom; }
};

// ============================================================================
// Simulation Class
// ============================================================================

/**
 * @brief Main simulation orchestrator
 */
class Simulation {
private:
    std::vector<CelestialBody> bodies;
    PhysicsEngine physics;
    SystemBuilder builder;
    EntityTracker tracker;
    Visualizer visualizer;
    InputHandler input;
    float time;
    float dt;
    bool paused;
    bool running;
    
public:
    Simulation() : time(0), dt(Config::Simulation::TIME_STEP), 
                   paused(false), running(true) {
        omp_set_num_threads(Config::Simulation::OMP_THREADS);
    }
    
    void initialize() {
        builder.build_complete_system(bodies);
        tracker.initialize(bodies);
    }
    
    void step() {
        if (!paused) {
            physics.integrate(bodies, dt);
            time += dt;
        }
    }
    
    void handle_input() {
        char key = input.get_key();
        
        switch (key) {
            case 'q':
            case 'Q':
                running = false;
                break;
            case ' ':
                paused = !paused;
                break;
            case 'w':
                visualizer.update_camera(
                    {visualizer.get_camera_pos().x, 
                     visualizer.get_camera_pos().y - 5.0f / visualizer.get_zoom()},
                    false);
                break;
            case 's':
                visualizer.update_camera(
                    {visualizer.get_camera_pos().x, 
                     visualizer.get_camera_pos().y + 5.0f / visualizer.get_zoom()},
                    false);
                break;
            case 'a':
                visualizer.update_camera(
                    {visualizer.get_camera_pos().x - 5.0f / visualizer.get_zoom(), 
                     visualizer.get_camera_pos().y},
                    false);
                break;
            case 'd':
                visualizer.update_camera(
                    {visualizer.get_camera_pos().x + 5.0f / visualizer.get_zoom(), 
                     visualizer.get_camera_pos().y},
                    false);
                break;
            case '+':
            case '=':
                visualizer.adjust_zoom(Config::Display::ZOOM_FACTOR);
                break;
            case '-':
            case '_':
                visualizer.adjust_zoom(1.0f / Config::Display::ZOOM_FACTOR);
                break;
            case 't':
            case 'T':
                tracker.cycle_target();
                break;
            case '3':
                tracker.enable_tracking();
                break;
            case '1':
                tracker.disable_tracking();
                break;
            case '0':
                visualizer.update_camera({0, 0}, false);
                visualizer.set_zoom(Config::Display::DEFAULT_ZOOM);
                break;
        }
    }
    
    void run_interactive() {
        std::cout << "\033[2J\033[H";  // Clear screen
        
        int frame = 0;
        auto last_frame = std::chrono::steady_clock::now();
        
        while (running) {
            // Handle input
            handle_input();
            
            // Update tracking camera
            if (tracker.is_tracking()) {
                int idx = tracker.get_target_index();
                if (idx >= 0 && idx < bodies.size()) {
                    visualizer.update_camera(bodies[idx].position, true);
                    visualizer.set_zoom(Config::Display::TRACKING_ZOOM);
                }
            }
            
            // Physics update
            step();
            
            // Render at target FPS
            frame++;
            if (frame % Config::Display::FRAME_SKIP == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_frame).count();
                
                if (elapsed < Config::Display::FRAME_TIME_MS) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(Config::Display::FRAME_TIME_MS - elapsed));
                }
                
                visualizer.render(bodies, tracker);
                last_frame = std::chrono::steady_clock::now();
            }
        }
        
        std::cout << "\033[2J\033[H";  // Clear screen on exit
    }
    
    const std::vector<CelestialBody>& get_bodies() const { return bodies; }
    float get_time() const { return time; }
    bool is_paused() const { return paused; }
    void toggle_pause() { paused = !paused; }
};

// ============================================================================
// Main Function
// ============================================================================

void print_usage() {
    std::cout << "\n=== Clean Solar System Simulation ===\n";
    std::cout << "Fully parameterized and refactored version\n\n";
    
    std::cout << "Usage: ./solar_system_clean [options]\n\n";
    
    std::cout << "Options:\n";
    std::cout << "  --interactive    Run interactive visualization (default)\n";
    std::cout << "  --benchmark      Run performance benchmark\n";
    std::cout << "  --test           Run basic test\n";
    std::cout << "  --help           Show this help\n\n";
    
    std::cout << "Interactive Controls:\n";
    std::cout << "  WASD    - Move camera\n";
    std::cout << "  +/-     - Zoom in/out\n";
    std::cout << "  T       - Cycle through named entities\n";
    std::cout << "  3       - Enable tracking mode\n";
    std::cout << "  1       - Disable tracking mode\n";
    std::cout << "  0       - Reset camera to origin\n";
    std::cout << "  Space   - Pause/unpause\n";
    std::cout << "  Q       - Quit\n\n";
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    bool interactive = true;
    bool benchmark = false;
    bool test = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--benchmark") {
            interactive = false;
            benchmark = true;
        } else if (arg == "--test") {
            interactive = false;
            test = true;
        } else if (arg == "--interactive") {
            interactive = true;
        }
    }
    
    std::cout << "\n=== Clean Solar System Simulation ===\n";
    std::cout << "Fully parameterized and refactored version\n\n";
    
    std::cout << "Configuration:\n";
    std::cout << "  Random asteroids: " << Config::Generation::RANDOM_ASTEROIDS << "\n";
    std::cout << "  Random KBOs: " << Config::Generation::RANDOM_KBOS << "\n";
    std::cout << "  Saturn ring particles: " << Config::Generation::SATURN_RING_PARTICLES << "\n";
    std::cout << "  Alpha Centauri distance: " << Config::Generation::ALPHA_CEN_DISTANCE << " AU\n";
    std::cout << "  OMP threads: " << Config::Simulation::OMP_THREADS << "\n\n";
    
    Simulation sim;
    
    std::cout << "Initializing simulation...\n";
    sim.initialize();
    
    std::cout << "Simulation initialized with " << sim.get_bodies().size() << " bodies\n\n";
    
    if (test) {
        std::cout << "Running basic test...\n";
        
        // Run for a short time
        for (int i = 0; i < 100; i++) {
            sim.step();
            if (i % 10 == 0) {
                std::cout << "  Step " << i << " - Time: " 
                          << std::fixed << std::setprecision(6) 
                          << sim.get_time() << " years\n";
            }
        }
        
        std::cout << "Test complete. Simulated " << sim.get_time() << " years\n";
        
    } else if (benchmark) {
        std::cout << "Running performance benchmark...\n";
        
        const int steps = 1000;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < steps; i++) {
            sim.step();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Benchmark results:\n";
        std::cout << "  Steps: " << steps << "\n";
        std::cout << "  Time: " << duration.count() << " ms\n";
        std::cout << "  Steps/second: " << (steps * 1000.0 / duration.count()) << "\n";
        std::cout << "  Bodies: " << sim.get_bodies().size() << "\n";
        std::cout << "  Updates/second: " 
                  << (sim.get_bodies().size() * steps * 1000.0 / duration.count()) << "\n";
        
    } else {
        std::cout << "Starting interactive visualization...\n";
        std::cout << "Press 'h' for help once running.\n\n";
        
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        sim.run_interactive();
        
        std::cout << "\nSimulation ended.\n";
        std::cout << "Final time: " << sim.get_time() << " years\n";
    }
    
    return 0;
}
