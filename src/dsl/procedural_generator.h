#pragma once

#include "sexpr.h"
#include "../physics/pools.h"
#include <random>
#include <functional>
#include <vector>
#include <cmath>

namespace digistar {

// Forward declaration
struct SimulationState;

namespace dsl {

/**
 * Procedural generation system for efficient mass particle creation
 * 
 * Features:
 * - Distribution functions (gaussian, power-law, uniform, etc.)
 * - Spatial patterns (grid, hexagonal, spiral, fractal)
 * - Template-based generation
 * - Batch creation optimization
 * - GPU-friendly data layout
 */

// Forward declarations
class Generator;
class Distribution;
class SpatialPattern;
class ParticleTemplate;

/**
 * Distribution types for particle properties
 */
enum class DistributionType {
    UNIFORM,           // Uniform distribution
    GAUSSIAN,          // Normal/Gaussian distribution
    POWER_LAW,         // Power law (e.g., Salpeter mass function)
    EXPONENTIAL,       // Exponential distribution
    POISSON,          // Poisson distribution
    CUSTOM            // Custom distribution function
};

/**
 * Spatial pattern types
 */
enum class PatternType {
    RANDOM,           // Random positions
    GRID,             // Regular grid
    HEXAGONAL,        // Hexagonal packing
    SPIRAL,           // Spiral pattern
    RADIAL,           // Radial distribution
    FRACTAL,          // Fractal patterns
    DISK,             // Disk/galaxy distribution
    SPHERE,           // Spherical distribution
    CUSTOM            // Custom pattern function
};

/**
 * Base distribution class
 */
class Distribution {
protected:
    std::mt19937& rng;
    
public:
    explicit Distribution(std::mt19937& gen) : rng(gen) {}
    virtual ~Distribution() = default;
    
    virtual float sample() = 0;
    virtual std::vector<float> sampleBatch(size_t count);
    virtual std::string toString() const = 0;
};

/**
 * Uniform distribution
 */
class UniformDistribution : public Distribution {
private:
    std::uniform_real_distribution<float> dist;
    float min_val, max_val;
    
public:
    UniformDistribution(std::mt19937& gen, float min, float max)
        : Distribution(gen), dist(min, max), min_val(min), max_val(max) {}
    
    float sample() override { return dist(rng); }
    std::string toString() const override;
};

/**
 * Gaussian/Normal distribution
 */
class GaussianDistribution : public Distribution {
private:
    std::normal_distribution<float> dist;
    float mean, stddev;
    
public:
    GaussianDistribution(std::mt19937& gen, float mu, float sigma)
        : Distribution(gen), dist(mu, sigma), mean(mu), stddev(sigma) {}
    
    float sample() override { return dist(rng); }
    std::string toString() const override;
};

/**
 * Power law distribution
 */
class PowerLawDistribution : public Distribution {
private:
    float min_val, max_val, exponent;
    std::uniform_real_distribution<float> uniform;
    
public:
    PowerLawDistribution(std::mt19937& gen, float min, float max, float exp)
        : Distribution(gen), min_val(min), max_val(max), exponent(exp),
          uniform(0.0f, 1.0f) {}
    
    float sample() override;
    std::string toString() const override;
};

/**
 * Salpeter Initial Mass Function (for stellar masses)
 */
class SalpeterIMF : public PowerLawDistribution {
public:
    SalpeterIMF(std::mt19937& gen, float min_mass = 0.1f, float max_mass = 100.0f)
        : PowerLawDistribution(gen, min_mass, max_mass, -2.35f) {}
    
    std::string toString() const override { return "SalpeterIMF"; }
};

/**
 * Spatial pattern generator
 */
class SpatialPattern {
protected:
    float center_x, center_y;
    float scale;
    std::mt19937& rng;
    
public:
    SpatialPattern(std::mt19937& gen, float cx, float cy, float s)
        : center_x(cx), center_y(cy), scale(s), rng(gen) {}
    virtual ~SpatialPattern() = default;
    
    struct Position {
        float x, y;
        Position() : x(0), y(0) {}  // Default constructor
        Position(float px, float py) : x(px), y(py) {}
    };
    
    virtual std::vector<Position> generate(size_t count) = 0;
    virtual std::string toString() const = 0;
};

/**
 * Random pattern
 */
class RandomPattern : public SpatialPattern {
private:
    std::uniform_real_distribution<float> dist;
    
public:
    RandomPattern(std::mt19937& gen, float cx, float cy, float radius)
        : SpatialPattern(gen, cx, cy, radius), dist(-1.0f, 1.0f) {}
    
    std::vector<Position> generate(size_t count) override;
    std::string toString() const override { return "Random"; }
};

/**
 * Grid pattern
 */
class GridPattern : public SpatialPattern {
private:
    size_t rows, cols;
    float spacing;
    
public:
    GridPattern(std::mt19937& gen, float cx, float cy, 
                size_t r, size_t c, float space)
        : SpatialPattern(gen, cx, cy, 1.0f),
          rows(r), cols(c), spacing(space) {}
    
    std::vector<Position> generate(size_t count) override;
    std::string toString() const override { return "Grid"; }
};

/**
 * Hexagonal packing pattern
 */
class HexagonalPattern : public SpatialPattern {
private:
    float spacing;
    
public:
    HexagonalPattern(std::mt19937& gen, float cx, float cy, float space)
        : SpatialPattern(gen, cx, cy, 1.0f), spacing(space) {}
    
    std::vector<Position> generate(size_t count) override;
    std::string toString() const override { return "Hexagonal"; }
};

/**
 * Spiral pattern (logarithmic spiral)
 */
class SpiralPattern : public SpatialPattern {
private:
    float tightness;  // How tightly wound the spiral is
    float arm_count;  // Number of spiral arms
    std::uniform_real_distribution<float> jitter;
    
public:
    SpiralPattern(std::mt19937& gen, float cx, float cy, float radius,
                  float tight = 0.3f, float arms = 2.0f)
        : SpatialPattern(gen, cx, cy, radius),
          tightness(tight), arm_count(arms), jitter(-0.1f, 0.1f) {}
    
    std::vector<Position> generate(size_t count) override;
    std::string toString() const override { return "Spiral"; }
};

/**
 * Disk/Galaxy pattern
 */
class DiskPattern : public SpatialPattern {
private:
    float inner_radius, outer_radius;
    float thickness;  // For 3D-like appearance
    std::uniform_real_distribution<float> uniform;
    std::normal_distribution<float> height_dist;
    
public:
    DiskPattern(std::mt19937& gen, float cx, float cy,
                float inner, float outer, float thick = 0.1f)
        : SpatialPattern(gen, cx, cy, 1.0f),
          inner_radius(inner), outer_radius(outer), thickness(thick),
          uniform(0.0f, 1.0f), height_dist(0.0f, thick) {}
    
    std::vector<Position> generate(size_t count) override;
    std::string toString() const override { return "Disk"; }
};

/**
 * Fractal pattern (using L-systems or IFS)
 */
class FractalPattern : public SpatialPattern {
public:
    enum FractalType {
        SIERPINSKI,
        DRAGON_CURVE,
        BARNSLEY_FERN,
        JULIA_SET
    };
    
private:
    FractalType type;
    int iterations;
    
public:
    FractalPattern(std::mt19937& gen, float cx, float cy, float scale,
                   FractalType t, int iter = 5)
        : SpatialPattern(gen, cx, cy, scale),
          type(t), iterations(iter) {}
    
    std::vector<Position> generate(size_t count) override;
    std::string toString() const override;
};

/**
 * Particle template for batch creation
 */
class ParticleTemplate {
public:
    // Property distributions
    std::shared_ptr<Distribution> mass_dist;
    std::shared_ptr<Distribution> radius_dist;
    std::shared_ptr<Distribution> temperature_dist;
    std::shared_ptr<Distribution> velocity_x_dist;
    std::shared_ptr<Distribution> velocity_y_dist;
    
    // Optional: orbital velocity calculation
    bool use_orbital_velocity = false;
    float central_mass = 0.0f;
    
    // Optional: color/type assignment
    std::function<uint32_t(float mass)> type_function;
    
    ParticleTemplate() = default;
    
    // Builder pattern for convenient construction
    ParticleTemplate& withMass(std::shared_ptr<Distribution> dist) {
        mass_dist = dist;
        return *this;
    }
    
    ParticleTemplate& withRadius(std::shared_ptr<Distribution> dist) {
        radius_dist = dist;
        return *this;
    }
    
    ParticleTemplate& withTemperature(std::shared_ptr<Distribution> dist) {
        temperature_dist = dist;
        return *this;
    }
    
    ParticleTemplate& withVelocity(std::shared_ptr<Distribution> vx,
                                   std::shared_ptr<Distribution> vy) {
        velocity_x_dist = vx;
        velocity_y_dist = vy;
        return *this;
    }
    
    ParticleTemplate& withOrbitalVelocity(float center_mass) {
        use_orbital_velocity = true;
        central_mass = center_mass;
        return *this;
    }
};

/**
 * Main procedural generator class
 */
class ProceduralGenerator {
private:
    SimulationState* sim_state;
    std::mt19937 rng;
    
    // Batch creation buffer for efficiency
    struct ParticleBatch {
        std::vector<float> x, y;
        std::vector<float> vx, vy;
        std::vector<float> mass, radius;
        std::vector<float> temperature;
        std::vector<uint32_t> type;
        
        void reserve(size_t count) {
            x.reserve(count);
            y.reserve(count);
            vx.reserve(count);
            vy.reserve(count);
            mass.reserve(count);
            radius.reserve(count);
            temperature.reserve(count);
            type.reserve(count);
        }
        
        void clear() {
            x.clear(); y.clear();
            vx.clear(); vy.clear();
            mass.clear(); radius.clear();
            temperature.clear(); type.clear();
        }
        
        size_t size() const { return x.size(); }
    };
    
    ParticleBatch batch_buffer;
    
    // Helper functions
    float calculateOrbitalVelocity(float distance, float central_mass) const;
    void applyRotation(float& x, float& y, float angle) const;
    
public:
    explicit ProceduralGenerator(SimulationState* state, unsigned int seed = 0);
    
    // Main generation functions
    std::vector<size_t> generate(const std::string& type,
                                 size_t count,
                                 const std::unordered_map<std::string, SExprPtr>& params);
    
    // Specific generators
    std::vector<size_t> generateCloud(size_t count,
                                      float cx, float cy, float radius,
                                      const ParticleTemplate& tmpl);
    
    std::vector<size_t> generateGalaxy(size_t count,
                                       float cx, float cy, float radius,
                                       size_t arm_count = 2,
                                       float rotation = 0.0f);
    
    std::vector<size_t> generateSolarSystem(float cx, float cy,
                                           float star_mass,
                                           const std::vector<std::pair<float, float>>& planets);
    
    std::vector<size_t> generateAsteroidBelt(size_t count,
                                            float cx, float cy,
                                            float inner_radius, float outer_radius,
                                            float central_mass);
    
    std::vector<size_t> generateCluster(size_t count,
                                       float cx, float cy, float radius,
                                       size_t subclusters = 5);
    
    std::vector<size_t> generateLattice(size_t width, size_t height,
                                       float cx, float cy, float spacing,
                                       const ParticleTemplate& tmpl);
    
    // Batch operations
    void beginBatch();
    void addToBatch(float x, float y, float vx, float vy,
                    float mass, float radius, float temp, uint32_t type = 0);
    std::vector<size_t> commitBatch();
    
    // Distribution factories
    std::shared_ptr<Distribution> createDistribution(const std::string& type,
                                                     const std::vector<float>& params);
    
    // Pattern factories
    std::shared_ptr<SpatialPattern> createPattern(const std::string& type,
                                                  float cx, float cy,
                                                  const std::vector<float>& params);
    
    // Template factories
    ParticleTemplate createTemplate(const std::unordered_map<std::string, SExprPtr>& params);
    
    // Utility functions
    void setSeed(unsigned int seed) { rng.seed(seed); }
    std::mt19937& getRNG() { return rng; }
};

/**
 * DSL integration for procedural generation
 */
class GeneratorDSL {
public:
    // Register generator functions in DSL environment
    static void registerFunctions(std::shared_ptr<Environment> env,
                                  ProceduralGenerator* generator);
    
    // Parse generation command from S-expression
    static std::vector<size_t> executeGeneration(SExprPtr expr,
                                                 ProceduralGenerator* generator);
    
    // Helper to extract parameters
    static std::unordered_map<std::string, float> extractParams(SExprPtr expr);
};

} // namespace dsl
} // namespace digistar