#include "procedural_generator.h"
#include "../backend/backend_interface.h"  // For SimulationState
#include <algorithm>
#include <cmath>
#include <sstream>

namespace digistar {
namespace dsl {

// Constants
constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float G = 6.67430e-11f; // Gravitational constant (SI units)

//=============================================================================
// Distribution implementations
//=============================================================================

std::vector<float> Distribution::sampleBatch(size_t count) {
    std::vector<float> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(sample());
    }
    return result;
}

std::string UniformDistribution::toString() const {
    std::stringstream ss;
    ss << "Uniform[" << min_val << "," << max_val << "]";
    return ss.str();
}

std::string GaussianDistribution::toString() const {
    std::stringstream ss;
    ss << "Gaussian(μ=" << mean << ",σ=" << stddev << ")";
    return ss.str();
}

float PowerLawDistribution::sample() {
    float u = uniform(rng);
    float norm = 1.0f / (1.0f + exponent);
    float min_term = std::pow(min_val, 1.0f + exponent);
    float max_term = std::pow(max_val, 1.0f + exponent);
    float value = std::pow(min_term + u * (max_term - min_term), norm);
    return value;
}

std::string PowerLawDistribution::toString() const {
    std::stringstream ss;
    ss << "PowerLaw(α=" << exponent << ")";
    return ss.str();
}

//=============================================================================
// Spatial pattern implementations
//=============================================================================

std::vector<SpatialPattern::Position> RandomPattern::generate(size_t count) {
    std::vector<Position> positions;
    positions.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        float r = scale * std::sqrt(dist(rng) * 0.5f + 0.5f); // Uniform disk
        float theta = TWO_PI * dist(rng);
        float x = center_x + r * std::cos(theta);
        float y = center_y + r * std::sin(theta);
        positions.emplace_back(x, y);
    }
    
    return positions;
}

std::vector<SpatialPattern::Position> GridPattern::generate(size_t count) {
    std::vector<Position> positions;
    positions.reserve(std::min(count, rows * cols));
    
    float start_x = center_x - (cols - 1) * spacing * 0.5f;
    float start_y = center_y - (rows - 1) * spacing * 0.5f;
    
    size_t generated = 0;
    for (size_t r = 0; r < rows && generated < count; ++r) {
        for (size_t c = 0; c < cols && generated < count; ++c) {
            float x = start_x + c * spacing;
            float y = start_y + r * spacing;
            positions.emplace_back(x, y);
            ++generated;
        }
    }
    
    return positions;
}

std::vector<SpatialPattern::Position> HexagonalPattern::generate(size_t count) {
    std::vector<Position> positions;
    positions.reserve(count);
    
    // Calculate dimensions for hexagonal grid
    size_t side = static_cast<size_t>(std::sqrt(count * 2.0f / std::sqrt(3.0f)));
    float dx = spacing;
    float dy = spacing * std::sqrt(3.0f) / 2.0f;
    
    size_t generated = 0;
    for (size_t row = 0; row < side * 2 && generated < count; ++row) {
        size_t cols_in_row = (row % 2 == 0) ? side : side - 1;
        float x_offset = (row % 2 == 0) ? 0 : dx * 0.5f;
        
        for (size_t col = 0; col < cols_in_row && generated < count; ++col) {
            float x = center_x + x_offset + col * dx - (side - 1) * dx * 0.5f;
            float y = center_y + row * dy - (side - 1) * dy;
            positions.emplace_back(x, y);
            ++generated;
        }
    }
    
    return positions;
}

std::vector<SpatialPattern::Position> SpiralPattern::generate(size_t count) {
    std::vector<Position> positions;
    positions.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(count);
        
        // Logarithmic spiral with multiple arms
        float angle = TWO_PI * t * 10.0f; // Number of rotations
        float r = scale * std::exp(tightness * angle / TWO_PI) * t;
        
        // Add arm offset
        float arm = std::floor(jitter(rng) * arm_count + arm_count) / arm_count;
        angle += TWO_PI * arm / arm_count;
        
        // Add some jitter for natural look
        r *= (1.0f + jitter(rng) * 0.2f);
        angle += jitter(rng) * 0.1f;
        
        float x = center_x + r * std::cos(angle);
        float y = center_y + r * std::sin(angle);
        positions.emplace_back(x, y);
    }
    
    return positions;
}

std::vector<SpatialPattern::Position> DiskPattern::generate(size_t count) {
    std::vector<Position> positions;
    positions.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        // Generate radius with higher density toward center
        float u = uniform(rng);
        float r = inner_radius + (outer_radius - inner_radius) * std::sqrt(u);
        
        // Random angle
        float theta = TWO_PI * uniform(rng);
        
        // Add thickness variation (simulating 3D disk)
        float height_offset = height_dist(rng);
        
        float x = center_x + r * std::cos(theta);
        float y = center_y + r * std::sin(theta) + height_offset;
        positions.emplace_back(x, y);
    }
    
    return positions;
}

std::vector<SpatialPattern::Position> FractalPattern::generate(size_t count) {
    std::vector<Position> positions;
    positions.reserve(count);
    
    switch (type) {
        case SIERPINSKI: {
            // Sierpinski triangle using chaos game
            float vertices[3][2] = {
                {center_x, center_y + scale},
                {center_x - scale * 0.866f, center_y - scale * 0.5f},
                {center_x + scale * 0.866f, center_y - scale * 0.5f}
            };
            
            float x = center_x, y = center_y;
            std::uniform_int_distribution<int> vertex_dist(0, 2);
            
            for (size_t i = 0; i < count; ++i) {
                int v = vertex_dist(rng);
                x = (x + vertices[v][0]) * 0.5f;
                y = (y + vertices[v][1]) * 0.5f;
                
                if (i >= 100) { // Skip first iterations for convergence
                    positions.emplace_back(x, y);
                }
            }
            break;
        }
        
        case BARNSLEY_FERN: {
            // Barnsley fern using iterated function system
            float x = 0, y = 0;
            std::uniform_real_distribution<float> prob(0, 1);
            
            for (size_t i = 0; i < count + 100; ++i) {
                float p = prob(rng);
                float nx, ny;
                
                if (p < 0.01f) {
                    nx = 0;
                    ny = 0.16f * y;
                } else if (p < 0.86f) {
                    nx = 0.85f * x + 0.04f * y;
                    ny = -0.04f * x + 0.85f * y + 1.6f;
                } else if (p < 0.93f) {
                    nx = 0.2f * x - 0.26f * y;
                    ny = 0.23f * x + 0.22f * y + 1.6f;
                } else {
                    nx = -0.15f * x + 0.28f * y;
                    ny = 0.26f * x + 0.24f * y + 0.44f;
                }
                
                x = nx;
                y = ny;
                
                if (i >= 100) { // Skip first iterations
                    positions.emplace_back(
                        center_x + x * scale * 10.0f,
                        center_y + y * scale * 10.0f
                    );
                }
            }
            break;
        }
        
        default:
            // Fall back to random for unimplemented types
            return RandomPattern(rng, center_x, center_y, scale).generate(count);
    }
    
    // Ensure we have exactly count positions
    while (positions.size() < count) {
        positions.push_back(positions[positions.size() % std::max(size_t(1), positions.size())]);
    }
    positions.resize(count);
    
    return positions;
}

std::string FractalPattern::toString() const {
    switch (type) {
        case SIERPINSKI: return "Sierpinski";
        case DRAGON_CURVE: return "DragonCurve";
        case BARNSLEY_FERN: return "BarnsleyFern";
        case JULIA_SET: return "JuliaSet";
        default: return "Fractal";
    }
}

//=============================================================================
// ProceduralGenerator implementation
//=============================================================================

ProceduralGenerator::ProceduralGenerator(SimulationState* state, unsigned int seed)
    : sim_state(state), rng(seed == 0 ? std::random_device{}() : seed) {
    batch_buffer.reserve(10000); // Pre-allocate for efficiency
}

float ProceduralGenerator::calculateOrbitalVelocity(float distance, float central_mass) const {
    // v = sqrt(G * M / r)
    // Using simplified units where G = 1
    return std::sqrt(central_mass / distance);
}

void ProceduralGenerator::applyRotation(float& x, float& y, float angle) const {
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);
    float nx = x * cos_a - y * sin_a;
    float ny = x * sin_a + y * cos_a;
    x = nx;
    y = ny;
}

std::vector<size_t> ProceduralGenerator::generate(
    const std::string& type,
    size_t count,
    const std::unordered_map<std::string, SExprPtr>& params) {
    
    // Extract common parameters
    float cx = 0, cy = 0, radius = 100;
    
    auto it = params.find("center");
    if (it != params.end() && it->second->isVector()) {
        const auto& v = it->second->asVector();
        if (v.size() >= 2) {
            cx = v[0];
            cy = v[1];
        }
    }
    
    it = params.find("radius");
    if (it != params.end() && it->second->isNumber()) {
        radius = it->second->asNumber();
    }
    
    // Dispatch to specific generator
    if (type == "cloud") {
        ParticleTemplate tmpl = createTemplate(params);
        return generateCloud(count, cx, cy, radius, tmpl);
    } else if (type == "galaxy") {
        size_t arms = 2;
        it = params.find("arms");
        if (it != params.end() && it->second->isNumber()) {
            arms = static_cast<size_t>(it->second->asNumber());
        }
        return generateGalaxy(count, cx, cy, radius, arms);
    } else if (type == "cluster") {
        size_t subclusters = 5;
        it = params.find("subclusters");
        if (it != params.end() && it->second->isNumber()) {
            subclusters = static_cast<size_t>(it->second->asNumber());
        }
        return generateCluster(count, cx, cy, radius, subclusters);
    }
    
    // Default: random cloud
    ParticleTemplate tmpl = createTemplate(params);
    return generateCloud(count, cx, cy, radius, tmpl);
}

std::vector<size_t> ProceduralGenerator::generateCloud(
    size_t count,
    float cx, float cy, float radius,
    const ParticleTemplate& tmpl) {
    
    beginBatch();
    
    // Generate positions
    RandomPattern pattern(rng, cx, cy, radius);
    auto positions = pattern.generate(count);
    
    // Generate properties for each particle
    for (const auto& pos : positions) {
        float mass = tmpl.mass_dist ? tmpl.mass_dist->sample() : 1.0f;
        float r = tmpl.radius_dist ? tmpl.radius_dist->sample() : 1.0f;
        float temp = tmpl.temperature_dist ? tmpl.temperature_dist->sample() : 300.0f;
        
        float vx = 0, vy = 0;
        if (tmpl.use_orbital_velocity) {
            float dist = std::sqrt((pos.x - cx) * (pos.x - cx) + 
                                  (pos.y - cy) * (pos.y - cy));
            if (dist > 0.01f) {
                float v_orbital = calculateOrbitalVelocity(dist, tmpl.central_mass);
                float angle = std::atan2(pos.y - cy, pos.x - cx) + PI/2;
                vx = v_orbital * std::cos(angle);
                vy = v_orbital * std::sin(angle);
            }
        } else {
            vx = tmpl.velocity_x_dist ? tmpl.velocity_x_dist->sample() : 0.0f;
            vy = tmpl.velocity_y_dist ? tmpl.velocity_y_dist->sample() : 0.0f;
        }
        
        uint32_t type = 0;
        if (tmpl.type_function) {
            type = tmpl.type_function(mass);
        }
        
        addToBatch(pos.x, pos.y, vx, vy, mass, r, temp, type);
    }
    
    return commitBatch();
}

std::vector<size_t> ProceduralGenerator::generateGalaxy(
    size_t count,
    float cx, float cy, float radius,
    size_t arm_count,
    float rotation) {
    
    beginBatch();
    
    // Create distributions for galaxy
    SalpeterIMF mass_dist(rng, 0.1f, 50.0f);
    UniformDistribution radius_dist(rng, 0.5f, 2.0f);
    GaussianDistribution temp_dist(rng, 5000.0f, 2000.0f);
    
    // Generate spiral pattern
    SpiralPattern pattern(rng, cx, cy, radius, 0.3f, static_cast<float>(arm_count));
    auto positions = pattern.generate(count);
    
    // Central black hole
    addToBatch(cx, cy, 0, 0, 1000000.0f, 10.0f, 0.0f, 1); // Type 1 = black hole
    
    // Generate stars
    for (const auto& pos : positions) {
        float mass = mass_dist.sample();
        float r = radius_dist.sample() * std::pow(mass, 0.8f); // Mass-radius relation
        float temp = temp_dist.sample() * std::pow(mass, 0.5f); // Mass-luminosity relation
        
        // Orbital velocity
        float dist = std::sqrt((pos.x - cx) * (pos.x - cx) + 
                              (pos.y - cy) * (pos.y - cy));
        float v_orbital = calculateOrbitalVelocity(dist, 1000000.0f);
        
        // Apply rotation curve (flat for galaxy)
        v_orbital *= std::min(1.0f, dist / (radius * 0.2f));
        
        float angle = std::atan2(pos.y - cy, pos.x - cx) + PI/2 + rotation;
        float vx = v_orbital * std::cos(angle);
        float vy = v_orbital * std::sin(angle);
        
        // Determine star type based on mass
        uint32_t type = 2; // Normal star
        if (mass > 20.0f) type = 3; // Blue giant
        else if (mass < 0.5f) type = 4; // Red dwarf
        
        addToBatch(pos.x, pos.y, vx, vy, mass, r, temp, type);
    }
    
    return commitBatch();
}

std::vector<size_t> ProceduralGenerator::generateSolarSystem(
    float cx, float cy,
    float star_mass,
    const std::vector<std::pair<float, float>>& planets) {
    
    beginBatch();
    
    // Add star
    addToBatch(cx, cy, 0, 0, star_mass, star_mass * 0.01f, 5778.0f, 3); // Sun-like star
    
    // Add planets
    for (const auto& [distance, mass] : planets) {
        // Random angle for initial position
        std::uniform_real_distribution<float> angle_dist(0, TWO_PI);
        float angle = angle_dist(rng);
        
        float x = cx + distance * std::cos(angle);
        float y = cy + distance * std::sin(angle);
        
        // Calculate orbital velocity
        float v_orbital = calculateOrbitalVelocity(distance, star_mass);
        float vx = -v_orbital * std::sin(angle);
        float vy = v_orbital * std::cos(angle);
        
        // Planet properties
        float radius = std::pow(mass, 0.5f) * 0.1f;
        float temp = 280.0f * std::pow(distance / 100.0f, -0.5f); // Temperature falls with distance
        
        uint32_t type = 5; // Rocky planet
        if (mass > 50.0f) type = 6; // Gas giant
        
        addToBatch(x, y, vx, vy, mass, radius, temp, type);
    }
    
    return commitBatch();
}

std::vector<size_t> ProceduralGenerator::generateAsteroidBelt(
    size_t count,
    float cx, float cy,
    float inner_radius, float outer_radius,
    float central_mass) {
    
    beginBatch();
    
    // Create distributions
    PowerLawDistribution mass_dist(rng, 0.001f, 0.1f, -2.5f);
    DiskPattern pattern(rng, cx, cy, inner_radius, outer_radius, 
                       (outer_radius - inner_radius) * 0.05f);
    
    auto positions = pattern.generate(count);
    
    for (const auto& pos : positions) {
        float mass = mass_dist.sample();
        float radius = std::pow(mass, 0.4f) * 0.5f;
        
        // Orbital velocity with some eccentricity
        float dist = std::sqrt((pos.x - cx) * (pos.x - cx) + 
                              (pos.y - cy) * (pos.y - cy));
        float v_orbital = calculateOrbitalVelocity(dist, central_mass);
        
        // Add some random eccentricity
        std::normal_distribution<float> ecc_dist(1.0f, 0.1f);
        v_orbital *= ecc_dist(rng);
        
        float angle = std::atan2(pos.y - cy, pos.x - cx) + PI/2;
        float vx = v_orbital * std::cos(angle);
        float vy = v_orbital * std::sin(angle);
        
        addToBatch(pos.x, pos.y, vx, vy, mass, radius, 200.0f, 7); // Type 7 = asteroid
    }
    
    return commitBatch();
}

std::vector<size_t> ProceduralGenerator::generateCluster(
    size_t count,
    float cx, float cy, float radius,
    size_t subclusters) {
    
    beginBatch();
    
    // Generate subcluster centers
    RandomPattern center_pattern(rng, cx, cy, radius * 0.7f);
    auto centers = center_pattern.generate(subclusters);
    
    // Particles per subcluster
    size_t per_cluster = count / subclusters;
    size_t remainder = count % subclusters;
    
    // Mass distribution
    GaussianDistribution mass_dist(rng, 1.0f, 0.3f);
    GaussianDistribution vel_dist(rng, 0.0f, 5.0f);
    
    for (size_t i = 0; i < subclusters; ++i) {
        size_t cluster_count = per_cluster + (i < remainder ? 1 : 0);
        float cluster_radius = radius * 0.3f;
        
        // Generate particles for this subcluster
        RandomPattern pattern(rng, centers[i].x, centers[i].y, cluster_radius);
        auto positions = pattern.generate(cluster_count);
        
        for (const auto& pos : positions) {
            float mass = std::abs(mass_dist.sample());
            float r = mass * 0.5f;
            
            // Velocity relative to subcluster center
            float vx = vel_dist.sample();
            float vy = vel_dist.sample();
            
            // Add bulk motion of subcluster
            float cluster_angle = std::atan2(centers[i].y - cy, centers[i].x - cx);
            float cluster_speed = 10.0f;
            vx += cluster_speed * std::cos(cluster_angle + PI/2);
            vy += cluster_speed * std::sin(cluster_angle + PI/2);
            
            addToBatch(pos.x, pos.y, vx, vy, mass, r, 1000.0f, 0);
        }
    }
    
    return commitBatch();
}

std::vector<size_t> ProceduralGenerator::generateLattice(
    size_t width, size_t height,
    float cx, float cy, float spacing,
    const ParticleTemplate& tmpl) {
    
    beginBatch();
    
    GridPattern pattern(rng, cx, cy, height, width, spacing);
    auto positions = pattern.generate(width * height);
    
    for (const auto& pos : positions) {
        float mass = tmpl.mass_dist ? tmpl.mass_dist->sample() : 1.0f;
        float r = tmpl.radius_dist ? tmpl.radius_dist->sample() : 0.5f;
        float temp = tmpl.temperature_dist ? tmpl.temperature_dist->sample() : 300.0f;
        float vx = tmpl.velocity_x_dist ? tmpl.velocity_x_dist->sample() : 0.0f;
        float vy = tmpl.velocity_y_dist ? tmpl.velocity_y_dist->sample() : 0.0f;
        
        addToBatch(pos.x, pos.y, vx, vy, mass, r, temp, 0);
    }
    
    return commitBatch();
}

void ProceduralGenerator::beginBatch() {
    batch_buffer.clear();
}

void ProceduralGenerator::addToBatch(float x, float y, float vx, float vy,
                                     float mass, float radius, float temp, uint32_t type) {
    batch_buffer.x.push_back(x);
    batch_buffer.y.push_back(y);
    batch_buffer.vx.push_back(vx);
    batch_buffer.vy.push_back(vy);
    batch_buffer.mass.push_back(mass);
    batch_buffer.radius.push_back(radius);
    batch_buffer.temperature.push_back(temp);
    batch_buffer.type.push_back(type);
}

std::vector<size_t> ProceduralGenerator::commitBatch() {
    if (!sim_state || batch_buffer.size() == 0) {
        return {};
    }
    
    std::vector<size_t> indices;
    indices.reserve(batch_buffer.size());
    
    // Get current particle pools
    auto& pools = sim_state->particles;
    size_t start_idx = pools.count;
    
    // Allocate space for new particles
    size_t new_count = start_idx + batch_buffer.size();
    if (new_count > pools.capacity) {
        // Need to resize pools - this should be handled by pool manager
        return indices; // Return empty for now
    }
    
    // Copy batch data to pools
    for (size_t i = 0; i < batch_buffer.size(); ++i) {
        size_t idx = start_idx + i;
        
        pools.pos_x[idx] = batch_buffer.x[i];
        pools.pos_y[idx] = batch_buffer.y[i];
        pools.vel_x[idx] = batch_buffer.vx[i];
        pools.vel_y[idx] = batch_buffer.vy[i];
        pools.mass[idx] = batch_buffer.mass[i];
        pools.radius[idx] = batch_buffer.radius[i];
        pools.temperature[idx] = batch_buffer.temperature[i];
        pools.material_type[idx] = batch_buffer.type[i];
        pools.alive[idx] = true;
        
        // Initialize other properties
        pools.force_x[idx] = 0.0f;
        pools.force_y[idx] = 0.0f;
        pools.charge[idx] = 0.0f;
        
        indices.push_back(idx);
    }
    
    // Update particle count
    pools.count = new_count;
    sim_state->stats.active_particles = new_count;
    
    // Clear batch buffer
    batch_buffer.clear();
    
    return indices;
}

std::shared_ptr<Distribution> ProceduralGenerator::createDistribution(
    const std::string& type,
    const std::vector<float>& params) {
    
    if (type == "uniform" && params.size() >= 2) {
        return std::make_shared<UniformDistribution>(rng, params[0], params[1]);
    } else if (type == "gaussian" && params.size() >= 2) {
        return std::make_shared<GaussianDistribution>(rng, params[0], params[1]);
    } else if (type == "power_law" && params.size() >= 3) {
        return std::make_shared<PowerLawDistribution>(rng, params[0], params[1], params[2]);
    } else if (type == "salpeter" && params.size() >= 2) {
        return std::make_shared<SalpeterIMF>(rng, params[0], params[1]);
    }
    
    // Default: uniform [0, 1]
    return std::make_shared<UniformDistribution>(rng, 0.0f, 1.0f);
}

std::shared_ptr<SpatialPattern> ProceduralGenerator::createPattern(
    const std::string& type,
    float cx, float cy,
    const std::vector<float>& params) {
    
    if (type == "random" && params.size() >= 1) {
        return std::make_shared<RandomPattern>(rng, cx, cy, params[0]);
    } else if (type == "grid" && params.size() >= 3) {
        return std::make_shared<GridPattern>(rng, cx, cy,
            static_cast<size_t>(params[0]), static_cast<size_t>(params[1]), params[2]);
    } else if (type == "hexagonal" && params.size() >= 1) {
        return std::make_shared<HexagonalPattern>(rng, cx, cy, params[0]);
    } else if (type == "spiral" && params.size() >= 1) {
        float tight = params.size() > 1 ? params[1] : 0.3f;
        float arms = params.size() > 2 ? params[2] : 2.0f;
        return std::make_shared<SpiralPattern>(rng, cx, cy, params[0], tight, arms);
    } else if (type == "disk" && params.size() >= 2) {
        float thick = params.size() > 2 ? params[2] : 0.1f;
        return std::make_shared<DiskPattern>(rng, cx, cy, params[0], params[1], thick);
    }
    
    // Default: random
    float radius = params.empty() ? 100.0f : params[0];
    return std::make_shared<RandomPattern>(rng, cx, cy, radius);
}

ParticleTemplate ProceduralGenerator::createTemplate(
    const std::unordered_map<std::string, SExprPtr>& params) {
    
    ParticleTemplate tmpl;
    
    // Mass distribution
    auto it = params.find("mass");
    if (it != params.end()) {
        if (it->second->isVector()) {
            const auto& v = it->second->asVector();
            if (v.size() >= 2) {
                tmpl.mass_dist = std::make_shared<UniformDistribution>(rng, v[0], v[1]);
            }
        } else if (it->second->isNumber()) {
            float mass = it->second->asNumber();
            tmpl.mass_dist = std::make_shared<UniformDistribution>(rng, mass, mass);
        }
    }
    
    // Radius distribution
    it = params.find("radius");
    if (it != params.end()) {
        if (it->second->isVector()) {
            const auto& v = it->second->asVector();
            if (v.size() >= 2) {
                tmpl.radius_dist = std::make_shared<UniformDistribution>(rng, v[0], v[1]);
            }
        } else if (it->second->isNumber()) {
            float r = it->second->asNumber();
            tmpl.radius_dist = std::make_shared<UniformDistribution>(rng, r, r);
        }
    }
    
    // Temperature
    it = params.find("temperature");
    if (it != params.end() && it->second->isNumber()) {
        float temp = it->second->asNumber();
        tmpl.temperature_dist = std::make_shared<UniformDistribution>(rng, temp, temp);
    }
    
    // Velocity
    it = params.find("velocity");
    if (it != params.end() && it->second->isVector()) {
        const auto& v = it->second->asVector();
        if (v.size() >= 2) {
            tmpl.velocity_x_dist = std::make_shared<UniformDistribution>(rng, -v[0], v[0]);
            tmpl.velocity_y_dist = std::make_shared<UniformDistribution>(rng, -v[1], v[1]);
        }
    }
    
    // Set defaults if not specified
    if (!tmpl.mass_dist) {
        tmpl.mass_dist = std::make_shared<UniformDistribution>(rng, 0.5f, 2.0f);
    }
    if (!tmpl.radius_dist) {
        tmpl.radius_dist = std::make_shared<UniformDistribution>(rng, 0.5f, 1.5f);
    }
    if (!tmpl.temperature_dist) {
        tmpl.temperature_dist = std::make_shared<UniformDistribution>(rng, 200.0f, 400.0f);
    }
    if (!tmpl.velocity_x_dist) {
        tmpl.velocity_x_dist = std::make_shared<UniformDistribution>(rng, -1.0f, 1.0f);
    }
    if (!tmpl.velocity_y_dist) {
        tmpl.velocity_y_dist = std::make_shared<UniformDistribution>(rng, -1.0f, 1.0f);
    }
    
    return tmpl;
}

} // namespace dsl
} // namespace digistar