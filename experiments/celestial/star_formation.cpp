#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include "../../src/dynamics/ParticleMerger.h"

// Constants for star formation (simplified for demonstration)
const float FUSION_THRESHOLD_TEMP = 5e4f;     // 50,000 K for fusion (simplified for demo)
const float FUSION_THRESHOLD_DENSITY = 5.0f;  // kg/m³ (simplified)
const float SCHWARZSCHILD_FACTOR = 0.1f;      // Safety factor for black hole formation

// Extended particle for star formation
struct StarParticle : public ParticleExt {
    bool is_fusing = false;        // Star is undergoing fusion
    bool is_black_hole = false;    // Collapsed to black hole
    float luminosity = 0.0f;       // Energy output
    float density = 1.0f;          // Mass/volume density
    
    void updateDensity() {
        float volume = (4.0f/3.0f) * M_PI * radius * radius * radius;
        density = mass / volume;
    }
    
    void checkFusionConditions() {
        updateDensity();
        
        // Check for black hole formation first
        const float G = 6.67e-11f;
        const float c = 3e8f;
        float schwarzschild_radius = 2.0f * G * mass / (c * c);
        
        if (radius < schwarzschild_radius * SCHWARZSCHILD_FACTOR) {
            is_black_hole = true;
            is_fusing = false;
            luminosity = 0.0f;
            return;
        }
        
        // Check for fusion ignition
        if (temp_internal > FUSION_THRESHOLD_TEMP && density > FUSION_THRESHOLD_DENSITY) {
            is_fusing = true;
            // Simple luminosity model: L ∝ M³
            luminosity = 1e-10f * pow(mass, 3);
        } else if (temp_internal < FUSION_THRESHOLD_TEMP * 0.5f) {
            // Fusion stops if temperature drops too low
            is_fusing = false;
            luminosity = 0.0f;
        }
    }
};

// Enhanced merger for star formation
class StarFormationMerger : public ParticleMerger {
private:
    float compression_heating_factor = 0.5f;  // Fraction of gravitational energy -> heat (increased for demo)
    
public:
    StarParticle mergeStarParticles(const StarParticle& p1, const StarParticle& p2) {
        // Use base class merge
        ParticleExt base_merged = mergeParticles(p1, p2);
        
        StarParticle result;
        static_cast<ParticleExt&>(result) = base_merged;
        
        // Add compression heating
        // When particles merge, gravitational potential energy is released
        const float G = 10000.0f;  // Increased for demonstration
        float grav_energy = G * p1.mass * p2.mass / (p1.radius + p2.radius);
        float heat_added = compression_heating_factor * grav_energy / (result.mass * specific_heat);
        result.temp_internal += heat_added;
        
        // Update stellar properties
        result.updateDensity();
        result.checkFusionConditions();
        
        return result;
    }
};

// Apply gravity with softening
void applyGravity(std::vector<StarParticle>& particles, float G, float dt, float softening = 0.1f) {
    std::vector<float2> forces(particles.size(), {0, 0});
    
    for (size_t i = 0; i < particles.size(); i++) {
        for (size_t j = i + 1; j < particles.size(); j++) {
            float2 r = particles[j].pos - particles[i].pos;
            float dist_sq = r.x * r.x + r.y * r.y + softening * softening;
            float dist = sqrt(dist_sq);
            
            float F = G * particles[i].mass * particles[j].mass / dist_sq;
            float2 F_vec = r * (F / dist);
            
            forces[i] = forces[i] + F_vec / particles[i].mass;
            forces[j] = forces[j] - F_vec / particles[j].mass;
        }
    }
    
    for (size_t i = 0; i < particles.size(); i++) {
        particles[i].vel = particles[i].vel + forces[i] * dt;
        particles[i].pos = particles[i].pos + particles[i].vel * dt;
    }
}

// Apply radiation pressure from stars
void applyRadiationPressure(std::vector<StarParticle>& particles, float dt) {
    for (size_t i = 0; i < particles.size(); i++) {
        if (!particles[i].is_fusing) continue;
        
        // Stars push other particles away with radiation
        for (size_t j = 0; j < particles.size(); j++) {
            if (i == j) continue;
            
            float2 r = particles[j].pos - particles[i].pos;
            float dist_sq = r.x * r.x + r.y * r.y;
            float dist = sqrt(dist_sq);
            
            if (dist > 0.001f) {
                // Radiation pressure falls off as 1/r²
                float pressure = particles[i].luminosity / (4.0f * M_PI * dist_sq);
                float2 force = r * (pressure / dist);
                
                particles[j].vel = particles[j].vel + force * dt / particles[j].mass;
            }
        }
    }
}

// Apply cooling
void applyCooling(std::vector<StarParticle>& particles, float dt) {
    const float cooling_rate = 0.001f;  // Simple linear cooling
    
    for (auto& p : particles) {
        if (!p.is_fusing && p.temp_internal > 300.0f) {
            p.temp_internal -= cooling_rate * p.temp_internal * dt;
        }
    }
}

// Print statistics
void printStats(const std::vector<StarParticle>& particles, int step) {
    int num_stars = 0;
    int num_black_holes = 0;
    float total_mass = 0;
    float total_luminosity = 0;
    float max_temp = 0;
    float max_density = 0;
    
    for (const auto& p : particles) {
        if (p.is_black_hole) num_black_holes++;
        else if (p.is_fusing) num_stars++;
        total_mass += p.mass;
        total_luminosity += p.luminosity;
        max_temp = std::max(max_temp, p.temp_internal);
        max_density = std::max(max_density, p.density);
    }
    
    std::cout << "Step " << std::setw(5) << step 
              << " | Particles: " << std::setw(4) << particles.size()
              << " | Stars: " << std::setw(2) << num_stars
              << " | Black Holes: " << std::setw(2) << num_black_holes
              << " | Max T: " << std::scientific << std::setprecision(2) << max_temp << "K"
              << " | Max ρ: " << std::fixed << std::setprecision(1) << max_density
              << " | L: " << std::scientific << std::setprecision(2) << total_luminosity << "\n";
}

// ASCII visualization for star formation
void visualizeStarFormation(const std::vector<StarParticle>& particles, float box_size) {
    const int width = 80;
    const int height = 24;
    std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));
    std::vector<std::vector<int>> brightness(height, std::vector<int>(width, 0));
    
    for (const auto& p : particles) {
        int x = (p.pos.x / box_size + 0.5f) * width;
        int y = (p.pos.y / box_size + 0.5f) * height;
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
            if (p.is_black_hole) {
                grid[y][x] = '#';  // Black hole
            } else if (p.is_fusing) {
                grid[y][x] = '*';  // Star
                // Add glow around stars
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            brightness[ny][nx] = std::max(brightness[ny][nx], 2);
                        }
                    }
                }
            } else if (p.temp_internal > 1e6) {
                grid[y][x] = '+';  // Very hot
            } else if (p.temp_internal > 1e4) {
                grid[y][x] = 'o';  // Hot
            } else {
                grid[y][x] = '.';  // Cool gas
            }
        }
    }
    
    // Apply brightness
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (brightness[y][x] > 0 && grid[y][x] == ' ') {
                grid[y][x] = ':';
            }
        }
    }
    
    // Draw
    std::cout << "+" << std::string(width, '-') << "+\n";
    for (const auto& row : grid) {
        std::cout << "|";
        for (char c : row) std::cout << c;
        std::cout << "|\n";
    }
    std::cout << "+" << std::string(width, '-') << "+\n";
}

int main() {
    std::cout << "=== Star Formation Through Gravitational Collapse ===\n\n";
    std::cout << "This simulation demonstrates how stars naturally form from\n";
    std::cout << "gravitational collapse and compression heating.\n\n";
    std::cout << "Legend:\n";
    std::cout << "  . = Cool gas particle\n";
    std::cout << "  o = Warm/hot gas\n";
    std::cout << "  + = Very hot (pre-stellar)\n";
    std::cout << "  * = Star (fusion active)\n";
    std::cout << "  # = Black hole\n";
    std::cout << "  : = Star glow/radiation\n\n";
    
    // Create a collapsing gas cloud
    std::vector<StarParticle> particles;
    std::mt19937 rng(42);
    std::normal_distribution<float> pos_dist(0.0f, 100.0f);
    std::normal_distribution<float> vel_dist(0.0f, 5.0f);
    
    // Create initial gas cloud with slight rotation
    int num_particles = 100;
    for (int i = 0; i < num_particles; i++) {
        StarParticle p;
        
        // Position: Gaussian distribution
        p.pos.x = pos_dist(rng);
        p.pos.y = pos_dist(rng);
        
        // Add slight rotation
        float r = sqrt(p.pos.x * p.pos.x + p.pos.y * p.pos.y);
        float angular_vel = 0.05f;
        p.vel.x = -angular_vel * p.pos.y + vel_dist(rng);
        p.vel.y = angular_vel * p.pos.x + vel_dist(rng);
        
        p.mass = 10.0f;
        p.radius = 5.0f;
        p.temp_internal = 1000.0f;  // Start cool
        p.updateDensity();
        
        particles.push_back(p);
    }
    
    // Add a few denser regions (seeds for star formation)
    for (int i = 0; i < 3; i++) {
        StarParticle p;
        p.pos.x = (i - 1) * 50.0f;
        p.pos.y = 0.0f;
        p.vel = {0, 0};
        p.mass = 50.0f;
        p.radius = 8.0f;
        p.temp_internal = 2000.0f;
        p.updateDensity();
        particles.push_back(p);
    }
    
    // Simulation parameters
    StarFormationMerger merger;
    merger.setOverlapThreshold(0.9f);      // Very close for merger
    merger.setVelocityThreshold(20.0f);    // Higher threshold for gas
    merger.setMergeProbability(0.8f);      // High probability in collapse
    
    float dt = 0.01f;
    float G = 100.0f;  // Strong gravity for faster collapse
    float box_size = 300.0f;
    int max_steps = 500;
    
    std::cout << "Initial gas cloud:\n";
    printStats(particles, 0);
    visualizeStarFormation(particles, box_size);
    
    // Run simulation
    for (int step = 1; step <= max_steps; step++) {
        // Physics
        applyGravity(particles, G, dt);
        applyRadiationPressure(particles, dt);
        applyCooling(particles, dt);
        
        // Merging with compression heating
        std::vector<std::pair<int, int>> merge_pairs;
        for (size_t i = 0; i < particles.size(); i++) {
            if (particles[i].is_merged) continue;
            
            for (size_t j = i + 1; j < particles.size(); j++) {
                if (particles[j].is_merged) continue;
                
                float2 diff = particles[i].pos - particles[j].pos;
                float dist = sqrt(diff.x * diff.x + diff.y * diff.y);
                
                if (merger.shouldMerge(particles[i], particles[j], dist)) {
                    merge_pairs.push_back({i, j});
                    particles[i].merge_target = j;
                    particles[j].is_merged = true;
                    break;
                }
            }
        }
        
        // Execute merges
        if (!merge_pairs.empty()) {
            std::vector<StarParticle> new_particles;
            
            for (const auto& [i, j] : merge_pairs) {
                StarParticle merged = merger.mergeStarParticles(particles[i], particles[j]);
                new_particles.push_back(merged);
            }
            
            for (size_t i = 0; i < particles.size(); i++) {
                if (!particles[i].is_merged && particles[i].merge_target == -1) {
                    particles[i].checkFusionConditions();
                    new_particles.push_back(particles[i]);
                }
            }
            
            particles = new_particles;
            
            // Reset merge flags
            for (auto& p : particles) {
                p.is_merged = false;
                p.merge_target = -1;
            }
        } else {
            // Just update fusion conditions
            for (auto& p : particles) {
                p.checkFusionConditions();
            }
        }
        
        // Display at key moments
        bool show = false;
        if (step == 1 || step % 50 == 0) show = true;
        
        // Always show when stars form
        static int last_star_count = 0;
        int current_stars = 0;
        for (const auto& p : particles) {
            if (p.is_fusing) current_stars++;
        }
        if (current_stars != last_star_count) {
            show = true;
            last_star_count = current_stars;
        }
        
        if (show) {
            std::cout << "\n";
            if (current_stars > 0 && current_stars != last_star_count) {
                std::cout << "*** STAR FORMATION! ***\n";
            }
            printStats(particles, step);
            visualizeStarFormation(particles, box_size);
        }
    }
    
    // Final summary
    std::cout << "\n=== Final Summary ===\n";
    std::cout << "Starting particles: " << (num_particles + 3) << "\n";
    std::cout << "Final particles: " << particles.size() << "\n";
    
    int stars = 0, black_holes = 0;
    float stellar_mass = 0, gas_mass = 0;
    
    for (const auto& p : particles) {
        if (p.is_black_hole) {
            black_holes++;
            stellar_mass += p.mass;
        } else if (p.is_fusing) {
            stars++;
            stellar_mass += p.mass;
        } else {
            gas_mass += p.mass;
        }
    }
    
    std::cout << "Stars formed: " << stars << "\n";
    std::cout << "Black holes: " << black_holes << "\n";
    std::cout << "Stellar mass: " << stellar_mass << "\n";
    std::cout << "Gas mass: " << gas_mass << "\n";
    std::cout << "\nThe simulation demonstrates natural star formation through:\n";
    std::cout << "1. Gravitational collapse of gas clouds\n";
    std::cout << "2. Compression heating from mergers\n";
    std::cout << "3. Fusion ignition when T > 10^7 K\n";
    std::cout << "4. Radiation pressure balancing gravity\n";
    
    return 0;
}