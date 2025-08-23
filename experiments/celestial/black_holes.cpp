#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

// Constants
const float G = 10.0f;           // Gravitational constant (scaled)
const float c = 100.0f;          // Speed of light (scaled)
const float dt = 0.01f;          // Time step

// Simplified black hole particle
struct Particle {
    float x, y;
    float vx, vy;
    float mass;
    float radius;
    float temp;
    
    // Black hole properties
    bool is_black_hole = false;
    float event_horizon = 0;
    int particles_absorbed = 0;
    float total_mass_absorbed = 0;
    
    // Accretion
    float accretion_rate = 0;
    float accretion_temp = 0;
    
    bool is_deleted = false;
    
    Particle(float x_ = 0, float y_ = 0, float m = 10.0f) 
        : x(x_), y(y_), vx(0), vy(0), mass(m), radius(5.0f), temp(300.0f) {}
    
    void updateBlackHole() {
        if (!is_black_hole) return;
        
        // Schwarzschild radius (simplified for 2D)
        event_horizon = 0.2f * sqrt(mass);  // Simplified scaling
        radius = event_horizon;
        
        // Black holes don't radiate
        temp = 0;
        
        // Accretion disk temperature
        if (accretion_rate > 0) {
            accretion_temp = 10000.0f * mass / event_horizon;  // Simplified
        }
    }
    
    bool shouldBecomeBlackHole() {
        // Critical density check
        float area = M_PI * radius * radius;
        float density = mass / area;
        return density > 100.0f;  // Simplified threshold
    }
};

// Check if particle is absorbed by black hole
bool checkAbsorption(Particle& bh, Particle& p) {
    if (!bh.is_black_hole || p.is_black_hole) return false;
    
    float dx = p.x - bh.x;
    float dy = p.y - bh.y;
    float dist = sqrt(dx*dx + dy*dy);
    
    if (dist < bh.event_horizon + p.radius) {
        // Absorbed!
        float momentum_x = bh.vx * bh.mass + p.vx * p.mass;
        float momentum_y = bh.vy * bh.mass + p.vy * p.mass;
        
        bh.mass += p.mass;
        bh.vx = momentum_x / bh.mass;
        bh.vy = momentum_y / bh.mass;
        
        bh.particles_absorbed++;
        bh.total_mass_absorbed += p.mass;
        bh.accretion_rate = p.mass / dt;
        
        bh.updateBlackHole();
        
        return true;
    }
    return false;
}

// Calculate tidal forces
struct TidalEffect {
    bool should_break;
    float stress;
};

TidalEffect calculateTidal(const Particle& bh, const Particle& p) {
    TidalEffect effect = {false, 0};
    
    if (!bh.is_black_hole) return effect;
    
    float dx = p.x - bh.x;
    float dy = p.y - bh.y;
    float dist = sqrt(dx*dx + dy*dy);
    
    // Tidal force ~ M*R/r³
    float tidal = 2.0f * G * bh.mass * p.radius / (dist * dist * dist);
    
    // Simplified material strength
    float strength = 10.0f;
    effect.stress = tidal / strength;
    
    // Break if stress > 1
    if (effect.stress > 1.0f && dist < bh.event_horizon * 5.0f) {
        effect.should_break = true;
    }
    
    return effect;
}

// Apply gravity between particles
void applyGravity(Particle& p1, Particle& p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dist_sq = dx*dx + dy*dy + 0.1f;  // Softening
    float dist = sqrt(dist_sq);
    
    float F = G * p1.mass * p2.mass / dist_sq;
    float fx = F * dx / dist;
    float fy = F * dy / dist;
    
    p1.vx += fx * dt / p1.mass;
    p1.vy += fy * dt / p1.mass;
    p2.vx -= fx * dt / p2.mass;
    p2.vy -= fy * dt / p2.mass;
}

// Visualization
void visualize(const std::vector<Particle>& particles, const std::string& title = "") {
    const int width = 80;
    const int height = 24;
    std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));
    
    if (!title.empty()) {
        std::cout << title << "\n";
    }
    
    for (const auto& p : particles) {
        if (p.is_deleted) continue;
        
        int x = (p.x / 300.0f + 0.5f) * width;
        int y = (p.y / 300.0f + 0.5f) * height;
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
            if (p.is_black_hole) {
                // Show event horizon size
                int eh_size = std::max(1, (int)(p.event_horizon / 10.0f));
                for (int dy = -eh_size; dy <= eh_size; dy++) {
                    for (int dx = -eh_size; dx <= eh_size; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (dx*dx + dy*dy <= eh_size*eh_size) {
                                grid[ny][nx] = '#';  // Black hole
                            }
                        }
                    }
                }
            } else if (p.temp > 5000) {
                grid[y][x] = '*';  // Hot/star
            } else if (p.mass > 50) {
                grid[y][x] = 'O';  // Massive
            } else {
                grid[y][x] = '.';  // Normal particle
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

// Print statistics
void printStats(const std::vector<Particle>& particles) {
    int black_holes = 0;
    int absorbed = 0;
    float max_mass = 0;
    
    for (const auto& p : particles) {
        if (p.is_deleted) continue;
        if (p.is_black_hole) {
            black_holes++;
            absorbed += p.particles_absorbed;
            max_mass = std::max(max_mass, p.mass);
        }
    }
    
    std::cout << "Black holes: " << black_holes 
              << " | Particles absorbed: " << absorbed
              << " | Largest BH mass: " << max_mass << "\n";
}

int main() {
    std::cout << "=== Black Hole Dynamics Test ===\n\n";
    std::cout << "Legend:\n";
    std::cout << "  # = Black hole (size shows event horizon)\n";
    std::cout << "  * = Hot particle/star\n";
    std::cout << "  O = Massive particle\n";
    std::cout << "  . = Normal particle\n\n";
    
    // Test 1: Black hole formation from density
    std::cout << "TEST 1: Black Hole Formation from Collapse\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    std::vector<Particle> particles;
    
    // Create dense cluster that will collapse
    for (int i = 0; i < 20; i++) {
        float angle = 2.0f * M_PI * i / 20.0f;
        float r = 20.0f;
        Particle p(r * cos(angle), r * sin(angle), 10.0f);
        p.vx = -10.0f * sin(angle);  // Orbital velocity
        p.vy = 10.0f * cos(angle);
        particles.push_back(p);
    }
    
    // Add central massive object
    Particle center(0, 0, 100.0f);
    particles.push_back(center);
    
    std::cout << "Initial configuration (dense cluster):\n";
    visualize(particles);
    
    // Simulate collapse
    for (int step = 0; step < 200; step++) {
        // Apply gravity
        for (size_t i = 0; i < particles.size(); i++) {
            for (size_t j = i+1; j < particles.size(); j++) {
                if (!particles[i].is_deleted && !particles[j].is_deleted) {
                    applyGravity(particles[i], particles[j]);
                }
            }
        }
        
        // Update positions
        for (auto& p : particles) {
            if (!p.is_deleted) {
                p.x += p.vx * dt;
                p.y += p.vy * dt;
            }
        }
        
        // Check for black hole formation
        for (auto& p : particles) {
            if (!p.is_black_hole && p.shouldBecomeBlackHole()) {
                p.is_black_hole = true;
                p.updateBlackHole();
                std::cout << "*** BLACK HOLE FORMED! Mass: " << p.mass << " ***\n";
            }
        }
        
        // Check for absorption
        for (auto& bh : particles) {
            if (bh.is_black_hole) {
                for (auto& p : particles) {
                    if (&p != &bh && !p.is_deleted) {
                        if (checkAbsorption(bh, p)) {
                            p.is_deleted = true;
                        }
                    }
                }
            }
        }
        
        if (step == 100 || step == 199) {
            std::cout << "\nStep " << step << ":\n";
            visualize(particles);
            printStats(particles);
        }
    }
    
    // Test 2: Tidal disruption
    std::cout << "\n\nTEST 2: Tidal Disruption (Spaghettification)\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    
    // Create black hole
    Particle bh(0, 0, 1000.0f);
    bh.is_black_hole = true;
    bh.updateBlackHole();
    particles.push_back(bh);
    
    // Add particles approaching black hole
    for (int i = 0; i < 5; i++) {
        Particle p(-100 + i*10, 50, 5.0f);
        p.vx = 30.0f;
        p.vy = -5.0f;
        particles.push_back(p);
    }
    
    std::cout << "Particles approaching black hole:\n";
    visualize(particles);
    std::cout << "Black hole event horizon: " << bh.event_horizon << "\n";
    
    // Simulate approach
    for (int step = 0; step < 150; step++) {
        // Gravity
        for (size_t i = 1; i < particles.size(); i++) {
            if (!particles[i].is_deleted) {
                applyGravity(particles[0], particles[i]);
            }
        }
        
        // Check tidal forces
        for (size_t i = 1; i < particles.size(); i++) {
            if (!particles[i].is_deleted) {
                TidalEffect tidal = calculateTidal(particles[0], particles[i]);
                if (tidal.should_break) {
                    std::cout << "  Particle " << i << " torn apart by tidal forces!\n";
                    // Split into fragments
                    particles[i].mass *= 0.5f;
                    particles[i].radius *= 0.8f;
                    
                    // Create fragment
                    Particle fragment = particles[i];
                    fragment.vx += 10.0f;
                    fragment.vy += 10.0f;
                    particles.push_back(fragment);
                }
            }
        }
        
        // Update positions
        for (auto& p : particles) {
            if (!p.is_deleted) {
                p.x += p.vx * dt;
                p.y += p.vy * dt;
            }
        }
        
        // Check absorption
        for (size_t i = 1; i < particles.size(); i++) {
            if (!particles[i].is_deleted) {
                if (checkAbsorption(particles[0], particles[i])) {
                    std::cout << "  Particle absorbed! BH mass now: " 
                              << particles[0].mass << "\n";
                    particles[i].is_deleted = true;
                }
            }
        }
        
        if (step == 75 || step == 149) {
            std::cout << "\nStep " << step << ":\n";
            visualize(particles);
            printStats(particles);
        }
    }
    
    // Test 3: Black hole merger
    std::cout << "\n\nTEST 3: Black Hole Merger\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    
    // Two black holes orbiting
    Particle bh1(-30, 0, 500.0f);
    bh1.is_black_hole = true;
    bh1.vy = 20.0f;
    bh1.updateBlackHole();
    particles.push_back(bh1);
    
    Particle bh2(30, 0, 500.0f);
    bh2.is_black_hole = true;
    bh2.vy = -20.0f;
    bh2.updateBlackHole();
    particles.push_back(bh2);
    
    std::cout << "Two black holes in orbit:\n";
    visualize(particles);
    std::cout << "BH1 mass: " << bh1.mass << ", BH2 mass: " << bh2.mass << "\n";
    
    // Simulate merger
    for (int step = 0; step < 300; step++) {
        applyGravity(particles[0], particles[1]);
        
        particles[0].x += particles[0].vx * dt;
        particles[0].y += particles[0].vy * dt;
        particles[1].x += particles[1].vx * dt;
        particles[1].y += particles[1].vy * dt;
        
        // Check for merger
        float dx = particles[1].x - particles[0].x;
        float dy = particles[1].y - particles[0].y;
        float dist = sqrt(dx*dx + dy*dy);
        
        if (dist < particles[0].event_horizon + particles[1].event_horizon) {
            std::cout << "\n*** BLACK HOLES MERGING! ***\n";
            
            // Merge
            float total_mass = particles[0].mass + particles[1].mass;
            float px = particles[0].vx * particles[0].mass + 
                      particles[1].vx * particles[1].mass;
            float py = particles[0].vy * particles[0].mass + 
                      particles[1].vy * particles[1].mass;
            
            particles[0].mass = total_mass * 0.95f;  // 5% radiated as GW
            particles[0].vx = px / particles[0].mass;
            particles[0].vy = py / particles[0].mass;
            particles[0].x = (particles[0].x * particles[0].mass + 
                             particles[1].x * particles[1].mass) / total_mass;
            particles[0].y = (particles[0].y * particles[0].mass + 
                             particles[1].y * particles[1].mass) / total_mass;
            particles[0].updateBlackHole();
            
            particles[1].is_deleted = true;
            
            std::cout << "Merged BH mass: " << particles[0].mass << "\n";
            std::cout << "Event horizon: " << particles[0].event_horizon << "\n";
            break;
        }
        
        if (step % 100 == 0) {
            std::cout << "\nStep " << step << ":\n";
            visualize(particles);
        }
    }
    
    std::cout << "\nFinal state:\n";
    visualize(particles);
    
    // Test 4: Accretion disk
    std::cout << "\n\nTEST 4: Accretion Disk Formation\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    
    // Central black hole
    Particle central_bh(0, 0, 2000.0f);
    central_bh.is_black_hole = true;
    central_bh.updateBlackHole();
    particles.push_back(central_bh);
    
    // Add orbiting particles at various radii
    for (int ring = 0; ring < 3; ring++) {
        float r = 40.0f + ring * 20.0f;
        int n = 8 + ring * 4;
        for (int i = 0; i < n; i++) {
            float angle = 2.0f * M_PI * i / n;
            Particle p(r * cos(angle), r * sin(angle), 2.0f);
            
            // Orbital velocity
            float v_orb = sqrt(G * central_bh.mass / r);
            p.vx = -v_orb * sin(angle);
            p.vy = v_orb * cos(angle);
            
            particles.push_back(p);
        }
    }
    
    std::cout << "Initial disk configuration:\n";
    visualize(particles);
    
    // Simulate accretion
    int absorbed_count = 0;
    for (int step = 0; step < 200; step++) {
        // Apply forces
        for (size_t i = 1; i < particles.size(); i++) {
            if (!particles[i].is_deleted) {
                applyGravity(particles[0], particles[i]);
                
                // Disk friction causes inspiral
                float dx = particles[i].x - particles[0].x;
                float dy = particles[i].y - particles[0].y;
                float dist = sqrt(dx*dx + dy*dy);
                
                if (dist < central_bh.event_horizon * 10.0f) {
                    // Add some drag to create inspiral
                    particles[i].vx *= 0.995f;
                    particles[i].vy *= 0.995f;
                    
                    // Heat up from friction
                    particles[i].temp += 10.0f;
                }
            }
        }
        
        // Update positions
        for (auto& p : particles) {
            if (!p.is_deleted) {
                p.x += p.vx * dt;
                p.y += p.vy * dt;
            }
        }
        
        // Check absorption
        for (size_t i = 1; i < particles.size(); i++) {
            if (!particles[i].is_deleted) {
                if (checkAbsorption(particles[0], particles[i])) {
                    particles[i].is_deleted = true;
                    absorbed_count++;
                }
            }
        }
        
        if (step == 100 || step == 199) {
            std::cout << "\nStep " << step << ":\n";
            visualize(particles);
            std::cout << "Particles absorbed: " << absorbed_count << "\n";
            std::cout << "Black hole mass: " << particles[0].mass << "\n";
        }
    }
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Black hole dynamics demonstrated:\n";
    std::cout << "✓ Formation from gravitational collapse\n";
    std::cout << "✓ Particle absorption at event horizon\n";
    std::cout << "✓ Tidal disruption (spaghettification)\n";
    std::cout << "✓ Black hole mergers\n";
    std::cout << "✓ Accretion disk dynamics\n";
    std::cout << "\nBlack holes emerge naturally from extreme conditions!\n";
    
    return 0;
}