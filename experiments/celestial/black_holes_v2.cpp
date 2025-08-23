#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

// Constants (scaled for dramatic effects)
const float G = 100.0f;          // Strong gravity
const float c = 1000.0f;         // Speed of light
const float dt = 0.001f;         // Small timestep for accuracy

// Particle structure
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
    
    bool is_deleted = false;
    
    Particle(float x_ = 0, float y_ = 0, float m = 10.0f) 
        : x(x_), y(y_), vx(0), vy(0), mass(m), radius(2.0f), temp(300.0f) {}
    
    void updateBlackHole() {
        if (!is_black_hole) return;
        
        // Event horizon grows with mass (more dramatic scaling)
        event_horizon = 2.0f * G * mass / (c * c) * 100.0f;  // Scaled up for visibility
        radius = event_horizon;
        temp = 0;  // Black holes are cold
    }
    
    bool shouldBecomeBlackHole() {
        // Lower threshold for more black hole formation
        float area = M_PI * radius * radius;
        float density = mass / area;
        return density > 10.0f || mass > 500.0f;
    }
};

// Fragment a particle due to tidal forces
std::vector<Particle> tidallySplitParticle(const Particle& p, const Particle& bh) {
    std::vector<Particle> fragments;
    
    // Split into 3-5 fragments
    int num_fragments = 3 + rand() % 3;
    float fragment_mass = p.mass / num_fragments;
    
    for (int i = 0; i < num_fragments; i++) {
        Particle frag;
        frag.mass = fragment_mass;
        frag.radius = p.radius * 0.6f;
        
        // Scatter fragments perpendicular to radial direction
        float angle = 2.0f * M_PI * i / num_fragments;
        float scatter_speed = 50.0f;
        
        frag.x = p.x + p.radius * cos(angle);
        frag.y = p.y + p.radius * sin(angle);
        
        // Velocity includes original plus scatter
        float radial_x = p.x - bh.x;
        float radial_y = p.y - bh.y;
        float r = sqrt(radial_x * radial_x + radial_y * radial_y);
        float tangent_x = -radial_y / r;
        float tangent_y = radial_x / r;
        
        frag.vx = p.vx + tangent_x * scatter_speed * (i % 2 == 0 ? 1 : -1);
        frag.vy = p.vy + tangent_y * scatter_speed * (i % 2 == 0 ? 1 : -1);
        
        frag.temp = p.temp * 2.0f;  // Heating from disruption
        
        fragments.push_back(frag);
    }
    
    return fragments;
}

// Check absorption
bool checkAbsorption(Particle& bh, Particle& p) {
    if (!bh.is_black_hole || p.is_black_hole || p.is_deleted) return false;
    
    float dx = p.x - bh.x;
    float dy = p.y - bh.y;
    float dist = sqrt(dx*dx + dy*dy);
    
    // Absorption when particle touches event horizon
    if (dist < bh.event_horizon + p.radius) {
        // Conservation of momentum
        float px_total = bh.vx * bh.mass + p.vx * p.mass;
        float py_total = bh.vy * bh.mass + p.vy * p.mass;
        
        bh.mass += p.mass;
        bh.vx = px_total / bh.mass;
        bh.vy = py_total / bh.mass;
        
        bh.particles_absorbed++;
        bh.total_mass_absorbed += p.mass;
        
        // Update event horizon immediately
        bh.updateBlackHole();
        
        std::cout << "  ABSORBED! BH mass: " << bh.mass 
                  << ", Event horizon: " << bh.event_horizon << "\n";
        
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
    
    if (!bh.is_black_hole || p.is_black_hole) return effect;
    
    float dx = p.x - bh.x;
    float dy = p.y - bh.y;
    float dist = sqrt(dx*dx + dy*dy);
    
    // Tidal force gradient
    float tidal_gradient = 2.0f * G * bh.mass / (dist * dist * dist);
    float tidal_force = tidal_gradient * p.radius;
    
    // Material strength (lower for easier breaking)
    float strength = 1.0f;
    effect.stress = tidal_force / strength;
    
    // Break when stressed and close enough
    float tidal_radius = bh.event_horizon * 3.0f;  // Roche limit
    if (effect.stress > 1.0f && dist < tidal_radius && p.mass > 2.0f) {
        effect.should_break = true;
    }
    
    return effect;
}

// Apply gravity
void applyGravity(Particle& p1, Particle& p2, float softening = 0.1f) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dist_sq = dx*dx + dy*dy + softening*softening;
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
        
        int x = (p.x / 200.0f + 0.5f) * width;
        int y = (p.y / 200.0f + 0.5f) * height;
        
        if (p.is_black_hole) {
            // Draw event horizon
            int eh_pixels = std::max(1, (int)(p.event_horizon / 5.0f));
            for (int dy = -eh_pixels; dy <= eh_pixels; dy++) {
                for (int dx = -eh_pixels; dx <= eh_pixels; dx++) {
                    if (dx*dx + dy*dy <= eh_pixels*eh_pixels) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            grid[ny][nx] = '#';
                        }
                    }
                }
            }
            // Center marker
            if (x >= 0 && x < width && y >= 0 && y < height) {
                grid[y][x] = '@';
            }
        } else {
            if (x >= 0 && x < width && y >= 0 && y < height) {
                if (p.temp > 1000) {
                    grid[y][x] = '*';  // Hot
                } else if (p.mass > 20) {
                    grid[y][x] = 'O';  // Massive
                } else {
                    grid[y][x] = '.';  // Normal
                }
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
    std::cout << "=== Enhanced Black Hole Dynamics Test ===\n\n";
    std::cout << "Legend:\n";
    std::cout << "  @ = Black hole center\n";
    std::cout << "  # = Event horizon\n";
    std::cout << "  * = Hot/disrupted particle\n";
    std::cout << "  O = Massive particle\n";
    std::cout << "  . = Normal particle\n\n";
    
    std::mt19937 rng(42);
    
    // Test 1: Tidal disruption with visible effects
    std::cout << "TEST 1: Tidal Disruption Event\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    std::vector<Particle> particles;
    
    // Create massive black hole
    Particle bh(0, 0, 1000.0f);
    bh.is_black_hole = true;
    bh.updateBlackHole();
    particles.push_back(bh);
    
    std::cout << "Black hole created: Mass=" << bh.mass 
              << ", Event horizon=" << bh.event_horizon << "\n\n";
    
    // Add approaching star
    Particle star(-50, 20, 30.0f);
    star.radius = 5.0f;
    star.vx = 40.0f;
    star.vy = -5.0f;
    star.temp = 5000.0f;
    particles.push_back(star);
    
    // Add orbiting particles
    for (int i = 0; i < 5; i++) {
        float angle = 2.0f * M_PI * i / 5.0f;
        float r = 30.0f;
        Particle p(r * cos(angle), r * sin(angle), 5.0f);
        
        // Orbital velocity
        float v_orb = sqrt(G * bh.mass / r);
        p.vx = -v_orb * sin(angle) * 0.8f;  // Slightly elliptical
        p.vy = v_orb * cos(angle) * 0.8f;
        particles.push_back(p);
    }
    
    std::cout << "Initial: Star approaching black hole with orbiting particles\n";
    visualize(particles);
    
    // Simulate
    for (int step = 0; step < 1000; step++) {
        // Apply gravity
        for (size_t i = 0; i < particles.size(); i++) {
            for (size_t j = i+1; j < particles.size(); j++) {
                if (!particles[i].is_deleted && !particles[j].is_deleted) {
                    applyGravity(particles[i], particles[j]);
                }
            }
        }
        
        // Check tidal disruption
        std::vector<Particle> new_fragments;
        for (size_t i = 1; i < particles.size(); i++) {
            if (!particles[i].is_deleted && !particles[i].is_black_hole) {
                TidalEffect tidal = calculateTidal(particles[0], particles[i]);
                
                if (tidal.should_break) {
                    std::cout << "*** TIDAL DISRUPTION! Particle " << i 
                              << " torn apart! ***\n";
                    
                    auto fragments = tidallySplitParticle(particles[i], particles[0]);
                    for (const auto& frag : fragments) {
                        new_fragments.push_back(frag);
                    }
                    particles[i].is_deleted = true;
                }
            }
        }
        
        // Add fragments
        for (const auto& frag : new_fragments) {
            particles.push_back(frag);
        }
        
        // Check absorption
        for (size_t i = 1; i < particles.size(); i++) {
            if (!particles[i].is_deleted) {
                if (checkAbsorption(particles[0], particles[i])) {
                    particles[i].is_deleted = true;
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
        
        // Show key moments
        if (step == 200 || step == 400 || step == 600 || step == 999) {
            std::cout << "\nStep " << step << ":\n";
            visualize(particles);
            
            int active = 0;
            for (const auto& p : particles) {
                if (!p.is_deleted && !p.is_black_hole) active++;
            }
            std::cout << "Active particles: " << active 
                      << " | BH absorbed: " << particles[0].particles_absorbed
                      << " | BH mass: " << particles[0].mass << "\n";
        }
    }
    
    // Test 2: Accretion disk with visible spiral
    std::cout << "\n\nTEST 2: Accretion Disk Formation\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    
    // Central black hole
    Particle central_bh(0, 0, 500.0f);
    central_bh.is_black_hole = true;
    central_bh.updateBlackHole();
    particles.push_back(central_bh);
    
    // Create disk with slight inward spiral
    for (int ring = 0; ring < 4; ring++) {
        float r = 20.0f + ring * 10.0f;
        int n = 6 + ring * 2;
        
        for (int i = 0; i < n; i++) {
            float angle = 2.0f * M_PI * i / n + ring * 0.1f;
            Particle p(r * cos(angle), r * sin(angle), 2.0f);
            
            // Orbital velocity with small inward component
            float v_orb = sqrt(G * central_bh.mass / r);
            p.vx = -v_orb * sin(angle) * 0.9f;  // Slightly slow = spiral in
            p.vy = v_orb * cos(angle) * 0.9f;
            p.temp = 300.0f + 1000.0f / r;  // Hotter closer to BH
            
            particles.push_back(p);
        }
    }
    
    std::cout << "Initial accretion disk:\n";
    visualize(particles);
    
    int total_absorbed = 0;
    for (int step = 0; step < 2000; step++) {
        // Gravity
        for (size_t i = 1; i < particles.size(); i++) {
            if (!particles[i].is_deleted) {
                applyGravity(particles[0], particles[i], 0.01f);
                
                // Disk viscosity causes inspiral
                float dx = particles[i].x;
                float dy = particles[i].y;
                float r = sqrt(dx*dx + dy*dy);
                
                if (r < central_bh.event_horizon * 5.0f) {
                    // Friction heating
                    particles[i].temp += 100.0f * dt;
                    
                    // Viscous drag
                    particles[i].vx *= (1.0f - 0.01f * dt);
                    particles[i].vy *= (1.0f - 0.01f * dt);
                }
            }
        }
        
        // Check absorption
        for (size_t i = 1; i < particles.size(); i++) {
            if (!particles[i].is_deleted) {
                if (checkAbsorption(particles[0], particles[i])) {
                    particles[i].is_deleted = true;
                    total_absorbed++;
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
        
        if (step == 500 || step == 1000 || step == 1999) {
            std::cout << "\nStep " << step << ":\n";
            visualize(particles);
            std::cout << "Total absorbed: " << total_absorbed 
                      << " | BH mass: " << particles[0].mass
                      << " | Event horizon: " << particles[0].event_horizon << "\n";
        }
    }
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Enhanced black hole dynamics demonstrated:\n";
    std::cout << "✓ Tidal disruption with visible fragmentation\n";
    std::cout << "✓ Particle absorption with mass growth\n";
    std::cout << "✓ Accretion disk with inspiral\n";
    std::cout << "✓ Event horizon growth\n";
    std::cout << "\nBlack holes create dramatic and dangerous dynamics!\n";
    
    return 0;
}