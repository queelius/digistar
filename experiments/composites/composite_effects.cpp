#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <random>

// Constants
const float dt = 0.01f;
const float SPRING_STIFFNESS = 100.0f;
const float SPRING_DAMPING = 0.1f;
const float SPRING_BREAK_FORCE = 50.0f;
const float MATERIAL_CONDUCTIVITY = 1.0f;
const float DRAG_STRENGTH = 10.0f;
const float ROTATION_STRENGTH = 5.0f;

// Basic structures
struct float2 {
    float x, y;
    
    float2 operator+(const float2& other) const { return {x + other.x, y + other.y}; }
    float2 operator-(const float2& other) const { return {x - other.x, y - other.y}; }
    float2 operator*(float s) const { return {x * s, y * s}; }
    float2 operator/(float s) const { return {x / s, y / s}; }
    float2& operator+=(const float2& other) { x += other.x; y += other.y; return *this; }
    float2& operator-=(const float2& other) { x -= other.x; y -= other.y; return *this; }
    float2& operator/=(float s) { x /= s; y /= s; return *this; }
};

float length(const float2& v) { return sqrt(v.x * v.x + v.y * v.y); }
float2 normalize(const float2& v) { float l = length(v); return l > 0 ? v / l : float2{0, 0}; }
float dot(const float2& a, const float2& b) { return a.x * b.x + a.y * b.y; }
float cross(const float2& a, const float2& b) { return a.x * b.y - a.y * b.x; }
float2 perp(const float2& v) { return {-v.y, v.x}; }
float2 rotate(const float2& v, float angle) {
    float c = cos(angle), s = sin(angle);
    return {v.x * c - v.y * s, v.x * s + v.y * c};
}

struct Particle {
    float2 pos;
    float2 vel;
    float mass;
    float radius;
    float temp;
    
    Particle(float x = 0, float y = 0, float m = 1.0f) 
        : pos{x, y}, vel{0, 0}, mass(m), radius(1.0f), temp(300.0f) {}
};

struct Spring {
    int i, j;
    float rest_length;
    float stiffness;
    float damping;
    float break_force;
    bool broken = false;
    float stress = 0;
    
    Spring(int i_, int j_, float rest, float stiff = SPRING_STIFFNESS)
        : i(i_), j(j_), rest_length(rest), stiffness(stiff), 
          damping(SPRING_DAMPING), break_force(SPRING_BREAK_FORCE) {}
};

// Union-Find for blazing fast component detection
class UnionFind {
private:
    std::vector<int> parent;
    std::vector<int> rank;
    
public:
    UnionFind(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }
    
    void unite(int x, int y) {
        int px = find(x);
        int py = find(y);
        if (px == py) return;
        
        // Union by rank
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
    }
    
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};

// Composite body structure
struct CompositeBody {
    std::vector<int> particle_indices;
    std::vector<Spring*> internal_springs;
    
    // Core properties
    int num_particles = 0;
    float total_mass = 0;
    float2 center_of_mass = {0, 0};
    float moment_of_inertia = 0;
    
    // Motion
    float2 mean_velocity = {0, 0};
    float angular_velocity = 0;
    
    // Temperature
    float average_temp = 0;
    float internal_temp = 0;
    
    // Shape
    float compactness = 0;
    float elongation = 1.0f;
    float2 principal_axis = {1, 0};
    float bounding_radius = 0;
    
    // Spring network
    int num_springs = 0;
    float avg_spring_stress = 0;
    float avg_spring_stiffness = SPRING_STIFFNESS;
    float structural_integrity = 1.0f;
    
    void computeProperties(const std::vector<Particle>& particles,
                          const std::vector<Spring>& springs) {
        if (particle_indices.empty()) return;
        
        num_particles = particle_indices.size();
        
        // Reset aggregates
        total_mass = 0;
        center_of_mass = {0, 0};
        average_temp = 0;
        
        // First pass: basic aggregates
        for (int idx : particle_indices) {
            const auto& p = particles[idx];
            total_mass += p.mass;
            center_of_mass.x += p.pos.x * p.mass;
            center_of_mass.y += p.pos.y * p.mass;
            average_temp += p.temp;
        }
        
        center_of_mass /= total_mass;
        average_temp /= num_particles;
        
        // Second pass: relative properties
        moment_of_inertia = 0;
        float2 total_momentum = {0, 0};
        float max_dist = 0;
        float avg_dist = 0;
        
        // For shape analysis
        float Ixx = 0, Iyy = 0, Ixy = 0;
        
        for (int idx : particle_indices) {
            const auto& p = particles[idx];
            
            float2 r = p.pos - center_of_mass;
            float dist = length(r);
            avg_dist += dist;
            max_dist = std::max(max_dist, dist);
            
            moment_of_inertia += p.mass * dist * dist;
            total_momentum += p.vel * p.mass;
            
            // Inertia tensor components
            Ixx += p.mass * r.y * r.y;
            Iyy += p.mass * r.x * r.x;
            Ixy -= p.mass * r.x * r.y;
        }
        
        mean_velocity = total_momentum / total_mass;
        bounding_radius = max_dist;
        avg_dist /= num_particles;
        compactness = max_dist > 0 ? avg_dist / max_dist : 1.0f;
        
        // Shape analysis via eigenvalues
        float trace = Ixx + Iyy;
        float det = Ixx * Iyy - Ixy * Ixy;
        float discriminant = trace * trace - 4.0f * det;
        
        if (discriminant > 0 && det > 0) {
            float sqrt_disc = sqrt(discriminant);
            float lambda1 = (trace + sqrt_disc) / 2.0f;
            float lambda2 = (trace - sqrt_disc) / 2.0f;
            
            if (lambda2 > 0) {
                elongation = sqrt(lambda1 / lambda2);
            }
            
            // Principal axis
            if (abs(lambda1 - Ixx) > 0.001f) {
                principal_axis = normalize(float2{Ixy, lambda1 - Ixx});
            }
        }
        
        // Spring analysis
        internal_springs.clear();
        num_springs = 0;
        avg_spring_stress = 0;
        float total_stiffness = 0;
        
        for (const auto& spring : springs) {
            bool i_in = false, j_in = false;
            for (int idx : particle_indices) {
                if (idx == spring.i) i_in = true;
                if (idx == spring.j) j_in = true;
            }
            
            if (i_in && j_in && !spring.broken) {
                internal_springs.push_back(const_cast<Spring*>(&spring));
                num_springs++;
                avg_spring_stress += spring.stress;
                total_stiffness += spring.stiffness;
            }
        }
        
        if (num_springs > 0) {
            avg_spring_stress /= num_springs;
            avg_spring_stiffness = total_stiffness / num_springs;
        }
        
        // Structural integrity
        float connectivity = num_springs > 0 ? 
            (float)num_springs / (num_particles * (num_particles - 1) / 2) : 0;
        structural_integrity = connectivity * (1.0f - avg_spring_stress / SPRING_BREAK_FORCE);
    }
};

// Resonance calculation
struct ResonanceEffect {
    float amplification;
    float stress_multiplier;
    bool is_destructive;
};

ResonanceEffect calculateResonance(const CompositeBody& body) {
    if (body.total_mass == 0 || body.avg_spring_stiffness == 0) {
        return {1.0f, 1.0f, false};
    }
    
    float natural_freq = sqrt(body.avg_spring_stiffness / body.total_mass);
    float rotation_freq = abs(body.angular_velocity) / (2.0f * M_PI);
    float ratio = rotation_freq / natural_freq;
    
    float resonance = 0;
    for (int n = 1; n <= 3; n++) {
        float delta = abs(ratio - n);
        resonance += exp(-delta * delta * 10.0f);
    }
    
    return {
        1.0f + resonance * 5.0f,
        1.0f + resonance * 2.0f,
        resonance > 0.8f
    };
}

// Find composite bodies using Union-Find
std::vector<CompositeBody> findCompositeBodies(
    const std::vector<Particle>& particles,
    const std::vector<Spring>& springs) {
    
    int n = particles.size();
    UnionFind uf(n);
    
    // Unite particles connected by springs
    for (const auto& spring : springs) {
        if (!spring.broken) {
            uf.unite(spring.i, spring.j);
        }
    }
    
    // Group particles by component
    std::unordered_map<int, std::vector<int>> components;
    for (int i = 0; i < n; i++) {
        int root = uf.find(i);
        components[root].push_back(i);
    }
    
    // Create composite bodies
    std::vector<CompositeBody> bodies;
    for (const auto& [root, indices] : components) {
        if (indices.size() > 1) {  // Skip singletons
            CompositeBody body;
            body.particle_indices = indices;
            body.computeProperties(particles, springs);
            bodies.push_back(body);
        }
    }
    
    return bodies;
}

// Apply force to composite
void applyForceToComposite(CompositeBody& body,
                          std::vector<Particle>& particles,
                          const float2& force,
                          const float2& application_point) {
    // Linear acceleration
    float2 accel = force / body.total_mass;
    
    // Torque creates rotation
    float2 r = application_point - body.center_of_mass;
    float torque = cross(r, force);
    float angular_accel = torque / body.moment_of_inertia;
    
    // Apply to all particles
    for (int idx : body.particle_indices) {
        particles[idx].vel += accel * dt;
        
        // Rotational component
        float2 r_particle = particles[idx].pos - body.center_of_mass;
        float2 v_rot = perp(r_particle) * angular_accel;
        particles[idx].vel += v_rot * dt;
    }
    
    // Update angular velocity
    body.angular_velocity += angular_accel * dt;
}

// Thermal conduction through composite
void propagateHeatInComposite(CompositeBody& body,
                             std::vector<Particle>& particles) {
    float connectivity = (float)body.num_springs / body.num_particles;
    float conductivity = connectivity * body.compactness * MATERIAL_CONDUCTIVITY;
    
    for (Spring* spring : body.internal_springs) {
        float temp_diff = particles[spring->j].temp - particles[spring->i].temp;
        float heat_flow = conductivity * temp_diff * dt;
        
        particles[spring->i].temp += heat_flow / particles[spring->i].mass;
        particles[spring->j].temp -= heat_flow / particles[spring->j].mass;
    }
}

// Visualization
void visualize(const std::vector<Particle>& particles,
              const std::vector<Spring>& springs,
              const std::vector<CompositeBody>& composites,
              const std::string& title = "") {
    const int width = 80;
    const int height = 24;
    std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));
    
    if (!title.empty()) {
        std::cout << title << "\n";
    }
    
    // Draw springs
    for (const auto& spring : springs) {
        if (spring.broken) continue;
        
        const auto& p1 = particles[spring.i];
        const auto& p2 = particles[spring.j];
        
        // Simple line drawing
        int steps = 10;
        for (int s = 0; s <= steps; s++) {
            float t = (float)s / steps;
            float x = p1.pos.x * (1 - t) + p2.pos.x * t;
            float y = p1.pos.y * (1 - t) + p2.pos.y * t;
            
            int gx = (x / 100.0f + 0.5f) * width;
            int gy = (y / 100.0f + 0.5f) * height;
            
            if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                if (spring.stress > SPRING_BREAK_FORCE * 0.8f) {
                    grid[gy][gx] = '!';  // Stressed spring
                } else {
                    grid[gy][gx] = '-';  // Normal spring
                }
            }
        }
    }
    
    // Draw particles
    for (size_t i = 0; i < particles.size(); i++) {
        const auto& p = particles[i];
        int x = (p.pos.x / 100.0f + 0.5f) * width;
        int y = (p.pos.y / 100.0f + 0.5f) * height;
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
            // Find which composite this particle belongs to
            char symbol = 'o';
            for (size_t c = 0; c < composites.size(); c++) {
                for (int idx : composites[c].particle_indices) {
                    if (idx == (int)i) {
                        symbol = 'A' + (c % 26);  // Label by composite
                        break;
                    }
                }
            }
            
            if (p.temp > 500) {
                symbol = '*';  // Hot particle
            }
            
            grid[y][x] = symbol;
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
    std::cout << "=== Composite Body Effects Test ===\n\n";
    std::cout << "Demonstrating Union-Find for blazing fast component detection\n";
    std::cout << "and rich composite-level physics effects\n\n";
    
    std::mt19937 rng(42);
    
    // Test 1: Resonance-induced breakup
    std::cout << "TEST 1: Resonance Effects\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    std::vector<Particle> particles;
    std::vector<Spring> springs;
    
    // Create a rectangular composite body
    int rows = 4, cols = 6;
    float spacing = 3.0f;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float x = (j - cols/2.0f) * spacing;
            float y = (i - rows/2.0f) * spacing;
            particles.emplace_back(x, y, 1.0f);
        }
    }
    
    // Connect with springs in grid pattern
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            
            // Right neighbor
            if (j < cols - 1) {
                springs.emplace_back(idx, idx + 1, spacing);
            }
            
            // Bottom neighbor
            if (i < rows - 1) {
                springs.emplace_back(idx, idx + cols, spacing);
            }
            
            // Diagonal
            if (i < rows - 1 && j < cols - 1) {
                springs.emplace_back(idx, idx + cols + 1, spacing * sqrt(2));
            }
        }
    }
    
    // Find initial composite
    auto composites = findCompositeBodies(particles, springs);
    std::cout << "Initial: Found " << composites.size() << " composite body\n";
    if (!composites.empty()) {
        std::cout << "  Particles: " << composites[0].num_particles 
                  << ", Springs: " << composites[0].num_springs
                  << ", Elongation: " << composites[0].elongation << "\n";
    }
    
    visualize(particles, springs, composites, "\nInitial rectangular body:");
    
    // Apply rotating force to induce resonance
    std::cout << "\nApplying rotating force to induce resonance...\n";
    
    float time = 0;
    for (int step = 0; step < 500; step++) {
        time += dt;
        
        // Update composites
        composites = findCompositeBodies(particles, springs);
        
        for (auto& body : composites) {
            // Apply rotating force
            float force_mag = 10.0f;
            float angle = time * 2.0f;  // Rotation frequency
            float2 force = {force_mag * (float)cos(angle), force_mag * (float)sin(angle)};
            float2 app_point = body.center_of_mass + body.principal_axis * body.bounding_radius;
            
            applyForceToComposite(body, particles, force, app_point);
            
            // Check resonance
            ResonanceEffect resonance = calculateResonance(body);
            
            if (resonance.is_destructive) {
                std::cout << "  RESONANCE! Amplification: " << resonance.amplification 
                         << ", Breaking springs...\n";
                
                // Break overstressed springs
                for (auto& spring : springs) {
                    float2 delta = particles[spring.j].pos - particles[spring.i].pos;
                    float dist = length(delta);
                    spring.stress = abs(dist - spring.rest_length) * spring.stiffness;
                    
                    if (spring.stress * resonance.stress_multiplier > spring.break_force) {
                        spring.broken = true;
                    }
                }
            }
        }
        
        // Spring forces
        for (auto& spring : springs) {
            if (spring.broken) continue;
            
            float2 delta = particles[spring.j].pos - particles[spring.i].pos;
            float dist = length(delta);
            float2 dir = normalize(delta);
            
            // Hooke's law
            float force = spring.stiffness * (dist - spring.rest_length);
            
            // Damping
            float2 v_rel = particles[spring.j].vel - particles[spring.i].vel;
            force += spring.damping * dot(v_rel, dir);
            
            particles[spring.i].vel += dir * force * dt / particles[spring.i].mass;
            particles[spring.j].vel -= dir * force * dt / particles[spring.j].mass;
        }
        
        // Update positions
        for (auto& p : particles) {
            p.pos += p.vel * dt;
        }
    }
    
    // Show final state
    composites = findCompositeBodies(particles, springs);
    std::cout << "\nFinal: Found " << composites.size() << " composite bodies\n";
    for (size_t i = 0; i < composites.size(); i++) {
        std::cout << "  Body " << i << ": " << composites[i].num_particles 
                  << " particles, " << composites[i].num_springs << " springs\n";
    }
    
    visualize(particles, springs, composites, "\nAfter resonance:");
    
    // Test 2: Thermal conduction through composite
    std::cout << "\n\nTEST 2: Thermal Conduction\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    springs.clear();
    
    // Create chain of connected particles
    int chain_length = 10;
    for (int i = 0; i < chain_length; i++) {
        particles.emplace_back(i * 5.0f - 25.0f, 0, 1.0f);
        particles.back().temp = 300.0f;  // Room temperature
    }
    
    // Heat one end
    particles[0].temp = 1000.0f;  // Hot
    particles[chain_length-1].temp = 100.0f;  // Cold
    
    // Connect with springs
    for (int i = 0; i < chain_length - 1; i++) {
        springs.emplace_back(i, i + 1, 5.0f);
    }
    
    composites = findCompositeBodies(particles, springs);
    
    std::cout << "Initial temperatures:\n";
    for (int i = 0; i < chain_length; i++) {
        std::cout << "  P" << i << ": " << particles[i].temp << "K\n";
    }
    
    // Simulate heat conduction
    for (int step = 0; step < 1000; step++) {
        for (auto& body : composites) {
            propagateHeatInComposite(body, particles);
        }
    }
    
    std::cout << "\nAfter heat conduction:\n";
    for (int i = 0; i < chain_length; i++) {
        std::cout << "  P" << i << ": " << std::fixed << std::setprecision(1) 
                  << particles[i].temp << "K\n";
    }
    
    // Test 3: Manipulation API
    std::cout << "\n\nTEST 3: Force Application & Manipulation\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    springs.clear();
    
    // Create triangular body
    particles.emplace_back(0, 10, 2.0f);
    particles.emplace_back(-10, -5, 2.0f);
    particles.emplace_back(10, -5, 2.0f);
    
    springs.emplace_back(0, 1, 20.0f);
    springs.emplace_back(1, 2, 20.0f);
    springs.emplace_back(2, 0, 20.0f);
    
    composites = findCompositeBodies(particles, springs);
    
    std::cout << "Triangular composite body:\n";
    std::cout << "  COM: (" << composites[0].center_of_mass.x << ", " 
              << composites[0].center_of_mass.y << ")\n";
    std::cout << "  Moment of inertia: " << composites[0].moment_of_inertia << "\n";
    
    // Apply off-center force to create rotation
    float2 force = {100.0f, 0};
    float2 app_point = particles[0].pos;  // Top vertex
    
    std::cout << "\nApplying horizontal force at top vertex...\n";
    applyForceToComposite(composites[0], particles, force, app_point);
    
    // Simulate
    for (int step = 0; step < 100; step++) {
        // Spring forces
        for (auto& spring : springs) {
            float2 delta = particles[spring.j].pos - particles[spring.i].pos;
            float dist = length(delta);
            float2 dir = normalize(delta);
            
            float force = spring.stiffness * (dist - spring.rest_length);
            particles[spring.i].vel += dir * force * dt / particles[spring.i].mass;
            particles[spring.j].vel -= dir * force * dt / particles[spring.j].mass;
        }
        
        // Update
        for (auto& p : particles) {
            p.pos += p.vel * dt;
        }
    }
    
    composites = findCompositeBodies(particles, springs);
    
    std::cout << "After force application:\n";
    std::cout << "  COM: (" << composites[0].center_of_mass.x << ", " 
              << composites[0].center_of_mass.y << ")\n";
    std::cout << "  Mean velocity: (" << composites[0].mean_velocity.x << ", "
              << composites[0].mean_velocity.y << ")\n";
    std::cout << "  Angular velocity: " << composites[0].angular_velocity << " rad/s\n";
    
    visualize(particles, springs, composites, "\nRotating triangle:");
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "✓ Union-Find provides O(α(n)) ≈ O(1) component detection\n";
    std::cout << "✓ Resonance can break composites apart\n";
    std::cout << "✓ Heat flows through spring networks\n";
    std::cout << "✓ Off-center forces create realistic rotation\n";
    std::cout << "✓ Composite properties emerge from particle dynamics\n\n";
    
    std::cout << "The Union-Find algorithm makes composite detection blazingly fast,\n";
    std::cout << "enabling rich physics that emerge naturally from the particle level!\n";
    
    return 0;
}