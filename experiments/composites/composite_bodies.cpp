#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>

const float BOLTZMANN = 1.38e-23f;

// Simple 2D vector
struct float2 {
    float x, y;
    
    float2 operator+(const float2& other) const { return {x + other.x, y + other.y}; }
    float2 operator-(const float2& other) const { return {x - other.x, y - other.y}; }
    float2 operator*(float s) const { return {x * s, y * s}; }
    float2 operator/(float s) const { return {x / s, y / s}; }
    float2& operator+=(const float2& other) { x += other.x; y += other.y; return *this; }
    float2& operator-=(const float2& other) { x -= other.x; y -= other.y; return *this; }
    float2& operator*=(float s) { x *= s; y *= s; return *this; }
    float2& operator/=(float s) { x /= s; y /= s; return *this; }
};

float length(const float2& v) { return sqrt(v.x * v.x + v.y * v.y); }
float dot(const float2& a, const float2& b) { return a.x * b.x + a.y * b.y; }

// Particle
struct Particle {
    float2 pos;
    float2 vel;
    float mass;
    float radius;
    float temp_internal;
    
    Particle(float x = 0, float y = 0, float m = 1.0f) 
        : pos{x, y}, vel{0, 0}, mass(m), radius(2.0f), temp_internal(300.0f) {}
};

// Spring
struct Spring {
    int i, j;
    float rest_length;
    float stiffness;
    float break_force;
    bool broken = false;
    
    Spring(int i_, int j_, float rest, float k = 100.0f) 
        : i(i_), j(j_), rest_length(rest), stiffness(k), break_force(k * rest * 0.5f) {}
};

// Union-Find for connected components
class UnionFind {
    std::vector<int> parent;
    std::vector<int> rank;
    
public:
    UnionFind(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; i++) parent[i] = i;
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }
    
    void unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;
        
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
    }
    
    bool connected(int x, int y) { return find(x) == find(y); }
};

// Composite body properties
struct CompositeBody {
    std::vector<int> particle_indices;
    
    // Aggregate properties
    float total_mass = 0;
    float2 center_of_mass = {0, 0};
    float2 mean_velocity = {0, 0};
    float moment_of_inertia = 0;
    float angular_velocity = 0;
    
    // Temperature
    float internal_temp = 0;  // From velocity dispersion
    float average_temp = 0;    // Mean particle temperature
    
    // Shape
    float compactness = 0;
    float elongation = 1.0f;
    float2 principal_axis = {1, 0};
    
    // Springs
    int num_springs = 0;
    float structural_integrity = 0;
    
    void computeProperties(const std::vector<Particle>& particles,
                          const std::vector<Spring>& springs) {
        if (particle_indices.empty()) return;
        
        // Reset
        total_mass = 0;
        center_of_mass = {0, 0};
        average_temp = 0;
        
        // First pass: basic aggregates
        for (int idx : particle_indices) {
            const auto& p = particles[idx];
            total_mass += p.mass;
            center_of_mass.x += p.pos.x * p.mass;
            center_of_mass.y += p.pos.y * p.mass;
            average_temp += p.temp_internal;
        }
        
        center_of_mass /= total_mass;
        average_temp /= particle_indices.size();
        
        // Mean velocity
        float2 total_momentum = {0, 0};
        for (int idx : particle_indices) {
            const auto& p = particles[idx];
            total_momentum.x += p.vel.x * p.mass;
            total_momentum.y += p.vel.y * p.mass;
        }
        mean_velocity = total_momentum / total_mass;
        
        // Second pass: relative properties
        float total_kinetic = 0;
        moment_of_inertia = 0;
        float angular_momentum = 0;
        
        // For shape analysis
        float Ixx = 0, Iyy = 0, Ixy = 0;
        
        for (int idx : particle_indices) {
            const auto& p = particles[idx];
            
            float2 r = p.pos - center_of_mass;
            float r_sq = r.x * r.x + r.y * r.y;
            
            float2 v_rel = p.vel - mean_velocity;
            total_kinetic += 0.5f * p.mass * dot(v_rel, v_rel);
            
            moment_of_inertia += p.mass * r_sq;
            angular_momentum += p.mass * (r.x * v_rel.y - r.y * v_rel.x);
            
            // Inertia tensor
            Ixx += p.mass * r.y * r.y;
            Iyy += p.mass * r.x * r.x;
            Ixy -= p.mass * r.x * r.y;
        }
        
        // Internal temperature from kinetic energy
        internal_temp = 2.0f * total_kinetic / (particle_indices.size() * BOLTZMANN * 1e20f);
        
        // Angular velocity
        if (moment_of_inertia > 0) {
            angular_velocity = angular_momentum / moment_of_inertia;
        }
        
        // Shape analysis
        float trace = Ixx + Iyy;
        float det = Ixx * Iyy - Ixy * Ixy;
        float disc = trace * trace - 4.0f * det;
        
        if (disc > 0 && det > 0) {
            float sqrt_disc = sqrt(disc);
            float lambda1 = (trace + sqrt_disc) / 2.0f;
            float lambda2 = (trace - sqrt_disc) / 2.0f;
            
            if (lambda2 > 0) {
                elongation = sqrt(lambda1 / lambda2);
            }
            
            // Principal axis
            if (abs(Ixy) > 0.001f) {
                principal_axis.x = lambda1 - Ixx;
                principal_axis.y = Ixy;
                float len = length(principal_axis);
                if (len > 0) principal_axis = principal_axis / len;
            }
        }
        
        // Compactness
        float avg_dist = 0, max_dist = 0;
        for (int idx : particle_indices) {
            float dist = length(particles[idx].pos - center_of_mass);
            avg_dist += dist;
            max_dist = std::max(max_dist, dist);
        }
        avg_dist /= particle_indices.size();
        compactness = (max_dist > 0) ? avg_dist / max_dist : 1.0f;
        
        // Spring analysis
        num_springs = 0;
        for (const auto& spring : springs) {
            if (spring.broken) continue;
            
            bool i_in = std::find(particle_indices.begin(), particle_indices.end(), 
                                 spring.i) != particle_indices.end();
            bool j_in = std::find(particle_indices.begin(), particle_indices.end(), 
                                 spring.j) != particle_indices.end();
            
            if (i_in && j_in) num_springs++;
        }
        
        // Structural integrity
        int max_springs = particle_indices.size() * (particle_indices.size() - 1) / 2;
        structural_integrity = (max_springs > 0) ? (float)num_springs / max_springs : 0;
    }
    
    void print() const {
        std::cout << "  Particles: " << particle_indices.size() 
                  << " | Mass: " << std::fixed << std::setprecision(1) << total_mass
                  << " | COM: (" << center_of_mass.x << ", " << center_of_mass.y << ")"
                  << " | Vel: (" << mean_velocity.x << ", " << mean_velocity.y << ")\n";
        std::cout << "  Internal T: " << std::scientific << std::setprecision(2) << internal_temp 
                  << "K | Avg T: " << average_temp << "K"
                  << " | Angular vel: " << std::fixed << std::setprecision(3) << angular_velocity 
                  << " rad/s\n";
        std::cout << "  Elongation: " << std::setprecision(2) << elongation 
                  << " | Compactness: " << compactness
                  << " | Springs: " << num_springs
                  << " | Integrity: " << std::setprecision(1) << structural_integrity * 100 << "%\n";
    }
};

// Find all composite bodies
std::vector<CompositeBody> findCompositeBodies(const std::vector<Particle>& particles,
                                               const std::vector<Spring>& springs) {
    int n = particles.size();
    UnionFind uf(n);
    
    // Unite connected particles
    for (const auto& spring : springs) {
        if (!spring.broken) {
            uf.unite(spring.i, spring.j);
        }
    }
    
    // Group by component
    std::unordered_map<int, std::vector<int>> components;
    for (int i = 0; i < n; i++) {
        components[uf.find(i)].push_back(i);
    }
    
    // Create composite bodies
    std::vector<CompositeBody> bodies;
    for (const auto& [root, indices] : components) {
        CompositeBody body;
        body.particle_indices = indices;
        body.computeProperties(particles, springs);
        bodies.push_back(body);
    }
    
    // Sort by size
    std::sort(bodies.begin(), bodies.end(), 
              [](const auto& a, const auto& b) { 
                  return a.particle_indices.size() > b.particle_indices.size(); 
              });
    
    return bodies;
}

// Apply spring forces
void applySpringForces(std::vector<Particle>& particles,
                      std::vector<Spring>& springs,
                      float dt) {
    for (auto& spring : springs) {
        if (spring.broken) continue;
        
        float2 r = particles[spring.j].pos - particles[spring.i].pos;
        float dist = length(r);
        
        if (dist < 0.001f) continue;
        
        float force_mag = spring.stiffness * (dist - spring.rest_length);
        
        // Check for breaking
        if (abs(force_mag) > spring.break_force) {
            spring.broken = true;
            std::cout << "  Spring broke between " << spring.i << " and " << spring.j << "!\n";
            continue;
        }
        
        float2 force = r * (force_mag / dist);
        
        particles[spring.i].vel += force * dt / particles[spring.i].mass;
        particles[spring.j].vel -= force * dt / particles[spring.j].mass;
    }
}

// Visualization
void visualize(const std::vector<Particle>& particles,
               const std::vector<CompositeBody>& bodies) {
    const int width = 60;
    const int height = 20;
    std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));
    
    // Assign each particle to its body
    std::vector<int> particle_body(particles.size(), -1);
    for (size_t b = 0; b < bodies.size(); b++) {
        for (int idx : bodies[b].particle_indices) {
            particle_body[idx] = b;
        }
    }
    
    // Draw particles colored by body
    for (size_t i = 0; i < particles.size(); i++) {
        int x = (particles[i].pos.x / 100.0f + 0.5f) * width;
        int y = (particles[i].pos.y / 100.0f + 0.5f) * height;
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int body = particle_body[i];
            if (body >= 0 && body < 10) {
                grid[y][x] = '0' + body;  // Number for body ID
            } else {
                grid[y][x] = '.';  // Singleton
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
    std::cout << "=== Composite Bodies Test ===\n\n";
    std::cout << "This test demonstrates connected component detection and\n";
    std::cout << "computation of composite body properties.\n\n";
    
    // Test 1: Simple connected components
    std::cout << "TEST 1: Component Detection\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    std::vector<Particle> particles;
    std::vector<Spring> springs;
    
    // Create three separate groups
    // Group 1: Triangle (0-2)
    particles.push_back(Particle(-20, 0, 2.0f));
    particles.push_back(Particle(-10, 0, 2.0f));
    particles.push_back(Particle(-15, 8, 2.0f));
    springs.push_back(Spring(0, 1, 10.0f));
    springs.push_back(Spring(1, 2, 10.0f));
    springs.push_back(Spring(2, 0, 10.0f));
    
    // Group 2: Line (3-6)
    for (int i = 0; i < 4; i++) {
        particles.push_back(Particle(10 + i*5, 0, 1.0f));
        if (i > 0) {
            springs.push_back(Spring(3+i-1, 3+i, 5.0f));
        }
    }
    
    // Group 3: Square (7-10)
    particles.push_back(Particle(-10, -20, 1.5f));
    particles.push_back(Particle(10, -20, 1.5f));
    particles.push_back(Particle(10, -10, 1.5f));
    particles.push_back(Particle(-10, -10, 1.5f));
    springs.push_back(Spring(7, 8, 20.0f));
    springs.push_back(Spring(8, 9, 10.0f));
    springs.push_back(Spring(9, 10, 20.0f));
    springs.push_back(Spring(10, 7, 10.0f));
    
    // Single particle (11)
    particles.push_back(Particle(0, 20, 5.0f));
    
    auto bodies = findCompositeBodies(particles, springs);
    
    std::cout << "Found " << bodies.size() << " composite bodies:\n\n";
    
    for (size_t i = 0; i < bodies.size(); i++) {
        std::cout << "Body " << i << ":\n";
        bodies[i].print();
        std::cout << "\n";
    }
    
    visualize(particles, bodies);
    std::cout << "Numbers show body ID, dots are singletons\n\n";
    
    // Test 2: Rotating body
    std::cout << "TEST 2: Rotating Composite\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    springs.clear();
    
    // Create spinning wheel
    int n = 8;
    for (int i = 0; i < n; i++) {
        float angle = 2.0f * M_PI * i / n;
        float r = 15.0f;
        Particle p(r * cos(angle), r * sin(angle), 1.0f);
        
        // Give it spin
        float v = 20.0f;
        p.vel.x = -v * sin(angle);
        p.vel.y = v * cos(angle);
        
        particles.push_back(p);
    }
    
    // Connect in a ring
    for (int i = 0; i < n; i++) {
        springs.push_back(Spring(i, (i+1) % n, 12.0f, 200.0f));
    }
    
    // Add spokes
    Particle center(0, 0, 2.0f);
    particles.push_back(center);
    for (int i = 0; i < n; i++) {
        springs.push_back(Spring(i, n, 15.0f, 100.0f));
    }
    
    std::cout << "Created spinning wheel structure\n\n";
    
    // Simulate and track properties
    float dt = 0.01f;
    for (int step = 0; step < 100; step++) {
        // Apply spring forces
        applySpringForces(particles, springs, dt);
        
        // Update positions
        for (auto& p : particles) {
            p.pos += p.vel * dt;
        }
        
        if (step % 50 == 0) {
            auto wheel = findCompositeBodies(particles, springs);
            std::cout << "Step " << step << ":\n";
            if (!wheel.empty()) {
                wheel[0].print();
            }
            visualize(particles, wheel);
        }
    }
    
    // Test 3: Structural failure
    std::cout << "\nTEST 3: Structural Failure\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    springs.clear();
    
    // Create bridge-like structure
    for (int i = 0; i < 10; i++) {
        particles.push_back(Particle(-25 + i*5, 0, 1.0f));
    }
    
    // Connect with springs
    for (int i = 0; i < 9; i++) {
        springs.push_back(Spring(i, i+1, 5.0f, 50.0f));  // Weak springs
    }
    
    // Apply increasing stress
    std::cout << "Applying stress to structure...\n\n";
    
    for (int i = 0; i < 10; i++) {
        // Pull ends apart
        particles[0].vel.x = -10.0f * i;
        particles[9].vel.x = 10.0f * i;
        
        // Simulate
        for (int step = 0; step < 10; step++) {
            applySpringForces(particles, springs, dt);
            for (auto& p : particles) {
                p.pos += p.vel * dt;
                p.vel *= 0.9f;  // Damping
            }
        }
        
        auto current_bodies = findCompositeBodies(particles, springs);
        if (current_bodies.size() > 1) {
            std::cout << "Structure broke into " << current_bodies.size() << " pieces!\n";
            for (size_t j = 0; j < current_bodies.size(); j++) {
                std::cout << "Piece " << j << ": " 
                          << current_bodies[j].particle_indices.size() << " particles\n";
            }
            break;
        }
    }
    
    // Test 4: Temperature from motion
    std::cout << "\nTEST 4: Internal Temperature\n";
    std::cout << std::string(50, '-') << "\n\n";
    
    particles.clear();
    springs.clear();
    
    // Create vibrating cluster
    for (int i = 0; i < 6; i++) {
        Particle p(0, 0, 1.0f);
        // Random velocities
        p.vel.x = (rand() % 100 - 50) / 10.0f;
        p.vel.y = (rand() % 100 - 50) / 10.0f;
        particles.push_back(p);
    }
    
    // Fully connected
    for (int i = 0; i < 6; i++) {
        for (int j = i+1; j < 6; j++) {
            springs.push_back(Spring(i, j, 10.0f, 20.0f));
        }
    }
    
    std::cout << "Cluster with random velocities:\n";
    auto cluster = findCompositeBodies(particles, springs);
    cluster[0].print();
    
    std::cout << "\nInternal temperature comes from velocity dispersion!\n";
    std::cout << "Higher internal temp = particles moving relative to COM\n";
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Composite body system demonstrates:\n";
    std::cout << "✓ Fast connected component detection (Union-Find)\n";
    std::cout << "✓ Aggregate property computation\n";
    std::cout << "✓ Shape analysis (elongation, compactness)\n";
    std::cout << "✓ Internal temperature from motion\n";
    std::cout << "✓ Structural integrity tracking\n";
    std::cout << "\nComposite bodies enable rigid body dynamics from particles!\n";
    
    return 0;
}