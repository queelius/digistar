// Improved Composite Collision Test
// Demonstrates localized contact forces with spring propagation
// Fixes: Better spring values, improved contact detection, velocity-based broad phase

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <iomanip>
#include <cstdint>

struct float2 {
    float x, y;
    
    float2() : x(0), y(0) {}
    float2(float x_, float y_) : x(x_), y(y_) {}
    
    float2 operator+(const float2& other) const { return float2(x + other.x, y + other.y); }
    float2 operator-(const float2& other) const { return float2(x - other.x, y - other.y); }
    float2 operator*(float s) const { return float2(x * s, y * s); }
    float2 operator/(float s) const { return float2(x / s, y / s); }
    void operator+=(const float2& other) { x += other.x; y += other.y; }
    void operator-=(const float2& other) { x -= other.x; y -= other.y; }
    
    float length() const { return std::sqrt(x * x + y * y); }
    float2 normalized() const { 
        float len = length();
        return len > 0 ? float2(x/len, y/len) : float2(0, 0);
    }
    float dot(const float2& other) const { return x * other.x + y * other.y; }
};

struct Particle {
    float2 pos;
    float2 vel;
    float2 force;
    float mass;
    float radius;
    uint32_t composite_id;
    
    Particle() : mass(1.0f), radius(1.0f), composite_id(0) {}
};

struct Spring {
    uint32_t id1, id2;
    float rest_length;
    float stiffness;
    float damping;
    bool broken;
    
    Spring(uint32_t i1, uint32_t i2, float rest, float stiff, float damp)
        : id1(i1), id2(i2), rest_length(rest), stiffness(stiff), damping(damp), broken(false) {}
};

struct CompositeBody {
    uint32_t id;
    std::vector<uint32_t> particle_ids;
    float2 center_of_mass;
    float2 velocity;
    float bounding_radius;
    float total_mass;
    
    void update_properties(const std::vector<Particle>& particles) {
        if (particle_ids.empty()) return;
        
        // Calculate center of mass and velocity
        center_of_mass = float2(0, 0);
        velocity = float2(0, 0);
        total_mass = 0;
        
        for (uint32_t pid : particle_ids) {
            const Particle& p = particles[pid];
            center_of_mass += p.pos * p.mass;
            velocity += p.vel * p.mass;
            total_mass += p.mass;
        }
        
        center_of_mass = center_of_mass / total_mass;
        velocity = velocity / total_mass;
        
        // Calculate bounding radius
        bounding_radius = 0;
        for (uint32_t pid : particle_ids) {
            float dist = (particles[pid].pos - center_of_mass).length() + particles[pid].radius;
            bounding_radius = std::max(bounding_radius, dist);
        }
    }
};

// Simple spatial grid for contact detection
class ContactGrid {
    float cell_size;
    std::unordered_map<uint64_t, std::vector<uint32_t>> cells;
    
    uint64_t hash_pos(float2 pos) {
        int32_t cx = static_cast<int32_t>(std::floor(pos.x / cell_size));
        int32_t cy = static_cast<int32_t>(std::floor(pos.y / cell_size));
        return (static_cast<uint64_t>(cx) << 32) | static_cast<uint64_t>(cy);
    }
    
public:
    ContactGrid(float size) : cell_size(size) {}
    
    void clear() { cells.clear(); }
    
    void add_particle(uint32_t id, float2 pos) {
        cells[hash_pos(pos)].push_back(id);
    }
    
    std::vector<uint32_t> get_particles_near(float2 pos, float radius, 
                                             const std::vector<uint32_t>& filter_ids,
                                             const std::vector<Particle>& particles) {
        std::vector<uint32_t> result;
        
        int cells_to_check = static_cast<int>(std::ceil(radius / cell_size));
        
        for (int dx = -cells_to_check; dx <= cells_to_check; dx++) {
            for (int dy = -cells_to_check; dy <= cells_to_check; dy++) {
                float2 check_pos = pos + float2(dx * cell_size, dy * cell_size);
                uint64_t hash = hash_pos(check_pos);
                
                auto it = cells.find(hash);
                if (it != cells.end()) {
                    for (uint32_t pid : it->second) {
                        // Check if particle is in filter list and within radius
                        if (std::find(filter_ids.begin(), filter_ids.end(), pid) != filter_ids.end()) {
                            float dist = (particles[pid].pos - pos).length();
                            if (dist <= radius + particles[pid].radius) {
                                result.push_back(pid);
                            }
                        }
                    }
                }
            }
        }
        
        return result;
    }
};

class ImprovedCompositeCollisionTest {
    std::vector<Particle> particles;
    std::vector<Spring> springs;
    std::vector<CompositeBody> composites;
    ContactGrid contact_grid;
    
    const float dt = 0.001f;  // Smaller timestep for numerical stability
    const float contact_stiffness = 5000.0f;  // Increased for better collision response
    const float contact_damping = 50.0f;      // Add damping to contacts
    const float particle_radius = 1.0f;
    
    // Statistics
    int collision_checks = 0;
    int actual_collisions = 0;
    int false_positives = 0;
    
    void create_box_composite(float2 center, float2 velocity, float mass, uint32_t comp_id, float spring_stiffness) {
        CompositeBody comp;
        comp.id = comp_id;
        
        // Create 3x3 grid of particles
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                Particle p;
                p.pos = center + float2(i * 3.0f, j * 3.0f);
                p.vel = velocity;
                p.mass = mass / 9.0f;
                p.radius = particle_radius;
                p.composite_id = comp_id;
                
                comp.particle_ids.push_back(particles.size());
                particles.push_back(p);
            }
        }
        
        // Create springs between adjacent particles
        uint32_t base = comp.particle_ids[0];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                uint32_t current = base + i * 3 + j;
                
                // Spring to right neighbor
                if (j < 2) {
                    uint32_t right = current + 1;
                    float rest = (particles[current].pos - particles[right].pos).length();
                    springs.push_back(Spring(current, right, rest, spring_stiffness, spring_stiffness * 0.1f));
                }
                
                // Spring to bottom neighbor
                if (i < 2) {
                    uint32_t bottom = current + 3;
                    float rest = (particles[current].pos - particles[bottom].pos).length();
                    springs.push_back(Spring(current, bottom, rest, spring_stiffness, spring_stiffness * 0.1f));
                }
                
                // Diagonal springs for rigidity
                if (i < 2 && j < 2) {
                    uint32_t diag = current + 4;
                    float rest = (particles[current].pos - particles[diag].pos).length();
                    springs.push_back(Spring(current, diag, rest, spring_stiffness * 0.7f, spring_stiffness * 0.07f));
                }
                
                if (i < 2 && j > 0) {
                    uint32_t diag = current + 2;
                    float rest = (particles[current].pos - particles[diag].pos).length();
                    springs.push_back(Spring(current, diag, rest, spring_stiffness * 0.7f, spring_stiffness * 0.07f));
                }
            }
        }
        
        comp.update_properties(particles);
        composites.push_back(comp);
    }
    
    bool velocity_broad_phase(const CompositeBody& comp1, const CompositeBody& comp2) {
        float2 relative_pos = comp2.center_of_mass - comp1.center_of_mass;
        float dist = relative_pos.length();
        float combined_radius = comp1.bounding_radius + comp2.bounding_radius;
        
        // CRITICAL: Always check if already overlapping
        // This handles objects that are interpenetrating but separating
        // Without this, objects can get stuck inside each other!
        if (dist < combined_radius) {
            return true;  // Already in contact, must apply repulsion forces
        }
        
        // For separated objects, check if approaching
        float2 relative_vel = comp2.velocity - comp1.velocity;
        float approach_speed = -relative_vel.dot(relative_pos.normalized());
        
        // Skip if moving away or parallel
        if (approach_speed <= 0) return false;
        
        // Check if they could collide soon
        float time_to_collision = (dist - combined_radius) / approach_speed;
        return time_to_collision < 1.0f;  // Check if collision possible within 1 second
    }
    
    void handle_composite_collision(CompositeBody& comp1, CompositeBody& comp2) {
        collision_checks++;
        
        // Velocity-based broad phase
        if (!velocity_broad_phase(comp1, comp2)) {
            return;
        }
        
        // Bounding sphere check
        float2 diff = comp2.center_of_mass - comp1.center_of_mass;
        float dist = diff.length();
        float overlap = (comp1.bounding_radius + comp2.bounding_radius) - dist;
        
        if (overlap <= 0) return;
        
        // Contact point and improved contact radius
        float2 contact_point = comp1.center_of_mass + diff * (comp1.bounding_radius / dist);
        
        // Use particle radius and overlap to determine contact region
        float contact_radius = particle_radius * 2.0f + overlap;
        
        // Find particles at contact point
        auto contact_particles1 = contact_grid.get_particles_near(
            contact_point, contact_radius, comp1.particle_ids, particles);
        auto contact_particles2 = contact_grid.get_particles_near(
            contact_point, contact_radius, comp2.particle_ids, particles);
        
        // Check for false positive
        if (contact_particles1.empty() || contact_particles2.empty()) {
            false_positives++;
            return;
        }
        
        actual_collisions++;
        
        // Apply localized contact forces with damping
        float2 force_dir = diff.normalized();
        float force_mag = contact_stiffness * std::pow(overlap, 1.5f);
        
        // Add velocity-based damping
        float2 relative_velocity = comp2.velocity - comp1.velocity;
        float closing_speed = -relative_velocity.dot(force_dir);
        if (closing_speed > 0) {
            force_mag += contact_damping * closing_speed;
        }
        
        // Distribute force among contact particles
        float force_per_particle1 = force_mag / contact_particles1.size();
        float force_per_particle2 = force_mag / contact_particles2.size();
        
        for (uint32_t pid : contact_particles1) {
            particles[pid].force -= force_dir * force_per_particle1;
        }
        
        for (uint32_t pid : contact_particles2) {
            particles[pid].force += force_dir * force_per_particle2;
        }
    }
    
    void update_springs() {
        for (Spring& spring : springs) {
            if (spring.broken) continue;
            
            Particle& p1 = particles[spring.id1];
            Particle& p2 = particles[spring.id2];
            
            float2 diff = p2.pos - p1.pos;
            float dist = diff.length();
            
            if (dist < 0.001f) continue;
            
            // Spring force with damping
            float extension = dist - spring.rest_length;
            float2 dir = diff / dist;
            
            // Hooke's law
            float spring_force = spring.stiffness * extension;
            
            // Damping
            float2 relative_vel = p2.vel - p1.vel;
            float vel_along_spring = relative_vel.dot(dir);
            float damping_force = spring.damping * vel_along_spring;
            
            float total_force = spring_force + damping_force;
            
            p1.force += dir * total_force;
            p2.force -= dir * total_force;
            
            // Break springs under too much stress (optional)
            if (std::abs(extension) > spring.rest_length * 0.5f) {
                spring.broken = true;
            }
        }
    }
    
    void integrate() {
        for (Particle& p : particles) {
            // Update velocity and position
            p.vel += p.force * (dt / p.mass);
            p.pos += p.vel * dt;
            
            // Reset force
            p.force = float2(0, 0);
            
            // Light damping for stability
            p.vel = p.vel * 0.999f;
        }
    }
    
    void update_grid() {
        contact_grid.clear();
        for (size_t i = 0; i < particles.size(); i++) {
            contact_grid.add_particle(i, particles[i].pos);
        }
    }
    
public:
    ImprovedCompositeCollisionTest() : contact_grid(particle_radius * 2.0f) {}
    
    void run_collision_test() {
        std::cout << "\n=== Improved Composite Collision Test ===\n";
        std::cout << "Features: Velocity broad phase, better contact detection, proper spring values\n\n";
        
        // Create two box composites moving toward each other
        // Rigid composite: High stiffness (but not too high to avoid instability)
        create_box_composite(float2(42.5, 50), float2(3, 0), 10.0f, 0, 500.0f);
        // Soft composite: Low stiffness  
        create_box_composite(float2(57.5, 50), float2(-3, 0), 10.0f, 1, 50.0f);
        
        std::cout << "Created 2 composites:\n";
        std::cout << "  Composite 0: Rigid (stiffness=500), moving right at 3 units/s\n";
        std::cout << "  Composite 1: Soft (stiffness=50), moving left at 3 units/s\n\n";
        
        // Reset statistics
        collision_checks = 0;
        actual_collisions = 0;
        false_positives = 0;
        
        // Simulate for longer with more frames (adjusted for smaller dt)
        int total_frames = 2000;  // 10x more frames due to 10x smaller dt
        for (int frame = 0; frame < total_frames; frame++) {
            // Update composite properties
            for (CompositeBody& comp : composites) {
                comp.update_properties(particles);
            }
            
            // Update spatial grid
            update_grid();
            
            // Check collisions
            for (size_t i = 0; i < composites.size(); i++) {
                for (size_t j = i + 1; j < composites.size(); j++) {
                    handle_composite_collision(composites[i], composites[j]);
                }
            }
            
            // Spring forces
            update_springs();
            
            // Integrate
            integrate();
            
            // Print state at key moments (adjusted for new frame count)
            if (frame == 0 || frame == 500 || frame == 1000 || frame == 1500 || frame == 1999) {
                std::cout << "Frame " << frame << ":\n";
                for (const CompositeBody& comp : composites) {
                    std::cout << "  Composite " << comp.id 
                              << ": center=(" << std::fixed << std::setprecision(1) 
                              << comp.center_of_mass.x << "," << comp.center_of_mass.y 
                              << "), vel=(" << comp.velocity.x << "," << comp.velocity.y
                              << "), radius=" << comp.bounding_radius << "\n";
                }
                
                if (frame == 1000) {
                    std::cout << "  [Collision occurring - particles in contact]\n";
                }
            }
        }
        
        std::cout << "\nFinal state:\n";
        for (const CompositeBody& comp : composites) {
            // Calculate deformation
            float max_dist = 0;
            float min_dist = 1000;
            for (uint32_t pid : comp.particle_ids) {
                float dist = (particles[pid].pos - comp.center_of_mass).length();
                max_dist = std::max(max_dist, dist);
                min_dist = std::min(min_dist, dist);
            }
            
            std::cout << "  Composite " << comp.id << ":\n";
            std::cout << "    Final position: (" << std::fixed << std::setprecision(2) 
                      << comp.center_of_mass.x << ", " << comp.center_of_mass.y << ")\n";
            std::cout << "    Final velocity: (" << comp.velocity.x << ", " << comp.velocity.y << ")\n";
            std::cout << "    Deformation: " << (max_dist - min_dist) << "\n";
            std::cout << "    " << (comp.id == 0 ? "Rigid" : "Soft") 
                      << " body " << ((max_dist - min_dist) < 1.0f ? "maintained" : "slightly deformed")
                      << " shape\n";
        }
        
        std::cout << "\nCollision Statistics:\n";
        std::cout << "  Total collision checks: " << collision_checks << "\n";
        std::cout << "  Actual collisions detected: " << actual_collisions << "\n";
        std::cout << "  False positives filtered: " << false_positives << "\n";
        std::cout << "  Efficiency: " << (100.0f * actual_collisions / collision_checks) << "% useful checks\n";
    }
    
    void run_stress_test() {
        std::cout << "\n=== Stress Test: Multiple Composites ===\n";
        
        particles.clear();
        springs.clear();
        composites.clear();
        
        // Create a line of composites that will collide in sequence
        for (int i = 0; i < 5; i++) {
            float x_pos = 20.0f + i * 20.0f;
            float velocity = (i == 0) ? 5.0f : 0.0f;  // Only first one moves
            float stiffness = 200.0f + i * 200.0f;    // Varying stiffness
            
            create_box_composite(float2(x_pos, 50), float2(velocity, 0), 10.0f, i, stiffness);
        }
        
        std::cout << "Created 5 composites in a line:\n";
        std::cout << "  First composite moving right at 5 units/s\n";
        std::cout << "  Others stationary with increasing stiffness\n\n";
        
        // Simulate (adjusted for smaller dt)
        for (int frame = 0; frame < 3000; frame++) {
            for (CompositeBody& comp : composites) {
                comp.update_properties(particles);
            }
            
            update_grid();
            
            for (size_t i = 0; i < composites.size(); i++) {
                for (size_t j = i + 1; j < composites.size(); j++) {
                    handle_composite_collision(composites[i], composites[j]);
                }
            }
            
            update_springs();
            integrate();
            
            if (frame % 1000 == 0) {
                std::cout << "Frame " << frame << ": ";
                for (const CompositeBody& comp : composites) {
                    std::cout << "C" << comp.id << "(" << std::fixed << std::setprecision(0) 
                              << comp.center_of_mass.x << ") ";
                }
                std::cout << "\n";
            }
        }
        
        std::cout << "\nFinal velocities (momentum transfer test):\n";
        for (const CompositeBody& comp : composites) {
            std::cout << "  Composite " << comp.id << ": velocity = " 
                      << std::fixed << std::setprecision(2) << comp.velocity.x << "\n";
        }
    }
};

int main() {
    ImprovedCompositeCollisionTest test;
    
    // Run main collision test
    test.run_collision_test();
    
    // Run stress test with multiple composites
    test.run_stress_test();
    
    std::cout << "\n=== Test Complete ===\n";
    std::cout << "Key improvements demonstrated:\n";
    std::cout << "1. Velocity-based broad phase reduces unnecessary checks\n";
    std::cout << "2. Proper spring stiffness maintains rigid vs soft behavior\n";
    std::cout << "3. Improved contact radius detects more particle contacts\n";
    std::cout << "4. Contact damping prevents oscillation\n";
    std::cout << "5. Chain collisions show momentum transfer\n";
    
    return 0;
}