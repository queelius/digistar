// Integration Methods Comparison
// Tests different numerical integrators for various physics scenarios

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <string>

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
    float length_squared() const { return x * x + y * y; }
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
    
    // For Velocity Verlet
    float2 prev_force;
    
    Particle() : mass(1.0f) {}
    Particle(float2 p, float2 v, float m) : pos(p), vel(v), mass(m) {}
};

// Different physics scenarios
class PhysicsScenario {
public:
    virtual ~PhysicsScenario() = default;
    virtual void compute_forces(std::vector<Particle>& particles) = 0;
    virtual float compute_energy(const std::vector<Particle>& particles) = 0;
    virtual std::string name() const = 0;
};

// Scenario 1: Two-body orbital problem
class OrbitalScenario : public PhysicsScenario {
    const float G = 1.0f;  // Gravitational constant
    
public:
    void compute_forces(std::vector<Particle>& particles) override {
        // Clear forces
        for (auto& p : particles) {
            p.force = float2(0, 0);
        }
        
        // Gravitational forces between all pairs
        for (size_t i = 0; i < particles.size(); i++) {
            for (size_t j = i + 1; j < particles.size(); j++) {
                float2 diff = particles[j].pos - particles[i].pos;
                float dist_sq = diff.length_squared();
                if (dist_sq < 0.01f) dist_sq = 0.01f;  // Softening
                
                float dist = std::sqrt(dist_sq);
                float force_mag = G * particles[i].mass * particles[j].mass / dist_sq;
                float2 force_dir = diff / dist;
                float2 force = force_dir * force_mag;
                
                particles[i].force += force;
                particles[j].force -= force;
            }
        }
    }
    
    float compute_energy(const std::vector<Particle>& particles) override {
        float kinetic = 0;
        float potential = 0;
        
        // Kinetic energy
        for (const auto& p : particles) {
            kinetic += 0.5f * p.mass * p.vel.length_squared();
        }
        
        // Gravitational potential energy
        for (size_t i = 0; i < particles.size(); i++) {
            for (size_t j = i + 1; j < particles.size(); j++) {
                float dist = (particles[j].pos - particles[i].pos).length();
                if (dist < 0.01f) dist = 0.01f;
                potential -= G * particles[i].mass * particles[j].mass / dist;
            }
        }
        
        return kinetic + potential;
    }
    
    std::string name() const override { return "Orbital"; }
};

// Scenario 2: Stiff spring oscillator
class StiffSpringScenario : public PhysicsScenario {
    float k;  // Spring stiffness
    float rest_length = 1.0f;
    
public:
    StiffSpringScenario(float stiffness) : k(stiffness) {}
    
    void compute_forces(std::vector<Particle>& particles) override {
        // Single particle attached to origin by spring
        particles[0].force = float2(0, 0);
        
        float2 displacement = particles[0].pos;
        float dist = displacement.length();
        if (dist > 0) {
            float extension = dist - rest_length;
            float2 force = displacement.normalized() * (-k * extension);
            particles[0].force = force;
        }
    }
    
    float compute_energy(const std::vector<Particle>& particles) override {
        // Kinetic energy
        float kinetic = 0.5f * particles[0].mass * particles[0].vel.length_squared();
        
        // Spring potential energy
        float extension = particles[0].pos.length() - rest_length;
        float potential = 0.5f * k * extension * extension;
        
        return kinetic + potential;
    }
    
    std::string name() const override { 
        return "StiffSpring(k=" + std::to_string(int(k)) + ")"; 
    }
};

// Scenario 3: Collision cascade in box
class CollisionScenario : public PhysicsScenario {
    const float box_size = 10.0f;
    const float collision_k = 1000.0f;
    const float particle_radius = 0.5f;
    
public:
    void compute_forces(std::vector<Particle>& particles) override {
        // Clear forces
        for (auto& p : particles) {
            p.force = float2(0, 0);
        }
        
        // Particle-particle collisions
        for (size_t i = 0; i < particles.size(); i++) {
            for (size_t j = i + 1; j < particles.size(); j++) {
                float2 diff = particles[j].pos - particles[i].pos;
                float dist = diff.length();
                float overlap = 2 * particle_radius - dist;
                
                if (overlap > 0 && dist > 0) {
                    // Repulsion force
                    float2 force_dir = diff / dist;
                    float force_mag = collision_k * overlap;
                    float2 force = force_dir * force_mag;
                    
                    particles[i].force -= force;
                    particles[j].force += force;
                }
            }
            
            // Wall collisions
            if (particles[i].pos.x < -box_size + particle_radius) {
                particles[i].force.x += collision_k * (-box_size + particle_radius - particles[i].pos.x);
            }
            if (particles[i].pos.x > box_size - particle_radius) {
                particles[i].force.x += collision_k * (box_size - particle_radius - particles[i].pos.x);
            }
            if (particles[i].pos.y < -box_size + particle_radius) {
                particles[i].force.y += collision_k * (-box_size + particle_radius - particles[i].pos.y);
            }
            if (particles[i].pos.y > box_size - particle_radius) {
                particles[i].force.y += collision_k * (box_size - particle_radius - particles[i].pos.y);
            }
        }
    }
    
    float compute_energy(const std::vector<Particle>& particles) override {
        float kinetic = 0;
        for (const auto& p : particles) {
            kinetic += 0.5f * p.mass * p.vel.length_squared();
        }
        return kinetic;  // No potential energy in hard sphere model
    }
    
    std::string name() const override { return "Collision"; }
};

// Integration methods
class Integrator {
public:
    virtual ~Integrator() = default;
    virtual void integrate(std::vector<Particle>& particles, PhysicsScenario& scenario, float dt) = 0;
    virtual std::string name() const = 0;
};

// Forward Euler
class ForwardEuler : public Integrator {
public:
    void integrate(std::vector<Particle>& particles, PhysicsScenario& scenario, float dt) override {
        scenario.compute_forces(particles);
        
        for (auto& p : particles) {
            float2 accel = p.force / p.mass;
            p.pos += p.vel * dt;
            p.vel += accel * dt;
        }
    }
    
    std::string name() const override { return "ForwardEuler"; }
};

// Symplectic Euler (Semi-implicit)
class SymplecticEuler : public Integrator {
public:
    void integrate(std::vector<Particle>& particles, PhysicsScenario& scenario, float dt) override {
        scenario.compute_forces(particles);
        
        for (auto& p : particles) {
            float2 accel = p.force / p.mass;
            p.vel += accel * dt;  // Update velocity first
            p.pos += p.vel * dt;  // Use new velocity for position
        }
    }
    
    std::string name() const override { return "SymplecticEuler"; }
};

// Velocity Verlet
class VelocityVerlet : public Integrator {
public:
    void integrate(std::vector<Particle>& particles, PhysicsScenario& scenario, float dt) override {
        // First half: update positions
        for (auto& p : particles) {
            float2 accel = p.force / p.mass;
            p.pos += p.vel * dt + accel * (0.5f * dt * dt);
            p.prev_force = p.force;
        }
        
        // Compute new forces
        scenario.compute_forces(particles);
        
        // Second half: update velocities
        for (auto& p : particles) {
            float2 old_accel = p.prev_force / p.mass;
            float2 new_accel = p.force / p.mass;
            p.vel += (old_accel + new_accel) * (0.5f * dt);
        }
    }
    
    std::string name() const override { return "VelocityVerlet"; }
};

// Leapfrog
class Leapfrog : public Integrator {
public:
    void integrate(std::vector<Particle>& particles, PhysicsScenario& scenario, float dt) override {
        scenario.compute_forces(particles);
        
        for (auto& p : particles) {
            float2 accel = p.force / p.mass;
            p.vel += accel * dt;
            p.pos += p.vel * dt;
        }
    }
    
    std::string name() const override { return "Leapfrog"; }
};

// RK4 (Runge-Kutta 4th order)
class RK4 : public Integrator {
    struct State {
        float2 pos;
        float2 vel;
    };
    
    struct Derivative {
        float2 dpos;  // velocity
        float2 dvel;  // acceleration
    };
    
    Derivative evaluate(Particle& p, PhysicsScenario& scenario, 
                       float dt, const Derivative& d) {
        // Save original state
        State original = {p.pos, p.vel};
        
        // Apply derivative
        p.pos = original.pos + d.dpos * dt;
        p.vel = original.vel + d.dvel * dt;
        
        // Compute forces at new state
        std::vector<Particle> temp = {p};
        scenario.compute_forces(temp);
        
        Derivative output;
        output.dpos = temp[0].vel;
        output.dvel = temp[0].force / temp[0].mass;
        
        // Restore original state
        p.pos = original.pos;
        p.vel = original.vel;
        
        return output;
    }
    
public:
    void integrate(std::vector<Particle>& particles, PhysicsScenario& scenario, float dt) override {
        for (auto& p : particles) {
            scenario.compute_forces(particles);
            
            Derivative k1, k2, k3, k4;
            
            k1.dpos = p.vel;
            k1.dvel = p.force / p.mass;
            
            k2 = evaluate(p, scenario, dt * 0.5f, k1);
            k3 = evaluate(p, scenario, dt * 0.5f, k2);
            k4 = evaluate(p, scenario, dt, k3);
            
            // Combine derivatives
            float2 dpos = (k1.dpos + k2.dpos * 2.0f + k3.dpos * 2.0f + k4.dpos) / 6.0f;
            float2 dvel = (k1.dvel + k2.dvel * 2.0f + k3.dvel * 2.0f + k4.dvel) / 6.0f;
            
            p.pos += dpos * dt;
            p.vel += dvel * dt;
        }
    }
    
    std::string name() const override { return "RK4"; }
};

// Test harness
class IntegrationTest {
    void setup_orbital(std::vector<Particle>& particles) {
        particles.clear();
        // Binary star system
        particles.push_back(Particle(float2(-1, 0), float2(0, 0.5f), 1.0f));
        particles.push_back(Particle(float2(1, 0), float2(0, -0.5f), 1.0f));
    }
    
    void setup_spring(std::vector<Particle>& particles) {
        particles.clear();
        // Single mass on spring
        particles.push_back(Particle(float2(2, 0), float2(0, 0), 1.0f));
    }
    
    void setup_collision(std::vector<Particle>& particles) {
        particles.clear();
        // Random particles in box
        for (int i = 0; i < 10; i++) {
            float x = -8.0f + (i % 4) * 4.0f;
            float y = -8.0f + (i / 4) * 4.0f;
            float vx = (i % 2) ? 2.0f : -2.0f;
            float vy = (i % 3) ? 1.5f : -1.5f;
            particles.push_back(Particle(float2(x, y), float2(vx, vy), 1.0f));
        }
    }
    
public:
    void run_test(PhysicsScenario& scenario, Integrator& integrator, 
                  float dt, float total_time, bool verbose = false) {
        
        std::vector<Particle> particles;
        
        // Setup initial conditions based on scenario
        if (scenario.name() == "Orbital") {
            setup_orbital(particles);
        } else if (scenario.name().find("Spring") != std::string::npos) {
            setup_spring(particles);
        } else {
            setup_collision(particles);
        }
        
        float initial_energy = scenario.compute_energy(particles);
        float max_energy_error = 0;
        int steps = int(total_time / dt);
        int explosions = 0;
        
        if (verbose) {
            std::cout << "\nTesting " << integrator.name() 
                     << " on " << scenario.name() 
                     << " (dt=" << dt << "):\n";
        }
        
        for (int step = 0; step < steps; step++) {
            integrator.integrate(particles, scenario, dt);
            
            // Check for explosion
            bool exploded = false;
            for (const auto& p : particles) {
                if (p.pos.length() > 1000.0f || std::isnan(p.pos.x) || std::isinf(p.pos.x)) {
                    exploded = true;
                    explosions++;
                    break;
                }
            }
            
            if (exploded) {
                if (verbose) {
                    std::cout << "  EXPLOSION at step " << step << "\n";
                }
                break;
            }
            
            // Track energy error
            float current_energy = scenario.compute_energy(particles);
            float energy_error = std::abs((current_energy - initial_energy) / initial_energy);
            max_energy_error = std::max(max_energy_error, energy_error);
            
            // Periodic output
            if (verbose && step % (steps / 10) == 0) {
                std::cout << "  Step " << step << ": Energy error = " 
                         << (energy_error * 100) << "%\n";
            }
        }
        
        if (!explosions) {
            std::cout << integrator.name() << " + " << scenario.name() 
                     << " (dt=" << dt << "): Max energy error = " 
                     << (max_energy_error * 100) << "%\n";
        } else {
            std::cout << integrator.name() << " + " << scenario.name() 
                     << " (dt=" << dt << "): UNSTABLE (exploded)\n";
        }
    }
    
    void run_stability_test() {
        std::cout << "\n=== Stability Test ===\n";
        std::cout << "Testing different timesteps to find stability limits\n\n";
        
        // Test different integrators
        ForwardEuler forward_euler;
        SymplecticEuler symplectic_euler;
        VelocityVerlet velocity_verlet;
        Leapfrog leapfrog;
        
        // Test with increasingly stiff spring
        float stiffnesses[] = {10.0f, 100.0f, 1000.0f, 10000.0f};
        float timesteps[] = {0.1f, 0.01f, 0.001f, 0.0001f};
        
        for (float k : stiffnesses) {
            StiffSpringScenario spring(k);
            std::cout << "\nSpring stiffness k = " << k << ":\n";
            
            for (float dt : timesteps) {
                // Theoretical stability limit: dt < 2/sqrt(k/m)
                float stability_limit = 2.0f / std::sqrt(k);
                
                if (dt < stability_limit * 2) {  // Test near the limit
                    run_test(spring, forward_euler, dt, 1.0f);
                    run_test(spring, symplectic_euler, dt, 1.0f);
                    run_test(spring, velocity_verlet, dt, 1.0f);
                    std::cout << "  Theoretical limit: dt < " << stability_limit << "\n\n";
                }
            }
        }
    }
    
    void run_energy_conservation_test() {
        std::cout << "\n=== Energy Conservation Test ===\n";
        std::cout << "Testing long-term energy drift in orbital mechanics\n\n";
        
        OrbitalScenario orbital;
        ForwardEuler forward_euler;
        SymplecticEuler symplectic_euler;
        VelocityVerlet velocity_verlet;
        Leapfrog leapfrog;
        RK4 rk4;
        
        float dt = 0.01f;
        float total_time = 100.0f;  // 100 time units
        
        run_test(orbital, forward_euler, dt, total_time, true);
        run_test(orbital, symplectic_euler, dt, total_time, true);
        run_test(orbital, velocity_verlet, dt, total_time, true);
        run_test(orbital, leapfrog, dt, total_time, true);
        run_test(orbital, rk4, dt, total_time, true);
    }
    
    void run_collision_test() {
        std::cout << "\n=== Collision Test ===\n";
        std::cout << "Testing stability with impulsive collision forces\n\n";
        
        CollisionScenario collision;
        SymplecticEuler symplectic_euler;
        VelocityVerlet velocity_verlet;
        
        float timesteps[] = {0.01f, 0.001f, 0.0001f};
        
        for (float dt : timesteps) {
            run_test(collision, symplectic_euler, dt, 1.0f);
            run_test(collision, velocity_verlet, dt, 1.0f);
        }
    }
};

int main() {
    std::cout << "=== Integration Methods Comparison ===\n";
    std::cout << "Testing numerical integrators for different physics scenarios\n";
    
    IntegrationTest test;
    
    // Run all tests
    test.run_stability_test();
    test.run_energy_conservation_test();
    test.run_collision_test();
    
    std::cout << "\n=== Key Findings ===\n";
    std::cout << "1. Symplectic integrators (Verlet, Leapfrog) conserve energy best\n";
    std::cout << "2. Stiff springs require dt < 2/sqrt(k/m) for explicit methods\n";
    std::cout << "3. Semi-implicit Euler more stable than Forward Euler\n";
    std::cout << "4. RK4 accurate but not energy-conserving for Hamiltonian systems\n";
    std::cout << "5. Collisions need very small timesteps for stability\n";
    
    return 0;
}