/**
 * DigiStar Graphics Viewer Demo
 * 
 * This example demonstrates the new SDL2-based graphics viewer with:
 * - Real-time particle rendering
 * - Interactive camera controls  
 * - Visual event system
 * - Performance monitoring
 * - Multiple rendering modes
 */

#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <cmath>

// DigiStar includes
#include "../src/viewer/graphics_viewer.h"
#include "../src/viewer/viewer_event_bridge.h"
#include "../src/backend/backend_interface.h"
#include "../src/backend/cpu_backend_reference.h"
#include "../src/physics/types.h"
#include "../src/physics/pools.h"

using namespace digistar;

/**
 * Simple scenario generator for demo purposes
 */
class DemoScenario {
public:
    virtual ~DemoScenario() = default;
    virtual void initialize(SimulationState& state) = 0;
    virtual void update(SimulationState& state, float dt) = 0;
    virtual std::string getName() const = 0;
};

/**
 * Solar system-like scenario
 */
class SolarSystemDemo : public DemoScenario {
private:
    std::mt19937 rng{42};
    float time_elapsed = 0.0f;
    
public:
    void initialize(SimulationState& state) override {
        // Clear existing particles
        state.particles.count = 0;
        state.springs.count = 0;
        state.contacts.count = 0;
        
        // Create central star
        size_t star_id = state.particles.count++;
        state.particles.pos_x[star_id] = 0.0f;
        state.particles.pos_y[star_id] = 0.0f;
        state.particles.vel_x[star_id] = 0.0f;
        state.particles.vel_y[star_id] = 0.0f;
        state.particles.mass[star_id] = 1000.0f;
        state.particles.radius[star_id] = 50.0f;
        state.particles.temperature[star_id] = 1.0f;
        state.particles.force_x[star_id] = 0.0f;
        state.particles.force_y[star_id] = 0.0f;
        
        // Create orbiting planets
        std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
        
        for (int i = 0; i < 8; i++) {
            size_t planet_id = state.particles.count++;
            
            // Orbital parameters
            float orbit_radius = 200.0f + i * 150.0f;
            float angle = angle_dist(rng);
            float orbital_speed = sqrt(10.0f / orbit_radius) * 200.0f;  // Simplified orbital mechanics
            
            // Position in orbit
            state.particles.pos_x[planet_id] = orbit_radius * cos(angle);
            state.particles.pos_y[planet_id] = orbit_radius * sin(angle);
            
            // Orbital velocity (perpendicular to radius)
            state.particles.vel_x[planet_id] = -orbital_speed * sin(angle);
            state.particles.vel_y[planet_id] = orbital_speed * cos(angle);
            
            // Planet properties
            state.particles.mass[planet_id] = 10.0f + i * 5.0f;
            state.particles.radius[planet_id] = 5.0f + i * 2.0f;
            state.particles.temperature[planet_id] = 0.1f;
            state.particles.force_x[planet_id] = 0.0f;
            state.particles.force_y[planet_id] = 0.0f;
        }
        
        // Add some asteroids
        for (int i = 0; i < 50; i++) {
            size_t asteroid_id = state.particles.count++;
            
            std::uniform_real_distribution<float> radius_dist(300.0f, 800.0f);
            std::uniform_real_distribution<float> mass_dist(0.1f, 2.0f);
            
            float orbit_radius = radius_dist(rng);
            float angle = angle_dist(rng);
            float orbital_speed = sqrt(10.0f / orbit_radius) * 200.0f;
            
            state.particles.pos_x[asteroid_id] = orbit_radius * cos(angle);
            state.particles.pos_y[asteroid_id] = orbit_radius * sin(angle);
            state.particles.vel_x[asteroid_id] = -orbital_speed * sin(angle) * 0.8f;  // Slightly elliptical
            state.particles.vel_y[asteroid_id] = orbital_speed * cos(angle) * 0.8f;
            
            state.particles.mass[asteroid_id] = mass_dist(rng);
            state.particles.radius[asteroid_id] = 1.0f;
            state.particles.temperature[asteroid_id] = 0.05f;
            state.particles.force_x[asteroid_id] = 0.0f;
            state.particles.force_y[asteroid_id] = 0.0f;
        }
        
        std::cout << "Initialized solar system with " << state.particles.count << " bodies\n";
    }
    
    void update(SimulationState& state, float dt) override {
        time_elapsed += dt;
        
        // Periodically add some visual events for demonstration
        static float last_event_time = 0.0f;
        if (time_elapsed - last_event_time > 3.0f) {
            // Find a random particle for event
            if (state.particles.count > 10) {
                std::uniform_int_distribution<size_t> particle_dist(10, state.particles.count - 1);
                size_t particle_id = particle_dist(rng);
                
                // Create a random event (for demo purposes)
                // In real simulation, events would come from physics
                // This is just for visual demonstration
                
                last_event_time = time_elapsed;
            }
        }
    }
    
    std::string getName() const override {
        return "Solar System Demo";
    }
};

/**
 * Particle collision demo with spring networks
 */
class CollisionDemo : public DemoScenario {
private:
    std::mt19937 rng{123};
    float time_elapsed = 0.0f;
    
public:
    void initialize(SimulationState& state) override {
        state.particles.count = 0;
        state.springs.count = 0;
        
        // Create two clusters that will collide
        createCluster(state, -400.0f, 0.0f, 100.0f, 0.0f, 30);  // Left cluster moving right
        createCluster(state, 400.0f, 0.0f, -100.0f, 0.0f, 30);  // Right cluster moving left
        
        std::cout << "Initialized collision demo with " << state.particles.count << " particles\n";
    }
    
    void createCluster(SimulationState& state, float center_x, float center_y, 
                      float vel_x, float vel_y, int count) {
        std::uniform_real_distribution<float> pos_dist(-50.0f, 50.0f);
        std::uniform_real_distribution<float> vel_dist(-10.0f, 10.0f);
        
        for (int i = 0; i < count; i++) {
            size_t particle_id = state.particles.count++;
            
            state.particles.pos_x[particle_id] = center_x + pos_dist(rng);
            state.particles.pos_y[particle_id] = center_y + pos_dist(rng);
            state.particles.vel_x[particle_id] = vel_x + vel_dist(rng);
            state.particles.vel_y[particle_id] = vel_y + vel_dist(rng);
            state.particles.mass[particle_id] = 5.0f;
            state.particles.radius[particle_id] = 3.0f;
            state.particles.temperature[particle_id] = 0.2f;
            state.particles.force_x[particle_id] = 0.0f;
            state.particles.force_y[particle_id] = 0.0f;
        }
        
        // Create springs within the cluster
        for (int i = state.particles.count - count; i < state.particles.count; i++) {
            for (int j = i + 1; j < state.particles.count; j++) {
                float dx = state.particles.pos_x[i] - state.particles.pos_x[j];
                float dy = state.particles.pos_y[i] - state.particles.pos_y[j];
                float distance = sqrt(dx*dx + dy*dy);
                
                if (distance < 30.0f && state.springs.count < state.springs.capacity) {
                    size_t spring_id = state.springs.count++;
                    state.springs.particle1_id[spring_id] = i;
                    state.springs.particle2_id[spring_id] = j;
                    state.springs.rest_length[spring_id] = distance;
                    state.springs.stiffness[spring_id] = 100.0f;
                    state.springs.damping[spring_id] = 5.0f;
                }
            }
        }
    }
    
    void update(SimulationState& state, float dt) override {
        time_elapsed += dt;
        // Events will be generated naturally by physics simulation
    }
    
    std::string getName() const override {
        return "Collision Demo";
    }
};

/**
 * Main demo application
 */
class ViewerDemo {
private:
    std::unique_ptr<GraphicsViewer> viewer;
    std::unique_ptr<ViewerEventBridge> event_bridge;
    std::unique_ptr<IBackend> physics_backend;
    std::unique_ptr<DemoScenario> current_scenario;
    
    SimulationState sim_state;
    SimulationConfig sim_config;
    PhysicsConfig physics_config;
    
    // Timing
    std::chrono::high_resolution_clock::time_point last_frame_time;
    float target_dt = 1.0f / 60.0f;  // 60 FPS target
    
    bool is_running = true;
    bool is_paused = false;
    int current_scenario_index = 0;
    
public:
    bool initialize() {
        std::cout << "Initializing DigiStar Graphics Viewer Demo...\n";
        
        // Initialize simulation configuration
        sim_config.max_particles = 10000;
        sim_config.max_springs = 50000;
        sim_config.max_contacts = 10000;
        sim_config.world_size = 5000.0f;
        sim_config.use_toroidal = true;
        
        // Initialize physics configuration
        physics_config.enabled_systems = PhysicsConfig::GRAVITY | 
                                        PhysicsConfig::CONTACTS | 
                                        PhysicsConfig::SPRINGS;
        physics_config.gravity_mode = PhysicsConfig::DIRECT_N2;  // Simple for demo
        
        // Initialize simulation state
        sim_state.particles.initialize(sim_config.max_particles);
        sim_state.springs.initialize(sim_config.max_springs);
        sim_state.contacts.initialize(sim_config.max_contacts);
        sim_state.composites.initialize(1000);
        
        // Initialize spatial indices (simplified for demo)
        sim_state.contact_index = std::make_unique<SpatialIndex>();
        
        // Create physics backend
        physics_backend = std::make_unique<CpuBackendReference>();
        if (!physics_backend) {
            std::cerr << "Failed to create physics backend\n";
            return false;
        }
        
        physics_backend->initialize(sim_config);
        
        // Create graphics viewer
        viewer = ViewerFactory::createDebugViewer();  // Use debug viewer for demo
        if (!viewer->initialize("DigiStar Graphics Demo", 1920, 1080, false)) {
            std::cerr << "Failed to initialize graphics viewer\n";
            return false;
        }
        
        // Create event bridge
        event_bridge = EventBridgeFactory::createPresentationBridge();
        
        // Note: Event bridge initialization would require actual event system
        // For this demo, we'll simulate events manually
        
        // Load initial scenario
        loadScenario(0);
        
        last_frame_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Initialization complete!\n";
        std::cout << "\nControls:\n";
        std::cout << "  WASD/Arrow Keys: Pan camera\n";
        std::cout << "  Mouse Wheel: Zoom\n";
        std::cout << "  Left Click+Drag: Pan\n";
        std::cout << "  Space: Center on center of mass\n";
        std::cout << "  1-2: Switch scenarios\n";
        std::cout << "  P: Pause/Resume\n";
        std::cout << "  R: Reset current scenario\n";
        std::cout << "  F: Toggle fullscreen\n";
        std::cout << "  H: Toggle help overlay\n";
        std::cout << "  ESC: Exit\n\n";
        
        return true;
    }
    
    void run() {
        while (is_running) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                current_time - last_frame_time);
            float dt = frame_duration.count() / 1000000.0f;
            
            // Cap frame time to prevent large steps
            dt = std::min(dt, 1.0f / 30.0f);
            
            // Process events and update
            if (!processInput()) {
                is_running = false;
                break;
            }
            
            if (!is_paused) {
                update(dt);
            }
            
            render();
            
            last_frame_time = current_time;
            
            // Simple frame rate limiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    void shutdown() {
        std::cout << "Shutting down demo...\n";
        
        if (event_bridge) {
            event_bridge->shutdown();
        }
        
        if (physics_backend) {
            physics_backend->shutdown();
        }
        
        if (viewer) {
            viewer->shutdown();
        }
        
        std::cout << "Shutdown complete.\n";
    }
    
private:
    bool processInput() {
        if (!viewer->processEvents()) {
            return false;
        }
        
        // Handle demo-specific input
        if (viewer->isKeyPressed(SDL_SCANCODE_1)) {
            loadScenario(0);
        }
        if (viewer->isKeyPressed(SDL_SCANCODE_2)) {
            loadScenario(1);
        }
        
        if (viewer->isKeyPressed(SDL_SCANCODE_P)) {
            is_paused = !is_paused;
            std::cout << (is_paused ? "Paused" : "Resumed") << "\n";
        }
        
        if (viewer->isKeyPressed(SDL_SCANCODE_R)) {
            resetCurrentScenario();
        }
        
        return viewer->isRunning();
    }
    
    void update(float dt) {
        // Update current scenario
        if (current_scenario) {
            current_scenario->update(sim_state, dt);
        }
        
        // Run physics simulation
        physics_backend->step(sim_state, physics_config, dt);
        
        // Update window title with stats
        std::string title = "DigiStar Graphics Demo - " + 
                          (current_scenario ? current_scenario->getName() : "No Scenario") +
                          " - Particles: " + std::to_string(sim_state.particles.count) +
                          " - Springs: " + std::to_string(sim_state.springs.count);
        viewer->setWindowTitle(title);
    }
    
    void render() {
        viewer->beginFrame();
        viewer->renderSimulation(sim_state);
        viewer->renderUI(sim_state);
        viewer->presentFrame();
    }
    
    void loadScenario(int index) {
        current_scenario_index = index;
        
        switch (index) {
            case 0:
                current_scenario = std::make_unique<SolarSystemDemo>();
                break;
            case 1:
                current_scenario = std::make_unique<CollisionDemo>();
                break;
            default:
                current_scenario = std::make_unique<SolarSystemDemo>();
                break;
        }
        
        if (current_scenario) {
            current_scenario->initialize(sim_state);
            std::cout << "Loaded scenario: " << current_scenario->getName() << "\n";
        }
    }
    
    void resetCurrentScenario() {
        if (current_scenario) {
            current_scenario->initialize(sim_state);
            std::cout << "Reset scenario: " << current_scenario->getName() << "\n";
        }
    }
};

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    
    try {
        ViewerDemo demo;
        
        if (!demo.initialize()) {
            std::cerr << "Failed to initialize demo\n";
            return -1;
        }
        
        demo.run();
        demo.shutdown();
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Demo crashed with exception: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "Demo crashed with unknown exception\n";
        return -1;
    }
}