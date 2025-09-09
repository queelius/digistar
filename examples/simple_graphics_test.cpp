/**
 * Simple Graphics Viewer Test
 * 
 * A minimal example demonstrating the DigiStar graphics viewer
 * with a basic particle system.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <cmath>

#include "../src/viewer/graphics_viewer.h"

using namespace digistar;

int main() {
    std::cout << "DigiStar Graphics Viewer - Simple Test\n";
    
    // Create and initialize viewer
    auto viewer = std::make_unique<GraphicsViewer>();
    if (!viewer->initialize("Simple Graphics Test", 1280, 720, false)) {
        std::cerr << "Failed to initialize graphics viewer\n";
        return -1;
    }
    
    // Set up basic viewer configuration
    viewer->setRenderMode(GraphicsViewer::RenderMode::PARTICLES_AND_SPRINGS);
    viewer->setParticleStyle(GraphicsViewer::ParticleStyle::CIRCLES);
    viewer->setColorScheme(GraphicsViewer::ColorScheme::VELOCITY_BASED);
    
    // Create a simple simulation state
    SimulationState sim_state;
    
    // Initialize particle pool
    const size_t max_particles = 1000;
    sim_state.particles.initialize(max_particles);
    sim_state.springs.initialize(max_particles * 2);
    sim_state.contacts.initialize(max_particles);
    sim_state.composites.initialize(100);
    
    // Create a simple particle system - spinning galaxy-like structure
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> radius_dist(50.0f, 300.0f);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> mass_dist(1.0f, 10.0f);
    
    const size_t num_particles = 500;
    for (size_t i = 0; i < num_particles && i < max_particles; i++) {
        float radius = radius_dist(rng);
        float angle = angle_dist(rng);
        float orbital_speed = sqrt(1000.0f / radius);  // Simple orbital mechanics
        
        sim_state.particles.pos_x[i] = radius * cos(angle);
        sim_state.particles.pos_y[i] = radius * sin(angle);
        
        // Add orbital velocity plus some random motion
        sim_state.particles.vel_x[i] = -orbital_speed * sin(angle) + (rng() % 20 - 10) * 0.1f;
        sim_state.particles.vel_y[i] = orbital_speed * cos(angle) + (rng() % 20 - 10) * 0.1f;
        
        sim_state.particles.mass[i] = mass_dist(rng);
        sim_state.particles.radius[i] = 2.0f + sim_state.particles.mass[i] * 0.5f;
        sim_state.particles.temperature[i] = 0.5f;
        sim_state.particles.force_x[i] = 0.0f;
        sim_state.particles.force_y[i] = 0.0f;
        
        sim_state.particles.count = i + 1;
    }
    
    // Add a few springs between nearby particles
    for (size_t i = 0; i < sim_state.particles.count && sim_state.springs.count < 50; i++) {
        for (size_t j = i + 1; j < sim_state.particles.count && sim_state.springs.count < 50; j++) {
            float dx = sim_state.particles.pos_x[i] - sim_state.particles.pos_x[j];
            float dy = sim_state.particles.pos_y[i] - sim_state.particles.pos_y[j];
            float distance = sqrt(dx*dx + dy*dy);
            
            if (distance < 50.0f) {  // Only connect nearby particles
                size_t spring_id = sim_state.springs.count++;
                sim_state.springs.particle1_id[spring_id] = i;
                sim_state.springs.particle2_id[spring_id] = j;
                sim_state.springs.rest_length[spring_id] = distance;
                sim_state.springs.stiffness[spring_id] = 10.0f;
                sim_state.springs.damping[spring_id] = 1.0f;
            }
        }
    }
    
    std::cout << "Created " << sim_state.particles.count << " particles and " 
              << sim_state.springs.count << " springs\n";
    
    // Main loop
    auto last_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nControls:\n";
    std::cout << "  WASD/Arrow Keys: Pan camera\n";
    std::cout << "  Mouse Wheel: Zoom\n"; 
    std::cout << "  Left Click+Drag: Pan\n";
    std::cout << "  Space: Center on center of mass\n";
    std::cout << "  H: Toggle help overlay\n";
    std::cout << "  ESC: Exit\n\n";
    
    while (viewer->isRunning()) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            current_time - last_time);
        float dt = duration.count() / 1000000.0f;
        last_time = current_time;
        
        // Process input
        if (!viewer->processEvents()) {
            break;
        }
        
        // Simple physics update - just move particles
        for (size_t i = 0; i < sim_state.particles.count; i++) {
            sim_state.particles.pos_x[i] += sim_state.particles.vel_x[i] * dt;
            sim_state.particles.pos_y[i] += sim_state.particles.vel_y[i] * dt;
        }
        
        // Add some visual events periodically
        static float event_timer = 0.0f;
        event_timer += dt;
        if (event_timer > 2.0f) {
            event_timer = 0.0f;
            
            // Add a random explosion event
            size_t random_particle = rng() % sim_state.particles.count;
            VisualEvent explosion(VisualEvent::EXPLOSION,
                                sim_state.particles.pos_x[random_particle],
                                sim_state.particles.pos_y[random_particle],
                                0.8f);
            explosion.color = {255, 100, 50, 255};
            explosion.max_radius = 100.0f;
            viewer->addVisualEvent(explosion);
        }
        
        // Update simulation stats for display
        sim_state.stats.active_particles = sim_state.particles.count;
        sim_state.stats.active_springs = sim_state.springs.count;
        sim_state.stats.total_energy = sim_state.particles.count * 100.0f;  // Fake energy
        
        // Calculate max velocity for stats
        float max_vel = 0.0f;
        for (size_t i = 0; i < sim_state.particles.count; i++) {
            float vel = sqrt(sim_state.particles.vel_x[i] * sim_state.particles.vel_x[i] +
                           sim_state.particles.vel_y[i] * sim_state.particles.vel_y[i]);
            max_vel = std::max(max_vel, vel);
        }
        sim_state.stats.max_velocity = max_vel;
        
        // Render
        viewer->beginFrame();
        viewer->renderSimulation(sim_state);
        viewer->renderUI(sim_state);
        viewer->presentFrame();
    }
    
    std::cout << "Shutting down...\n";
    viewer->shutdown();
    
    return 0;
}