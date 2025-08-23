#include "simulation/simulation.h"
#include <iostream>
#include <cstring>

using namespace digistar;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "\nOptions:\n"
              << "  --scenario <name>     Load scenario: default, solar_system, galaxy_collision,\n"
              << "                        asteroid_field, spring_network, composite_test, stress_test\n"
              << "  --backend <type>      Backend: cpu (reference), openmp, cuda\n"
              << "  --particles <count>   Maximum particle count (default: 100000)\n"
              << "  --timestep <dt>       Simulation timestep (default: 0.01)\n"
              << "  --fps <rate>          Target frame rate (default: 60)\n"
              << "  --width <cols>        Terminal width (default: 120)\n"
              << "  --height <rows>       Terminal height (default: 40)\n"
              << "  --no-render           Disable rendering (benchmark mode)\n"
              << "  --no-color            Disable ANSI colors\n"
              << "  --pause               Start paused\n"
              << "  --pm-gravity          Use Particle-Mesh gravity solver\n"
              << "  --help                Show this help message\n"
              << "\n" << InputHandler::getHelpText();
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    Simulation::Config config;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--scenario") == 0 && i + 1 < argc) {
            config.scenario_name = argv[++i];
        }
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            std::string backend = argv[++i];
            if (backend == "cpu" || backend == "reference") {
                config.backend_type = BackendFactory::Type::CPU;
            } else if (backend == "openmp") {
                config.backend_type = BackendFactory::Type::CPU_SIMD;
            } else if (backend == "cuda") {
                config.backend_type = BackendFactory::Type::CUDA;
            } else {
                std::cerr << "Unknown backend: " << backend << std::endl;
                return 1;
            }
        }
        else if (strcmp(argv[i], "--particles") == 0 && i + 1 < argc) {
            config.max_particles = std::stoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--timestep") == 0 && i + 1 < argc) {
            config.timestep = std::stof(argv[++i]);
        }
        else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            config.target_fps = std::stof(argv[++i]);
        }
        else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            config.renderer_config.width = std::stoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            config.renderer_config.height = std::stoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--no-render") == 0) {
            config.enable_rendering = false;
        }
        else if (strcmp(argv[i], "--no-color") == 0) {
            config.use_ansi_colors = false;
        }
        else if (strcmp(argv[i], "--pause") == 0) {
            config.pause_on_start = true;
        }
        else if (strcmp(argv[i], "--pm-gravity") == 0) {
            config.physics.gravity_mode = PhysicsConfig::PARTICLE_MESH;
        }
        else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Print startup info
    std::cout << "=== DigiStar Physics Simulation ===" << std::endl;
    std::cout << "Scenario: " << config.scenario_name << std::endl;
    std::cout << "Backend: " << (config.backend_type == BackendFactory::Type::CPU ? 
                                  "CPU Reference" : "CPU OpenMP") << std::endl;
    std::cout << "Max particles: " << config.max_particles << std::endl;
    std::cout << "Timestep: " << config.timestep << std::endl;
    std::cout << "Target FPS: " << config.target_fps << std::endl;
    
    if (config.enable_rendering) {
        std::cout << "Display: " << config.renderer_config.width << "x" 
                  << config.renderer_config.height << std::endl;
    } else {
        std::cout << "Rendering: Disabled (benchmark mode)" << std::endl;
    }
    
    std::cout << "\nPress 'h' for help on controls\n" << std::endl;
    
    // Create and run simulation
    try {
        Simulation sim(config);
        
        if (!sim.initialize()) {
            std::cerr << "Failed to initialize simulation" << std::endl;
            return 1;
        }
        
        sim.run();
        
        // Print final stats
        std::cout << "\n=== Simulation Complete ===" << std::endl;
        std::cout << "Simulation time: " << sim.getSimulationTime() << " seconds" << std::endl;
        std::cout << "Real time: " << sim.getRealTimeElapsed() << " seconds" << std::endl;
        std::cout << "Average FPS: " << sim.getFPS() << std::endl;
        std::cout << "Average physics time: " << sim.getPhysicsTime() << " ms" << std::endl;
        std::cout << "Average render time: " << sim.getRenderTime() << " ms" << std::endl;
        
        auto stats = sim.getStats();
        std::cout << "Final particle count: " << stats.active_particles << std::endl;
        std::cout << "Final spring count: " << stats.active_springs << std::endl;
        std::cout << "Total energy: " << stats.total_energy << " J" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}