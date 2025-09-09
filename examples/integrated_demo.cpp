/**
 * DigiStar Integrated Simulation Demo
 * 
 * This example demonstrates the full integration of all DigiStar components:
 * - Physics backend with multiple systems
 * - Event system with real-time event processing
 * - DSL runtime with reactive scripting
 * - Performance monitoring and statistics
 * 
 * The demo creates a dynamic particle system with various physics interactions
 * and shows how to use the integrated simulation API.
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <signal.h>
#include <atomic>

#include "../src/simulation/integrated_simulation.h"
#include "../src/simulation/simulation_builder.h"
#include "../src/config/simulation_config.h"
#include "../src/events/event_consumer.h"

using namespace digistar;

// Global flag for clean shutdown
std::atomic<bool> keep_running{true};

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down gracefully..." << std::endl;
    keep_running = false;
}

// Event monitoring class
class EventMonitor {
public:
    EventMonitor(SharedMemoryEventSystem* event_system) 
        : event_system_(event_system) {
        
        if (event_system && event_system->is_valid()) {
            consumer_ = std::make_unique<EventConsumer>(event_system->get_buffer(), "demo_monitor");
        }
    }
    
    void processEvents() {
        if (!consumer_) return;
        
        Event event;
        while (consumer_->tryRead(event)) {
            processEvent(event);
        }
    }
    
    void printStatistics() const {
        std::cout << "\n=== Event Statistics ===\n";
        std::cout << "Events processed: " << events_processed_ << "\n";
        std::cout << "Collisions: " << collision_count_ << "\n";
        std::cout << "Spring events: " << spring_count_ << "\n";
        std::cout << "Particle events: " << particle_count_ << "\n";
        std::cout << "High-energy events: " << high_energy_count_ << "\n";
        std::cout << "========================\n\n";
    }

private:
    SharedMemoryEventSystem* event_system_;
    std::unique_ptr<EventConsumer> consumer_;
    
    // Statistics
    size_t events_processed_ = 0;
    size_t collision_count_ = 0;
    size_t spring_count_ = 0;
    size_t particle_count_ = 0;
    size_t high_energy_count_ = 0;
    
    void processEvent(const Event& event) {
        events_processed_++;
        
        switch (event.type) {
        case EventType::SOFT_CONTACT:
        case EventType::HARD_COLLISION:
            collision_count_++;
            if (event.magnitude > 10000.0f) {
                high_energy_count_++;
                std::cout << "[HIGH ENERGY] Collision with energy: " << event.magnitude 
                         << " at (" << event.x << ", " << event.y << ")\n";
            }
            break;
            
        case EventType::SPRING_CREATED:
        case EventType::SPRING_BROKEN:
            spring_count_++;
            std::cout << "[SPRING] " << (event.type == EventType::SPRING_CREATED ? "Created" : "Broken")
                     << " spring between particles " << event.primary_id << " and " << event.secondary_id << "\n";
            break;
            
        case EventType::PARTICLE_MERGE:
        case EventType::PARTICLE_FISSION:
            particle_count_++;
            std::cout << "[PARTICLE] " << (event.type == EventType::PARTICLE_MERGE ? "Merge" : "Fission")
                     << " event at (" << event.x << ", " << event.y << ") with magnitude " << event.magnitude << "\n";
            break;
            
        case EventType::COMPOSITE_FORMED:
            std::cout << "[COMPOSITE] New composite body formed with " 
                     << event.data.composite.particle_count << " particles\n";
            break;
            
        case EventType::TICK_COMPLETE:
            // Don't print tick events - too verbose
            break;
            
        default:
            std::cout << "[EVENT] Type " << static_cast<int>(event.type) 
                     << " at (" << event.x << ", " << event.y << ")\n";
            break;
        }
    }
};

// DSL script for dynamic behavior
const std::string DEMO_DSL_SCRIPT = R"(
; Demo DSL Script for Integrated Simulation
; This script demonstrates reactive programming with DigiStar

; Create initial particle cluster
(on-start
  (begin
    (log "Creating initial particle cluster...")
    (create-particle-cluster [0 0] 100 50.0)
    (create-particle-cluster [1000 1000] 50 30.0)
    (create-particle-cluster [-500 500] 75 40.0)))

; React to high-energy collisions
(on-event 'hard-collision
  (lambda (event)
    (when (> (event-magnitude event) 10000)
      (begin
        (log (string-append "High-energy collision detected: " 
                           (number->string (event-magnitude event))))
        (create-explosion (event-position event) (* 0.1 (event-magnitude event)))))))

; Automatic spring formation based on proximity and velocity
(on-event 'soft-contact
  (lambda (event)
    (let ((p1 (event-primary-id event))
          (p2 (event-secondary-id event))
          (rel-velocity (relative-velocity p1 p2)))
      (when (< (magnitude rel-velocity) 5.0)
        (create-spring p1 p2 1000.0 50.0)
        (log (string-append "Auto-spring created between " 
                           (number->string p1) " and " (number->string p2)))))))

; Periodic system monitoring
(schedule 'monitor-system 5000
  (lambda ()
    (let ((stats (simulation-stats)))
      (log (string-append "Active particles: " (number->string (stats-particles stats))))
      (log (string-append "Active springs: " (number->string (stats-springs stats))))
      (log (string-append "Total energy: " (number->string (stats-energy stats)))))))

; Temperature-based phase transitions
(on-particle-property-change 'temperature
  (lambda (particle old-temp new-temp)
    (cond
      ; Melting point
      [(and (< old-temp 1000) (>= new-temp 1000))
       (set-particle-phase particle 'liquid)
       (emit-event 'phase-transition 
         :particle particle 
         :old-phase 'solid 
         :new-phase 'liquid)]
      
      ; Boiling point
      [(and (< old-temp 2000) (>= new-temp 2000))
       (set-particle-phase particle 'gas)
       (emit-event 'phase-transition 
         :particle particle 
         :old-phase 'liquid 
         :new-phase 'gas)])))

; Composite body detection and management
(on-event 'composite-formed
  (lambda (event)
    (let ((composite-id (event-primary-id event))
          (particle-count (event-data-field event 'particle-count)))
      (log (string-append "New composite body formed with " 
                         (number->string particle-count) " particles"))
      
      ; Large composites get special treatment
      (when (> particle-count 20)
        (set-composite-property composite-id 'structural-integrity 2.0)
        (log "Large composite detected - increased structural integrity")))))
)";

void runDemo() {
    std::cout << "=== DigiStar Integrated Simulation Demo ===\n\n";
    
    try {
        // Create simulation with comprehensive configuration
        auto simulation = SimulationBuilder()
            .withPreset(SimulationPreset::DEVELOPMENT)
            .withMaxParticles(10000)
            .withMaxSprings(50000)
            .withMaxContacts(5000)
            .withWorldSize(5000.0f)
            .enableGravity(true)
            .enableContacts(true)
            .enableSprings(true)
            .enableThermal(true)
            .withGravityMode(PhysicsConfig::DIRECT_N2)
            .withIntegrator(PhysicsConfig::VELOCITY_VERLET)
            .withTargetFPS(60.0f)
            .withEventSystem("digistar_demo")
            .withDSL(true)
            .withScriptContent("demo_script", DEMO_DSL_SCRIPT)
            .enableMonitoring(true)
            .withMonitoringInterval(std::chrono::milliseconds(1000))
            .enablePerformanceLogging(true)
            .continueOnDSLError(true)
            .onStart([]() {
                std::cout << "[SIMULATION] Started successfully!\n";
            })
            .onStop([]() {
                std::cout << "[SIMULATION] Stopped.\n";
            })
            .onError([](const std::string& error) {
                std::cout << "[SIMULATION ERROR] " << error << "\n";
            })
            .onStatsUpdate([](const IntegratedSimulationStats& stats) {
                static int update_count = 0;
                if (++update_count % 10 == 0) {  // Print every 10 updates
                    std::cout << "[STATS] Particles: " << stats.backend_stats.active_particles
                             << ", Springs: " << stats.backend_stats.active_springs
                             << ", FPS: " << std::fixed << std::setprecision(1) << stats.current_fps
                             << ", Energy: " << std::scientific << stats.backend_stats.total_energy << "\n";
                }
            })
            .buildAndInitialize();
        
        // Set up event monitoring
        EventMonitor event_monitor(simulation->getEventSystem());
        
        std::cout << "Configuration Summary:\n" << SimulationBuilder().getSummary() << "\n";
        std::cout << "Starting simulation... (Press Ctrl+C to stop)\n\n";
        
        // Start simulation
        simulation->start();
        
        // Main loop
        auto last_stats_print = std::chrono::steady_clock::now();
        auto last_event_check = std::chrono::steady_clock::now();
        
        while (keep_running && simulation->isRunning()) {
            auto now = std::chrono::steady_clock::now();
            
            // Process events regularly
            if (now - last_event_check >= std::chrono::milliseconds(100)) {
                event_monitor.processEvents();
                last_event_check = now;
            }
            
            // Print detailed statistics every 10 seconds
            if (now - last_stats_print >= std::chrono::seconds(10)) {
                std::cout << "\n" << simulation->getPerformanceReport() << "\n";
                event_monitor.printStatistics();
                last_stats_print = now;
            }
            
            // Check for errors
            const std::string& last_error = simulation->getLastError();
            if (!last_error.empty()) {
                static std::string prev_error;
                if (last_error != prev_error) {
                    std::cout << "[ERROR] " << last_error << "\n";
                    prev_error = last_error;
                }
            }
            
            // Sleep to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        // Clean shutdown
        std::cout << "\nStopping simulation...\n";
        simulation->stop();
        
        // Final statistics
        std::cout << "\n=== Final Statistics ===\n";
        std::cout << simulation->getPerformanceReport() << "\n";
        event_monitor.printStatistics();
        
    } catch (const std::exception& e) {
        std::cerr << "Demo failed: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Set up signal handlers for clean shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "DigiStar Integrated Simulation Demo\n";
    std::cout << "Build: " << __DATE__ << " " << __TIME__ << "\n\n";
    
    // Parse command line arguments
    bool verbose = false;
    std::string config_file;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--config" || arg == "-c") {
            if (i + 1 < argc) {
                config_file = argv[++i];
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --verbose, -v     Enable verbose output\n";
            std::cout << "  --config FILE     Load configuration from file\n";
            std::cout << "  --help, -h        Show this help\n";
            return 0;
        }
    }
    
    if (verbose) {
        std::cout << "Verbose mode enabled\n";
    }
    
    if (!config_file.empty()) {
        std::cout << "Loading configuration from: " << config_file << "\n";
        // In a full implementation, we'd load the config file here
    }
    
    // Run the demo
    runDemo();
    
    std::cout << "Demo completed.\n";
    return 0;
}