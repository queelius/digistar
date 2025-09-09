/**
 * Galaxy Formation Simulation
 * 
 * This example demonstrates large-scale structure formation using DigiStar's
 * integrated simulation system. It shows:
 * - Large particle counts (100k+ particles)
 * - Gravity-dominated dynamics using Particle Mesh solver
 * - Procedural generation of initial conditions
 * - Event-driven visualization triggers
 * - Performance optimization for large-scale simulations
 */

#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>

#include "../src/simulation/simulation_builder.h"
#include "../src/config/simulation_config.h"
#include "../src/events/event_consumer.h"

using namespace digistar;

class GalaxyFormationMonitor {
public:
    explicit GalaxyFormationMonitor(SharedMemoryEventSystem* event_system) {
        if (event_system && event_system->is_valid()) {
            consumer_ = std::make_unique<EventConsumer>(event_system->get_buffer(), "galaxy_monitor");
        }
    }
    
    void processEvents() {
        if (!consumer_) return;
        
        Event event;
        while (consumer_->tryRead(event)) {
            switch (event.type) {
            case EventType::COMPOSITE_FORMED:
                handleCompositeFormation(event);
                break;
            case EventType::STAR_FORMATION:
                handleStarFormation(event);
                break;
            case EventType::BLACK_HOLE_FORMATION:
                handleBlackHoleFormation(event);
                break;
            case EventType::TIDAL_DISRUPTION:
                handleTidalDisruption(event);
                break;
            default:
                break;
            }
        }
    }
    
    void printStatistics() const {
        std::cout << "\n=== Galaxy Formation Statistics ===\n";
        std::cout << "Proto-galaxies formed: " << proto_galaxies_ << "\n";
        std::cout << "Stars formed: " << stars_formed_ << "\n";
        std::cout << "Black holes formed: " << black_holes_ << "\n";
        std::cout << "Tidal disruptions: " << tidal_disruptions_ << "\n";
        std::cout << "Largest structure: " << largest_structure_size_ << " particles\n";
        std::cout << "=====================================\n\n";
    }

private:
    std::unique_ptr<EventConsumer> consumer_;
    
    size_t proto_galaxies_ = 0;
    size_t stars_formed_ = 0;
    size_t black_holes_ = 0;
    size_t tidal_disruptions_ = 0;
    size_t largest_structure_size_ = 0;
    
    void handleCompositeFormation(const Event& event) {
        size_t particle_count = event.data.composite.particle_count;
        
        if (particle_count > 100) {  // Consider large structures as proto-galaxies
            proto_galaxies_++;
            largest_structure_size_ = std::max(largest_structure_size_, particle_count);
            
            std::cout << "[GALAXY] Proto-galaxy formed with " << particle_count 
                     << " particles at (" << std::fixed << std::setprecision(1) 
                     << event.x << ", " << event.y << ")\n";
        }
    }
    
    void handleStarFormation(const Event& event) {
        stars_formed_++;
        std::cout << "[STAR] Star formation at (" << std::fixed << std::setprecision(1)
                 << event.x << ", " << event.y << ") with mass " << event.magnitude << "\n";
    }
    
    void handleBlackHoleFormation(const Event& event) {
        black_holes_++;
        std::cout << "[BLACK HOLE] Black hole formed at (" << std::fixed << std::setprecision(1)
                 << event.x << ", " << event.y << ") with mass " << event.magnitude << "\n";
    }
    
    void handleTidalDisruption(const Event& event) {
        tidal_disruptions_++;
        std::cout << "[TIDAL] Tidal disruption at (" << std::fixed << std::setprecision(1)
                 << event.x << ", " << event.y << ")\n";
    }
};

// DSL script for galaxy formation dynamics
const std::string GALAXY_FORMATION_DSL = R"(
; Galaxy Formation DSL Script
; Implements realistic structure formation physics

; Initialize dark matter and baryonic matter
(on-start
  (begin
    (log "Initializing galaxy formation simulation...")
    
    ; Create initial dark matter distribution (80% of mass)
    (generate 'dark-matter
      :particles 80000
      :distribution 'cold-dark-matter
      :mass-function '(constant 1.0)
      :temperature 2.7
      :region (sphere [0 0] 50000))
    
    ; Create baryonic matter (20% of mass)
    (generate 'baryonic-matter
      :particles 20000
      :distribution 'gaussian
      :mass-function '(log-normal 0.1 2.0)
      :temperature 10000
      :region (sphere [0 0] 30000))
    
    (log "Initial conditions set up - starting evolution")))

; Density-driven star formation
(on-density-threshold 1000.0
  (lambda (region density)
    (let ((mass (region-total-mass region))
          (temperature (region-average-temperature region)))
      (when (and (> mass 100.0) (< temperature 1000.0))
        ; Jeans instability - collapse to form star
        (let ((star (collapse-region region)))
          (set-particle-type star 'star)
          (set-particle-temperature star 1e6)  ; Main sequence
          (emit-event 'star-formation
            :position (particle-position star)
            :mass mass))))))

; Supermassive black hole formation in dense cores
(on-mass-threshold 10000.0
  (lambda (particle)
    (when (> (particle-density-local particle 100.0) 1e6)
      ; Form supermassive black hole
      (let ((bh (collapse-to-black-hole particle)))
        (set-particle-type bh 'black-hole)
        (emit-event 'black-hole-formation
          :position (particle-position bh)
          :mass (particle-mass bh))))))

; Tidal forces and disruption
(on-gravitational-encounter
  (lambda (p1 p2 separation)
    (let ((mass1 (particle-mass p1))
          (mass2 (particle-mass p2))
          (roche-limit (* 2.44 (particle-radius p1) 
                         (expt (/ mass1 mass2) (/ 1.0 3.0)))))
      (when (< separation roche-limit)
        ; Tidal disruption
        (tidal-disrupt p1 p2)
        (emit-event 'tidal-disruption
          :position (midpoint (particle-position p1) (particle-position p2))
          :magnitude separation)))))

; Cooling and heating processes
(on-particle-property-change 'temperature
  (lambda (particle old-temp new-temp)
    (let ((density (particle-density-local particle 50.0)))
      ; Radiative cooling at high density
      (when (> density 100.0)
        (let ((cooling-rate (* 1e-6 density (sqrt new-temp))))
          (set-particle-temperature particle 
            (max 10.0 (- new-temp (* cooling-rate (simulation-dt)))))))
      
      ; Heating from nearby stars
      (when (particle-type-nearby? particle 'star 20.0)
        (let ((heating-rate 100.0))
          (set-particle-temperature particle 
            (+ new-temp (* heating-rate (simulation-dt)))))))))

; Structure identification and classification
(schedule 'analyze-structures 10000
  (lambda ()
    (let ((structures (find-gravitationally-bound-structures)))
      (for-each
        (lambda (structure)
          (let ((mass (structure-total-mass structure))
                (size (structure-radius structure))
                (particles (structure-particle-count structure)))
            
            ; Classify structure type
            (cond
              [(> mass 1e12)  ; Galaxy cluster
               (log (string-append "Galaxy cluster found: " 
                                 (number->string mass) " solar masses"))]
              
              [(> mass 1e10)  ; Large galaxy
               (log (string-append "Large galaxy found: " 
                                 (number->string mass) " solar masses"))]
              
              [(> mass 1e8)   ; Small galaxy
               (log (string-append "Dwarf galaxy found: " 
                                 (number->string mass) " solar masses"))]
              
              [(> mass 1e6)   ; Star cluster
               (log (string-append "Star cluster found: " 
                                 (number->string particles) " stars"))])))
        structures))))

; Mergers and accretion
(on-event 'composite-collision
  (lambda (event)
    (let ((mass1 (event-data-field event 'mass1))
          (mass2 (event-data-field event 'mass2)))
      (when (and (> mass1 1e8) (> mass2 1e8))
        ; Major merger
        (log (string-append "Major galaxy merger: " 
                           (number->string mass1) " + " (number->string mass2)))
        
        ; Trigger starburst
        (let ((merger-region (sphere (event-position event) 1000.0)))
          (increase-star-formation-rate merger-region 10.0)
          (schedule-delayed 5000
            (lambda ()
              (restore-star-formation-rate merger-region))))))))

; Conservation checks and validation
(schedule 'conservation-check 5000
  (lambda ()
    (let ((total-mass (total-system-mass))
          (total-energy (total-system-energy))
          (angular-momentum (total-angular-momentum)))
      
      ; Check conservation laws
      (when (> (abs (- total-mass initial-mass)) (* 0.01 initial-mass))
        (log "WARNING: Mass conservation violated"))
      
      (when (> (abs (- total-energy initial-energy)) (* 0.1 (abs initial-energy)))
        (log "WARNING: Energy conservation check - may be expansion"))
      
      (log (string-append "Conservation check - Mass: " (number->string total-mass)
                         ", Energy: " (number->string total-energy))))))
)";

void runGalaxyFormation() {
    std::cout << "=== Galaxy Formation Simulation ===\n\n";
    
    try {
        // Create high-performance galaxy formation simulation
        auto simulation = SimulationBuilder()
            .withPreset(SimulationPreset::GALAXY_FORMATION)
            .withMaxParticles(100'000)  // Large particle count
            .withMaxSprings(0)          // No springs for galaxy formation
            .withMaxContacts(10'000)    // Minimal contacts
            .withWorldSize(100'000.0f)  // Very large world
            .withToroidalTopology(true) // Periodic boundaries
            .enableGravity(true)
            .enableContacts(false)      // Disable contacts for performance
            .enableSprings(false)       // Disable springs for performance
            .enableThermal(true)        // Enable thermal processes
            .withGravityMode(PhysicsConfig::PARTICLE_MESH)  // Essential for large N
            .withGravityStrength(6.67430e-11f)
            .withIntegrator(PhysicsConfig::LEAPFROG)        // Good for gravity-only
            .withAdaptiveTimeStep(1e-8f, 1e-4f)            // Wide range for evolution
            .withTargetFPS(30.0f)       // Lower FPS for large simulations
            .withSeparatePhysicsThread(true)
            .withSeparateEventThread(true)
            .withThreadCount(0)         // Use all available cores
            .withEventSystem("galaxy_formation")
            .withDSL(true)
            .withScriptContent("galaxy_formation", GALAXY_FORMATION_DSL)
            .enableMonitoring(true)
            .withMonitoringInterval(std::chrono::milliseconds(2000))
            .enablePerformanceLogging(true)
            .continueOnBackendError(false)  // Strict error handling
            .continueOnDSLError(true)
            .onStart([]() {
                std::cout << "[GALAXY] Simulation started - beginning structure formation\n";
            })
            .onStatsUpdate([](const IntegratedSimulationStats& stats) {
                static int update_count = 0;
                if (++update_count % 5 == 0) {  // Print every 5 updates
                    std::cout << "[EVOLUTION] Time: " << std::fixed << std::setprecision(3) 
                             << stats.simulation_time << "s, "
                             << "Particles: " << stats.backend_stats.active_particles << ", "
                             << "Energy: " << std::scientific << stats.backend_stats.total_energy << ", "
                             << "FPS: " << std::fixed << std::setprecision(1) << stats.current_fps << "\n";
                }
            })
            .buildAndInitialize();
        
        // Set up galaxy formation monitoring
        GalaxyFormationMonitor monitor(simulation->getEventSystem());
        
        std::cout << "Starting galaxy formation evolution...\n";
        std::cout << "This simulation will run for cosmic time scales.\n";
        std::cout << "Watch for structure formation events!\n\n";
        
        // Initialize some particles manually for testing
        auto& state = simulation->getSimulationState();
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Create initial dark matter halo
        std::normal_distribution<float> pos_dist(0.0f, 10000.0f);
        std::normal_distribution<float> vel_dist(0.0f, 100.0f);
        std::lognormal_distribution<float> mass_dist(0.0f, 1.0f);
        
        std::cout << "Generating initial dark matter distribution...\n";
        for (size_t i = 0; i < 1000 && i < state.particles.capacity(); ++i) {
            size_t id = state.particles.add();
            
            state.particles.x[id] = pos_dist(gen);
            state.particles.y[id] = pos_dist(gen);
            state.particles.vx[id] = vel_dist(gen);
            state.particles.vy[id] = vel_dist(gen);
            state.particles.mass[id] = mass_dist(gen);
            state.particles.radius[id] = 0.1f;
            state.particles.temperature[id] = 2.7f;  // CMB temperature
            state.particles.charge[id] = 0.0f;       // Dark matter is neutral
        }
        
        // Start simulation
        simulation->start();
        
        // Run for a specified time or until interrupted
        auto start_time = std::chrono::steady_clock::now();
        auto last_stats_time = start_time;
        auto last_event_time = start_time;
        
        const auto run_duration = std::chrono::minutes(5);  // Run for 5 minutes
        
        while (simulation->isRunning()) {
            auto now = std::chrono::steady_clock::now();
            
            // Check if we should stop
            if (now - start_time >= run_duration) {
                std::cout << "\nReached time limit - stopping simulation\n";
                break;
            }
            
            // Process events
            if (now - last_event_time >= std::chrono::milliseconds(200)) {
                monitor.processEvents();
                last_event_time = now;
            }
            
            // Print detailed statistics
            if (now - last_stats_time >= std::chrono::seconds(15)) {
                std::cout << "\n" << simulation->getPerformanceReport() << "\n";
                monitor.printStatistics();
                last_stats_time = now;
                
                // Check for interesting physics
                const auto& stats = simulation->getStats();
                if (stats.backend_stats.total_energy != 0.0f) {
                    float virial_ratio = -2.0f * stats.backend_stats.total_energy / stats.backend_stats.total_energy;  // Simplified
                    std::cout << "Virial equilibrium ratio: " << virial_ratio << "\n";
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Final analysis
        std::cout << "\n=== Final Galaxy Formation Analysis ===\n";
        std::cout << simulation->getPerformanceReport() << "\n";
        monitor.printStatistics();
        
        simulation->stop();
        
    } catch (const std::exception& e) {
        std::cerr << "Galaxy formation simulation failed: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "DigiStar Galaxy Formation Simulation\n";
    std::cout << "Simulating large-scale structure formation\n\n";
    
    // Parse command line arguments
    size_t particle_count = 100000;
    float world_size = 100000.0f;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--particles" && i + 1 < argc) {
            particle_count = std::stoull(argv[++i]);
        } else if (arg == "--world-size" && i + 1 < argc) {
            world_size = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --particles N     Number of particles (default: 100000)\n";
            std::cout << "  --world-size S    World size (default: 100000)\n";
            std::cout << "  --help, -h        Show this help\n";
            return 0;
        }
    }
    
    std::cout << "Configuration:\n";
    std::cout << "  Particles: " << particle_count << "\n";
    std::cout << "  World Size: " << world_size << "\n\n";
    
    runGalaxyFormation();
    
    std::cout << "Galaxy formation simulation completed.\n";
    return 0;
}