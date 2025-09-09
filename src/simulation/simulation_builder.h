#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

#include "integrated_simulation.h"
#include "../backend/backend_interface.h"

namespace digistar {

/**
 * Preset configurations for common simulation scenarios
 */
enum class SimulationPreset {
    MINIMAL,              ///< Minimal setup for testing
    DEVELOPMENT,          ///< Development setup with full monitoring
    PERFORMANCE,          ///< Optimized for performance
    GALAXY_FORMATION,     ///< Large-scale structure formation
    PLANETARY_SYSTEM,     ///< Solar system dynamics
    PARTICLE_PHYSICS,     ///< High-energy particle interactions
    DISTRIBUTED_CLUSTER   ///< Multi-node distributed simulation
};

/**
 * Fluent API builder for creating integrated simulations
 * 
 * This class provides a convenient, readable way to configure and create
 * DigiStar simulations. It uses the builder pattern with method chaining
 * to allow for flexible configuration.
 * 
 * Example usage:
 * ```cpp
 * auto sim = SimulationBuilder()
 *     .withPreset(SimulationPreset::GALAXY_FORMATION)
 *     .withBackend(BackendType::CUDA)
 *     .withMaxParticles(10'000'000)
 *     .withEventSystem("galaxy_events")
 *     .withScript("scripts/galaxy.dsl")
 *     .enableMonitoring()
 *     .build();
 * ```
 */
class SimulationBuilder {
public:
    /**
     * Constructor - starts with default configuration
     */
    SimulationBuilder();
    
    // === Preset Configuration ===
    
    /**
     * Apply a preset configuration
     */
    SimulationBuilder& withPreset(SimulationPreset preset);
    
    /**
     * Load configuration from JSON file
     */
    SimulationBuilder& fromConfigFile(const std::string& filename);
    
    /**
     * Load configuration from JSON string
     */
    SimulationBuilder& fromConfigString(const std::string& json);
    
    // === Backend Configuration ===
    
    /**
     * Set the physics backend type
     */
    SimulationBuilder& withBackend(BackendFactory::Type backend_type);
    
    /**
     * Set maximum number of particles
     */
    SimulationBuilder& withMaxParticles(size_t max_particles);
    
    /**
     * Set maximum number of springs
     */
    SimulationBuilder& withMaxSprings(size_t max_springs);
    
    /**
     * Set maximum number of active contacts
     */
    SimulationBuilder& withMaxContacts(size_t max_contacts);
    
    /**
     * Set world size for toroidal wrapping
     */
    SimulationBuilder& withWorldSize(float world_size);
    
    /**
     * Enable/disable toroidal topology
     */
    SimulationBuilder& withToroidalTopology(bool enable = true);
    
    // === Physics Configuration ===
    
    /**
     * Enable specific physics systems
     */
    SimulationBuilder& enableGravity(bool enable = true);
    SimulationBuilder& enableContacts(bool enable = true);
    SimulationBuilder& enableSprings(bool enable = true);
    SimulationBuilder& enableThermal(bool enable = true);
    SimulationBuilder& enableFusion(bool enable = true);
    
    /**
     * Set gravity parameters
     */
    SimulationBuilder& withGravityMode(PhysicsConfig::GravityMode mode);
    SimulationBuilder& withGravityStrength(float strength);
    
    /**
     * Set integrator type and timestep
     */
    SimulationBuilder& withIntegrator(PhysicsConfig::IntegratorType integrator);
    SimulationBuilder& withTimeStep(float dt);
    SimulationBuilder& withAdaptiveTimeStep(float min_dt, float max_dt);
    
    // === Event System Configuration ===
    
    /**
     * Enable event system with shared memory name
     */
    SimulationBuilder& withEventSystem(const std::string& shm_name = "");
    
    /**
     * Disable event system
     */
    SimulationBuilder& withoutEvents();
    
    /**
     * Configure event filtering
     */
    SimulationBuilder& withEventFilter(const std::vector<EventType>& enabled_types);
    
    /**
     * Enable spatial event filtering
     */
    SimulationBuilder& withSpatialEventFilter(float x, float y, float radius);
    
    // === DSL Configuration ===
    
    /**
     * Enable DSL with startup scripts
     */
    SimulationBuilder& withDSL(bool enable = true);
    
    /**
     * Add startup script
     */
    SimulationBuilder& withScript(const std::string& filename);
    SimulationBuilder& withScriptContent(const std::string& name, const std::string& content);
    
    /**
     * Set DSL script directory
     */
    SimulationBuilder& withScriptDirectory(const std::string& directory);
    
    // === Performance Configuration ===
    
    /**
     * Set target FPS
     */
    SimulationBuilder& withTargetFPS(float fps);
    
    /**
     * Enable multithreading options
     */
    SimulationBuilder& withSeparatePhysicsThread(bool enable = true);
    SimulationBuilder& withSeparateEventThread(bool enable = true);
    
    /**
     * Set thread count (0 = auto-detect)
     */
    SimulationBuilder& withThreadCount(size_t count);
    
    /**
     * Enable SIMD optimizations
     */
    SimulationBuilder& withSIMD(bool enable = true);
    
    // === Monitoring Configuration ===
    
    /**
     * Enable performance monitoring
     */
    SimulationBuilder& enableMonitoring(bool enable = true);
    
    /**
     * Set monitoring interval
     */
    SimulationBuilder& withMonitoringInterval(std::chrono::milliseconds interval);
    
    /**
     * Enable performance logging
     */
    SimulationBuilder& enablePerformanceLogging(bool enable = true);
    
    // === Error Handling ===
    
    /**
     * Configure error handling behavior
     */
    SimulationBuilder& continueOnBackendError(bool continue_on_error = true);
    SimulationBuilder& continueOnDSLError(bool continue_on_error = true);
    
    /**
     * Set custom error handler
     */
    SimulationBuilder& withErrorHandler(std::function<void(const std::string&)> handler);
    
    // === Event Handlers ===
    
    /**
     * Set lifecycle event handlers
     */
    SimulationBuilder& onStart(std::function<void()> handler);
    SimulationBuilder& onStop(std::function<void()> handler);
    SimulationBuilder& onPause(std::function<void()> handler);
    SimulationBuilder& onResume(std::function<void()> handler);
    SimulationBuilder& onError(std::function<void(const std::string&)> handler);
    SimulationBuilder& onStatsUpdate(std::function<void(const IntegratedSimulationStats&)> handler);
    SimulationBuilder& onFrameComplete(std::function<void(float)> handler);
    
    // === Build Methods ===
    
    /**
     * Build and return the configured simulation
     * @throws std::runtime_error if configuration is invalid
     */
    std::unique_ptr<IntegratedSimulation> build();
    
    /**
     * Build and initialize the simulation
     * @throws std::runtime_error if build or initialization fails
     */
    std::unique_ptr<IntegratedSimulation> buildAndInitialize();
    
    /**
     * Build, initialize, and start the simulation
     * @throws std::runtime_error if any step fails
     */
    std::unique_ptr<IntegratedSimulation> buildAndStart();
    
    // === Validation ===
    
    /**
     * Validate current configuration
     * @return vector of validation errors (empty if valid)
     */
    std::vector<std::string> validate() const;
    
    /**
     * Check if configuration is valid
     */
    bool isValid() const { return validate().empty(); }
    
    // === Introspection ===
    
    /**
     * Get current configuration as JSON string
     */
    std::string toJson() const;
    
    /**
     * Get summary of current configuration
     */
    std::string getSummary() const;
    
    /**
     * Get the built configuration (without building simulation)
     */
    IntegratedSimulationConfig getConfig() const { return config_; }
    
    // === Static Factory Methods ===
    
    /**
     * Create builder with preset
     */
    static SimulationBuilder fromPreset(SimulationPreset preset);
    
    /**
     * Create builder from configuration file
     */
    static SimulationBuilder fromConfig(const std::string& filename);
    
    /**
     * Create builder for galaxy formation simulation
     */
    static SimulationBuilder forGalaxyFormation();
    
    /**
     * Create builder for planetary system simulation
     */
    static SimulationBuilder forPlanetarySystem();
    
    /**
     * Create builder for particle physics simulation
     */
    static SimulationBuilder forParticlePhysics();
    
    /**
     * Create minimal builder for testing
     */
    static SimulationBuilder minimal();
    
private:
    IntegratedSimulationConfig config_;
    SimulationEventHandlers event_handlers_;
    std::function<void(const std::string&)> error_handler_;
    
    // Helper methods
    void applyPreset(SimulationPreset preset);
    void validateAndThrow() const;
    std::string formatValidationErrors(const std::vector<std::string>& errors) const;
    
    // Configuration helpers
    void configureForGalaxyFormation();
    void configureForPlanetarySystem();
    void configureForParticlePhysics();
    void configureForDevelopment();
    void configureForPerformance();
    void configureMinimal();
    
    // JSON serialization helpers
    void parseConfigFromJson(const std::string& json);
    std::string configToJson() const;
};

/**
 * Quick-start functions for common scenarios
 */
namespace QuickStart {
    
    /**
     * Create a minimal simulation for testing
     */
    std::unique_ptr<IntegratedSimulation> createMinimal();
    
    /**
     * Create a galaxy formation simulation
     */
    std::unique_ptr<IntegratedSimulation> createGalaxyFormation(size_t particles = 100000);
    
    /**
     * Create a solar system simulation
     */
    std::unique_ptr<IntegratedSimulation> createSolarSystem();
    
    /**
     * Create a particle physics simulation
     */
    std::unique_ptr<IntegratedSimulation> createParticlePhysics();
    
    /**
     * Create from configuration file
     */
    std::unique_ptr<IntegratedSimulation> fromConfigFile(const std::string& filename);
    
} // namespace QuickStart

/**
 * Configuration validation helpers
 */
namespace ConfigValidation {
    
    /**
     * Validate backend configuration
     */
    std::vector<std::string> validateBackendConfig(const SimulationConfig& config);
    
    /**
     * Validate physics configuration
     */
    std::vector<std::string> validatePhysicsConfig(const PhysicsConfig& config);
    
    /**
     * Validate event system configuration
     */
    std::vector<std::string> validateEventConfig(const IntegratedSimulationConfig& config);
    
    /**
     * Validate DSL configuration
     */
    std::vector<std::string> validateDSLConfig(const IntegratedSimulationConfig& config);
    
    /**
     * Check system requirements for configuration
     */
    std::vector<std::string> checkSystemRequirements(const IntegratedSimulationConfig& config);
    
} // namespace ConfigValidation

} // namespace digistar