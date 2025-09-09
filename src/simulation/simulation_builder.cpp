#include "simulation_builder.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>

namespace digistar {

SimulationBuilder::SimulationBuilder() {
    // Set up default configuration
    configureMinimal();
}

SimulationBuilder& SimulationBuilder::withPreset(SimulationPreset preset) {
    applyPreset(preset);
    return *this;
}

SimulationBuilder& SimulationBuilder::fromConfigFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open configuration file: " + filename);
    }
    
    std::string json((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    
    try {
        parseConfigFromJson(json);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse configuration file '" + filename + "': " + e.what());
    }
    
    return *this;
}

SimulationBuilder& SimulationBuilder::fromConfigString(const std::string& json) {
    try {
        parseConfigFromJson(json);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse configuration JSON: " + std::string(e.what()));
    }
    
    return *this;
}

SimulationBuilder& SimulationBuilder::withBackend(BackendFactory::Type backend_type) {
    config_.backend_type = backend_type;
    return *this;
}

SimulationBuilder& SimulationBuilder::withMaxParticles(size_t max_particles) {
    config_.simulation_config.max_particles = max_particles;
    return *this;
}

SimulationBuilder& SimulationBuilder::withMaxSprings(size_t max_springs) {
    config_.simulation_config.max_springs = max_springs;
    return *this;
}

SimulationBuilder& SimulationBuilder::withMaxContacts(size_t max_contacts) {
    config_.simulation_config.max_contacts = max_contacts;
    return *this;
}

SimulationBuilder& SimulationBuilder::withWorldSize(float world_size) {
    config_.simulation_config.world_size = world_size;
    return *this;
}

SimulationBuilder& SimulationBuilder::withToroidalTopology(bool enable) {
    config_.simulation_config.use_toroidal = enable;
    return *this;
}

SimulationBuilder& SimulationBuilder::enableGravity(bool enable) {
    if (enable) {
        config_.physics_config.enabled_systems |= PhysicsConfig::GRAVITY;
    } else {
        config_.physics_config.enabled_systems &= ~PhysicsConfig::GRAVITY;
    }
    return *this;
}

SimulationBuilder& SimulationBuilder::enableContacts(bool enable) {
    if (enable) {
        config_.physics_config.enabled_systems |= PhysicsConfig::CONTACTS;
    } else {
        config_.physics_config.enabled_systems &= ~PhysicsConfig::CONTACTS;
    }
    return *this;
}

SimulationBuilder& SimulationBuilder::enableSprings(bool enable) {
    if (enable) {
        config_.physics_config.enabled_systems |= PhysicsConfig::SPRINGS;
    } else {
        config_.physics_config.enabled_systems &= ~PhysicsConfig::SPRINGS;
    }
    return *this;
}

SimulationBuilder& SimulationBuilder::enableThermal(bool enable) {
    if (enable) {
        config_.physics_config.enabled_systems |= PhysicsConfig::THERMAL;
    } else {
        config_.physics_config.enabled_systems &= ~PhysicsConfig::THERMAL;
    }
    return *this;
}

SimulationBuilder& SimulationBuilder::enableFusion(bool enable) {
    if (enable) {
        config_.physics_config.enabled_systems |= PhysicsConfig::FUSION;
    } else {
        config_.physics_config.enabled_systems &= ~PhysicsConfig::FUSION;
    }
    return *this;
}

SimulationBuilder& SimulationBuilder::withGravityMode(PhysicsConfig::GravityMode mode) {
    config_.physics_config.gravity_mode = mode;
    return *this;
}

SimulationBuilder& SimulationBuilder::withGravityStrength(float strength) {
    config_.physics_config.gravity_strength = strength;
    return *this;
}

SimulationBuilder& SimulationBuilder::withIntegrator(PhysicsConfig::IntegratorType integrator) {
    config_.physics_config.default_integrator = integrator;
    return *this;
}

SimulationBuilder& SimulationBuilder::withTimeStep(float dt) {
    config_.target_fps = 1.0f / dt;
    config_.adaptive_timestep = false;
    return *this;
}

SimulationBuilder& SimulationBuilder::withAdaptiveTimeStep(float min_dt, float max_dt) {
    config_.adaptive_timestep = true;
    config_.min_dt = min_dt;
    config_.max_dt = max_dt;
    return *this;
}

SimulationBuilder& SimulationBuilder::withEventSystem(const std::string& shm_name) {
    config_.enable_events = true;
    config_.event_shm_name = shm_name.empty() ? "digistar_events" : shm_name;
    config_.auto_create_event_system = true;
    return *this;
}

SimulationBuilder& SimulationBuilder::withoutEvents() {
    config_.enable_events = false;
    return *this;
}

SimulationBuilder& SimulationBuilder::withDSL(bool enable) {
    config_.enable_dsl = enable;
    return *this;
}

SimulationBuilder& SimulationBuilder::withScript(const std::string& filename) {
    config_.startup_scripts.push_back(filename);
    return *this;
}

SimulationBuilder& SimulationBuilder::withScriptContent(const std::string& name, const std::string& content) {
    // For script content, we'll need to write it to a temporary file
    std::string temp_filename = "/tmp/digistar_" + name + "_" + std::to_string(std::hash<std::string>{}(content)) + ".dsl";
    
    std::ofstream temp_file(temp_filename);
    if (temp_file.is_open()) {
        temp_file << content;
        temp_file.close();
        config_.startup_scripts.push_back(temp_filename);
    }
    
    return *this;
}

SimulationBuilder& SimulationBuilder::withScriptDirectory(const std::string& directory) {
    config_.dsl_script_directory = directory;
    return *this;
}

SimulationBuilder& SimulationBuilder::withTargetFPS(float fps) {
    config_.target_fps = fps;
    return *this;
}

SimulationBuilder& SimulationBuilder::withSeparatePhysicsThread(bool enable) {
    config_.use_separate_physics_thread = enable;
    return *this;
}

SimulationBuilder& SimulationBuilder::withSeparateEventThread(bool enable) {
    config_.use_separate_event_thread = enable;
    return *this;
}

SimulationBuilder& SimulationBuilder::withThreadCount(size_t count) {
    config_.simulation_config.num_threads = count;
    return *this;
}

SimulationBuilder& SimulationBuilder::withSIMD(bool enable) {
    config_.simulation_config.enable_simd = enable;
    return *this;
}

SimulationBuilder& SimulationBuilder::enableMonitoring(bool enable) {
    config_.enable_monitoring = enable;
    return *this;
}

SimulationBuilder& SimulationBuilder::withMonitoringInterval(std::chrono::milliseconds interval) {
    config_.monitor_interval = interval;
    return *this;
}

SimulationBuilder& SimulationBuilder::enablePerformanceLogging(bool enable) {
    config_.log_performance_stats = enable;
    return *this;
}

SimulationBuilder& SimulationBuilder::continueOnBackendError(bool continue_on_error) {
    config_.continue_on_backend_error = continue_on_error;
    return *this;
}

SimulationBuilder& SimulationBuilder::continueOnDSLError(bool continue_on_error) {
    config_.continue_on_dsl_error = continue_on_error;
    return *this;
}

SimulationBuilder& SimulationBuilder::withErrorHandler(std::function<void(const std::string&)> handler) {
    error_handler_ = handler;
    return *this;
}

SimulationBuilder& SimulationBuilder::onStart(std::function<void()> handler) {
    event_handlers_.on_start = handler;
    return *this;
}

SimulationBuilder& SimulationBuilder::onStop(std::function<void()> handler) {
    event_handlers_.on_stop = handler;
    return *this;
}

SimulationBuilder& SimulationBuilder::onPause(std::function<void()> handler) {
    event_handlers_.on_pause = handler;
    return *this;
}

SimulationBuilder& SimulationBuilder::onResume(std::function<void()> handler) {
    event_handlers_.on_resume = handler;
    return *this;
}

SimulationBuilder& SimulationBuilder::onError(std::function<void(const std::string&)> handler) {
    event_handlers_.on_error = handler;
    return *this;
}

SimulationBuilder& SimulationBuilder::onStatsUpdate(std::function<void(const IntegratedSimulationStats&)> handler) {
    event_handlers_.on_stats_update = handler;
    return *this;
}

SimulationBuilder& SimulationBuilder::onFrameComplete(std::function<void(float)> handler) {
    event_handlers_.on_frame_complete = handler;
    return *this;
}

std::unique_ptr<IntegratedSimulation> SimulationBuilder::build() {
    validateAndThrow();
    
    auto simulation = std::make_unique<IntegratedSimulation>(config_);
    
    // Set event handlers
    simulation->setEventHandlers(event_handlers_);
    
    // Set error handler if provided
    if (error_handler_) {
        simulation->setErrorHandler(error_handler_);
    }
    
    return simulation;
}

std::unique_ptr<IntegratedSimulation> SimulationBuilder::buildAndInitialize() {
    auto simulation = build();
    
    if (!simulation->initialize()) {
        throw std::runtime_error("Failed to initialize simulation: " + simulation->getLastError());
    }
    
    return simulation;
}

std::unique_ptr<IntegratedSimulation> SimulationBuilder::buildAndStart() {
    auto simulation = buildAndInitialize();
    simulation->start();
    return simulation;
}

std::vector<std::string> SimulationBuilder::validate() const {
    std::vector<std::string> errors;
    
    // Validate backend configuration
    auto backend_errors = ConfigValidation::validateBackendConfig(config_.simulation_config);
    errors.insert(errors.end(), backend_errors.begin(), backend_errors.end());
    
    // Validate physics configuration
    auto physics_errors = ConfigValidation::validatePhysicsConfig(config_.physics_config);
    errors.insert(errors.end(), physics_errors.begin(), physics_errors.end());
    
    // Validate event configuration
    auto event_errors = ConfigValidation::validateEventConfig(config_);
    errors.insert(errors.end(), event_errors.begin(), event_errors.end());
    
    // Validate DSL configuration
    auto dsl_errors = ConfigValidation::validateDSLConfig(config_);
    errors.insert(errors.end(), dsl_errors.begin(), dsl_errors.end());
    
    // Check system requirements
    auto system_errors = ConfigValidation::checkSystemRequirements(config_);
    errors.insert(errors.end(), system_errors.begin(), system_errors.end());
    
    return errors;
}

std::string SimulationBuilder::toJson() const {
    return configToJson();
}

std::string SimulationBuilder::getSummary() const {
    std::stringstream ss;
    
    ss << "=== DigiStar Simulation Configuration Summary ===\n";
    
    // Backend info
    ss << "Backend: " << BackendFactory::getDescription(config_.backend_type) << "\n";
    ss << "Max Particles: " << config_.simulation_config.max_particles << "\n";
    ss << "Max Springs: " << config_.simulation_config.max_springs << "\n";
    ss << "World Size: " << config_.simulation_config.world_size << "\n";
    
    // Physics systems
    ss << "\nPhysics Systems:\n";
    ss << "  Gravity: " << ((config_.physics_config.enabled_systems & PhysicsConfig::GRAVITY) ? "Yes" : "No") << "\n";
    ss << "  Contacts: " << ((config_.physics_config.enabled_systems & PhysicsConfig::CONTACTS) ? "Yes" : "No") << "\n";
    ss << "  Springs: " << ((config_.physics_config.enabled_systems & PhysicsConfig::SPRINGS) ? "Yes" : "No") << "\n";
    ss << "  Thermal: " << ((config_.physics_config.enabled_systems & PhysicsConfig::THERMAL) ? "Yes" : "No") << "\n";
    
    // Performance settings
    ss << "\nPerformance:\n";
    ss << "  Target FPS: " << config_.target_fps << "\n";
    ss << "  Adaptive Timestep: " << (config_.adaptive_timestep ? "Yes" : "No") << "\n";
    ss << "  Separate Physics Thread: " << (config_.use_separate_physics_thread ? "Yes" : "No") << "\n";
    
    // Features
    ss << "\nFeatures:\n";
    ss << "  Events: " << (config_.enable_events ? "Yes (" + config_.event_shm_name + ")" : "No") << "\n";
    ss << "  DSL: " << (config_.enable_dsl ? "Yes" : "No") << "\n";
    if (!config_.startup_scripts.empty()) {
        ss << "  Startup Scripts: " << config_.startup_scripts.size() << "\n";
    }
    ss << "  Monitoring: " << (config_.enable_monitoring ? "Yes" : "No") << "\n";
    
    return ss.str();
}

// Static factory methods
SimulationBuilder SimulationBuilder::withPreset(SimulationPreset preset) {
    SimulationBuilder builder;
    builder.applyPreset(preset);
    return builder;
}

SimulationBuilder SimulationBuilder::fromConfig(const std::string& filename) {
    SimulationBuilder builder;
    builder.fromConfigFile(filename);
    return builder;
}

SimulationBuilder SimulationBuilder::forGalaxyFormation() {
    SimulationBuilder builder;
    builder.configureForGalaxyFormation();
    return builder;
}

SimulationBuilder SimulationBuilder::forPlanetarySystem() {
    SimulationBuilder builder;
    builder.configureForPlanetarySystem();
    return builder;
}

SimulationBuilder SimulationBuilder::forParticlePhysics() {
    SimulationBuilder builder;
    builder.configureForParticlePhysics();
    return builder;
}

SimulationBuilder SimulationBuilder::minimal() {
    SimulationBuilder builder;
    builder.configureMinimal();
    return builder;
}

// Private implementation methods
void SimulationBuilder::applyPreset(SimulationPreset preset) {
    switch (preset) {
    case SimulationPreset::MINIMAL:
        configureMinimal();
        break;
    case SimulationPreset::DEVELOPMENT:
        configureForDevelopment();
        break;
    case SimulationPreset::PERFORMANCE:
        configureForPerformance();
        break;
    case SimulationPreset::GALAXY_FORMATION:
        configureForGalaxyFormation();
        break;
    case SimulationPreset::PLANETARY_SYSTEM:
        configureForPlanetarySystem();
        break;
    case SimulationPreset::PARTICLE_PHYSICS:
        configureForParticlePhysics();
        break;
    case SimulationPreset::DISTRIBUTED_CLUSTER:
        configureForPerformance();  // Start with performance config
        config_.use_separate_physics_thread = true;
        config_.use_separate_event_thread = true;
        break;
    }
}

void SimulationBuilder::validateAndThrow() const {
    auto errors = validate();
    if (!errors.empty()) {
        throw std::runtime_error("Invalid configuration:\n" + formatValidationErrors(errors));
    }
}

std::string SimulationBuilder::formatValidationErrors(const std::vector<std::string>& errors) const {
    std::stringstream ss;
    for (size_t i = 0; i < errors.size(); ++i) {
        ss << "  " << (i + 1) << ". " << errors[i] << "\n";
    }
    return ss.str();
}

void SimulationBuilder::configureForGalaxyFormation() {
    config_.backend_type = BackendFactory::Type::CPU_SIMD;
    config_.simulation_config.max_particles = 1'000'000;
    config_.simulation_config.max_springs = 0;  // No springs for galaxy formation
    config_.simulation_config.world_size = 100'000.0f;  // Large world
    config_.physics_config.enabled_systems = PhysicsConfig::GRAVITY | PhysicsConfig::THERMAL;
    config_.physics_config.gravity_mode = PhysicsConfig::PARTICLE_MESH;
    config_.enable_events = true;
    config_.enable_dsl = true;
    config_.target_fps = 30.0f;  // Lower FPS for large simulations
    config_.use_separate_physics_thread = true;
}

void SimulationBuilder::configureForPlanetarySystem() {
    config_.backend_type = BackendFactory::Type::CPU;
    config_.simulation_config.max_particles = 10'000;
    config_.simulation_config.max_springs = 50'000;
    config_.simulation_config.world_size = 10'000.0f;
    config_.physics_config.enabled_systems = PhysicsConfig::GRAVITY | PhysicsConfig::CONTACTS | PhysicsConfig::SPRINGS;
    config_.physics_config.gravity_mode = PhysicsConfig::DIRECT_N2;  // Small N, direct is fine
    config_.enable_events = true;
    config_.enable_dsl = true;
    config_.target_fps = 60.0f;
}

void SimulationBuilder::configureForParticlePhysics() {
    config_.backend_type = BackendFactory::Type::CPU_SIMD;
    config_.simulation_config.max_particles = 100'000;
    config_.simulation_config.max_springs = 1'000'000;
    config_.simulation_config.max_contacts = 100'000;
    config_.simulation_config.world_size = 1000.0f;
    config_.physics_config.enabled_systems = PhysicsConfig::CONTACTS | PhysicsConfig::SPRINGS | 
                                            PhysicsConfig::THERMAL | PhysicsConfig::FUSION | PhysicsConfig::FISSION;
    config_.enable_events = true;
    config_.enable_dsl = true;
    config_.target_fps = 120.0f;  // High FPS for particle interactions
}

void SimulationBuilder::configureForDevelopment() {
    config_.backend_type = BackendFactory::Type::CPU;
    config_.simulation_config.max_particles = 10'000;
    config_.enable_events = true;
    config_.enable_dsl = true;
    config_.enable_monitoring = true;
    config_.log_performance_stats = true;
    config_.continue_on_backend_error = true;
    config_.continue_on_dsl_error = true;
}

void SimulationBuilder::configureForPerformance() {
    config_.backend_type = BackendFactory::Type::CPU_SIMD;
    config_.simulation_config.max_particles = 1'000'000;
    config_.enable_events = false;  // Disable for max performance
    config_.enable_dsl = false;
    config_.enable_monitoring = false;
    config_.use_separate_physics_thread = true;
    config_.target_fps = 120.0f;
}

void SimulationBuilder::configureMinimal() {
    config_.backend_type = BackendFactory::Type::CPU;
    config_.simulation_config.max_particles = 1000;
    config_.simulation_config.max_springs = 5000;
    config_.simulation_config.max_contacts = 1000;
    config_.simulation_config.world_size = 1000.0f;
    config_.physics_config.enabled_systems = PhysicsConfig::GRAVITY | PhysicsConfig::CONTACTS;
    config_.enable_events = false;
    config_.enable_dsl = false;
    config_.enable_monitoring = false;
    config_.target_fps = 60.0f;
    config_.use_separate_physics_thread = false;
}

void SimulationBuilder::parseConfigFromJson(const std::string& json) {
    // This is a simplified implementation - in production you'd use a proper JSON parser
    // For now, just demonstrate the concept
    std::cout << "[SimulationBuilder] JSON parsing not fully implemented - using defaults" << std::endl;
}

std::string SimulationBuilder::configToJson() const {
    // Simplified JSON serialization - in production you'd use a proper JSON library
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"backend_type\": \"" << static_cast<int>(config_.backend_type) << "\",\n";
    ss << "  \"max_particles\": " << config_.simulation_config.max_particles << ",\n";
    ss << "  \"max_springs\": " << config_.simulation_config.max_springs << ",\n";
    ss << "  \"world_size\": " << config_.simulation_config.world_size << ",\n";
    ss << "  \"enable_events\": " << (config_.enable_events ? "true" : "false") << ",\n";
    ss << "  \"enable_dsl\": " << (config_.enable_dsl ? "true" : "false") << ",\n";
    ss << "  \"target_fps\": " << config_.target_fps << "\n";
    ss << "}";
    return ss.str();
}

// QuickStart namespace implementation
namespace QuickStart {

std::unique_ptr<IntegratedSimulation> createMinimal() {
    return SimulationBuilder::minimal().buildAndInitialize();
}

std::unique_ptr<IntegratedSimulation> createGalaxyFormation(size_t particles) {
    return SimulationBuilder::forGalaxyFormation()
        .withMaxParticles(particles)
        .buildAndInitialize();
}

std::unique_ptr<IntegratedSimulation> createSolarSystem() {
    return SimulationBuilder::forPlanetarySystem()
        .withMaxParticles(50)  // Sun + planets + moons + asteroids
        .buildAndInitialize();
}

std::unique_ptr<IntegratedSimulation> createParticlePhysics() {
    return SimulationBuilder::forParticlePhysics()
        .buildAndInitialize();
}

std::unique_ptr<IntegratedSimulation> fromConfigFile(const std::string& filename) {
    return SimulationBuilder::fromConfig(filename).buildAndInitialize();
}

} // namespace QuickStart

// ConfigValidation namespace implementation
namespace ConfigValidation {

std::vector<std::string> validateBackendConfig(const SimulationConfig& config) {
    std::vector<std::string> errors;
    
    if (config.max_particles == 0) {
        errors.push_back("max_particles must be greater than 0");
    }
    
    if (config.max_particles > 100'000'000) {
        errors.push_back("max_particles exceeds reasonable limit (100M)");
    }
    
    if (config.world_size <= 0.0f) {
        errors.push_back("world_size must be positive");
    }
    
    if (config.contact_cell_size <= 0.0f) {
        errors.push_back("contact_cell_size must be positive");
    }
    
    return errors;
}

std::vector<std::string> validatePhysicsConfig(const PhysicsConfig& config) {
    std::vector<std::string> errors;
    
    if (config.gravity_strength < 0.0f) {
        errors.push_back("gravity_strength cannot be negative");
    }
    
    if (config.contact_stiffness <= 0.0f) {
        errors.push_back("contact_stiffness must be positive");
    }
    
    if (config.spring_break_strain <= 0.0f) {
        errors.push_back("spring_break_strain must be positive");
    }
    
    return errors;
}

std::vector<std::string> validateEventConfig(const IntegratedSimulationConfig& config) {
    std::vector<std::string> errors;
    
    if (config.enable_events && config.event_shm_name.empty()) {
        errors.push_back("event_shm_name cannot be empty when events are enabled");
    }
    
    return errors;
}

std::vector<std::string> validateDSLConfig(const IntegratedSimulationConfig& config) {
    std::vector<std::string> errors;
    
    if (config.enable_dsl) {
        for (const auto& script : config.startup_scripts) {
            if (!std::filesystem::exists(script)) {
                errors.push_back("Startup script not found: " + script);
            }
        }
        
        if (!std::filesystem::exists(config.dsl_script_directory)) {
            errors.push_back("DSL script directory not found: " + config.dsl_script_directory);
        }
    }
    
    return errors;
}

std::vector<std::string> checkSystemRequirements(const IntegratedSimulationConfig& config) {
    std::vector<std::string> errors;
    
    // Check available memory (simplified)
    if (config.simulation_config.max_particles > 1'000'000 && 
        config.backend_type == BackendFactory::Type::CPU) {
        errors.push_back("Large particle counts may require GPU backend for optimal performance");
    }
    
    if (config.target_fps > 120.0f) {
        errors.push_back("Very high FPS targets may not be achievable");
    }
    
    return errors;
}

} // namespace ConfigValidation

} // namespace digistar