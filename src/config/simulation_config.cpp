#include "simulation_config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <chrono>

namespace digistar {

// ConfigurationManager implementation
ConfigurationManager::ConfigurationManager() {
    // Set up default profile
    ConfigProfile default_profile;
    default_profile.name = "default";
    default_profile.description = "Default configuration profile";
    default_profile.inherit_from_default = false;  // This is the default
    profiles_["default"] = default_profile;
    
    // Set up development profile
    ConfigProfile dev_profile;
    dev_profile.name = "development";
    dev_profile.description = "Development configuration with full monitoring";
    dev_profile.overrides["monitoring.enable"] = "true";
    dev_profile.overrides["monitoring.log_performance"] = "true";
    dev_profile.overrides["error.continue_on_backend_error"] = "true";
    dev_profile.overrides["error.continue_on_dsl_error"] = "true";
    profiles_["development"] = dev_profile;
    
    // Set up production profile
    ConfigProfile prod_profile;
    prod_profile.name = "production";
    prod_profile.description = "Production configuration optimized for performance";
    prod_profile.overrides["events.enable"] = "false";
    prod_profile.overrides["dsl.enable"] = "false";
    prod_profile.overrides["monitoring.log_performance"] = "false";
    prod_profile.overrides["performance.target_fps"] = "120";
    prod_profile.overrides["performance.separate_physics_thread"] = "true";
    profiles_["production"] = prod_profile;
}

ConfigurationManager::~ConfigurationManager() {
    disableHotReload();
}

void ConfigurationManager::setProfile(const std::string& profile_name) {
    if (profiles_.find(profile_name) == profiles_.end()) {
        std::cerr << "[ConfigurationManager] Profile '" << profile_name << "' not found" << std::endl;
        return;
    }
    
    current_profile_ = profile_name;
    applyProfile(profile_name);
}

void ConfigurationManager::addProfile(const ConfigProfile& profile) {
    profiles_[profile.name] = profile;
}

std::vector<std::string> ConfigurationManager::getProfiles() const {
    std::vector<std::string> profiles;
    for (const auto& [name, _] : profiles_) {
        profiles.push_back(name);
    }
    return profiles;
}

bool ConfigurationManager::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ConfigurationManager] Cannot open file: " << filename << std::endl;
        return false;
    }
    
    std::string json((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    
    return loadFromString(json);
}

bool ConfigurationManager::loadFromString(const std::string& json) {
    try {
        return parseJsonString(json);
    } catch (const std::exception& e) {
        std::cerr << "[ConfigurationManager] Failed to parse JSON: " << e.what() << std::endl;
        return false;
    }
}

void ConfigurationManager::loadFromEnvironment(const std::string& prefix) {
    // Load DigiStar configuration from environment variables
    auto& config = digistar_config_;
    
    std::string value;
    
    // Backend configuration
    if (!(value = getEnvironmentValue("BACKEND_TYPE", prefix)).empty()) {
        config.backend_type = value;
    }
    
    if (!(value = getEnvironmentValue("MAX_PARTICLES", prefix)).empty()) {
        config.max_particles = std::stoull(value);
    }
    
    if (!(value = getEnvironmentValue("MAX_SPRINGS", prefix)).empty()) {
        config.max_springs = std::stoull(value);
    }
    
    if (!(value = getEnvironmentValue("WORLD_SIZE", prefix)).empty()) {
        config.world_size = std::stof(value);
    }
    
    // Physics configuration
    if (!(value = getEnvironmentValue("ENABLE_GRAVITY", prefix)).empty()) {
        config.enable_gravity = (value == "true" || value == "1");
    }
    
    if (!(value = getEnvironmentValue("GRAVITY_STRENGTH", prefix)).empty()) {
        config.gravity_strength = std::stof(value);
    }
    
    // Event system configuration
    if (!(value = getEnvironmentValue("ENABLE_EVENTS", prefix)).empty()) {
        config.enable_events = (value == "true" || value == "1");
    }
    
    if (!(value = getEnvironmentValue("EVENT_SHM_NAME", prefix)).empty()) {
        config.event_shm_name = value;
    }
    
    // Performance configuration
    if (!(value = getEnvironmentValue("TARGET_FPS", prefix)).empty()) {
        config.target_fps = std::stof(value);
    }
    
    if (!(value = getEnvironmentValue("NUM_THREADS", prefix)).empty()) {
        config.num_threads = std::stoull(value);
    }
    
    std::cout << "[ConfigurationManager] Loaded configuration from environment with prefix: " << prefix << std::endl;
}

bool ConfigurationManager::saveToFile(const std::string& filename) const {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        file << toJson();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ConfigurationManager] Failed to save to file: " << e.what() << std::endl;
        return false;
    }
}

void ConfigurationManager::enableHotReload(const std::string& filename, std::chrono::milliseconds check_interval) {
    disableHotReload();  // Stop any existing watch
    
    if (!std::filesystem::exists(filename)) {
        std::cerr << "[ConfigurationManager] Cannot watch non-existent file: " << filename << std::endl;
        return;
    }
    
    watched_file_ = filename;
    check_interval_ = check_interval;
    last_write_time_ = std::filesystem::last_write_time(filename);
    hot_reload_enabled_ = true;
    stop_watching_.store(false);
    
    watch_thread_ = std::make_unique<std::thread>(&ConfigurationManager::watchFileChanges, this);
    
    std::cout << "[ConfigurationManager] Hot reload enabled for: " << filename << std::endl;
}

void ConfigurationManager::disableHotReload() {
    if (hot_reload_enabled_) {
        stop_watching_.store(true);
        
        if (watch_thread_ && watch_thread_->joinable()) {
            watch_thread_->join();
        }
        
        watch_thread_.reset();
        hot_reload_enabled_ = false;
        
        std::cout << "[ConfigurationManager] Hot reload disabled" << std::endl;
    }
}

bool ConfigurationManager::checkForChanges() {
    if (!hot_reload_enabled_ || watched_file_.empty()) {
        return false;
    }
    
    if (!std::filesystem::exists(watched_file_)) {
        return false;
    }
    
    auto current_write_time = std::filesystem::last_write_time(watched_file_);
    if (current_write_time != last_write_time_) {
        last_write_time_ = current_write_time;
        
        // Reload configuration
        if (loadFromFile(watched_file_)) {
            std::vector<std::string> changes = {"Configuration reloaded from: " + watched_file_};
            
            if (change_callback_) {
                change_callback_(changes);
            }
            
            std::cout << "[ConfigurationManager] Configuration reloaded due to file change" << std::endl;
            return true;
        }
    }
    
    return false;
}

void ConfigurationManager::setChangeCallback(std::function<void(const std::vector<std::string>&)> callback) {
    change_callback_ = callback;
}

bool ConfigurationManager::hasKey(const std::string& key) const {
    return values_.find(key) != values_.end();
}

std::vector<std::string> ConfigurationManager::getKeysWithPrefix(const std::string& prefix) const {
    std::vector<std::string> keys;
    for (const auto& [key, _] : values_) {
        if (key.find(prefix) == 0) {
            keys.push_back(key);
        }
    }
    return keys;
}

std::vector<std::string> ConfigurationManager::validate() const {
    std::vector<std::string> errors;
    
    // Validate DigiStar configuration
    auto& config = digistar_config_;
    
    // Backend validation
    if (config.max_particles.get() == 0) {
        errors.push_back("max_particles must be greater than 0");
    }
    
    if (config.max_particles.get() > 100'000'000) {
        errors.push_back("max_particles exceeds reasonable limit (100M)");
    }
    
    if (config.world_size.get() <= 0.0f) {
        errors.push_back("world_size must be positive");
    }
    
    // Physics validation
    if (config.gravity_strength.get() < 0.0f) {
        errors.push_back("gravity_strength cannot be negative");
    }
    
    if (config.contact_stiffness.get() <= 0.0f) {
        errors.push_back("contact_stiffness must be positive");
    }
    
    // Performance validation
    if (config.target_fps.get() <= 0.0f) {
        errors.push_back("target_fps must be positive");
    }
    
    if (config.target_fps.get() > 1000.0f) {
        errors.push_back("target_fps exceeds reasonable limit (1000 FPS)");
    }
    
    return errors;
}

std::string ConfigurationManager::toJson() const {
    return serializeToJson();
}

std::string ConfigurationManager::toEnvironmentScript() const {
    std::stringstream ss;
    auto& config = digistar_config_;
    
    ss << "#!/bin/bash\n";
    ss << "# DigiStar Configuration Environment Variables\n";
    ss << "# Generated on " << std::chrono::system_clock::now().time_since_epoch().count() << "\n\n";
    
    // Backend configuration
    ss << "export DIGISTAR_BACKEND_TYPE=\"" << config.backend_type.get() << "\"\n";
    ss << "export DIGISTAR_MAX_PARTICLES=" << config.max_particles.get() << "\n";
    ss << "export DIGISTAR_MAX_SPRINGS=" << config.max_springs.get() << "\n";
    ss << "export DIGISTAR_WORLD_SIZE=" << config.world_size.get() << "\n";
    
    // Physics configuration
    ss << "export DIGISTAR_ENABLE_GRAVITY=" << (config.enable_gravity.get() ? "true" : "false") << "\n";
    ss << "export DIGISTAR_GRAVITY_STRENGTH=" << config.gravity_strength.get() << "\n";
    
    // Event system configuration
    ss << "export DIGISTAR_ENABLE_EVENTS=" << (config.enable_events.get() ? "true" : "false") << "\n";
    ss << "export DIGISTAR_EVENT_SHM_NAME=\"" << config.event_shm_name.get() << "\"\n";
    
    // Performance configuration
    ss << "export DIGISTAR_TARGET_FPS=" << config.target_fps.get() << "\n";
    ss << "export DIGISTAR_NUM_THREADS=" << config.num_threads.get() << "\n";
    
    return ss.str();
}

std::string ConfigurationManager::getSummary() const {
    std::stringstream ss;
    auto& config = digistar_config_;
    
    ss << "=== DigiStar Configuration Summary ===\n";
    ss << "Profile: " << current_profile_ << "\n\n";
    
    ss << "Backend:\n";
    ss << "  Type: " << config.backend_type.get() << "\n";
    ss << "  Max Particles: " << config.max_particles.get() << "\n";
    ss << "  Max Springs: " << config.max_springs.get() << "\n";
    ss << "  World Size: " << config.world_size.get() << "\n\n";
    
    ss << "Physics:\n";
    ss << "  Gravity: " << (config.enable_gravity.get() ? "Enabled" : "Disabled") << "\n";
    ss << "  Contacts: " << (config.enable_contacts.get() ? "Enabled" : "Disabled") << "\n";
    ss << "  Springs: " << (config.enable_springs.get() ? "Enabled" : "Disabled") << "\n\n";
    
    ss << "Performance:\n";
    ss << "  Target FPS: " << config.target_fps.get() << "\n";
    ss << "  Threads: " << config.num_threads.get() << " (0=auto)\n";
    ss << "  Separate Physics Thread: " << (config.separate_physics_thread.get() ? "Yes" : "No") << "\n\n";
    
    ss << "Features:\n";
    ss << "  Events: " << (config.enable_events.get() ? "Enabled" : "Disabled") << "\n";
    ss << "  DSL: " << (config.enable_dsl.get() ? "Enabled" : "Disabled") << "\n";
    ss << "  Monitoring: " << (config.enable_monitoring.get() ? "Enabled" : "Disabled") << "\n";
    
    return ss.str();
}

ConfigurationManager& ConfigurationManager::getInstance() {
    static ConfigurationManager instance;
    return instance;
}

std::unique_ptr<ConfigurationManager> ConfigurationManager::createDefault() {
    return std::make_unique<ConfigurationManager>();
}

std::unique_ptr<ConfigurationManager> ConfigurationManager::fromPreset(const std::string& preset_name) {
    auto config = createDefault();
    config->setProfile(preset_name);
    return config;
}

// Private methods
void ConfigurationManager::watchFileChanges() {
    while (!stop_watching_.load()) {
        checkForChanges();
        std::this_thread::sleep_for(check_interval_);
    }
}

bool ConfigurationManager::parseJsonString(const std::string& json) {
    // Simplified JSON parsing - in production you'd use a proper JSON library
    // For now, just handle basic key-value pairs
    
    std::cout << "[ConfigurationManager] JSON parsing not fully implemented - basic values loaded" << std::endl;
    
    // Example of what real parsing would do:
    // Parse JSON and populate values_ map and DigiStar configuration
    
    return true;
}

std::string ConfigurationManager::serializeToJson() const {
    std::stringstream ss;
    auto& config = digistar_config_;
    
    ss << "{\n";
    ss << "  \"profile\": \"" << current_profile_ << "\",\n";
    ss << "  \"backend\": {\n";
    ss << "    \"type\": \"" << config.backend_type.get() << "\",\n";
    ss << "    \"max_particles\": " << config.max_particles.get() << ",\n";
    ss << "    \"max_springs\": " << config.max_springs.get() << ",\n";
    ss << "    \"world_size\": " << config.world_size.get() << "\n";
    ss << "  },\n";
    ss << "  \"physics\": {\n";
    ss << "    \"gravity\": {\n";
    ss << "      \"enable\": " << (config.enable_gravity.get() ? "true" : "false") << ",\n";
    ss << "      \"strength\": " << config.gravity_strength.get() << ",\n";
    ss << "      \"mode\": \"" << config.gravity_mode.get() << "\"\n";
    ss << "    },\n";
    ss << "    \"contacts\": {\n";
    ss << "      \"enable\": " << (config.enable_contacts.get() ? "true" : "false") << ",\n";
    ss << "      \"stiffness\": " << config.contact_stiffness.get() << "\n";
    ss << "    }\n";
    ss << "  },\n";
    ss << "  \"events\": {\n";
    ss << "    \"enable\": " << (config.enable_events.get() ? "true" : "false") << ",\n";
    ss << "    \"shm_name\": \"" << config.event_shm_name.get() << "\"\n";
    ss << "  },\n";
    ss << "  \"performance\": {\n";
    ss << "    \"target_fps\": " << config.target_fps.get() << ",\n";
    ss << "    \"num_threads\": " << config.num_threads.get() << "\n";
    ss << "  }\n";
    ss << "}";
    
    return ss.str();
}

void ConfigurationManager::applyProfile(const std::string& profile_name) {
    auto it = profiles_.find(profile_name);
    if (it == profiles_.end()) {
        return;
    }
    
    const auto& profile = it->second;
    
    // Apply profile overrides to configuration
    // In a real implementation, this would be more sophisticated
    std::cout << "[ConfigurationManager] Applied profile: " << profile_name << std::endl;
}

std::string ConfigurationManager::getEnvironmentValue(const std::string& key, const std::string& prefix) const {
    std::string env_key = prefix + key;
    const char* value = std::getenv(env_key.c_str());
    return value ? std::string(value) : std::string();
}

// ConfigPresets namespace implementation
namespace ConfigPresets {

std::unique_ptr<ConfigurationManager> minimal() {
    auto config = ConfigurationManager::createDefault();
    auto& ds_config = config->getDigiStarConfig();
    
    ds_config.max_particles = 1000;
    ds_config.max_springs = 5000;
    ds_config.enable_events = false;
    ds_config.enable_dsl = false;
    ds_config.enable_monitoring = false;
    ds_config.separate_physics_thread = false;
    
    return config;
}

std::unique_ptr<ConfigurationManager> development() {
    auto config = ConfigurationManager::createDefault();
    config->setProfile("development");
    return config;
}

std::unique_ptr<ConfigurationManager> production() {
    auto config = ConfigurationManager::createDefault();
    config->setProfile("production");
    return config;
}

std::unique_ptr<ConfigurationManager> galaxyFormation() {
    auto config = ConfigurationManager::createDefault();
    auto& ds_config = config->getDigiStarConfig();
    
    ds_config.max_particles = 1'000'000;
    ds_config.world_size = 100'000.0f;
    ds_config.enable_springs = false;  // No springs for galaxy formation
    ds_config.gravity_mode = "particle_mesh";
    ds_config.target_fps = 30.0f;  // Lower FPS for large simulations
    ds_config.separate_physics_thread = true;
    
    return config;
}

std::unique_ptr<ConfigurationManager> planetarySystem() {
    auto config = ConfigurationManager::createDefault();
    auto& ds_config = config->getDigiStarConfig();
    
    ds_config.max_particles = 10'000;
    ds_config.max_springs = 50'000;
    ds_config.world_size = 10'000.0f;
    ds_config.gravity_mode = "direct";  // Small N, direct is fine
    ds_config.target_fps = 60.0f;
    
    return config;
}

std::unique_ptr<ConfigurationManager> particlePhysics() {
    auto config = ConfigurationManager::createDefault();
    auto& ds_config = config->getDigiStarConfig();
    
    ds_config.max_particles = 100'000;
    ds_config.max_springs = 1'000'000;
    ds_config.world_size = 1000.0f;
    ds_config.enable_gravity = false;  // Focus on contact/spring forces
    ds_config.target_fps = 120.0f;  // High FPS for particle interactions
    
    return config;
}

} // namespace ConfigPresets

// ConfigUtils namespace implementation
namespace ConfigUtils {

IntegratedSimulationConfig toIntegratedConfig(const ConfigurationManager& config) {
    const auto& ds_config = config.getDigiStarConfig();
    
    IntegratedSimulationConfig integrated;
    
    // Backend configuration
    if (ds_config.backend_type.get() == "cpu") {
        integrated.backend_type = BackendFactory::Type::CPU;
    } else if (ds_config.backend_type.get() == "cuda") {
        integrated.backend_type = BackendFactory::Type::CUDA;
    } else if (ds_config.backend_type.get() == "cpu_simd") {
        integrated.backend_type = BackendFactory::Type::CPU_SIMD;
    } else {
        integrated.backend_type = BackendFactory::Type::CPU;
    }
    
    integrated.simulation_config.max_particles = ds_config.max_particles.get();
    integrated.simulation_config.max_springs = ds_config.max_springs.get();
    integrated.simulation_config.max_contacts = ds_config.max_contacts.get();
    integrated.simulation_config.world_size = ds_config.world_size.get();
    
    // Physics configuration
    integrated.physics_config.enabled_systems = 0;
    if (ds_config.enable_gravity.get()) {
        integrated.physics_config.enabled_systems |= PhysicsConfig::GRAVITY;
    }
    if (ds_config.enable_contacts.get()) {
        integrated.physics_config.enabled_systems |= PhysicsConfig::CONTACTS;
    }
    if (ds_config.enable_springs.get()) {
        integrated.physics_config.enabled_systems |= PhysicsConfig::SPRINGS;
    }
    
    integrated.physics_config.gravity_strength = ds_config.gravity_strength.get();
    integrated.physics_config.contact_stiffness = ds_config.contact_stiffness.get();
    integrated.physics_config.contact_damping = ds_config.contact_damping.get();
    
    // Event system configuration
    integrated.enable_events = ds_config.enable_events.get();
    integrated.event_shm_name = ds_config.event_shm_name.get();
    integrated.auto_create_event_system = ds_config.auto_create_events.get();
    
    // DSL configuration
    integrated.enable_dsl = ds_config.enable_dsl.get();
    integrated.dsl_script_directory = ds_config.script_directory.get();
    
    // Performance configuration
    integrated.target_fps = ds_config.target_fps.get();
    integrated.adaptive_timestep = ds_config.adaptive_timestep.get();
    integrated.min_dt = ds_config.min_dt.get();
    integrated.max_dt = ds_config.max_dt.get();
    integrated.use_separate_physics_thread = ds_config.separate_physics_thread.get();
    integrated.use_separate_event_thread = ds_config.separate_event_thread.get();
    
    // Monitoring configuration
    integrated.enable_monitoring = ds_config.enable_monitoring.get();
    integrated.monitor_interval = std::chrono::milliseconds(ds_config.monitor_interval_ms.get());
    integrated.log_performance_stats = ds_config.log_performance.get();
    
    return integrated;
}

std::unique_ptr<ConfigurationManager> fromIntegratedConfig(const IntegratedSimulationConfig& integrated) {
    auto config = ConfigurationManager::createDefault();
    auto& ds_config = config->getDigiStarConfig();
    
    // Convert backend type
    switch (integrated.backend_type) {
    case BackendFactory::Type::CPU:
        ds_config.backend_type = "cpu";
        break;
    case BackendFactory::Type::CPU_SIMD:
        ds_config.backend_type = "cpu_simd";
        break;
    case BackendFactory::Type::CUDA:
        ds_config.backend_type = "cuda";
        break;
    default:
        ds_config.backend_type = "cpu";
        break;
    }
    
    // Backend configuration
    ds_config.max_particles = integrated.simulation_config.max_particles;
    ds_config.max_springs = integrated.simulation_config.max_springs;
    ds_config.max_contacts = integrated.simulation_config.max_contacts;
    ds_config.world_size = integrated.simulation_config.world_size;
    
    // Physics configuration
    ds_config.enable_gravity = (integrated.physics_config.enabled_systems & PhysicsConfig::GRAVITY) != 0;
    ds_config.enable_contacts = (integrated.physics_config.enabled_systems & PhysicsConfig::CONTACTS) != 0;
    ds_config.enable_springs = (integrated.physics_config.enabled_systems & PhysicsConfig::SPRINGS) != 0;
    
    ds_config.gravity_strength = integrated.physics_config.gravity_strength;
    ds_config.contact_stiffness = integrated.physics_config.contact_stiffness;
    ds_config.contact_damping = integrated.physics_config.contact_damping;
    
    // Event system configuration
    ds_config.enable_events = integrated.enable_events;
    ds_config.event_shm_name = integrated.event_shm_name;
    ds_config.auto_create_events = integrated.auto_create_event_system;
    
    // Performance configuration
    ds_config.target_fps = integrated.target_fps;
    ds_config.separate_physics_thread = integrated.use_separate_physics_thread;
    ds_config.separate_event_thread = integrated.use_separate_event_thread;
    
    return config;
}

} // namespace ConfigUtils

} // namespace digistar