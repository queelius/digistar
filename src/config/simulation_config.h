#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

namespace digistar {

/**
 * Configuration management system for DigiStar simulations
 * 
 * This system provides:
 * - JSON-based configuration files
 * - Environment variable support
 * - Configuration validation
 * - Hot reloading capabilities
 * - Profile management (dev, production, test)
 * - Configuration merging and inheritance
 */

/**
 * Configuration value that can be loaded from multiple sources
 */
template<typename T>
class ConfigValue {
public:
    ConfigValue(const std::string& key, const T& default_value, const std::string& description = "")
        : key_(key), value_(default_value), default_value_(default_value), description_(description) {}
    
    // Get current value
    const T& get() const { return value_; }
    operator const T&() const { return value_; }
    
    // Set value
    void set(const T& value) { value_ = value; }
    ConfigValue& operator=(const T& value) { set(value); return *this; }
    
    // Metadata
    const std::string& key() const { return key_; }
    const std::string& description() const { return description_; }
    const T& defaultValue() const { return default_value_; }
    bool isDefault() const { return value_ == default_value_; }
    
    // Validation
    using ValidatorFunc = std::function<bool(const T&, std::string&)>;
    void setValidator(ValidatorFunc validator) { validator_ = validator; }
    bool validate(std::string& error) const {
        if (validator_) {
            return validator_(value_, error);
        }
        return true;
    }
    
private:
    std::string key_;
    T value_;
    T default_value_;
    std::string description_;
    ValidatorFunc validator_;
};

/**
 * Configuration section grouping related settings
 */
class ConfigSection {
public:
    explicit ConfigSection(const std::string& name) : name_(name) {}
    
    const std::string& name() const { return name_; }
    
    // Add configuration values
    template<typename T>
    ConfigValue<T>& add(const std::string& key, const T& default_value, const std::string& description = "") {
        auto config_value = std::make_unique<ConfigValue<T>>(key, default_value, description);
        auto* ptr = config_value.get();
        values_[key] = std::move(config_value);
        return *ptr;
    }
    
    // Get values
    template<typename T>
    ConfigValue<T>* get(const std::string& key) {
        auto it = values_.find(key);
        if (it != values_.end()) {
            return dynamic_cast<ConfigValue<T>*>(it->second.get());
        }
        return nullptr;
    }
    
    // List all keys
    std::vector<std::string> getKeys() const {
        std::vector<std::string> keys;
        for (const auto& [key, _] : values_) {
            keys.push_back(key);
        }
        return keys;
    }
    
    // Validation
    std::vector<std::string> validate() const {
        std::vector<std::string> errors;
        for (const auto& [key, value] : values_) {
            std::string error;
            // Note: This is simplified - in reality we'd need type-erased validation
            errors.push_back("Validation for " + key + " not implemented");
        }
        return errors;
    }
    
private:
    std::string name_;
    std::unordered_map<std::string, std::unique_ptr<void, void(*)(void*)>> values_;
};

/**
 * Configuration profile (e.g., development, production, test)
 */
struct ConfigProfile {
    std::string name;
    std::string description;
    std::unordered_map<std::string, std::string> overrides;  // key -> value
    bool inherit_from_default = true;
};

/**
 * Main configuration manager
 */
class ConfigurationManager {
public:
    /**
     * Constructor
     */
    ConfigurationManager();
    
    /**
     * Destructor
     */
    ~ConfigurationManager();
    
    // === Profile Management ===
    
    /**
     * Set active configuration profile
     */
    void setProfile(const std::string& profile_name);
    
    /**
     * Get current profile name
     */
    const std::string& getCurrentProfile() const { return current_profile_; }
    
    /**
     * Add configuration profile
     */
    void addProfile(const ConfigProfile& profile);
    
    /**
     * List available profiles
     */
    std::vector<std::string> getProfiles() const;
    
    // === Configuration Loading ===
    
    /**
     * Load configuration from JSON file
     */
    bool loadFromFile(const std::string& filename);
    
    /**
     * Load configuration from JSON string
     */
    bool loadFromString(const std::string& json);
    
    /**
     * Load configuration from environment variables
     * Variables should be prefixed with DIGISTAR_ (e.g., DIGISTAR_MAX_PARTICLES=1000000)
     */
    void loadFromEnvironment(const std::string& prefix = "DIGISTAR_");
    
    /**
     * Save current configuration to file
     */
    bool saveToFile(const std::string& filename) const;
    
    // === Hot Reloading ===
    
    /**
     * Enable hot reloading of configuration file
     */
    void enableHotReload(const std::string& filename, std::chrono::milliseconds check_interval = std::chrono::milliseconds(1000));
    
    /**
     * Disable hot reloading
     */
    void disableHotReload();
    
    /**
     * Check for configuration changes (manual trigger)
     */
    bool checkForChanges();
    
    /**
     * Set callback for configuration changes
     */
    void setChangeCallback(std::function<void(const std::vector<std::string>&)> callback);
    
    // === Configuration Access ===
    
    /**
     * Get configuration value
     */
    template<typename T>
    T getValue(const std::string& key, const T& default_value = T{}) const {
        // This is simplified - in reality we'd parse the key path and look up the value
        return default_value;
    }
    
    /**
     * Set configuration value
     */
    template<typename T>
    void setValue(const std::string& key, const T& value) {
        // Implementation would set the value and potentially trigger callbacks
    }
    
    /**
     * Check if key exists
     */
    bool hasKey(const std::string& key) const;
    
    /**
     * Get all keys with given prefix
     */
    std::vector<std::string> getKeysWithPrefix(const std::string& prefix) const;
    
    // === Validation ===
    
    /**
     * Validate current configuration
     */
    std::vector<std::string> validate() const;
    
    /**
     * Check if configuration is valid
     */
    bool isValid() const { return validate().empty(); }
    
    // === Serialization ===
    
    /**
     * Export configuration as JSON
     */
    std::string toJson() const;
    
    /**
     * Export configuration as environment variables
     */
    std::string toEnvironmentScript() const;
    
    /**
     * Get configuration summary
     */
    std::string getSummary() const;
    
    // === Built-in Configurations ===
    
    /**
     * Get DigiStar-specific configuration sections
     */
    struct DigiStarConfig {
        // Backend configuration
        ConfigValue<std::string> backend_type{"backend.type", "cpu", "Physics backend type (cpu, cuda, distributed)"};
        ConfigValue<size_t> max_particles{"backend.max_particles", 100000, "Maximum number of particles"};
        ConfigValue<size_t> max_springs{"backend.max_springs", 500000, "Maximum number of springs"};
        ConfigValue<size_t> max_contacts{"backend.max_contacts", 100000, "Maximum number of active contacts"};
        ConfigValue<float> world_size{"backend.world_size", 10000.0f, "World size for toroidal wrapping"};
        
        // Physics configuration
        ConfigValue<bool> enable_gravity{"physics.gravity.enable", true, "Enable gravity calculations"};
        ConfigValue<float> gravity_strength{"physics.gravity.strength", 6.67430e-11f, "Gravitational constant"};
        ConfigValue<std::string> gravity_mode{"physics.gravity.mode", "particle_mesh", "Gravity algorithm (direct, particle_mesh)"};
        
        ConfigValue<bool> enable_contacts{"physics.contacts.enable", true, "Enable contact forces"};
        ConfigValue<float> contact_stiffness{"physics.contacts.stiffness", 1000.0f, "Contact stiffness coefficient"};
        ConfigValue<float> contact_damping{"physics.contacts.damping", 10.0f, "Contact damping coefficient"};
        
        ConfigValue<bool> enable_springs{"physics.springs.enable", true, "Enable spring forces"};
        ConfigValue<float> spring_break_strain{"physics.springs.break_strain", 0.5f, "Spring breaking strain"};
        ConfigValue<float> spring_formation_distance{"physics.springs.formation_distance", 10.0f, "Distance for automatic spring formation"};
        
        // Event system configuration
        ConfigValue<bool> enable_events{"events.enable", true, "Enable event system"};
        ConfigValue<std::string> event_shm_name{"events.shm_name", "digistar_events", "Shared memory name for events"};
        ConfigValue<bool> auto_create_events{"events.auto_create", true, "Auto-create event buffer"};
        ConfigValue<size_t> event_buffer_size{"events.buffer_size", 65536, "Event buffer capacity"};
        
        // DSL configuration
        ConfigValue<bool> enable_dsl{"dsl.enable", true, "Enable DSL scripting"};
        ConfigValue<std::string> script_directory{"dsl.script_directory", "scripts/", "Directory for DSL scripts"};
        
        // Performance configuration
        ConfigValue<float> target_fps{"performance.target_fps", 60.0f, "Target frames per second"};
        ConfigValue<bool> adaptive_timestep{"performance.adaptive_timestep", false, "Use adaptive timestep"};
        ConfigValue<float> min_dt{"performance.min_dt", 1e-6f, "Minimum timestep (adaptive mode)"};
        ConfigValue<float> max_dt{"performance.max_dt", 1e-2f, "Maximum timestep (adaptive mode)"};
        ConfigValue<bool> separate_physics_thread{"performance.separate_physics_thread", true, "Use separate thread for physics"};
        ConfigValue<bool> separate_event_thread{"performance.separate_event_thread", false, "Use separate thread for events"};
        ConfigValue<size_t> num_threads{"performance.num_threads", 0, "Number of threads (0 = auto-detect)"};
        
        // Monitoring configuration
        ConfigValue<bool> enable_monitoring{"monitoring.enable", true, "Enable performance monitoring"};
        ConfigValue<int> monitor_interval_ms{"monitoring.interval_ms", 1000, "Monitoring interval in milliseconds"};
        ConfigValue<bool> log_performance{"monitoring.log_performance", false, "Log performance statistics"};
        ConfigValue<std::string> log_file{"monitoring.log_file", "digistar.log", "Log file path"};
        
        // Error handling
        ConfigValue<bool> continue_on_backend_error{"error.continue_on_backend_error", false, "Continue simulation on backend errors"};
        ConfigValue<bool> continue_on_dsl_error{"error.continue_on_dsl_error", true, "Continue simulation on DSL errors"};
    };
    
    /**
     * Get DigiStar configuration instance
     */
    DigiStarConfig& getDigiStarConfig() { return digistar_config_; }
    const DigiStarConfig& getDigiStarConfig() const { return digistar_config_; }
    
    // === Static Helpers ===
    
    /**
     * Get global configuration instance
     */
    static ConfigurationManager& getInstance();
    
    /**
     * Load default DigiStar configuration
     */
    static std::unique_ptr<ConfigurationManager> createDefault();
    
    /**
     * Create configuration from preset
     */
    static std::unique_ptr<ConfigurationManager> fromPreset(const std::string& preset_name);
    
private:
    std::string current_profile_ = "default";
    std::unordered_map<std::string, ConfigProfile> profiles_;
    
    // Configuration data (simplified - would be more structured in practice)
    std::unordered_map<std::string, std::string> values_;
    
    // Hot reloading
    bool hot_reload_enabled_ = false;
    std::string watched_file_;
    std::chrono::milliseconds check_interval_{1000};
    std::chrono::system_clock::time_point last_write_time_;
    std::unique_ptr<std::thread> watch_thread_;
    std::atomic<bool> stop_watching_{false};
    std::function<void(const std::vector<std::string>&)> change_callback_;
    
    // Built-in configurations
    DigiStarConfig digistar_config_;
    
    // Internal methods
    void watchFileChanges();
    bool parseJsonString(const std::string& json);
    std::string serializeToJson() const;
    void applyProfile(const std::string& profile_name);
    std::string getEnvironmentValue(const std::string& key, const std::string& prefix) const;
};

/**
 * Configuration presets for common scenarios
 */
namespace ConfigPresets {
    
    /**
     * Get minimal configuration for testing
     */
    std::unique_ptr<ConfigurationManager> minimal();
    
    /**
     * Get development configuration with full monitoring
     */
    std::unique_ptr<ConfigurationManager> development();
    
    /**
     * Get production configuration optimized for performance
     */
    std::unique_ptr<ConfigurationManager> production();
    
    /**
     * Get configuration for galaxy formation simulations
     */
    std::unique_ptr<ConfigurationManager> galaxyFormation();
    
    /**
     * Get configuration for planetary system simulations
     */
    std::unique_ptr<ConfigurationManager> planetarySystem();
    
    /**
     * Get configuration for particle physics simulations
     */
    std::unique_ptr<ConfigurationManager> particlePhysics();
    
} // namespace ConfigPresets

/**
 * Configuration utilities
 */
namespace ConfigUtils {
    
    /**
     * Convert configuration to simulation config structs
     */
    IntegratedSimulationConfig toIntegratedConfig(const ConfigurationManager& config);
    
    /**
     * Create configuration manager from integrated config
     */
    std::unique_ptr<ConfigurationManager> fromIntegratedConfig(const IntegratedSimulationConfig& config);
    
    /**
     * Merge two configuration managers
     */
    std::unique_ptr<ConfigurationManager> merge(const ConfigurationManager& base, 
                                               const ConfigurationManager& overlay);
    
    /**
     * Validate configuration against schema
     */
    std::vector<std::string> validateAgainstSchema(const ConfigurationManager& config, 
                                                  const std::string& schema_file);
    
    /**
     * Generate configuration documentation
     */
    std::string generateDocumentation(const ConfigurationManager& config);
    
} // namespace ConfigUtils

} // namespace digistar