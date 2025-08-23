#pragma once

#include <memory>
#include <string>
#include <chrono>
#include "../backend/backend_interface.h"
#include "../visualization/ascii_renderer.h"

namespace digistar {

// Main simulation class that ties everything together
class Simulation {
public:
    struct Config {
        // Simulation parameters
        float timestep = 0.01f;
        float target_fps = 60.0f;
        bool fixed_timestep = true;
        bool pause_on_start = false;
        
        // Physics configuration
        PhysicsConfig physics;
        
        // Backend selection
        BackendFactory::Type backend_type = BackendFactory::Type::CPU;
        
        // World setup
        float world_size = 10000.0f;
        bool use_toroidal = true;
        
        // Capacity limits
        size_t max_particles = 100000;
        size_t max_springs = 500000;
        size_t max_contacts = 10000;
        
        // Scenario to load
        std::string scenario_name = "default";
        
        // Rendering
        bool enable_rendering = true;
        bool use_ansi_colors = true;
        AsciiRenderer::Config renderer_config;
        
        // Logging
        bool enable_logging = false;
        std::string log_file = "simulation.log";
        
        // Checkpointing
        bool enable_checkpoints = false;
        float checkpoint_interval = 60.0f;  // seconds
        std::string checkpoint_dir = "./checkpoints";
    };
    
    enum class State {
        UNINITIALIZED,
        READY,
        RUNNING,
        PAUSED,
        STOPPED,
        ERROR
    };
    
private:
    Config config;
    State state = State::UNINITIALIZED;
    
    // Core components
    std::unique_ptr<IBackend> backend;
    std::unique_ptr<SimulationState> sim_state;
    std::unique_ptr<AsciiRenderer> renderer;
    std::unique_ptr<TerminalDisplay> terminal;
    
    // Timing
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point last_frame_time;
    float simulation_time = 0;
    float real_time_elapsed = 0;
    size_t frame_count = 0;
    float current_fps = 0;
    
    // Performance tracking
    float avg_physics_time = 0;
    float avg_render_time = 0;
    float time_scale = 1.0f;  // Speed up/slow down simulation
    
    // Checkpoint management
    std::chrono::high_resolution_clock::time_point last_checkpoint;
    
    // Internal methods
    void initializeBackend();
    void initializeSimulationState();
    void initializeRenderer();
    void loadScenario(const std::string& name);
    void updateTiming();
    void handleCheckpointing();
    
public:
    Simulation(const Config& cfg = Config());
    ~Simulation();
    
    // Lifecycle
    bool initialize();
    void shutdown();
    void reset();
    
    // Main loop
    void run();
    void step();
    void pause() { state = State::PAUSED; }
    void resume() { state = State::RUNNING; }
    void stop() { state = State::STOPPED; }
    
    // State management
    State getState() const { return state; }
    bool isRunning() const { return state == State::RUNNING; }
    bool isPaused() const { return state == State::PAUSED; }
    
    // Time control
    void setTimeScale(float scale) { time_scale = scale; }
    float getTimeScale() const { return time_scale; }
    float getSimulationTime() const { return simulation_time; }
    float getRealTimeElapsed() const { return real_time_elapsed; }
    
    // Performance metrics
    float getFPS() const { return current_fps; }
    float getPhysicsTime() const { return avg_physics_time; }
    float getRenderTime() const { return avg_render_time; }
    SimulationStats getStats() const { return backend ? backend->getStats() : SimulationStats(); }
    
    // Particle management
    uint32_t addParticle(float x, float y, float vx, float vy, float mass, float radius);
    void removeParticle(uint32_t id);
    void addSpring(uint32_t p1, uint32_t p2, float stiffness = 100.0f, float damping = 1.0f);
    void breakSpring(uint32_t spring_id);
    
    // Force application
    void applyForce(uint32_t particle_id, float fx, float fy);
    void applyImpulse(uint32_t particle_id, float ix, float iy);
    void applyTorque(uint32_t composite_id, float torque);
    
    // Camera control
    void setCameraCenter(float x, float y);
    void setCameraScale(float scale);
    void trackParticle(int32_t particle_id);
    void zoomIn() { if (renderer) renderer->zoomIn(); }
    void zoomOut() { if (renderer) renderer->zoomOut(); }
    
    // Configuration
    Config& getConfig() { return config; }
    const Config& getConfig() const { return config; }
    
    // Serialization
    bool saveState(const std::string& filename);
    bool loadState(const std::string& filename);
    bool saveCheckpoint();
    bool loadCheckpoint(const std::string& filename);
    
    // Direct access (for advanced usage)
    SimulationState* getSimulationState() { return sim_state.get(); }
    IBackend* getBackend() { return backend.get(); }
    AsciiRenderer* getRenderer() { return renderer.get(); }
};

// Scenario loader - creates predefined simulation setups
class ScenarioLoader {
public:
    static void loadDefault(SimulationState& state);
    static void loadSolarSystem(SimulationState& state);
    static void loadGalaxyCollision(SimulationState& state);
    static void loadAsteroidField(SimulationState& state);
    static void loadSpringNetwork(SimulationState& state);
    static void loadCompositeTest(SimulationState& state);
    static void loadStressTest(SimulationState& state, size_t particle_count);
    
    // Load from file
    static bool loadFromFile(SimulationState& state, const std::string& filename);
    static bool saveToFile(const SimulationState& state, const std::string& filename);
};

// Input handler for interactive control
class InputHandler {
public:
    enum class Command {
        NONE,
        QUIT,
        PAUSE,
        STEP,
        ZOOM_IN,
        ZOOM_OUT,
        PAN_LEFT,
        PAN_RIGHT,
        PAN_UP,
        PAN_DOWN,
        SPEED_UP,
        SLOW_DOWN,
        RESET_TIME,
        TOGGLE_GRID,
        TOGGLE_SPRINGS,
        TOGGLE_VELOCITIES,
        TOGGLE_FORCES,
        TOGGLE_TEMPERATURE,
        TOGGLE_COMPOSITES,
        TOGGLE_STATS,
        CYCLE_TRACKING,
        SAVE_STATE,
        LOAD_STATE,
        SPAWN_PARTICLE,
        DELETE_PARTICLE,
        APPLY_FORCE
    };
    
private:
    bool quit_requested = false;
    Command last_command = Command::NONE;
    
    // For non-blocking input
    bool setupTerminal();
    void restoreTerminal();
    char getCharNonBlocking();
    
public:
    InputHandler();
    ~InputHandler();
    
    Command pollCommand();
    bool shouldQuit() const { return quit_requested; }
    
    // Process command on simulation
    void processCommand(Command cmd, Simulation& sim);
    
    // Get help text
    static std::string getHelpText();
};

} // namespace digistar