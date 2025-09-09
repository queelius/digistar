#pragma once

#include "sexpr.h"
#include "command.h"
#include "thread_pool.h"
#include "../core/simulation.h"
#include <unordered_map>
#include <chrono>
#include <atomic>

namespace digistar {
namespace dsl {

// Forward declaration
class ScriptManager;

// Script execution context
class Script {
public:
    enum State {
        IDLE,
        RUNNING,
        PAUSED,
        COMPLETED,
        ERROR
    };
    
private:
    std::string name;
    std::string code;
    std::shared_ptr<Environment> env;
    State state = IDLE;
    
    // For persistent scripts
    bool persistent = false;
    float update_interval = 0.0f;  // 0 = every frame
    std::chrono::steady_clock::time_point last_update;
    
    // Statistics
    size_t execution_count = 0;
    std::chrono::milliseconds total_time{0};
    
    // Error info
    std::string last_error;
    
    // Parent manager
    ScriptManager* manager = nullptr;
    
public:
    Script(const std::string& n, const std::string& c, bool p = false)
        : name(n), code(c), persistent(p) {
        env = std::make_shared<Environment>();
    }
    
    // Execute the script
    void execute(CommandQueue& queue);
    
    // Getters
    const std::string& getName() const { return name; }
    const std::string& getCode() const { return code; }
    State getState() const { return state; }
    bool isPersistent() const { return persistent; }
    bool needsUpdate(float dt) const;
    const std::string& getLastError() const { return last_error; }
    
    // Statistics
    size_t getExecutionCount() const { return execution_count; }
    std::chrono::milliseconds getTotalTime() const { return total_time; }
    float getAverageTime() const {
        return execution_count > 0 ? 
            total_time.count() / static_cast<float>(execution_count) : 0.0f;
    }
    
    // Control
    void pause() { state = PAUSED; }
    void resume() { if (state == PAUSED) state = IDLE; }
    void stop() { state = COMPLETED; }
    
    // Set manager
    void setManager(ScriptManager* m) { manager = m; }
};

// Manages script execution with thread pool
class ScriptManager {
private:
    // Command queue for all scripts
    CommandQueue command_queue;
    
    // Active scripts
    std::unordered_map<std::string, std::shared_ptr<Script>> scripts;
    mutable std::mutex scripts_mutex;
    
    // Base environment with built-in functions
    std::shared_ptr<Environment> base_env;
    
    // Simulation reference (read-only)
    const Simulation* simulation = nullptr;
    
    // Statistics
    std::atomic<size_t> total_scripts_run{0};
    std::atomic<size_t> scripts_running{0};
    
    // Configuration
    bool auto_execute = true;  // Automatically execute scripts each frame
    size_t max_concurrent = 10;  // Max scripts running at once
    
    void setupBaseEnvironment();
    
public:
    ScriptManager();
    ~ScriptManager();
    
    // Set simulation reference
    void setSimulation(const Simulation* sim) { simulation = sim; }
    
    // Thread pool configuration
    void setThreadCount(size_t count) {
        ScriptExecutor::set_thread_count(count);
    }
    
    size_t getThreadCount() const {
        return ScriptExecutor::get_thread_count();
    }
    
    // Script management
    std::shared_ptr<Script> loadScript(const std::string& name, 
                                       const std::string& code,
                                       bool persistent = false);
    
    std::shared_ptr<Script> loadScriptFile(const std::string& name,
                                           const std::string& filename,
                                           bool persistent = false);
    
    void removeScript(const std::string& name);
    std::shared_ptr<Script> getScript(const std::string& name);
    std::vector<std::string> getScriptNames() const;
    
    // Execute a single script
    std::future<void> executeScript(const std::string& name);
    
    // Execute code directly (one-shot)
    std::future<void> executeCode(const std::string& code);
    
    // Update all persistent scripts (called each frame)
    void update(float dt);
    
    // Apply all pending commands to simulation
    void applyCommands(Simulation& sim) {
        command_queue.executeAll(sim);
    }
    
    // Control
    void pauseAll();
    void resumeAll();
    void stopAll();
    void clearAll();
    
    // Statistics
    size_t getTotalScriptsRun() const { return total_scripts_run.load(); }
    size_t getScriptsRunning() const { return scripts_running.load(); }
    size_t getPendingCommands() const { return command_queue.size(); }
    
    // Access to command queue (for scripts)
    CommandQueue& getCommandQueue() { return command_queue; }
    const Simulation* getSimulation() const { return simulation; }
    std::shared_ptr<Environment> getBaseEnvironment() const { return base_env; }
};

// RAII helper for script execution tracking
class ScriptExecutionGuard {
private:
    std::atomic<size_t>& counter;
    
public:
    explicit ScriptExecutionGuard(std::atomic<size_t>& c) : counter(c) {
        counter++;
    }
    
    ~ScriptExecutionGuard() {
        counter--;
    }
};

} // namespace dsl
} // namespace digistar