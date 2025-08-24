#pragma once

#include "sexpr.h"
#include "command.h"
#include "../core/simulation.h"
#include <functional>
#include <memory>
#include <atomic>
#include <thread>
#include <future>

namespace digistar {
namespace dsl {

// Forward declarations
class DslEvaluator;

// Script execution context (simplified - no continuations needed!)
class ScriptContext {
public:
    std::string name;
    std::shared_ptr<Environment> env;
    
    // For persistent scripts
    bool is_persistent = false;
    bool is_active = true;
    float update_interval = 0.0f;  // 0 = every frame
    float last_update = 0.0f;
    
    // Script-local state
    std::unordered_map<std::string, SExprPtr> state;
    
    ScriptContext(const std::string& n = "anonymous") 
        : name(n), env(std::make_shared<Environment>()) {}
    
    ScriptContext(const std::string& n, std::shared_ptr<Environment> e)
        : name(n), env(e) {}
};

// Built-in function type (now returns values, not commands)
using BuiltinFunc = std::function<SExprPtr(
    const std::vector<SExprPtr>& args,
    ScriptContext& ctx
)>;

// Thread pool for script execution
class ScriptThreadPool {
private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};
    
    void worker();
    
public:
    explicit ScriptThreadPool(size_t num_threads = 0);  // 0 = auto-detect
    ~ScriptThreadPool();
    
    // Submit task for execution
    template<typename F>
    auto submit(F&& f) -> std::future<decltype(f())>;
    
    // Get number of threads
    size_t size() const { return threads.size(); }
    
    // Reconfigure thread count
    void resize(size_t num_threads);
};

// DSL Evaluator with command queue architecture
class DslEvaluator {
private:
    // Simulation reference (read-only access)
    const Simulation* simulation = nullptr;
    
    // Command queue for mutations
    CommandQueue command_queue;
    CommandFactory command_factory{command_queue};
    
    // Thread pool for script execution
    std::unique_ptr<ScriptThreadPool> thread_pool;
    
    // Built-in functions
    std::unordered_map<std::string, BuiltinFunc> builtins;
    
    // Base environment with built-ins
    std::shared_ptr<Environment> base_env;
    
    // Active persistent scripts
    std::vector<std::shared_ptr<ScriptContext>> persistent_scripts;
    std::mutex scripts_mutex;
    
    // ID mapping for provisional IDs
    std::unordered_map<int, int> provisional_to_actual;
    std::mutex id_map_mutex;
    
    // Helper functions
    void registerBuiltins();
    void setupBaseEnvironment();
    
    // Evaluation core (pure - no side effects)
    SExprPtr evalList(const std::vector<SExprPtr>& list, ScriptContext& ctx);
    SExprPtr evalAtom(SExprPtr expr, ScriptContext& ctx);
    bool isTruthy(SExprPtr expr) const;
    
public:
    DslEvaluator(size_t num_threads = 1);
    ~DslEvaluator() = default;
    
    // Connect to simulation (read-only access)
    void setSimulation(const Simulation* sim) { simulation = sim; }
    
    // Configure thread pool
    void setThreadCount(size_t num_threads);
    size_t getThreadCount() const { return thread_pool->size(); }
    
    // Main evaluation (generates commands, doesn't execute them)
    SExprPtr eval(SExprPtr expr, ScriptContext& ctx);
    SExprPtr eval(SExprPtr expr);  // With fresh context
    
    // Execute string/file
    SExprPtr evalString(const std::string& code);
    std::vector<SExprPtr> evalFile(const std::string& filename);
    
    // Script management
    void addPersistentScript(const std::string& name, SExprPtr code);
    void removeScript(const std::string& name);
    
    // Called each frame by simulation
    void updateScripts(float dt);
    void applyCommands(Simulation& sim);
    
    // Special forms
    SExprPtr evalIf(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalLet(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalLambda(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalDefine(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalSet(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalBegin(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalWhen(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalQuote(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    
    // Apply function/closure
    SExprPtr apply(SExprPtr func, const std::vector<SExprPtr>& args, ScriptContext& ctx);
    
    // Provisional ID management
    void mapProvisionalId(int provisional, int actual);
    int resolveId(int id) const;
    
    // Access for built-ins
    CommandFactory& getCommandFactory() { return command_factory; }
    const Simulation* getSimulation() const { return simulation; }
};

// ============ Built-in Functions ============

// These now generate commands instead of directly modifying state

namespace builtins {

// Particle creation
SExprPtr particle(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr cloud(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr star(const std::vector<SExprPtr>& args, ScriptContext& ctx);

// Spring creation
SExprPtr spring(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr spring_mesh(const std::vector<SExprPtr>& args, ScriptContext& ctx);

// Particle control
SExprPtr set_velocity(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr apply_force(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr destroy(const std::vector<SExprPtr>& args, ScriptContext& ctx);

// Queries (read-only, safe to do directly)
SExprPtr find(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr measure(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr distance(const std::vector<SExprPtr>& args, ScriptContext& ctx);

// System control
SExprPtr set_gravity(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr set_dt(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr pause(const std::vector<SExprPtr>& args, ScriptContext& ctx);

// Thread control
SExprPtr set_script_threads(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr get_script_threads(const std::vector<SExprPtr>& args, ScriptContext& ctx);

// Math functions (pure, no commands)
SExprPtr add(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr subtract(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr multiply(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr divide(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr random_num(const std::vector<SExprPtr>& args, ScriptContext& ctx);

// Vector operations
SExprPtr vec_add(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr vec_sub(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr vec_scale(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr vec_dot(const std::vector<SExprPtr>& args, ScriptContext& ctx);
SExprPtr vec_length(const std::vector<SExprPtr>& args, ScriptContext& ctx);

} // namespace builtins

} // namespace dsl
} // namespace digistar