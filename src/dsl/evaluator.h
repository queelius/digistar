#pragma once

#include "sexpr.h"
#include "../physics/pools.h"
#include "../backend/backend_interface.h"
#include <functional>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <queue>
#include <random>

namespace digistar {
namespace dsl {

// Forward declarations
class DslEvaluator;
using ContinuationFunc = std::function<SExprPtr(DslEvaluator&)>;

// Script execution context
class ScriptContext {
public:
    std::string name;
    std::shared_ptr<Environment> env;
    
    // For persistent/background scripts
    bool is_persistent = false;
    bool is_active = true;
    float update_interval = 0.0f;  // 0 = every frame
    float last_update = 0.0f;
    
    // Continuation for tail recursion/loops
    ContinuationFunc continuation = nullptr;
    
    // Script state for behaviors
    std::unordered_map<std::string, SExprPtr> state;
    
    ScriptContext(const std::string& n = "anonymous") 
        : name(n), env(std::make_shared<Environment>()) {}
    
    ScriptContext(const std::string& n, std::shared_ptr<Environment> e)
        : name(n), env(e) {}
};

// Built-in function type
using BuiltinFunc = std::function<SExprPtr(
    DslEvaluator& eval,
    const std::vector<SExprPtr>& args,
    ScriptContext& ctx
)>;

// DSL Evaluator - interprets S-expressions and modifies simulation state
class DslEvaluator {
private:
    SimulationState* sim_state = nullptr;
    PhysicsConfig* physics_config = nullptr;
    
    // Built-in functions
    std::unordered_map<std::string, BuiltinFunc> builtins;
    
    // Base environment containing all built-ins and primitives
    std::shared_ptr<Environment> base_env;
    
    // Active script contexts (for persistent behaviors)
    std::vector<std::shared_ptr<ScriptContext>> active_scripts;
    
    // Random number generation
    std::mt19937 rng{std::random_device{}()};
    
    // Performance tracking
    size_t eval_count = 0;
    float total_eval_time = 0;
    
    // Helper functions
    void registerBuiltins();
    void registerCreationFunctions();
    void registerQueryFunctions();
    void registerControlFunctions();
    void registerBehaviorFunctions();
    void registerMathFunctions();
    
    // Evaluation helpers
    SExprPtr evalList(const std::vector<SExprPtr>& list, ScriptContext& ctx);
    SExprPtr evalAtom(SExprPtr expr, ScriptContext& ctx);
    bool isTruthy(SExprPtr expr) const;
    
    // Particle creation helpers
    size_t createParticle(float x, float y, float vx, float vy, 
                         float mass, float radius, float temp);
    size_t createCloud(float cx, float cy, float radius, size_t count,
                      float mass_min, float mass_max, float temp);
    size_t createOrbitalSystem(float cx, float cy, float central_mass,
                              const std::vector<SExprPtr>& orbits);
    
    // Spring/composite creation
    void createSpring(size_t p1, size_t p2, float stiffness, float damping,
                     float rest_length = -1.0f);
    void createSpringMesh(const std::vector<size_t>& particles, 
                         float stiffness, float damping);
    size_t createComposite(const std::vector<size_t>& particles);
    
    // Query helpers
    std::vector<size_t> findParticlesInRegion(float x, float y, float radius);
    std::vector<size_t> findParticlesByProperty(
        const std::string& prop, float min_val, float max_val);
    SExprPtr measureProperty(const std::string& prop, 
                            const std::vector<size_t>& particles);
    
public:
    DslEvaluator();
    ~DslEvaluator() = default;
    
    // Connect to simulation
    void setSimulationState(SimulationState* state) { sim_state = state; }
    void setPhysicsConfig(PhysicsConfig* config) { physics_config = config; }
    
    // Main evaluation
    SExprPtr eval(SExprPtr expr, ScriptContext& ctx);
    SExprPtr eval(SExprPtr expr);  // With default context
    
    // Execute string/file
    SExprPtr evalString(const std::string& code);
    SExprPtr evalFile(const std::string& filename);
    
    // Script management
    std::shared_ptr<ScriptContext> createScript(
        const std::string& name, SExprPtr code, bool persistent = false);
    void stopScript(const std::string& name);
    void updateScripts(float dt);  // Call each frame for persistent scripts
    
    // Special forms
    SExprPtr evalIf(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalLet(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalLambda(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalDefine(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalSet(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalQuote(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalLoop(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    SExprPtr evalWhen(const std::vector<SExprPtr>& args, ScriptContext& ctx);
    
    // Apply a closure/function
    SExprPtr apply(SExprPtr func, const std::vector<SExprPtr>& args, ScriptContext& ctx);
    
    // Tail recursion support
    SExprPtr evalWithTailRecursion(SExprPtr expr, ScriptContext& ctx);
    
    // Access to simulation state (for built-ins)
    SimulationState* getState() { return sim_state; }
    PhysicsConfig* getConfig() { return physics_config; }
    
    // Performance stats
    size_t getEvalCount() const { return eval_count; }
    float getAverageEvalTime() const { 
        return eval_count > 0 ? total_eval_time / eval_count : 0; 
    }
};

// Example built-in functions

// Creation functions
SExprPtr builtin_particle(DslEvaluator& eval, const std::vector<SExprPtr>& args, 
                          ScriptContext& ctx);
SExprPtr builtin_cloud(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                       ScriptContext& ctx);
SExprPtr builtin_star(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                      ScriptContext& ctx);
SExprPtr builtin_planet(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                        ScriptContext& ctx);
SExprPtr builtin_spring(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                        ScriptContext& ctx);

// Query functions  
SExprPtr builtin_query(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                       ScriptContext& ctx);
SExprPtr builtin_measure(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                         ScriptContext& ctx);
SExprPtr builtin_find(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                      ScriptContext& ctx);

// Control functions
SExprPtr builtin_set_velocity(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                              ScriptContext& ctx);
SExprPtr builtin_apply_force(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                             ScriptContext& ctx);
SExprPtr builtin_explode(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                         ScriptContext& ctx);

// Behavior functions (for persistent scripts)
SExprPtr builtin_watch(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                       ScriptContext& ctx);
SExprPtr builtin_rule(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                      ScriptContext& ctx);
SExprPtr builtin_trigger(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                         ScriptContext& ctx);

// Math/utility functions
SExprPtr builtin_random(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                        ScriptContext& ctx);
SExprPtr builtin_distance(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                          ScriptContext& ctx);
SExprPtr builtin_vector_add(DslEvaluator& eval, const std::vector<SExprPtr>& args,
                            ScriptContext& ctx);

} // namespace dsl
} // namespace digistar