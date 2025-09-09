#pragma once

#include "sexpr.h"
#include "pattern_matcher.h"
#include "procedural_generator.h"
#include "bytecode_compiler.h"
#include "bytecode_vm.h"
#include "event_bridge.h"
#include "../physics/pools.h"
#include <memory>
#include <string>
#include <chrono>

namespace digistar {

// Forward declaration
struct SimulationState;

namespace dsl {

/**
 * Unified DSL runtime combining all enhanced features
 * 
 * This is the main interface for the DigiStar DSL system, providing:
 * - Pattern matching
 * - Procedural generation
 * - Bytecode compilation and execution
 * - Event system integration
 * - Performance optimization
 */
class DSLRuntime {
public:
    // Performance tracking
    struct Performance {
        size_t total_compilations = 0;
        size_t total_executions = 0;
        std::chrono::microseconds total_compile_time{0};
        std::chrono::microseconds total_execution_time{0};
        size_t cache_hits = 0;
        size_t cache_misses = 0;
    };

private:
    // Core components
    SimulationState* sim_state;
    std::unique_ptr<BytecodeCompiler> compiler;
    std::unique_ptr<BytecodeVM> vm;
    std::unique_ptr<ProceduralGenerator> generator;
    std::unique_ptr<EventBridge> event_bridge;
    
    // Environment
    std::shared_ptr<Environment> global_env;
    
    // Script management
    struct LoadedScript {
        std::string name;
        std::string source;
        std::shared_ptr<BytecodeChunk> compiled;
        std::chrono::steady_clock::time_point last_run;
        size_t run_count = 0;
        bool auto_run = false;
        std::chrono::milliseconds interval{0};
    };
    
    std::unordered_map<std::string, LoadedScript> scripts;
    
    Performance performance;
    
    // Helper methods
    void registerBuiltinFunctions();
    void registerPatternMatchingForms();
    void registerGeneratorFunctions();
    void registerEventHandlers();
    
    std::shared_ptr<BytecodeChunk> compileScript(const std::string& source,
                                                 const std::string& name);
    
public:
    explicit DSLRuntime(SimulationState* state,
                       const std::string& event_shm_name = "digistar_events");
    ~DSLRuntime();
    
    // Script execution
    SExprPtr execute(const std::string& source);
    SExprPtr executeFile(const std::string& filename);
    SExprPtr executeCompiled(std::shared_ptr<BytecodeChunk> chunk);
    
    // Script management
    void loadScript(const std::string& name, const std::string& source,
                   bool compile_now = true);
    void loadScriptFile(const std::string& name, const std::string& filename);
    void runScript(const std::string& name);
    void scheduleScript(const std::string& name, std::chrono::milliseconds interval);
    void unloadScript(const std::string& name);
    
    // Pattern matching interface
    MatchResult match(SExprPtr value, PatternPtr pattern);
    SExprPtr matchCase(SExprPtr value, 
                       const std::vector<std::pair<PatternPtr, SExprPtr>>& cases);
    
    // Procedural generation interface
    std::vector<size_t> generate(const std::string& type, size_t count,
                                 const std::unordered_map<std::string, SExprPtr>& params);
    std::vector<size_t> generateGalaxy(size_t stars, float x, float y, float radius);
    std::vector<size_t> generateSolarSystem(float x, float y);
    std::vector<size_t> generateAsteroidField(size_t count, float x, float y,
                                              float inner_r, float outer_r);
    
    // Event handling interface
    void onEvent(EventType type, PatternPtr pattern, SExprPtr action);
    void onEvent(EventType type, const std::string& pattern_expr,
                const std::string& action_expr);
    void emitEvent(EventType type, const std::unordered_map<std::string, SExprPtr>& data);
    
    // Advanced features
    void defineFunction(const std::string& name, SExprPtr params, SExprPtr body);
    void defineMacro(const std::string& name, SExprPtr params, SExprPtr body);
    void definePattern(const std::string& name, PatternPtr pattern);
    
    // Environment access
    void setGlobal(const std::string& name, SExprPtr value);
    SExprPtr getGlobal(const std::string& name) const;
    std::shared_ptr<Environment> getEnvironment() const { return global_env; }
    
    // Component access
    ProceduralGenerator* getGenerator() { return generator.get(); }
    EventBridge* getEventBridge() { return event_bridge.get(); }
    BytecodeVM* getVM() { return vm.get(); }
    BytecodeCompiler* getCompiler() { return compiler.get(); }
    
    // Performance monitoring
    const Performance& getPerformance() const { return performance; }
    void resetPerformance() { performance = Performance(); }
    std::string getPerformanceReport() const;
    
    // Update loop integration
    void update(float dt);
    void processScheduledScripts();
    
    // Debugging
    void enableTracing(bool enable);
    void dumpState(const std::string& filename) const;
    std::string getStackTrace() const;
};

// ============================================================================
// DSLRuntime Implementation (Stub)
// ============================================================================

inline DSLRuntime::DSLRuntime(SimulationState* state,
                              const std::string& event_shm_name) 
    : sim_state(state) {
    // Initialize components
    compiler = std::make_unique<BytecodeCompiler>();
    vm = std::make_unique<BytecodeVM>(state);
    generator = std::make_unique<ProceduralGenerator>(state);
    
    // Create event ring buffer for testing
    auto buffer = new EventRingBuffer();
    event_bridge = std::make_unique<EventBridge>(buffer, state);
    
    global_env = std::make_shared<Environment>();
    vm->setGlobalEnv(global_env);
    
    // Register built-in functions
    registerBuiltinFunctions();
}

inline DSLRuntime::~DSLRuntime() = default;

inline SExprPtr DSLRuntime::execute(const std::string& source) {
    // Parse the source
    auto expr = SExprParser::parseString(source);
    if (!expr) {
        return SExpr::makeNil();
    }
    
    // Compile and execute
    auto chunk = compiler->compile(expr, "eval");
    return vm->execute(chunk);
}

inline void DSLRuntime::loadScript(const std::string& name, 
                                   const std::string& source,
                                   bool compile_now) {
    LoadedScript script;
    script.name = name;
    script.source = source;
    
    if (compile_now) {
        script.compiled = compileScript(source, name);
    }
    
    scripts[name] = script;
}

inline void DSLRuntime::runScript(const std::string& name) {
    auto it = scripts.find(name);
    if (it == scripts.end()) {
        return;
    }
    
    auto& script = it->second;
    
    // Compile if needed
    if (!script.compiled) {
        script.compiled = compileScript(script.source, name);
    }
    
    // Execute
    vm->execute(script.compiled);
    
    // Update stats
    script.run_count++;
    script.last_run = std::chrono::steady_clock::now();
    performance.total_executions++;
}

inline std::shared_ptr<BytecodeChunk> DSLRuntime::compileScript(const std::string& source,
                                                                const std::string& name) {
    auto expr = SExprParser::parseString(source);
    if (!expr) {
        return nullptr;
    }
    return compiler->compile(expr, name);
}

inline void DSLRuntime::registerBuiltinFunctions() {
    // Register basic functions in the global environment
    // This would be fully implemented in production
}

inline MatchResult DSLRuntime::match(SExprPtr value, PatternPtr pattern) {
    return PatternMatcher::matchPattern(pattern, value);
}

inline std::vector<size_t> DSLRuntime::generateGalaxy(size_t count, float x, float y, float radius) {
    return generator->generateGalaxy(count, x, y, radius);
}

// Note: getPerformance() implementation is already inline in the class declaration

/**
 * DSL REPL (Read-Eval-Print Loop) for interactive development
 */
class DSLRepl {
private:
    DSLRuntime* runtime;
    bool running = false;
    std::string prompt = "digistar> ";
    std::vector<std::string> history;
    size_t history_index = 0;
    
    // REPL commands
    void handleCommand(const std::string& cmd);
    void showHelp();
    void listScripts();
    void showPerformance();
    
public:
    explicit DSLRepl(DSLRuntime* rt) : runtime(rt) {}
    
    void run();
    void stop() { running = false; }
    void setPrompt(const std::string& p) { prompt = p; }
    
    // History management
    void addToHistory(const std::string& line);
    std::string getPreviousHistory();
    std::string getNextHistory();
};

/**
 * DSL standard library
 */
class DSLStandardLibrary {
public:
    // Load standard library into environment
    static void load(std::shared_ptr<Environment> env);
    
    // Math functions
    static void loadMathFunctions(std::shared_ptr<Environment> env);
    
    // List operations
    static void loadListFunctions(std::shared_ptr<Environment> env);
    
    // String operations
    static void loadStringFunctions(std::shared_ptr<Environment> env);
    
    // Physics helpers
    static void loadPhysicsFunctions(std::shared_ptr<Environment> env);
    
    // Utility functions
    static void loadUtilityFunctions(std::shared_ptr<Environment> env);
};

/**
 * Example DSL programs
 */
class DSLExamples {
public:
    // Galaxy formation
    static std::string galaxyFormation() {
        return R"(
; Create a spiral galaxy with realistic stellar distribution
(define (create-galaxy x y radius arms)
  (let ((stars (generate 'galaxy
                :particles 100000
                :center [x y]
                :radius radius
                :arms arms
                :distribution 'salpeter-mass)))
    ; Add event handlers for stellar evolution
    (on-event 'thermal-explosion
      (match event
        [(thermal-explosion ?star (> temp 1e7))
         (trigger-supernova star)]))
    stars))

; Create multiple galaxies
(create-galaxy 0 0 1000 2)
(create-galaxy 5000 5000 800 3)
)";
    }
    
    // Collision handling
    static std::string collisionHandling() {
        return R"(
; Advanced collision handling with pattern matching
(on-event 'collision
  (match event
    ; High-energy collision creates explosion
    [(collision ?p1 ?p2 (> ?energy 10000))
     (begin
       (create-explosion p1 p2 energy)
       (emit-event 'high-energy-collision 
         :particles [p1 p2]
         :energy energy))]
    
    ; Medium energy creates sparks
    [(collision ?p1 ?p2 (and (> ?energy 1000) (<= ?energy 10000)))
     (create-sparks p1 p2 (* energy 0.1))]
    
    ; Low energy is elastic
    [(collision ?p1 ?p2 _)
     (elastic-collision p1 p2)]))
)";
    }
    
    // Procedural asteroid field
    static std::string asteroidField() {
        return R"(
; Generate realistic asteroid belt
(define (create-asteroid-belt star-pos inner-radius outer-radius)
  (generate 'asteroids
    :particles 10000
    :pattern 'disk
    :inner-radius inner-radius
    :outer-radius outer-radius
    :mass-distribution '(power-law 0.001 1.0 -2.5)
    :orbital-velocity star-pos))

; Create belt around star at origin
(create-asteroid-belt [0 0] 200 400)
)";
    }
    
    // Reactive behaviors
    static std::string reactiveBehaviors() {
        return R"(
; Temperature-based phase transitions
(when-change 'particle.temperature
  (lambda (old new particle)
    (cond
      ; Melting point
      [(and (< old 1500) (>= new 1500))
       (set-phase particle 'liquid)]
      
      ; Boiling point
      [(and (< old 3000) (>= new 3000))
       (set-phase particle 'gas)]
      
      ; Fusion ignition
      [(and (< old 1e7) (>= new 1e7))
       (begin
         (set-type particle 'star)
         (emit-event 'fusion-ignition :particle particle))])))

; Spring stress monitoring
(when-change 'spring.stress
  (lambda (old new spring)
    (when (> new (* 0.9 (spring-break-force spring)))
      (emit-event 'spring-critical :spring spring :stress new))))
)";
    }
};

} // namespace dsl
} // namespace digistar