#include "script_manager.h"
#include <fstream>
#include <sstream>

namespace digistar {
namespace dsl {

// ============ Script Implementation ============

void Script::execute(CommandQueue& queue) {
    if (state == PAUSED || state == COMPLETED || state == ERROR) {
        return;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    state = RUNNING;
    
    try {
        // Parse the code
        SExprParser parser(code);
        auto expressions = parser.parseAll();
        
        // Create command factory for this script
        CommandFactory factory(queue);
        
        // Execute each expression
        for (const auto& expr : expressions) {
            // Here we would evaluate the expression
            // For now, let's just handle simple cases
            
            // Check if it's a particle creation
            if (expr->isList() && expr->length() > 0) {
                auto first = expr->nth(0);
                if (first->isSymbol()) {
                    const std::string& sym = first->asSymbol();
                    
                    if (sym == "particle") {
                        // Parse particle creation
                        // (particle :mass 1.0 :pos [0 0] :vel [1 0])
                        KeywordArgs args(expr->asList());
                        
                        double mass = args.getNumber("mass", 1.0);
                        auto pos = args.getVector("pos", {0, 0});
                        auto vel = args.getVector("vel", {0, 0});
                        double temp = args.getNumber("temp", 300);
                        
                        factory.createParticle(mass, 
                            Vec2(pos[0], pos[1]), 
                            Vec2(vel[0], vel[1]), 
                            temp);
                    }
                    else if (sym == "cloud") {
                        // Parse cloud creation
                        KeywordArgs args(expr->asList());
                        
                        auto center = args.getVector("center", {0, 0});
                        double radius = args.getNumber("radius", 100);
                        int count = static_cast<int>(args.getNumber("n", 100));
                        double mass_min = args.getNumber("mass-min", 1.0);
                        double mass_max = args.getNumber("mass-max", 1.0);
                        
                        factory.createCloud(
                            Vec2(center[0], center[1]),
                            radius, count, mass_min, mass_max);
                    }
                    else if (sym == "set-velocity") {
                        // (set-velocity particle-id [vx vy])
                        if (expr->length() >= 3) {
                            int id = static_cast<int>(expr->nth(1)->asNumber());
                            auto vel = expr->nth(2)->asVector();
                            factory.setVelocity(id, Vec2(vel[0], vel[1]));
                        }
                    }
                    else if (sym == "apply-force") {
                        // (apply-force particle-id [fx fy])
                        if (expr->length() >= 3) {
                            int id = static_cast<int>(expr->nth(1)->asNumber());
                            auto force = expr->nth(2)->asVector();
                            factory.applyForce(id, Vec2(force[0], force[1]));
                        }
                    }
                }
            }
        }
        
        state = persistent ? IDLE : COMPLETED;
        execution_count++;
        
    } catch (const std::exception& e) {
        state = ERROR;
        last_error = e.what();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    total_time += std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    last_update = end_time;
}

bool Script::needsUpdate(float dt) const {
    if (!persistent || state != IDLE) {
        return false;
    }
    
    if (update_interval <= 0) {
        return true;  // Update every frame
    }
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_update).count();
    
    return elapsed >= update_interval * 1000;
}

// ============ ScriptManager Implementation ============

ScriptManager::ScriptManager() {
    setupBaseEnvironment();
    ScriptExecutor::initialize(0);  // Auto-detect thread count
}

ScriptManager::~ScriptManager() {
    stopAll();
    ScriptExecutor::get().wait_all();
}

void ScriptManager::setupBaseEnvironment() {
    base_env = std::make_shared<Environment>();
    
    // Add built-in constants
    base_env->define("pi", SExpr::makeNumber(3.14159265358979323846));
    base_env->define("e", SExpr::makeNumber(2.71828182845904523536));
    base_env->define("G", SExpr::makeNumber(6.67430e-11));  // Gravitational constant
    
    // Add built-in variables that will be updated each frame
    base_env->define("*sim-time*", SExpr::makeNumber(0));
    base_env->define("*dt*", SExpr::makeNumber(0.01));
    base_env->define("*particle-count*", SExpr::makeNumber(0));
    base_env->define("*spring-count*", SExpr::makeNumber(0));
}

std::shared_ptr<Script> ScriptManager::loadScript(const std::string& name,
                                                  const std::string& code,
                                                  bool persistent) {
    std::lock_guard<std::mutex> lock(scripts_mutex);
    
    auto script = std::make_shared<Script>(name, code, persistent);
    script->setManager(this);
    scripts[name] = script;
    
    return script;
}

std::shared_ptr<Script> ScriptManager::loadScriptFile(const std::string& name,
                                                      const std::string& filename,
                                                      bool persistent) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open script file: " + filename);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    
    return loadScript(name, buffer.str(), persistent);
}

void ScriptManager::removeScript(const std::string& name) {
    std::lock_guard<std::mutex> lock(scripts_mutex);
    
    auto it = scripts.find(name);
    if (it != scripts.end()) {
        it->second->stop();
        scripts.erase(it);
    }
}

std::shared_ptr<Script> ScriptManager::getScript(const std::string& name) {
    std::lock_guard<std::mutex> lock(scripts_mutex);
    
    auto it = scripts.find(name);
    return (it != scripts.end()) ? it->second : nullptr;
}

std::vector<std::string> ScriptManager::getScriptNames() const {
    std::lock_guard<std::mutex> lock(scripts_mutex);
    
    std::vector<std::string> names;
    names.reserve(scripts.size());
    
    for (const auto& [name, script] : scripts) {
        names.push_back(name);
    }
    
    return names;
}

std::future<void> ScriptManager::executeScript(const std::string& name) {
    auto script = getScript(name);
    if (!script) {
        throw std::runtime_error("Script not found: " + name);
    }
    
    return ScriptExecutor::get().submit([this, script]() {
        ScriptExecutionGuard guard(scripts_running);
        script->execute(command_queue);
        total_scripts_run++;
    });
}

std::future<void> ScriptManager::executeCode(const std::string& code) {
    static std::atomic<int> anonymous_id{0};
    std::string name = "anonymous_" + std::to_string(anonymous_id++);
    
    auto script = std::make_shared<Script>(name, code, false);
    script->setManager(this);
    
    return ScriptExecutor::get().submit([this, script]() {
        ScriptExecutionGuard guard(scripts_running);
        script->execute(command_queue);
        total_scripts_run++;
    });
}

void ScriptManager::update(float dt) {
    if (!auto_execute) {
        return;
    }
    
    // Update base environment with current simulation state
    if (simulation) {
        base_env->set("*sim-time*", SExpr::makeNumber(dt));  // Update with actual time
        base_env->set("*dt*", SExpr::makeNumber(simulation->getDt()));
        base_env->set("*particle-count*", 
            SExpr::makeNumber(static_cast<double>(simulation->getParticleCount())));
        base_env->set("*spring-count*", 
            SExpr::makeNumber(static_cast<double>(simulation->getSpringCount())));
    }
    
    // Collect scripts that need updating
    std::vector<std::shared_ptr<Script>> to_execute;
    
    {
        std::lock_guard<std::mutex> lock(scripts_mutex);
        for (const auto& [name, script] : scripts) {
            if (script->needsUpdate(dt)) {
                to_execute.push_back(script);
            }
        }
    }
    
    // Submit scripts to thread pool
    for (const auto& script : to_execute) {
        if (scripts_running < max_concurrent) {
            ScriptExecutor::get().submit_detached([this, script]() {
                ScriptExecutionGuard guard(scripts_running);
                script->execute(command_queue);
                total_scripts_run++;
            });
        }
    }
}

void ScriptManager::pauseAll() {
    std::lock_guard<std::mutex> lock(scripts_mutex);
    for (auto& [name, script] : scripts) {
        script->pause();
    }
}

void ScriptManager::resumeAll() {
    std::lock_guard<std::mutex> lock(scripts_mutex);
    for (auto& [name, script] : scripts) {
        script->resume();
    }
}

void ScriptManager::stopAll() {
    std::lock_guard<std::mutex> lock(scripts_mutex);
    for (auto& [name, script] : scripts) {
        script->stop();
    }
}

void ScriptManager::clearAll() {
    stopAll();
    ScriptExecutor::get().wait_all();
    
    std::lock_guard<std::mutex> lock(scripts_mutex);
    scripts.clear();
}

} // namespace dsl
} // namespace digistar