#include "evaluator.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>

namespace digistar {
namespace dsl {

// ============ DslEvaluator Implementation ============

DslEvaluator::DslEvaluator() {
    // Create base environment
    base_env = std::make_shared<Environment>();
    
    // Register all built-ins
    registerBuiltins();
}

void DslEvaluator::registerBuiltins() {
    registerCreationFunctions();
    registerQueryFunctions();
    registerControlFunctions();
    registerBehaviorFunctions();
    registerMathFunctions();
    
    // After registering all builtins, add them to base environment
    // We'll store them as special values that the evaluator recognizes
    for (const auto& [name, func] : builtins) {
        // For now, we'll handle built-ins specially in eval
        // Later we could wrap them in a special BuiltinClosure type
        base_env->define(name, SExpr::makeSymbol("<builtin:" + name + ">"));
    }
}

void DslEvaluator::registerCreationFunctions() {
    // Particle creation
    builtins["particle"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args, 
                                  ScriptContext& ctx) -> SExprPtr {
        KeywordArgs kwargs(args);
        
        auto pos = kwargs.getVector("pos", {0, 0});
        auto vel = kwargs.getVector("vel", {0, 0});
        float mass = kwargs.getNumber("mass", 1.0);
        float radius = kwargs.getNumber("radius", 1.0);
        float temp = kwargs.getNumber("temp", 300.0);
        
        size_t id = createParticle(pos[0], pos[1], vel[0], vel[1], 
                                   mass, radius, temp);
        return SExpr::makeNumber(static_cast<double>(id));
    };
    
    // Cloud of particles
    builtins["cloud"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                               ScriptContext& ctx) -> SExprPtr {
        KeywordArgs kwargs(args);
        
        size_t n = static_cast<size_t>(kwargs.getNumber("n", 100));
        auto center = kwargs.getVector("center", {0, 0});
        float radius = kwargs.getNumber("radius", 100.0);
        float mass_min = kwargs.getNumber("mass-min", 0.1);
        float mass_max = kwargs.getNumber("mass-max", 10.0);
        float temp = kwargs.getNumber("temp", 10.0);
        
        size_t first_id = createCloud(center[0], center[1], radius, n,
                                      mass_min, mass_max, temp);
        
        // Return list of IDs
        std::vector<SExprPtr> ids;
        for (size_t i = 0; i < n; i++) {
            ids.push_back(SExpr::makeNumber(static_cast<double>(first_id + i)));
        }
        return SExpr::makeList(ids);
    };
    
    // Star (massive hot particle)
    builtins["star"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                              ScriptContext& ctx) -> SExprPtr {
        KeywordArgs kwargs(args);
        
        auto pos = kwargs.getVector("pos", {0, 0});
        float mass = kwargs.getNumber("mass", 1.989e30);  // Solar mass
        float temp = kwargs.getNumber("temp", 5778);       // Solar temp
        float radius = kwargs.getNumber("radius", 6.96e8); // Solar radius
        
        size_t id = createParticle(pos[0], pos[1], 0, 0, mass, radius, temp);
        return SExpr::makeNumber(static_cast<double>(id));
    };
    
    // Spring connection
    builtins["spring"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                                ScriptContext& ctx) -> SExprPtr {
        if (args.size() < 2) return SExpr::makeNil();
        
        size_t p1 = static_cast<size_t>(eval.eval(args[0], ctx)->asNumber());
        size_t p2 = static_cast<size_t>(eval.eval(args[1], ctx)->asNumber());
        
        KeywordArgs kwargs(std::vector<SExprPtr>(args.begin() + 2, args.end()));
        float stiffness = kwargs.getNumber("stiffness", 1000.0);
        float damping = kwargs.getNumber("damping", 10.0);
        float rest_length = kwargs.getNumber("rest-length", -1.0);
        
        createSpring(p1, p2, stiffness, damping, rest_length);
        return SExpr::makeBool(true);
    };
    
    // Composite body from particles
    builtins["composite"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                                   ScriptContext& ctx) -> SExprPtr {
        if (args.empty()) return SExpr::makeNil();
        
        std::vector<size_t> particles;
        for (auto& arg : args) {
            auto evaluated = eval.eval(arg, ctx);
            if (evaluated->isList()) {
                for (auto& elem : evaluated->asList()) {
                    if (elem->isNumber()) {
                        particles.push_back(static_cast<size_t>(elem->asNumber()));
                    }
                }
            } else if (evaluated->isNumber()) {
                particles.push_back(static_cast<size_t>(evaluated->asNumber()));
            }
        }
        
        size_t comp_id = createComposite(particles);
        return SExpr::makeNumber(static_cast<double>(comp_id));
    };
}

void DslEvaluator::registerQueryFunctions() {
    // Find particles in region
    builtins["find"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                              ScriptContext& ctx) -> SExprPtr {
        KeywordArgs kwargs(args);
        
        auto center = kwargs.getVector("center", {0, 0});
        float radius = kwargs.getNumber("radius", 100.0);
        
        auto particles = findParticlesInRegion(center[0], center[1], radius);
        
        std::vector<SExprPtr> ids;
        for (size_t id : particles) {
            ids.push_back(SExpr::makeNumber(static_cast<double>(id)));
        }
        return SExpr::makeList(ids);
    };
    
    // Measure property
    builtins["measure"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                                 ScriptContext& ctx) -> SExprPtr {
        if (args.empty()) return SExpr::makeNil();
        
        std::string prop = eval.eval(args[0], ctx)->asSymbol();
        
        std::vector<size_t> particles;
        if (args.size() > 1) {
            auto target = eval.eval(args[1], ctx);
            if (target->isList()) {
                for (auto& elem : target->asList()) {
                    if (elem->isNumber()) {
                        particles.push_back(static_cast<size_t>(elem->asNumber()));
                    }
                }
            }
        }
        
        return measureProperty(prop, particles);
    };
    
    // Query with conditions
    builtins["query"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                               ScriptContext& ctx) -> SExprPtr {
        // Complex query system - simplified for now
        KeywordArgs kwargs(args);
        
        std::string prop = kwargs.getString("property", "mass");
        float min_val = kwargs.getNumber("min", -INFINITY);
        float max_val = kwargs.getNumber("max", INFINITY);
        
        auto particles = findParticlesByProperty(prop, min_val, max_val);
        
        std::vector<SExprPtr> ids;
        for (size_t id : particles) {
            ids.push_back(SExpr::makeNumber(static_cast<double>(id)));
        }
        return SExpr::makeList(ids);
    };
}

void DslEvaluator::registerControlFunctions() {
    // Set velocity
    builtins["set-velocity"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                                      ScriptContext& ctx) -> SExprPtr {
        if (args.size() < 2) return SExpr::makeNil();
        
        size_t id = static_cast<size_t>(eval.eval(args[0], ctx)->asNumber());
        auto vel = eval.eval(args[1], ctx)->asVector();
        
        if (id < sim_state->particles.count) {
            sim_state->particles.vel_x[id] = vel[0];
            sim_state->particles.vel_y[id] = vel[1];
        }
        
        return SExpr::makeBool(true);
    };
    
    // Apply force
    builtins["apply-force"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                                     ScriptContext& ctx) -> SExprPtr {
        if (args.size() < 2) return SExpr::makeNil();
        
        size_t id = static_cast<size_t>(eval.eval(args[0], ctx)->asNumber());
        auto force = eval.eval(args[1], ctx)->asVector();
        
        if (id < sim_state->particles.count) {
            sim_state->particles.force_x[id] += force[0];
            sim_state->particles.force_y[id] += force[1];
        }
        
        return SExpr::makeBool(true);
    };
    
    // Explode particle
    builtins["explode"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                                 ScriptContext& ctx) -> SExprPtr {
        if (args.empty()) return SExpr::makeNil();
        
        size_t id = static_cast<size_t>(eval.eval(args[0], ctx)->asNumber());
        
        KeywordArgs kwargs(std::vector<SExprPtr>(args.begin() + 1, args.end()));
        float energy = kwargs.getNumber("energy", 1e6);
        size_t fragments = static_cast<size_t>(kwargs.getNumber("fragments", 10));
        
        if (id < sim_state->particles.count) {
            float px = sim_state->particles.pos_x[id];
            float py = sim_state->particles.pos_y[id];
            float mass = sim_state->particles.mass[id];
            float fragment_mass = mass / fragments;
            
            // Mark original as deleted
            sim_state->particles.alive[id] = false;
            
            // Create fragments
            for (size_t i = 0; i < fragments; i++) {
                float angle = 2 * M_PI * i / fragments;
                float speed = std::sqrt(2 * energy / (fragments * fragment_mass));
                
                createParticle(px, py, 
                              speed * std::cos(angle), speed * std::sin(angle),
                              fragment_mass, 0.1, 1000.0);
            }
        }
        
        return SExpr::makeBool(true);
    };
}

void DslEvaluator::registerBehaviorFunctions() {
    // Watch expression continuously
    builtins["watch"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                               ScriptContext& ctx) -> SExprPtr {
        if (args.empty()) return SExpr::makeNil();
        
        KeywordArgs kwargs(args);
        float interval = kwargs.getNumber("every", 0.1);
        
        // Store watch expression in context for continuous evaluation
        ctx.state["watch_expr"] = args[0];
        ctx.state["watch_interval"] = SExpr::makeNumber(interval);
        ctx.is_persistent = true;
        ctx.update_interval = interval;
        
        // Set up continuation
        ctx.continuation = [args](DslEvaluator& eval) -> SExprPtr {
            // This will be called every update
            ScriptContext temp_ctx;
            auto result = eval.eval(args[0], temp_ctx);
            std::cout << "Watch: " << result->toString() << std::endl;
            return result;
        };
        
        return SExpr::makeBool(true);
    };
    
    // Define behavioral rule
    builtins["rule"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                              ScriptContext& ctx) -> SExprPtr {
        if (args.size() < 2) return SExpr::makeNil();
        
        auto name = eval.eval(args[0], ctx)->asSymbol();
        auto condition = args[1];
        auto action = args.size() > 2 ? args[2] : SExpr::makeNil();
        
        // Store rule in context
        ctx.state["rule_name"] = SExpr::makeSymbol(name);
        ctx.state["rule_condition"] = condition;
        ctx.state["rule_action"] = action;
        ctx.is_persistent = true;
        
        // Set up continuation for rule checking
        ctx.continuation = [condition, action](DslEvaluator& eval) -> SExprPtr {
            ScriptContext temp_ctx;
            auto cond_result = eval.eval(condition, temp_ctx);
            
            if (eval.isTruthy(cond_result)) {
                return eval.eval(action, temp_ctx);
            }
            return SExpr::makeNil();
        };
        
        return SExpr::makeBool(true);
    };
    
    // Trigger event when condition met
    builtins["trigger"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                                 ScriptContext& ctx) -> SExprPtr {
        if (args.size() < 2) return SExpr::makeNil();
        
        auto condition = eval.eval(args[0], ctx);
        
        if (isTruthy(condition)) {
            // Execute all remaining args as actions
            for (size_t i = 1; i < args.size(); i++) {
                eval.eval(args[i], ctx);
            }
            return SExpr::makeBool(true);
        }
        
        return SExpr::makeBool(false);
    };
}

void DslEvaluator::registerMathFunctions() {
    // Random number
    builtins["random"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                                ScriptContext& ctx) -> SExprPtr {
        float min_val = 0.0;
        float max_val = 1.0;
        
        if (args.size() >= 1) {
            max_val = eval.eval(args[0], ctx)->asNumber();
        }
        if (args.size() >= 2) {
            min_val = max_val;
            max_val = eval.eval(args[1], ctx)->asNumber();
        }
        
        std::uniform_real_distribution<float> dist(min_val, max_val);
        return SExpr::makeNumber(dist(rng));
    };
    
    // Distance between particles
    builtins["distance"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                                  ScriptContext& ctx) -> SExprPtr {
        if (args.size() < 2) return SExpr::makeNil();
        
        size_t id1 = static_cast<size_t>(eval.eval(args[0], ctx)->asNumber());
        size_t id2 = static_cast<size_t>(eval.eval(args[1], ctx)->asNumber());
        
        if (id1 < sim_state->particles.count && id2 < sim_state->particles.count) {
            float dx = sim_state->particles.pos_x[id2] - sim_state->particles.pos_x[id1];
            float dy = sim_state->particles.pos_y[id2] - sim_state->particles.pos_y[id1];
            return SExpr::makeNumber(std::sqrt(dx*dx + dy*dy));
        }
        
        return SExpr::makeNumber(0.0);
    };
    
    // Basic arithmetic
    builtins["+"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                          ScriptContext& ctx) -> SExprPtr {
        double sum = 0;
        for (auto& arg : args) {
            sum += eval.eval(arg, ctx)->asNumber();
        }
        return SExpr::makeNumber(sum);
    };
    
    builtins["-"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                          ScriptContext& ctx) -> SExprPtr {
        if (args.empty()) return SExpr::makeNumber(0);
        double result = eval.eval(args[0], ctx)->asNumber();
        for (size_t i = 1; i < args.size(); i++) {
            result -= eval.eval(args[i], ctx)->asNumber();
        }
        return SExpr::makeNumber(result);
    };
    
    builtins["*"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                          ScriptContext& ctx) -> SExprPtr {
        double product = 1;
        for (auto& arg : args) {
            product *= eval.eval(arg, ctx)->asNumber();
        }
        return SExpr::makeNumber(product);
    };
    
    builtins["/"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                          ScriptContext& ctx) -> SExprPtr {
        if (args.empty()) return SExpr::makeNumber(1);
        double result = eval.eval(args[0], ctx)->asNumber();
        for (size_t i = 1; i < args.size(); i++) {
            result /= eval.eval(args[i], ctx)->asNumber();
        }
        return SExpr::makeNumber(result);
    };
    
    // Comparisons
    builtins[">"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                          ScriptContext& ctx) -> SExprPtr {
        if (args.size() < 2) return SExpr::makeBool(true);
        double a = eval.eval(args[0], ctx)->asNumber();
        double b = eval.eval(args[1], ctx)->asNumber();
        return SExpr::makeBool(a > b);
    };
    
    builtins["<"] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                          ScriptContext& ctx) -> SExprPtr {
        if (args.size() < 2) return SExpr::makeBool(true);
        double a = eval.eval(args[0], ctx)->asNumber();
        double b = eval.eval(args[1], ctx)->asNumber();
        return SExpr::makeBool(a < b);
    };
    
    builtins["="] = [this](DslEvaluator& eval, const std::vector<SExprPtr>& args,
                          ScriptContext& ctx) -> SExprPtr {
        if (args.size() < 2) return SExpr::makeBool(true);
        auto a = eval.eval(args[0], ctx);
        auto b = eval.eval(args[1], ctx);
        return SExpr::makeBool(a->equals(*b));
    };
}

// Main evaluation function
SExprPtr DslEvaluator::eval(SExprPtr expr, ScriptContext& ctx) {
    eval_count++;
    auto start = std::chrono::high_resolution_clock::now();
    
    SExprPtr result = evalWithTailRecursion(expr, ctx);
    
    auto end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float, std::milli>(end - start).count();
    total_eval_time += elapsed;
    
    return result;
}

SExprPtr DslEvaluator::evalWithTailRecursion(SExprPtr expr, ScriptContext& ctx) {
    SExprPtr current = expr;
    
    while (true) {
        if (!current) return SExpr::makeNil();
        
        if (current->isNil()) {
            return current;
        }
        
        if (current->isAtom()) {
            return evalAtom(current, ctx);
        }
        
        if (current->isList()) {
            auto result = evalList(current->asList(), ctx);
            
            // Check for tail recursion
            if (ctx.continuation) {
                auto cont = ctx.continuation;
                ctx.continuation = nullptr;
                current = cont(*this);
                continue;  // Loop for tail recursion
            }
            
            return result;
        }
        
        return SExpr::makeNil();
    }
}

SExprPtr DslEvaluator::evalAtom(SExprPtr expr, ScriptContext& ctx) {
    if (expr->isNumber() || expr->isBool() || expr->isVector() || 
        expr->isString() || expr->isClosure()) {
        return expr;  // Self-evaluating
    }
    
    if (expr->isSymbol()) {
        const std::string& sym = expr->asSymbol();
        
        // Keywords (start with :) evaluate to themselves
        if (!sym.empty() && sym[0] == ':') {
            return expr;
        }
        
        // Look up in environment
        auto value = ctx.env->lookup(sym);
        if (value) {
            return value;
        }
        
        // Unbound variable is an error
        // For now, print error and return nil to allow continuation
        // In a production system, we'd throw an exception
        std::cerr << "Error: Unbound variable: " << sym << std::endl;
        return SExpr::makeNil();
    }
    
    return expr;
}

SExprPtr DslEvaluator::evalList(const std::vector<SExprPtr>& list, ScriptContext& ctx) {
    if (list.empty()) return SExpr::makeNil();
    
    auto first = list[0];
    
    // Special forms (don't evaluate args first)
    if (first->isSymbol()) {
        const std::string& sym = first->asSymbol();
        std::vector<SExprPtr> args(list.begin() + 1, list.end());
        
        if (sym == "if") return evalIf(args, ctx);
        if (sym == "let") return evalLet(args, ctx);
        if (sym == "lambda") return evalLambda(args, ctx);
        if (sym == "define") return evalDefine(args, ctx);
        if (sym == "set!") return evalSet(args, ctx);
        if (sym == "quote") return evalQuote(args, ctx);
        if (sym == "loop") return evalLoop(args, ctx);
        if (sym == "when") return evalWhen(args, ctx);
        
        // Check for built-in function
        auto it = builtins.find(sym);
        if (it != builtins.end()) {
            return it->second(*this, args, ctx);
        }
    }
    
    // Function application
    auto func = eval(first, ctx);
    
    // Evaluate arguments
    std::vector<SExprPtr> eval_args;
    for (size_t i = 1; i < list.size(); i++) {
        eval_args.push_back(eval(list[i], ctx));
    }
    
    // Apply function
    return apply(func, eval_args, ctx);
}

// Special forms
SExprPtr DslEvaluator::evalIf(const std::vector<SExprPtr>& args, ScriptContext& ctx) {
    if (args.size() < 2) return SExpr::makeNil();
    
    auto condition = eval(args[0], ctx);
    
    if (isTruthy(condition)) {
        return eval(args[1], ctx);
    } else if (args.size() > 2) {
        return eval(args[2], ctx);
    }
    
    return SExpr::makeNil();
}

SExprPtr DslEvaluator::evalLet(const std::vector<SExprPtr>& args, ScriptContext& ctx) {
    if (args.size() < 2) return SExpr::makeNil();
    
    // Create new environment
    auto new_env = ctx.env->extend();
    ScriptContext new_ctx = ctx;
    new_ctx.env = new_env;
    
    // Bind variables
    if (args[0]->isList()) {
        auto bindings = args[0]->asList();
        for (size_t i = 0; i < bindings.size(); i += 2) {
            if (i + 1 < bindings.size() && bindings[i]->isSymbol()) {
                auto value = eval(bindings[i + 1], new_ctx);
                new_env->define(bindings[i]->asSymbol(), value);
            }
        }
    }
    
    // Evaluate body
    SExprPtr result = SExpr::makeNil();
    for (size_t i = 1; i < args.size(); i++) {
        result = eval(args[i], new_ctx);
    }
    
    return result;
}

SExprPtr DslEvaluator::evalDefine(const std::vector<SExprPtr>& args, ScriptContext& ctx) {
    if (args.size() < 2) return SExpr::makeNil();
    
    if (args[0]->isSymbol()) {
        auto value = eval(args[1], ctx);
        ctx.env->define(args[0]->asSymbol(), value);
        return value;
    }
    
    return SExpr::makeNil();
}

SExprPtr DslEvaluator::evalSet(const std::vector<SExprPtr>& args, ScriptContext& ctx) {
    if (args.size() < 2) return SExpr::makeNil();
    
    if (args[0]->isSymbol()) {
        const std::string& name = args[0]->asSymbol();
        auto value = eval(args[1], ctx);
        
        // Try to set the binding
        if (!ctx.env->set(name, value)) {
            std::cerr << "Error: set! of unbound variable: " << name << std::endl;
            return SExpr::makeNil();
        }
        
        return value;
    }
    
    return SExpr::makeNil();
}

SExprPtr DslEvaluator::evalQuote(const std::vector<SExprPtr>& args, ScriptContext& ctx) {
    if (args.empty()) return SExpr::makeNil();
    return args[0];
}

SExprPtr DslEvaluator::evalLoop(const std::vector<SExprPtr>& args, ScriptContext& ctx) {
    // Infinite loop with break condition
    while (true) {
        for (auto& expr : args) {
            auto result = eval(expr, ctx);
            
            // Check for break
            if (result->isSymbol() && result->asSymbol() == "break") {
                return SExpr::makeNil();
            }
        }
        
        // Allow other scripts to run
        if (ctx.is_persistent) {
            break;  // Will continue next update
        }
    }
    
    return SExpr::makeNil();
}

SExprPtr DslEvaluator::evalWhen(const std::vector<SExprPtr>& args, ScriptContext& ctx) {
    if (args.empty()) return SExpr::makeNil();
    
    auto condition = eval(args[0], ctx);
    
    if (isTruthy(condition)) {
        SExprPtr result = SExpr::makeNil();
        for (size_t i = 1; i < args.size(); i++) {
            result = eval(args[i], ctx);
        }
        return result;
    }
    
    return SExpr::makeNil();
}

SExprPtr DslEvaluator::evalLambda(const std::vector<SExprPtr>& args, ScriptContext& ctx) {
    if (args.size() < 2) return SExpr::makeNil();
    
    // First arg should be parameter list
    if (!args[0]->isList()) return SExpr::makeNil();
    
    // Extract parameter names
    std::vector<std::string> params;
    for (auto& param : args[0]->asList()) {
        if (!param->isSymbol()) {
            std::cerr << "Error: Lambda parameter must be a symbol" << std::endl;
            return SExpr::makeNil();
        }
        params.push_back(param->asSymbol());
    }
    
    // Body is the rest of the arguments (implicit begin)
    std::vector<SExprPtr> body_exprs(args.begin() + 1, args.end());
    SExprPtr body;
    if (body_exprs.size() == 1) {
        body = body_exprs[0];
    } else {
        // Multiple expressions - wrap in begin
        body_exprs.insert(body_exprs.begin(), SExpr::makeSymbol("begin"));
        body = SExpr::makeList(body_exprs);
    }
    
    // Create closure capturing current environment
    auto closure = std::make_shared<Closure>(params, body, ctx.env);
    return SExpr::makeClosure(closure);
}

SExprPtr DslEvaluator::apply(SExprPtr func, const std::vector<SExprPtr>& args, ScriptContext& ctx) {
    if (!func) return SExpr::makeNil();
    
    // Check if it's a closure
    if (func->isClosure()) {
        auto closure = func->asClosure();
        
        // Check arity
        if (!closure->isVariadic() && args.size() != closure->getArity()) {
            std::cerr << "Error: Arity mismatch. Expected " << closure->getArity() 
                     << " arguments, got " << args.size() << std::endl;
            return SExpr::makeNil();
        }
        
        // Create new environment extending closure's environment
        auto new_env = std::make_shared<Environment>(closure->getEnv());
        
        // Bind parameters to arguments
        const auto& params = closure->getParams();
        for (size_t i = 0; i < params.size() && i < args.size(); i++) {
            new_env->define(params[i], args[i]);
        }
        
        // If variadic, bind remaining args as a list
        if (closure->isVariadic() && args.size() >= closure->getArity()) {
            std::vector<SExprPtr> rest_args(args.begin() + closure->getArity(), args.end());
            new_env->define("...", SExpr::makeList(rest_args));
        }
        
        // Evaluate body in new environment
        ScriptContext new_ctx("lambda-call", new_env);
        return eval(closure->getBody(), new_ctx);
    }
    
    // Check if it's a built-in (stored as special symbol)
    if (func->isSymbol()) {
        const std::string& sym = func->asSymbol();
        if (sym.substr(0, 9) == "<builtin:") {
            std::string builtin_name = sym.substr(9, sym.length() - 10);
            auto it = builtins.find(builtin_name);
            if (it != builtins.end()) {
                // Built-ins expect unevaluated args, but we have evaluated ones
                // This is a design issue we need to resolve
                return it->second(*this, args, ctx);
            }
        }
    }
    
    std::cerr << "Error: Attempt to apply non-function" << std::endl;
    return SExpr::makeNil();
}

// Helper functions
bool DslEvaluator::isTruthy(SExprPtr expr) const {
    if (!expr || expr->isNil()) return false;
    if (expr->isBool()) return expr->asBool();
    if (expr->isNumber()) return expr->asNumber() != 0.0;
    if (expr->isList()) return !expr->asList().empty();
    return true;
}

SExprPtr DslEvaluator::eval(SExprPtr expr) {
    // Use a global context with base environment for REPL-style evaluation
    static std::shared_ptr<ScriptContext> global_ctx = nullptr;
    if (!global_ctx) {
        global_ctx = std::make_shared<ScriptContext>("global", base_env);
    }
    return eval(expr, *global_ctx);
}

SExprPtr DslEvaluator::evalString(const std::string& code) {
    auto expr = SExprParser::parseString(code);
    return eval(expr);
}

SExprPtr DslEvaluator::evalFile(const std::string& filename) {
    auto expressions = SExprParser::parseFile(filename);
    
    SExprPtr result = SExpr::makeNil();
    for (auto& expr : expressions) {
        result = eval(expr);
    }
    
    return result;
}

// Script management
std::shared_ptr<ScriptContext> DslEvaluator::createScript(
    const std::string& name, SExprPtr code, bool persistent) {
    
    auto ctx = std::make_shared<ScriptContext>(name);
    ctx->is_persistent = persistent;
    
    if (persistent) {
        active_scripts.push_back(ctx);
    }
    
    eval(code, *ctx);
    
    return ctx;
}

void DslEvaluator::stopScript(const std::string& name) {
    active_scripts.erase(
        std::remove_if(active_scripts.begin(), active_scripts.end(),
                      [&name](const std::shared_ptr<ScriptContext>& ctx) {
                          return ctx->name == name;
                      }),
        active_scripts.end()
    );
}

void DslEvaluator::updateScripts(float dt) {
    for (auto& ctx : active_scripts) {
        if (!ctx->is_active) continue;
        
        ctx->last_update += dt;
        
        if (ctx->last_update >= ctx->update_interval) {
            ctx->last_update = 0;
            
            // Execute continuation if present
            if (ctx->continuation) {
                ctx->continuation(*this);
            }
        }
    }
}

// Particle creation helpers
size_t DslEvaluator::createParticle(float x, float y, float vx, float vy,
                                    float mass, float radius, float temp) {
    if (!sim_state) return 0;
    
    auto& p = sim_state->particles;
    if (p.count >= p.capacity) return 0;
    
    size_t id = p.count++;
    p.pos_x[id] = x;
    p.pos_y[id] = y;
    p.vel_x[id] = vx;
    p.vel_y[id] = vy;
    p.mass[id] = mass;
    p.radius[id] = radius;
    p.temperature[id] = temp;
    p.charge[id] = 0;
    p.material_type[id] = 0;
    p.alive[id] = true;
    p.force_x[id] = 0;
    p.force_y[id] = 0;
    
    return id;
}

size_t DslEvaluator::createCloud(float cx, float cy, float radius, size_t count,
                                 float mass_min, float mass_max, float temp) {
    size_t first_id = sim_state->particles.count;
    
    std::uniform_real_distribution<float> r_dist(0, radius);
    std::uniform_real_distribution<float> theta_dist(0, 2 * M_PI);
    std::uniform_real_distribution<float> mass_dist(mass_min, mass_max);
    
    for (size_t i = 0; i < count; i++) {
        float r = r_dist(rng);
        float theta = theta_dist(rng);
        float mass = mass_dist(rng);
        
        createParticle(cx + r * std::cos(theta), 
                      cy + r * std::sin(theta),
                      0, 0, mass, mass * 0.1, temp);
    }
    
    return first_id;
}

size_t DslEvaluator::createOrbitalSystem(float cx, float cy, float central_mass,
                                         const std::vector<SExprPtr>& orbits) {
    // Create central body
    size_t central = createParticle(cx, cy, 0, 0, central_mass, 
                                    central_mass * 0.01, 5000);
    
    // Create orbiting bodies
    for (auto& orbit_spec : orbits) {
        if (!orbit_spec->isList()) continue;
        
        KeywordArgs kwargs(orbit_spec->asList());
        float a = kwargs.getNumber("a", 1.0);  // Semi-major axis
        float e = kwargs.getNumber("e", 0.0);  // Eccentricity
        float mass = kwargs.getNumber("mass", 1.0);
        float angle = kwargs.getNumber("angle", 0.0);
        
        // Calculate orbital velocity (assuming 2D circular for simplicity)
        float orbital_speed = std::sqrt(physics_config->gravity_strength * central_mass / a);
        
        float px = cx + a * std::cos(angle);
        float py = cy + a * std::sin(angle);
        float vx = -orbital_speed * std::sin(angle);
        float vy = orbital_speed * std::cos(angle);
        
        createParticle(px, py, vx, vy, mass, mass * 0.01, 300);
    }
    
    return central;
}

void DslEvaluator::createSpring(size_t p1, size_t p2, float stiffness, 
                                float damping, float rest_length) {
    if (!sim_state) return;
    
    auto& s = sim_state->springs;
    if (s.count >= s.capacity) return;
    
    if (rest_length < 0) {
        // Calculate from current positions
        float dx = sim_state->particles.pos_x[p2] - sim_state->particles.pos_x[p1];
        float dy = sim_state->particles.pos_y[p2] - sim_state->particles.pos_y[p1];
        rest_length = std::sqrt(dx*dx + dy*dy);
    }
    
    size_t id = s.count++;
    s.particle1_id[id] = p1;
    s.particle2_id[id] = p2;
    s.stiffness[id] = stiffness;
    s.damping[id] = damping;
    s.rest_length[id] = rest_length;
    s.strain[id] = 0.0;  // Current strain
    s.active[id] = true;
}

void DslEvaluator::createSpringMesh(const std::vector<size_t>& particles,
                                    float stiffness, float damping) {
    // Connect all particles to each other
    for (size_t i = 0; i < particles.size(); i++) {
        for (size_t j = i + 1; j < particles.size(); j++) {
            createSpring(particles[i], particles[j], stiffness, damping);
        }
    }
}

size_t DslEvaluator::createComposite(const std::vector<size_t>& particles) {
    if (!sim_state || particles.empty()) return 0;
    
    auto& c = sim_state->composites;
    if (c.count >= c.capacity) return 0;
    
    size_t id = c.count++;
    
    // Calculate center of mass
    float total_mass = 0;
    float com_x = 0, com_y = 0;
    
    for (size_t pid : particles) {
        float mass = sim_state->particles.mass[pid];
        total_mass += mass;
        com_x += mass * sim_state->particles.pos_x[pid];
        com_y += mass * sim_state->particles.pos_y[pid];
    }
    
    c.center_x[id] = com_x / total_mass;
    c.center_y[id] = com_y / total_mass;
    c.total_mass[id] = total_mass;
    c.particle_count[id] = particles.size();
    
    // Mark particles as part of composite
    for (size_t pid : particles) {
        sim_state->particles.composite_id[pid] = id;
    }
    
    return id;
}

// Query helpers
std::vector<size_t> DslEvaluator::findParticlesInRegion(float x, float y, float radius) {
    std::vector<size_t> result;
    if (!sim_state) return result;
    
    float r2 = radius * radius;
    
    for (size_t i = 0; i < sim_state->particles.count; i++) {
        if (!sim_state->particles.alive[i]) continue;
        
        float dx = sim_state->particles.pos_x[i] - x;
        float dy = sim_state->particles.pos_y[i] - y;
        
        if (dx*dx + dy*dy <= r2) {
            result.push_back(i);
        }
    }
    
    return result;
}

std::vector<size_t> DslEvaluator::findParticlesByProperty(
    const std::string& prop, float min_val, float max_val) {
    
    std::vector<size_t> result;
    if (!sim_state) return result;
    
    for (size_t i = 0; i < sim_state->particles.count; i++) {
        if (!sim_state->particles.alive[i]) continue;
        
        float value = 0;
        
        if (prop == "mass") {
            value = sim_state->particles.mass[i];
        } else if (prop == "temperature" || prop == "temp") {
            value = sim_state->particles.temperature[i];
        } else if (prop == "radius") {
            value = sim_state->particles.radius[i];
        } else if (prop == "speed") {
            float vx = sim_state->particles.vel_x[i];
            float vy = sim_state->particles.vel_y[i];
            value = std::sqrt(vx*vx + vy*vy);
        } else if (prop == "charge") {
            value = sim_state->particles.charge[i];
        }
        
        if (value >= min_val && value <= max_val) {
            result.push_back(i);
        }
    }
    
    return result;
}

SExprPtr DslEvaluator::measureProperty(const std::string& prop,
                                       const std::vector<size_t>& particles) {
    if (!sim_state || particles.empty()) return SExpr::makeNil();
    
    if (prop == "center-of-mass" || prop == "com") {
        float total_mass = 0;
        float com_x = 0, com_y = 0;
        
        for (size_t id : particles) {
            if (id >= sim_state->particles.count) continue;
            float mass = sim_state->particles.mass[id];
            total_mass += mass;
            com_x += mass * sim_state->particles.pos_x[id];
            com_y += mass * sim_state->particles.pos_y[id];
        }
        
        if (total_mass > 0) {
            std::vector<double> com = {com_x / total_mass, com_y / total_mass};
            return SExpr::makeVector(com);
        }
    } else if (prop == "total-mass") {
        float total = 0;
        for (size_t id : particles) {
            if (id < sim_state->particles.count) {
                total += sim_state->particles.mass[id];
            }
        }
        return SExpr::makeNumber(total);
    } else if (prop == "average-temperature" || prop == "avg-temp") {
        float sum = 0;
        size_t count = 0;
        for (size_t id : particles) {
            if (id < sim_state->particles.count) {
                sum += sim_state->particles.temperature[id];
                count++;
            }
        }
        return SExpr::makeNumber(count > 0 ? sum / count : 0);
    } else if (prop == "kinetic-energy") {
        float ke = 0;
        for (size_t id : particles) {
            if (id < sim_state->particles.count) {
                float vx = sim_state->particles.vel_x[id];
                float vy = sim_state->particles.vel_y[id];
                float mass = sim_state->particles.mass[id];
                ke += 0.5 * mass * (vx*vx + vy*vy);
            }
        }
        return SExpr::makeNumber(ke);
    }
    
    return SExpr::makeNil();
}

} // namespace dsl
} // namespace digistar