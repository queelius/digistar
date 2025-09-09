#pragma once

#include "bytecode_compiler.h"
#include "sexpr.h"
#include "../physics/pools.h"
#include <stack>
#include <vector>
#include <functional>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include <atomic>

namespace digistar {

// Forward declaration
struct SimulationState;

namespace dsl {

/**
 * Stack frame for function calls
 */
struct CallFrame {
    std::shared_ptr<BytecodeChunk> chunk;  // Current chunk
    size_t ip;                             // Instruction pointer
    size_t stack_base;                     // Stack base for locals
    std::shared_ptr<Environment> env;      // Environment for closures
    
    CallFrame(std::shared_ptr<BytecodeChunk> c, size_t base = 0)
        : chunk(c), ip(0), stack_base(base) {}
};

/**
 * VM execution state
 */
class VMState {
public:
    // Stack
    std::vector<SExprPtr> stack;
    size_t stack_top = 0;
    
    // Call stack
    std::vector<CallFrame> call_stack;
    
    // Global environment
    std::shared_ptr<Environment> global_env;
    
    // Performance counters
    struct Stats {
        size_t instructions_executed = 0;
        size_t function_calls = 0;
        size_t tail_calls_optimized = 0;
        size_t pattern_matches = 0;
        std::chrono::microseconds execution_time{0};
    } stats;
    
    // Error handling
    bool has_error = false;
    std::string error_message;
    
    VMState() : global_env(std::make_shared<Environment>()) {
        stack.reserve(1024);  // Pre-allocate stack space
        call_stack.reserve(64); // Pre-allocate call stack
    }
    
    // Stack operations
    void push(SExprPtr value) {
        if (stack_top >= stack.size()) {
            stack.resize(stack_top + 1);
        }
        stack[stack_top++] = value;
    }
    
    SExprPtr pop() {
        if (stack_top == 0) {
            throw std::runtime_error("Stack underflow");
        }
        return stack[--stack_top];
    }
    
    SExprPtr peek(size_t offset = 0) const {
        if (stack_top <= offset) {
            throw std::runtime_error("Stack underflow");
        }
        return stack[stack_top - 1 - offset];
    }
    
    void drop(size_t count = 1) {
        if (stack_top < count) {
            throw std::runtime_error("Stack underflow");
        }
        stack_top -= count;
    }
    
    // Call frame operations
    void pushFrame(std::shared_ptr<BytecodeChunk> chunk) {
        call_stack.emplace_back(chunk, stack_top);
    }
    
    void popFrame() {
        if (!call_stack.empty()) {
            call_stack.pop_back();
        }
    }
    
    CallFrame& currentFrame() {
        if (call_stack.empty()) {
            throw std::runtime_error("No active call frame");
        }
        return call_stack.back();
    }
    
    const CallFrame& currentFrame() const {
        if (call_stack.empty()) {
            throw std::runtime_error("No active call frame");
        }
        return call_stack.back();
    }
    
    // Error handling
    void setError(const std::string& msg) {
        has_error = true;
        error_message = msg;
    }
    
    void clearError() {
        has_error = false;
        error_message.clear();
    }
};

/**
 * Bytecode virtual machine
 */
class BytecodeVM {
private:
    SimulationState* sim_state;
    VMState state;
    
    // Built-in function registry
    using BuiltinFunc = std::function<SExprPtr(BytecodeVM&, size_t arg_count)>;
    std::unordered_map<std::string, BuiltinFunc> builtins;
    
    // Execution methods
    bool executeInstruction(const Instruction& inst);
    bool executeArithmetic(OpCode op);
    bool executeComparison(OpCode op);
    bool executeLogical(OpCode op);
    bool executeJump(const Instruction& inst);
    bool executeCall(size_t arg_count);
    bool executeTailCall(size_t arg_count);
    bool executePatternMatch(const Instruction& inst);
    
    // Helper methods
    double toNumber(SExprPtr value);
    bool toBool(SExprPtr value);
    SExprPtr makeNumber(double n);
    SExprPtr makeBool(bool b);
    
    // Built-in implementations
    void registerBuiltins();
    SExprPtr builtinAdd(size_t arg_count);
    SExprPtr builtinSub(size_t arg_count);
    SExprPtr builtinMul(size_t arg_count);
    SExprPtr builtinDiv(size_t arg_count);
    SExprPtr builtinCreateParticle(size_t arg_count);
    SExprPtr builtinQueryParticles(size_t arg_count);
    SExprPtr builtinEmitEvent(size_t arg_count);
    
public:
    explicit BytecodeVM(SimulationState* state = nullptr);
    
    // Main execution interface
    SExprPtr execute(std::shared_ptr<BytecodeChunk> chunk);
    SExprPtr executeWithTimeout(std::shared_ptr<BytecodeChunk> chunk,
                                std::chrono::milliseconds timeout);
    
    // Step execution for debugging
    bool step();
    void reset();
    
    // State access
    VMState& getState() { return state; }
    const VMState& getState() const { return state; }
    
    // Environment management
    void setGlobalEnv(std::shared_ptr<Environment> env) {
        state.global_env = env;
    }
    
    std::shared_ptr<Environment> getGlobalEnv() const {
        return state.global_env;
    }
    
    // Built-in management
    void registerBuiltin(const std::string& name, BuiltinFunc func) {
        builtins[name] = func;
    }
    
    bool hasBuiltin(const std::string& name) const {
        return builtins.find(name) != builtins.end();
    }
    
    // Performance monitoring
    const VMState::Stats& getStats() const { return state.stats; }
    void resetStats() { state.stats = VMState::Stats(); }
    
    // Debugging
    std::string stackTrace() const;
    std::string currentInstruction() const;
    void enableTracing(bool enable);
    
private:
    bool tracing_enabled = false;
    void trace(const std::string& msg);
};

// ============================================================================
// BytecodeVM Implementation
// ============================================================================

inline BytecodeVM::BytecodeVM(SimulationState* s) 
    : sim_state(s) {
    registerBuiltins();
    state.global_env = std::make_shared<Environment>();
}

inline SExprPtr BytecodeVM::execute(std::shared_ptr<BytecodeChunk> chunk) {
    if (!chunk) {
        return SExpr::makeNil();
    }
    
    state.clearError();
    state.pushFrame(chunk);
    
    while (!state.call_stack.empty() && !state.has_error) {
        auto& frame = state.currentFrame();
        
        if (frame.ip >= frame.chunk->instructions.size()) {
            // Function completed
            if (state.call_stack.size() == 1) {
                // Top-level function, return result
                break;
            }
            state.popFrame();
            continue;
        }
        
        const auto& inst = frame.chunk->instructions[frame.ip++];
        if (!executeInstruction(inst)) {
            break;
        }
    }
    
    // Clean up and return result
    state.call_stack.clear();
    if (state.stack_top > 0) {
        return state.pop();
    }
    return SExpr::makeNil();
}

inline SExprPtr BytecodeVM::executeWithTimeout(std::shared_ptr<BytecodeChunk> chunk,
                                               std::chrono::milliseconds timeout) {
    // Simple timeout implementation - in production would use interrupts
    auto future = std::async(std::launch::async, [this, chunk]() {
        return execute(chunk);
    });
    
    if (future.wait_for(timeout) == std::future_status::timeout) {
        // Timeout occurred
        state.setError("Execution timeout");
        return SExpr::makeNil();
    }
    
    return future.get();
}

inline bool BytecodeVM::executeInstruction(const Instruction& inst) {
    // Stub implementation - would need full implementation
    switch (inst.opcode) {
        case OpCode::PUSH_CONST:
            if (inst.arg1 < state.currentFrame().chunk->constants.size()) {
                state.push(state.currentFrame().chunk->constants[inst.arg1]);
            }
            break;
            
        case OpCode::PUSH_NIL:
            state.push(SExpr::makeNil());
            break;
            
        case OpCode::POP:
            state.pop();
            break;
            
        case OpCode::ADD:
        case OpCode::SUB:
        case OpCode::MUL:
        case OpCode::DIV:
            return executeArithmetic(inst.opcode);
            
        case OpCode::EQ:
        case OpCode::NE:
        case OpCode::LT:
        case OpCode::LE:
        case OpCode::GT:
        case OpCode::GE:
            return executeComparison(inst.opcode);
            
        case OpCode::HALT:
            return false;
            
        default:
            // Not implemented
            break;
    }
    return true;
}

inline bool BytecodeVM::executeArithmetic(OpCode op) {
    if (state.stack_top < 2) {
        state.setError("Stack underflow in arithmetic operation");
        return false;
    }
    
    auto b = state.pop();
    auto a = state.pop();
    
    double na = toNumber(a);
    double nb = toNumber(b);
    double result = 0;
    
    switch (op) {
        case OpCode::ADD: result = na + nb; break;
        case OpCode::SUB: result = na - nb; break;
        case OpCode::MUL: result = na * nb; break;
        case OpCode::DIV: 
            if (nb == 0) {
                state.setError("Division by zero");
                return false;
            }
            result = na / nb; 
            break;
        default:
            state.setError("Unknown arithmetic operation");
            return false;
    }
    
    state.push(makeNumber(result));
    return true;
}

inline bool BytecodeVM::executeComparison(OpCode op) {
    if (state.stack_top < 2) {
        state.setError("Stack underflow in comparison operation");
        return false;
    }
    
    auto b = state.pop();
    auto a = state.pop();
    
    double na = toNumber(a);
    double nb = toNumber(b);
    bool result = false;
    
    switch (op) {
        case OpCode::EQ: result = (na == nb); break;
        case OpCode::NE: result = (na != nb); break;
        case OpCode::LT: result = (na < nb); break;
        case OpCode::LE: result = (na <= nb); break;
        case OpCode::GT: result = (na > nb); break;
        case OpCode::GE: result = (na >= nb); break;
        default:
            state.setError("Unknown comparison operation");
            return false;
    }
    
    state.push(makeBool(result));
    return true;
}

inline double BytecodeVM::toNumber(SExprPtr value) {
    if (value && value->isNumber()) {
        return value->asNumber();
    }
    return 0.0;
}

inline bool BytecodeVM::toBool(SExprPtr value) {
    if (!value || value->isNil()) return false;
    if (value->isBool()) return value->asBool();
    if (value->isNumber()) return value->asNumber() != 0;
    return true;
}

inline SExprPtr BytecodeVM::makeNumber(double n) {
    return SExpr::makeNumber(n);
}

inline SExprPtr BytecodeVM::makeBool(bool b) {
    return SExpr::makeBool(b);
}

inline void BytecodeVM::registerBuiltins() {
    // Register basic built-in functions
    // These would be implemented properly in a real implementation
}

inline bool BytecodeVM::step() {
    // Single-step execution for debugging
    if (state.call_stack.empty() || state.has_error) {
        return false;
    }
    
    auto& frame = state.currentFrame();
    if (frame.ip >= frame.chunk->instructions.size()) {
        return false;
    }
    
    const auto& inst = frame.chunk->instructions[frame.ip++];
    return executeInstruction(inst);
}

inline void BytecodeVM::reset() {
    state = VMState();
}

inline bool BytecodeVM::executeLogical(OpCode op) {
    // Stub implementation
    return true;
}

inline bool BytecodeVM::executeJump(const Instruction& inst) {
    // Stub implementation
    return true;
}

inline bool BytecodeVM::executeCall(size_t arg_count) {
    // Stub implementation
    return true;
}

inline bool BytecodeVM::executeTailCall(size_t arg_count) {
    // Stub implementation
    return true;
}

inline bool BytecodeVM::executePatternMatch(const Instruction& inst) {
    // Stub implementation
    return true;
}

/**
 * VM execution context for concurrent execution
 */
class VMExecutor {
private:
    std::vector<std::unique_ptr<BytecodeVM>> vm_pool;
    std::queue<std::pair<std::shared_ptr<BytecodeChunk>, std::function<void(SExprPtr)>>> pending_tasks;
    std::mutex task_mutex;
    std::condition_variable task_cv;
    std::atomic<bool> should_stop{false};
    std::vector<std::thread> worker_threads;
    
    void workerLoop();
    
public:
    explicit VMExecutor(size_t thread_count = 4, SimulationState* state = nullptr);
    ~VMExecutor();
    
    // Async execution
    std::future<SExprPtr> executeAsync(std::shared_ptr<BytecodeChunk> chunk);
    void executeCallback(std::shared_ptr<BytecodeChunk> chunk,
                        std::function<void(SExprPtr)> callback);
    
    // Batch execution
    std::vector<SExprPtr> executeBatch(const std::vector<std::shared_ptr<BytecodeChunk>>& chunks);
    
    // Management
    void stop();
    size_t pendingTasks() const;
};

/**
 * JIT compiler interface (future extension)
 */
class JITCompiler {
public:
    // Compile hot paths to native code
    void* compileToNative(const BytecodeChunk& chunk, size_t start, size_t end);
    
    // Execute native code
    SExprPtr executeNative(void* native_code, VMState& state);
    
    // Profile-guided optimization
    void profileExecution(const BytecodeChunk& chunk, const VMState::Stats& stats);
    bool shouldJIT(const BytecodeChunk& chunk, size_t instruction_count);
};

} // namespace dsl
} // namespace digistar