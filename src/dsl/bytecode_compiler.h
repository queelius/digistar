#pragma once

#include "sexpr.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <stack>

namespace digistar {
namespace dsl {

/**
 * Bytecode instruction set for DigiStar DSL VM
 * 
 * Design principles:
 * - Stack-based architecture for simplicity
 * - Compact encoding (32-bit instructions)
 * - Direct threading support for performance
 * - Cache-friendly instruction layout
 */

enum class OpCode : uint8_t {
    // Stack operations
    PUSH_CONST,      // Push constant onto stack
    PUSH_VAR,        // Push variable value
    PUSH_NIL,        // Push nil onto stack
    POP,             // Pop and discard
    DUP,             // Duplicate top of stack
    SWAP,            // Swap top two elements
    ROT,             // Rotate top 3 elements
    
    // Arithmetic
    ADD,             // Pop 2, push sum
    SUB,             // Pop 2, push difference  
    MUL,             // Pop 2, push product
    DIV,             // Pop 2, push quotient
    MOD,             // Pop 2, push remainder
    NEG,             // Negate top
    
    // Comparison
    EQ,              // Equal
    NE,              // Not equal
    LT,              // Less than
    LE,              // Less or equal
    GT,              // Greater than
    GE,              // Greater or equal
    
    // Logical
    AND,             // Logical AND
    OR,              // Logical OR
    NOT,             // Logical NOT
    
    // Control flow
    JMP,             // Unconditional jump
    JMP_IF,          // Jump if true
    JMP_UNLESS,      // Jump if false
    CALL,            // Call function
    TAIL_CALL,       // Tail call optimization
    RET,             // Return from function
    
    // Variable operations
    LOAD_LOCAL,      // Load local variable
    STORE_LOCAL,     // Store local variable
    LOAD_GLOBAL,     // Load global variable
    STORE_GLOBAL,    // Store global variable
    LOAD_UPVAL,      // Load upvalue (closure)
    STORE_UPVAL,     // Store upvalue
    
    // Object operations
    MAKE_LIST,       // Create list from N stack items
    MAKE_VECTOR,     // Create vector from N stack items
    GET_ELEMENT,     // Get list/vector element
    SET_ELEMENT,     // Set list/vector element
    LIST_LENGTH,     // Get list length
    
    // Function operations
    MAKE_CLOSURE,    // Create closure
    LOAD_BUILTIN,    // Load built-in function
    
    // Pattern matching
    MATCH_BEGIN,     // Start pattern match
    MATCH_LITERAL,   // Match literal value
    MATCH_VAR,       // Match and bind variable
    MATCH_TYPE,      // Match type
    MATCH_LIST,      // Match list pattern
    MATCH_FAIL,      // Pattern match failed
    MATCH_END,       // End pattern match
    
    // Special operations
    HALT,            // Stop execution
    NOP,             // No operation
    DEBUG,           // Debug breakpoint
    
    // Extended operations for DSL
    CREATE_PARTICLE, // Create particle(s)
    QUERY_PARTICLES, // Query particle state
    EMIT_EVENT,      // Emit event
    WAIT,            // Wait for time/event
    SPAWN,           // Spawn concurrent task
    
    // Optimization hints
    LOOP_BEGIN,      // Loop start marker
    LOOP_END,        // Loop end marker
    HOT_PATH,        // Frequently executed code
};

/**
 * Bytecode instruction format
 */
struct Instruction {
    OpCode opcode;
    uint8_t arg1;    // First argument (register, constant index, etc.)
    uint16_t arg2;   // Second argument (offset, count, etc.)
    
    Instruction(OpCode op = OpCode::NOP, uint8_t a1 = 0, uint16_t a2 = 0)
        : opcode(op), arg1(a1), arg2(a2) {}
    
    uint32_t encode() const {
        return (static_cast<uint32_t>(opcode) << 24) |
               (static_cast<uint32_t>(arg1) << 16) |
               static_cast<uint32_t>(arg2);
    }
    
    static Instruction decode(uint32_t encoded) {
        return Instruction(
            static_cast<OpCode>((encoded >> 24) & 0xFF),
            static_cast<uint8_t>((encoded >> 16) & 0xFF),
            static_cast<uint16_t>(encoded & 0xFFFF)
        );
    }
};

/**
 * Compiled bytecode chunk
 */
class BytecodeChunk {
public:
    std::vector<Instruction> instructions;
    std::vector<SExprPtr> constants;       // Constant pool
    std::vector<std::string> variables;    // Variable names
    std::vector<std::string> debug_info;   // Debug information
    
    // Metadata
    std::string name;
    size_t max_stack_depth = 0;
    size_t local_count = 0;
    bool is_optimized = false;
    
    // Add instruction
    size_t emit(OpCode op, uint8_t arg1 = 0, uint16_t arg2 = 0) {
        size_t pos = instructions.size();
        instructions.emplace_back(op, arg1, arg2);
        return pos;
    }
    
    // Add constant and return its index
    uint16_t addConstant(SExprPtr value) {
        constants.push_back(value);
        return static_cast<uint16_t>(constants.size() - 1);
    }
    
    // Add variable and return its index
    uint16_t addVariable(const std::string& name) {
        auto it = std::find(variables.begin(), variables.end(), name);
        if (it != variables.end()) {
            return static_cast<uint16_t>(std::distance(variables.begin(), it));
        }
        variables.push_back(name);
        return static_cast<uint16_t>(variables.size() - 1);
    }
    
    // Patch jump instruction
    void patchJump(size_t instruction_idx, size_t target) {
        instructions[instruction_idx].arg2 = static_cast<uint16_t>(target);
    }
    
    // Get current instruction pointer
    size_t currentIP() const { return instructions.size(); }
    
    // Disassemble for debugging
    std::string disassemble() const;
};

/**
 * Compilation context
 */
class CompilationContext {
public:
    struct Scope {
        std::unordered_map<std::string, uint16_t> locals;
        size_t local_count = 0;
        bool is_function = false;
    };
    
    std::stack<Scope> scopes;
    std::shared_ptr<BytecodeChunk> chunk;
    size_t current_stack_depth = 0;
    size_t max_stack_depth = 0;
    
    // Loop tracking for optimizations
    struct LoopInfo {
        size_t start;
        size_t continue_target;
        std::vector<size_t> break_targets;
    };
    std::stack<LoopInfo> loops;
    
    CompilationContext() : chunk(std::make_shared<BytecodeChunk>()) {
        pushScope(); // Global scope
    }
    
    explicit CompilationContext(std::shared_ptr<BytecodeChunk> ch) : chunk(ch) {
        pushScope(); // Global scope
    }
    
    void pushScope(bool is_function = false) {
        Scope scope;
        scope.is_function = is_function;
        if (!scopes.empty() && !is_function) {
            // Inherit locals from parent scope
            scope.locals = scopes.top().locals;
            scope.local_count = scopes.top().local_count;
        }
        scopes.push(scope);
    }
    
    void popScope() {
        if (!scopes.empty()) {
            scopes.pop();
        }
    }
    
    uint16_t defineLocal(const std::string& name) {
        if (scopes.empty()) return 0;
        
        auto& scope = scopes.top();
        uint16_t index = static_cast<uint16_t>(scope.local_count++);
        scope.locals[name] = index;
        return index;
    }
    
    std::optional<uint16_t> resolveLocal(const std::string& name) {
        if (scopes.empty()) return std::nullopt;
        
        const auto& locals = scopes.top().locals;
        auto it = locals.find(name);
        if (it != locals.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    void trackStackDepth(int delta) {
        current_stack_depth += delta;
        if (current_stack_depth > max_stack_depth) {
            max_stack_depth = current_stack_depth;
        }
    }
    
    // Emit instructions
    size_t emit(OpCode op, uint16_t arg1 = 0, uint16_t arg2 = 0) {
        size_t offset = chunk->instructions.size();
        chunk->instructions.push_back({op, arg1, arg2});
        
        // Track stack depth changes
        switch (op) {
            case OpCode::PUSH_CONST:
            case OpCode::PUSH_VAR:
            case OpCode::PUSH_NIL:
            case OpCode::LOAD_LOCAL:
            case OpCode::LOAD_GLOBAL:
                trackStackDepth(1);
                break;
            case OpCode::POP:
            case OpCode::STORE_LOCAL:
            case OpCode::STORE_GLOBAL:
                trackStackDepth(-1);
                break;
            case OpCode::ADD:
            case OpCode::SUB:
            case OpCode::MUL:
            case OpCode::DIV:
            case OpCode::EQ:
            case OpCode::NE:
            case OpCode::LT:
            case OpCode::LE:
            case OpCode::GT:
            case OpCode::GE:
                trackStackDepth(-1); // Two operands consumed, one result pushed
                break;
            default:
                break;
        }
        
        return offset;
    }
    
    // Patch jump instructions
    void patchJump(size_t offset) {
        if (offset >= chunk->instructions.size()) return;
        
        size_t jump_target = chunk->instructions.size();
        auto& inst = chunk->instructions[offset];
        
        // Store the jump offset in arg1
        inst.arg1 = static_cast<uint16_t>(jump_target - offset - 1);
    }
};

/**
 * Bytecode compiler
 */
class BytecodeCompiler {
private:
    std::unordered_map<std::string, std::shared_ptr<BytecodeChunk>> compiled_cache;
    
    // Compilation methods
    void compileExpression(SExprPtr expr, CompilationContext& ctx);
    void compileList(SExprPtr expr, CompilationContext& ctx);
    void compileSpecialForm(const std::string& form, 
                           const std::vector<SExprPtr>& args,
                           CompilationContext& ctx);
    void compileFunctionCall(SExprPtr func,
                            const std::vector<SExprPtr>& args,
                            CompilationContext& ctx);
    
    // Special form compilers
    void compileIf(const std::vector<SExprPtr>& args, CompilationContext& ctx);
    void compileLet(const std::vector<SExprPtr>& args, CompilationContext& ctx);
    void compileLambda(const std::vector<SExprPtr>& args, CompilationContext& ctx);
    void compileDefine(const std::vector<SExprPtr>& args, CompilationContext& ctx);
    void compileBegin(const std::vector<SExprPtr>& args, CompilationContext& ctx);
    void compileWhile(const std::vector<SExprPtr>& args, CompilationContext& ctx);
    void compileMatch(const std::vector<SExprPtr>& args, CompilationContext& ctx);
    
    // Pattern compilation
    void compilePattern(SExprPtr pattern, CompilationContext& ctx);
    
    // Optimization passes
    void optimizeTailCalls(BytecodeChunk& chunk);
    void eliminateDeadCode(BytecodeChunk& chunk);
    void foldConstants(BytecodeChunk& chunk);
    void inlineSmallFunctions(BytecodeChunk& chunk);
    
    // Helper methods
    bool isSpecialForm(const std::string& name) const;
    bool isTailPosition(const CompilationContext& ctx) const;
    
public:
    BytecodeCompiler() = default;
    
    // Main compilation interface
    std::shared_ptr<BytecodeChunk> compile(SExprPtr expr,
                                          const std::string& name = "anonymous");
    
    // Compile multiple expressions (module)
    std::shared_ptr<BytecodeChunk> compileModule(const std::vector<SExprPtr>& exprs,
                                                const std::string& name = "module");
    
    // Compile with optimization
    std::shared_ptr<BytecodeChunk> compileOptimized(SExprPtr expr,
                                                   const std::string& name = "anonymous");
    
    // Cache management
    void cacheChunk(const std::string& key, std::shared_ptr<BytecodeChunk> chunk) {
        compiled_cache[key] = chunk;
    }
    
    std::shared_ptr<BytecodeChunk> getCachedChunk(const std::string& key) {
        auto it = compiled_cache.find(key);
        return (it != compiled_cache.end()) ? it->second : nullptr;
    }
    
    void clearCache() { compiled_cache.clear(); }
    
    // Debugging
    std::string disassemble(const BytecodeChunk& chunk) const;
    void dumpChunk(const BytecodeChunk& chunk, const std::string& filename) const;
};

// ============================================================================
// BytecodeCompiler Implementation
// ============================================================================

inline std::shared_ptr<BytecodeChunk> BytecodeCompiler::compile(SExprPtr expr,
                                                               const std::string& name) {
    if (!expr) {
        return nullptr;
    }
    
    // Create new chunk
    auto chunk = std::make_shared<BytecodeChunk>();
    chunk->name = name;
    
    // Create compilation context
    CompilationContext ctx(chunk);
    
    // Compile the expression
    compileExpression(expr, ctx);
    
    // Add halt instruction
    ctx.emit(OpCode::HALT);
    
    // Set metadata
    chunk->max_stack_depth = ctx.max_stack_depth;
    
    return chunk;
}

inline void BytecodeCompiler::compileExpression(SExprPtr expr, CompilationContext& ctx) {
    if (!expr) {
        ctx.emit(OpCode::PUSH_NIL);
        return;
    }
    
    if (expr->isNumber()) {
        auto idx = ctx.chunk->addConstant(expr);
        ctx.emit(OpCode::PUSH_CONST, idx);
    } else if (expr->isSymbol()) {
        // For now, just push as constant
        auto idx = ctx.chunk->addConstant(expr);
        ctx.emit(OpCode::PUSH_CONST, idx);
    } else if (expr->isBool()) {
        auto idx = ctx.chunk->addConstant(expr);
        ctx.emit(OpCode::PUSH_CONST, idx);
    } else if (expr->isString()) {
        auto idx = ctx.chunk->addConstant(expr);
        ctx.emit(OpCode::PUSH_CONST, idx);
    } else if (expr->isList()) {
        compileList(expr, ctx);
    } else {
        ctx.emit(OpCode::PUSH_NIL);
    }
}

inline void BytecodeCompiler::compileList(SExprPtr expr, CompilationContext& ctx) {
    const auto& list = expr->asList();
    if (list.empty()) {
        ctx.emit(OpCode::PUSH_NIL);
        return;
    }
    
    // Check for special forms
    if (list[0]->isSymbol()) {
        const std::string& op = list[0]->asSymbol();
        
        // Handle arithmetic operations
        if (op == "+") {
            // Compile arguments
            for (size_t i = 1; i < list.size(); ++i) {
                compileExpression(list[i], ctx);
            }
            // Emit ADD instructions
            for (size_t i = 2; i < list.size(); ++i) {
                ctx.emit(OpCode::ADD);
            }
        } else if (op == "-") {
            if (list.size() >= 2) {
                compileExpression(list[1], ctx);
                for (size_t i = 2; i < list.size(); ++i) {
                    compileExpression(list[i], ctx);
                    ctx.emit(OpCode::SUB);
                }
            }
        } else if (op == "*") {
            for (size_t i = 1; i < list.size(); ++i) {
                compileExpression(list[i], ctx);
            }
            for (size_t i = 2; i < list.size(); ++i) {
                ctx.emit(OpCode::MUL);
            }
        } else if (op == "/") {
            if (list.size() >= 2) {
                compileExpression(list[1], ctx);
                for (size_t i = 2; i < list.size(); ++i) {
                    compileExpression(list[i], ctx);
                    ctx.emit(OpCode::DIV);
                }
            }
        } else if (op == ">") {
            if (list.size() == 3) {
                compileExpression(list[1], ctx);
                compileExpression(list[2], ctx);
                ctx.emit(OpCode::GT);
            }
        } else if (op == "<") {
            if (list.size() == 3) {
                compileExpression(list[1], ctx);
                compileExpression(list[2], ctx);
                ctx.emit(OpCode::LT);
            }
        } else if (op == "if") {
            // Simple if compilation
            if (list.size() >= 3) {
                compileExpression(list[1], ctx);  // condition
                auto jump_if_false = ctx.emit(OpCode::JMP_UNLESS, 0);
                compileExpression(list[2], ctx);  // then branch
                
                if (list.size() > 3) {
                    auto jump_end = ctx.emit(OpCode::JMP, 0);
                    ctx.patchJump(jump_if_false);
                    compileExpression(list[3], ctx);  // else branch
                    ctx.patchJump(jump_end);
                } else {
                    ctx.patchJump(jump_if_false);
                    ctx.emit(OpCode::PUSH_NIL);
                }
            }
        } else {
            // Generic function call - compile all arguments
            for (size_t i = 1; i < list.size(); ++i) {
                compileExpression(list[i], ctx);
            }
            // For now, just keep the last value
            for (size_t i = 1; i < list.size() - 1; ++i) {
                ctx.emit(OpCode::POP);
            }
        }
    } else {
        // Not a function call, compile as list literal
        for (const auto& elem : list) {
            compileExpression(elem, ctx);
        }
    }
}

inline bool BytecodeCompiler::isSpecialForm(const std::string& name) const {
    static const std::unordered_set<std::string> special_forms = {
        "if", "let", "lambda", "define", "begin", "while", "match",
        "quote", "set!", "and", "or"
    };
    return special_forms.find(name) != special_forms.end();
}

/**
 * Bytecode optimizer
 */
class BytecodeOptimizer {
private:
    // Analysis passes
    struct BasicBlock {
        size_t start;
        size_t end;
        std::vector<size_t> predecessors;
        std::vector<size_t> successors;
        bool is_loop_header = false;
    };
    
    std::vector<BasicBlock> analyzeControlFlow(const BytecodeChunk& chunk);
    std::unordered_map<size_t, int> analyzeStackEffect(const BytecodeChunk& chunk);
    
public:
    // Optimization passes
    void optimizeChunk(BytecodeChunk& chunk);
    void performPeepholeOptimization(BytecodeChunk& chunk);
    void performLoopOptimization(BytecodeChunk& chunk);
    void performInlining(BytecodeChunk& chunk);
    
    // Analysis
    bool isHotPath(const BytecodeChunk& chunk, size_t start, size_t end);
    bool canInline(const BytecodeChunk& chunk);
};

} // namespace dsl
} // namespace digistar