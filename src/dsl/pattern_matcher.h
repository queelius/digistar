#pragma once

#include "sexpr.h"
#include <functional>
#include <optional>
#include <regex>

namespace digistar {
namespace dsl {

/**
 * Pattern matching system for S-expressions
 * 
 * Supports:
 * - Literal matching (exact values)
 * - Variable binding (capture values)
 * - Wildcards (match anything)
 * - Guard conditions (predicates)
 * - Nested patterns
 * - List patterns with rest capture
 * - Type patterns
 */

// Forward declarations
class Pattern;
class PatternMatcher;
using PatternPtr = std::shared_ptr<Pattern>;
using GuardFunc = std::function<bool(const std::unordered_map<std::string, SExprPtr>&)>;

/**
 * Matching result containing captured bindings
 */
struct MatchResult {
    bool success;
    std::unordered_map<std::string, SExprPtr> bindings;
    
    MatchResult() : success(false) {}
    MatchResult(bool s) : success(s) {}
    MatchResult(const std::unordered_map<std::string, SExprPtr>& b) 
        : success(true), bindings(b) {}
    
    // Convenience accessors
    SExprPtr get(const std::string& name) const {
        auto it = bindings.find(name);
        return (it != bindings.end()) ? it->second : nullptr;
    }
    
    bool has(const std::string& name) const {
        return bindings.find(name) != bindings.end();
    }
};

/**
 * Base pattern class
 */
class Pattern {
public:
    enum Type {
        LITERAL,     // Exact value match
        VARIABLE,    // Capture to variable
        WILDCARD,    // Match anything (_)
        TYPE,        // Match specific type
        LIST,        // Match list structure
        CONS,        // Match head and tail (x:xs)
        GUARD,       // Pattern with guard condition
        OR,          // Alternative patterns
        AND,         // All patterns must match
        PREDICATE    // Custom predicate function
    };
    
    virtual ~Pattern() = default;
    virtual Type getType() const = 0;
    virtual MatchResult match(SExprPtr expr) const = 0;
    virtual std::string toString() const = 0;
};

/**
 * Literal pattern - matches exact values
 */
class LiteralPattern : public Pattern {
private:
    SExprPtr value;
    
public:
    explicit LiteralPattern(SExprPtr val) : value(val) {}
    
    Type getType() const override { return LITERAL; }
    MatchResult match(SExprPtr expr) const override;
    std::string toString() const override { return value->toString(); }
};

/**
 * Variable pattern - captures value to named variable
 */
class VariablePattern : public Pattern {
private:
    std::string name;
    
public:
    explicit VariablePattern(const std::string& n) : name(n) {}
    
    Type getType() const override { return VARIABLE; }
    MatchResult match(SExprPtr expr) const override;
    std::string toString() const override { return name; }
};

/**
 * Wildcard pattern - matches anything without capturing
 */
class WildcardPattern : public Pattern {
public:
    Type getType() const override { return WILDCARD; }
    MatchResult match(SExprPtr expr) const override { return MatchResult(true); }
    std::string toString() const override { return "_"; }
};

/**
 * Type pattern - matches specific S-expression types
 */
class TypePattern : public Pattern {
public:
    enum TypeConstraint {
        NUMBER,
        SYMBOL,
        STRING,
        BOOL,
        VECTOR,
        LIST,
        CLOSURE,
        NIL
    };
    
private:
    TypeConstraint constraint;
    std::optional<std::string> capture_name;
    
public:
    TypePattern(TypeConstraint c, const std::string& name = "")
        : constraint(c), capture_name(name.empty() ? std::nullopt : std::optional(name)) {}
    
    Type getType() const override { return TYPE; }
    MatchResult match(SExprPtr expr) const override;
    std::string toString() const override;
};

/**
 * List pattern - matches list structures
 */
class ListPattern : public Pattern {
private:
    std::vector<PatternPtr> element_patterns;
    std::optional<std::string> rest_capture; // For matching remaining elements
    
public:
    ListPattern(const std::vector<PatternPtr>& patterns, 
                const std::string& rest = "")
        : element_patterns(patterns),
          rest_capture(rest.empty() ? std::nullopt : std::optional(rest)) {}
    
    Type getType() const override { return LIST; }
    MatchResult match(SExprPtr expr) const override;
    std::string toString() const override;
};

/**
 * Cons pattern - matches head and tail of list (like x:xs in Haskell)
 */
class ConsPattern : public Pattern {
private:
    PatternPtr head_pattern;
    PatternPtr tail_pattern;
    
public:
    ConsPattern(PatternPtr head, PatternPtr tail)
        : head_pattern(head), tail_pattern(tail) {}
    
    Type getType() const override { return CONS; }
    MatchResult match(SExprPtr expr) const override;
    std::string toString() const override;
};

/**
 * Guard pattern - pattern with additional condition
 */
class GuardPattern : public Pattern {
private:
    PatternPtr base_pattern;
    GuardFunc guard;
    std::string guard_desc;
    
public:
    GuardPattern(PatternPtr base, GuardFunc g, const std::string& desc = "")
        : base_pattern(base), guard(g), guard_desc(desc) {}
    
    Type getType() const override { return GUARD; }
    MatchResult match(SExprPtr expr) const override;
    std::string toString() const override;
};

/**
 * Or pattern - matches if any sub-pattern matches
 */
class OrPattern : public Pattern {
private:
    std::vector<PatternPtr> alternatives;
    
public:
    OrPattern(const std::vector<PatternPtr>& alts) : alternatives(alts) {}
    
    Type getType() const override { return OR; }
    MatchResult match(SExprPtr expr) const override;
    std::string toString() const override;
};

/**
 * And pattern - all sub-patterns must match
 */
class AndPattern : public Pattern {
private:
    std::vector<PatternPtr> patterns;
    
public:
    AndPattern(const std::vector<PatternPtr>& pats) : patterns(pats) {}
    
    Type getType() const override { return AND; }
    MatchResult match(SExprPtr expr) const override;
    std::string toString() const override;
};

/**
 * Predicate pattern - custom matching function
 */
class PredicatePattern : public Pattern {
private:
    std::function<MatchResult(SExprPtr)> predicate;
    std::string description;
    
public:
    PredicatePattern(std::function<MatchResult(SExprPtr)> pred,
                    const std::string& desc = "custom")
        : predicate(pred), description(desc) {}
    
    Type getType() const override { return PREDICATE; }
    MatchResult match(SExprPtr expr) const override { return predicate(expr); }
    std::string toString() const override { return "<" + description + ">"; }
};

/**
 * Pattern builder for convenient pattern construction
 */
class PatternBuilder {
public:
    // Basic patterns
    static PatternPtr literal(SExprPtr value);
    static PatternPtr literal(double n);
    static PatternPtr literal(const std::string& s);
    static PatternPtr literal(bool b);
    
    static PatternPtr var(const std::string& name);
    static PatternPtr wildcard();
    
    // Type patterns
    static PatternPtr number(const std::string& capture = "");
    static PatternPtr symbol(const std::string& capture = "");
    static PatternPtr string(const std::string& capture = "");
    static PatternPtr list(const std::string& capture = "");
    
    // Structural patterns
    static PatternPtr list(const std::vector<PatternPtr>& elements);
    static PatternPtr listWithRest(const std::vector<PatternPtr>& elements,
                                   const std::string& rest);
    static PatternPtr cons(PatternPtr head, PatternPtr tail);
    
    // Composite patterns
    static PatternPtr withGuard(PatternPtr base, GuardFunc guard,
                                const std::string& desc = "");
    static PatternPtr anyOf(const std::vector<PatternPtr>& alternatives);
    static PatternPtr allOf(const std::vector<PatternPtr>& patterns);
    
    // Predicate patterns
    static PatternPtr predicate(std::function<bool(SExprPtr)> pred,
                                const std::string& desc = "");
    static PatternPtr range(double min, double max, const std::string& capture = "");
    static PatternPtr regex(const std::string& pattern, const std::string& capture = "");
    
    // Common patterns
    static PatternPtr eventPattern(const std::string& event_type);
    static PatternPtr collision(const std::string& p1 = "p1", 
                                const std::string& p2 = "p2",
                                const std::string& energy = "energy");
};

/**
 * Pattern matching engine
 */
class PatternMatcher {
private:
    std::vector<std::pair<PatternPtr, std::function<SExprPtr(const MatchResult&)>>> cases;
    std::optional<std::function<SExprPtr()>> default_case;
    
public:
    PatternMatcher() = default;
    
    // Add a pattern case with action
    PatternMatcher& addCase(PatternPtr pattern,
                            std::function<SExprPtr(const MatchResult&)> action);
    
    // Add default case
    PatternMatcher& setDefault(std::function<SExprPtr()> action);
    
    // Match expression against all patterns
    SExprPtr match(SExprPtr expr);
    
    // Static helper for single pattern matching
    static MatchResult matchPattern(PatternPtr pattern, SExprPtr expr) {
        return pattern->match(expr);
    }
    
    // Parse pattern from S-expression notation
    static PatternPtr parsePattern(SExprPtr pattern_expr);
};

/**
 * Pattern matching macros for DSL integration
 */
class PatternMacros {
public:
    // Register pattern matching macros in environment
    static void registerMacros(std::shared_ptr<Environment> env);
    
    // Pattern matching forms
    static SExprPtr matchMacro(const std::vector<SExprPtr>& args,
                               std::shared_ptr<Environment> env);
    static SExprPtr whenMatchMacro(const std::vector<SExprPtr>& args,
                                   std::shared_ptr<Environment> env);
    static SExprPtr defineMatchMacro(const std::vector<SExprPtr>& args,
                                    std::shared_ptr<Environment> env);
};

} // namespace dsl
} // namespace digistar