#include "pattern_matcher.h"
#include <sstream>
#include <algorithm>

namespace digistar {
namespace dsl {

//=============================================================================
// LiteralPattern implementation
//=============================================================================

MatchResult LiteralPattern::match(SExprPtr expr) const {
    if (!expr || !value) {
        return MatchResult(false);
    }
    
    // Use S-expression equality comparison
    if (expr->equals(*value)) {
        return MatchResult(true);
    }
    
    return MatchResult(false);
}

//=============================================================================
// VariablePattern implementation
//=============================================================================

MatchResult VariablePattern::match(SExprPtr expr) const {
    if (!expr) {
        return MatchResult(false);
    }
    
    // Variable always matches and captures the value
    std::unordered_map<std::string, SExprPtr> bindings;
    bindings[name] = expr;
    return MatchResult(bindings);
}

//=============================================================================
// TypePattern implementation
//=============================================================================

MatchResult TypePattern::match(SExprPtr expr) const {
    if (!expr) {
        return MatchResult(false);
    }
    
    bool matches = false;
    
    switch (constraint) {
        case NUMBER:
            matches = expr->isNumber();
            break;
        case SYMBOL:
            matches = expr->isSymbol();
            break;
        case STRING:
            matches = expr->isString();
            break;
        case BOOL:
            matches = expr->isBool();
            break;
        case VECTOR:
            matches = expr->isVector();
            break;
        case LIST:
            matches = expr->isList();
            break;
        case CLOSURE:
            matches = expr->isClosure();
            break;
        case NIL:
            matches = expr->isNil();
            break;
    }
    
    if (matches) {
        if (capture_name.has_value()) {
            std::unordered_map<std::string, SExprPtr> bindings;
            bindings[capture_name.value()] = expr;
            return MatchResult(bindings);
        }
        return MatchResult(true);
    }
    
    return MatchResult(false);
}

std::string TypePattern::toString() const {
    std::string type_str;
    switch (constraint) {
        case NUMBER: type_str = "number"; break;
        case SYMBOL: type_str = "symbol"; break;
        case STRING: type_str = "string"; break;
        case BOOL: type_str = "bool"; break;
        case VECTOR: type_str = "vector"; break;
        case LIST: type_str = "list"; break;
        case CLOSURE: type_str = "closure"; break;
        case NIL: type_str = "nil"; break;
    }
    
    if (capture_name.has_value()) {
        return "<" + type_str + ":" + capture_name.value() + ">";
    }
    return "<" + type_str + ">";
}

//=============================================================================
// ListPattern implementation
//=============================================================================

MatchResult ListPattern::match(SExprPtr expr) const {
    if (!expr || !expr->isList()) {
        return MatchResult(false);
    }
    
    const auto& list = expr->asList();
    size_t pattern_count = element_patterns.size();
    size_t list_size = list.size();
    
    // Check size constraints
    if (rest_capture.has_value()) {
        // With rest capture, list must have at least pattern_count elements
        if (list_size < pattern_count) {
            return MatchResult(false);
        }
    } else {
        // Without rest capture, sizes must match exactly
        if (list_size != pattern_count) {
            return MatchResult(false);
        }
    }
    
    // Match each pattern against corresponding element
    std::unordered_map<std::string, SExprPtr> all_bindings;
    
    for (size_t i = 0; i < pattern_count; ++i) {
        MatchResult result = element_patterns[i]->match(list[i]);
        if (!result.success) {
            return MatchResult(false);
        }
        
        // Merge bindings
        for (const auto& [key, value] : result.bindings) {
            if (all_bindings.find(key) != all_bindings.end()) {
                // Variable already bound - check consistency
                if (!all_bindings[key]->equals(*value)) {
                    return MatchResult(false);
                }
            } else {
                all_bindings[key] = value;
            }
        }
    }
    
    // Capture rest if specified
    if (rest_capture.has_value() && list_size > pattern_count) {
        std::vector<SExprPtr> rest_elements(list.begin() + pattern_count, list.end());
        all_bindings[rest_capture.value()] = SExpr::makeList(rest_elements);
    }
    
    return MatchResult(all_bindings);
}

std::string ListPattern::toString() const {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < element_patterns.size(); ++i) {
        if (i > 0) ss << " ";
        ss << element_patterns[i]->toString();
    }
    if (rest_capture.has_value()) {
        ss << " . " << rest_capture.value();
    }
    ss << ")";
    return ss.str();
}

//=============================================================================
// ConsPattern implementation
//=============================================================================

MatchResult ConsPattern::match(SExprPtr expr) const {
    if (!expr || !expr->isList() || expr->length() == 0) {
        return MatchResult(false);
    }
    
    // Match head
    MatchResult head_result = head_pattern->match(expr->car());
    if (!head_result.success) {
        return MatchResult(false);
    }
    
    // Match tail
    MatchResult tail_result = tail_pattern->match(expr->cdr());
    if (!tail_result.success) {
        return MatchResult(false);
    }
    
    // Merge bindings
    std::unordered_map<std::string, SExprPtr> all_bindings = head_result.bindings;
    for (const auto& [key, value] : tail_result.bindings) {
        if (all_bindings.find(key) != all_bindings.end()) {
            // Check consistency
            if (!all_bindings[key]->equals(*value)) {
                return MatchResult(false);
            }
        } else {
            all_bindings[key] = value;
        }
    }
    
    return MatchResult(all_bindings);
}

std::string ConsPattern::toString() const {
    return head_pattern->toString() + ":" + tail_pattern->toString();
}

//=============================================================================
// GuardPattern implementation
//=============================================================================

MatchResult GuardPattern::match(SExprPtr expr) const {
    // First match the base pattern
    MatchResult base_result = base_pattern->match(expr);
    if (!base_result.success) {
        return MatchResult(false);
    }
    
    // Then check the guard condition
    if (guard(base_result.bindings)) {
        return base_result;
    }
    
    return MatchResult(false);
}

std::string GuardPattern::toString() const {
    std::string result = base_pattern->toString();
    if (!guard_desc.empty()) {
        result += " when " + guard_desc;
    }
    return result;
}

//=============================================================================
// OrPattern implementation
//=============================================================================

MatchResult OrPattern::match(SExprPtr expr) const {
    // Try each alternative in order
    for (const auto& pattern : alternatives) {
        MatchResult result = pattern->match(expr);
        if (result.success) {
            return result;
        }
    }
    
    return MatchResult(false);
}

std::string OrPattern::toString() const {
    std::stringstream ss;
    ss << "(or";
    for (const auto& pattern : alternatives) {
        ss << " " << pattern->toString();
    }
    ss << ")";
    return ss.str();
}

//=============================================================================
// AndPattern implementation
//=============================================================================

MatchResult AndPattern::match(SExprPtr expr) const {
    std::unordered_map<std::string, SExprPtr> all_bindings;
    
    // All patterns must match
    for (const auto& pattern : patterns) {
        MatchResult result = pattern->match(expr);
        if (!result.success) {
            return MatchResult(false);
        }
        
        // Merge bindings, checking for consistency
        for (const auto& [key, value] : result.bindings) {
            if (all_bindings.find(key) != all_bindings.end()) {
                if (!all_bindings[key]->equals(*value)) {
                    return MatchResult(false);
                }
            } else {
                all_bindings[key] = value;
            }
        }
    }
    
    return MatchResult(all_bindings);
}

std::string AndPattern::toString() const {
    std::stringstream ss;
    ss << "(and";
    for (const auto& pattern : patterns) {
        ss << " " << pattern->toString();
    }
    ss << ")";
    return ss.str();
}

//=============================================================================
// PatternBuilder implementation
//=============================================================================

PatternPtr PatternBuilder::literal(SExprPtr value) {
    return std::make_shared<LiteralPattern>(value);
}

PatternPtr PatternBuilder::literal(double n) {
    return std::make_shared<LiteralPattern>(SExpr::makeNumber(n));
}

PatternPtr PatternBuilder::literal(const std::string& s) {
    return std::make_shared<LiteralPattern>(SExpr::makeSymbol(s));
}

PatternPtr PatternBuilder::literal(bool b) {
    return std::make_shared<LiteralPattern>(SExpr::makeBool(b));
}

PatternPtr PatternBuilder::var(const std::string& name) {
    return std::make_shared<VariablePattern>(name);
}

PatternPtr PatternBuilder::wildcard() {
    return std::make_shared<WildcardPattern>();
}

PatternPtr PatternBuilder::number(const std::string& capture) {
    return std::make_shared<TypePattern>(TypePattern::NUMBER, capture);
}

PatternPtr PatternBuilder::symbol(const std::string& capture) {
    return std::make_shared<TypePattern>(TypePattern::SYMBOL, capture);
}

PatternPtr PatternBuilder::string(const std::string& capture) {
    return std::make_shared<TypePattern>(TypePattern::STRING, capture);
}

PatternPtr PatternBuilder::list(const std::string& capture) {
    return std::make_shared<TypePattern>(TypePattern::LIST, capture);
}

PatternPtr PatternBuilder::list(const std::vector<PatternPtr>& elements) {
    return std::make_shared<ListPattern>(elements);
}

PatternPtr PatternBuilder::listWithRest(const std::vector<PatternPtr>& elements,
                                        const std::string& rest) {
    return std::make_shared<ListPattern>(elements, rest);
}

PatternPtr PatternBuilder::cons(PatternPtr head, PatternPtr tail) {
    return std::make_shared<ConsPattern>(head, tail);
}

PatternPtr PatternBuilder::withGuard(PatternPtr base, GuardFunc guard,
                                     const std::string& desc) {
    return std::make_shared<GuardPattern>(base, guard, desc);
}

PatternPtr PatternBuilder::anyOf(const std::vector<PatternPtr>& alternatives) {
    return std::make_shared<OrPattern>(alternatives);
}

PatternPtr PatternBuilder::allOf(const std::vector<PatternPtr>& patterns) {
    return std::make_shared<AndPattern>(patterns);
}

PatternPtr PatternBuilder::predicate(std::function<bool(SExprPtr)> pred,
                                     const std::string& desc) {
    auto matcher = [pred](SExprPtr expr) -> MatchResult {
        if (pred(expr)) {
            return MatchResult(true);
        }
        return MatchResult(false);
    };
    return std::make_shared<PredicatePattern>(matcher, desc);
}

PatternPtr PatternBuilder::range(double min, double max, const std::string& capture) {
    auto matcher = [min, max, capture](SExprPtr expr) -> MatchResult {
        if (expr && expr->isNumber()) {
            double val = expr->asNumber();
            if (val >= min && val <= max) {
                if (!capture.empty()) {
                    std::unordered_map<std::string, SExprPtr> bindings;
                    bindings[capture] = expr;
                    return MatchResult(bindings);
                }
                return MatchResult(true);
            }
        }
        return MatchResult(false);
    };
    
    std::stringstream desc;
    desc << "range[" << min << "," << max << "]";
    return std::make_shared<PredicatePattern>(matcher, desc.str());
}

PatternPtr PatternBuilder::regex(const std::string& pattern, const std::string& capture) {
    std::regex re(pattern);
    
    auto matcher = [re, capture](SExprPtr expr) -> MatchResult {
        if (expr && (expr->isString() || expr->isSymbol())) {
            std::string str = expr->isString() ? expr->asString() : expr->asSymbol();
            if (std::regex_match(str, re)) {
                if (!capture.empty()) {
                    std::unordered_map<std::string, SExprPtr> bindings;
                    bindings[capture] = expr;
                    return MatchResult(bindings);
                }
                return MatchResult(true);
            }
        }
        return MatchResult(false);
    };
    
    return std::make_shared<PredicatePattern>(matcher, "regex:" + pattern);
}

PatternPtr PatternBuilder::eventPattern(const std::string& event_type) {
    // Pattern for matching event structures: (event-type data...)
    return list({
        literal(event_type),
        wildcard()  // Match any data
    });
}

PatternPtr PatternBuilder::collision(const std::string& p1, 
                                     const std::string& p2,
                                     const std::string& energy) {
    // Pattern for collision events: (collision particle1 particle2 energy-value)
    return list({
        literal("collision"),
        var(p1),
        var(p2),
        number(energy)
    });
}

//=============================================================================
// PatternMatcher implementation
//=============================================================================

PatternMatcher& PatternMatcher::addCase(PatternPtr pattern,
                                        std::function<SExprPtr(const MatchResult&)> action) {
    cases.emplace_back(pattern, action);
    return *this;
}

PatternMatcher& PatternMatcher::setDefault(std::function<SExprPtr()> action) {
    default_case = action;
    return *this;
}

SExprPtr PatternMatcher::match(SExprPtr expr) {
    // Try each pattern in order
    for (const auto& [pattern, action] : cases) {
        MatchResult result = pattern->match(expr);
        if (result.success) {
            return action(result);
        }
    }
    
    // Use default case if no pattern matched
    if (default_case.has_value()) {
        return default_case.value()();
    }
    
    // No match found
    return SExpr::makeNil();
}

PatternPtr PatternMatcher::parsePattern(SExprPtr pattern_expr) {
    if (!pattern_expr) {
        return PatternBuilder::wildcard();
    }
    
    // Handle atoms
    if (pattern_expr->isAtom()) {
        if (pattern_expr->isSymbol()) {
            const std::string& sym = pattern_expr->asSymbol();
            
            // Special symbols
            if (sym == "_") {
                return PatternBuilder::wildcard();
            } else if (sym.length() > 0 && sym[0] == '?') {
                // Variable pattern: ?name
                return PatternBuilder::var(sym.substr(1));
            } else if (sym.length() > 0 && sym[0] == ':') {
                // Type pattern: :number, :string, etc.
                std::string type = sym.substr(1);
                if (type == "number") return PatternBuilder::number();
                if (type == "string") return PatternBuilder::string();
                if (type == "symbol") return PatternBuilder::symbol();
                if (type == "list") return PatternBuilder::list();
            }
            // Otherwise literal symbol
            return PatternBuilder::literal(pattern_expr);
        }
        // Other atoms are literals
        return PatternBuilder::literal(pattern_expr);
    }
    
    // Handle lists
    if (pattern_expr->isList()) {
        const auto& list = pattern_expr->asList();
        
        if (list.empty()) {
            // Empty list pattern
            return PatternBuilder::list(std::vector<PatternPtr>{});
        }
        
        // Check for special forms
        if (list[0]->isSymbol()) {
            const std::string& head = list[0]->asSymbol();
            
            if (head == "quote" && list.size() == 2) {
                // Quoted literal
                return PatternBuilder::literal(list[1]);
            } else if (head == "or" && list.size() > 1) {
                // Or pattern
                std::vector<PatternPtr> alternatives;
                for (size_t i = 1; i < list.size(); ++i) {
                    alternatives.push_back(parsePattern(list[i]));
                }
                return PatternBuilder::anyOf(alternatives);
            } else if (head == "and" && list.size() > 1) {
                // And pattern
                std::vector<PatternPtr> patterns;
                for (size_t i = 1; i < list.size(); ++i) {
                    patterns.push_back(parsePattern(list[i]));
                }
                return PatternBuilder::allOf(patterns);
            }
        }
        
        // Regular list pattern
        std::vector<PatternPtr> element_patterns;
        std::string rest_var;
        
        for (size_t i = 0; i < list.size(); ++i) {
            if (i == list.size() - 2 && 
                list[i]->isSymbol() && 
                list[i]->asSymbol() == ".") {
                // Rest pattern: (a b . rest)
                if (list[i+1]->isSymbol() && list[i+1]->asSymbol()[0] == '?') {
                    rest_var = list[i+1]->asSymbol().substr(1);
                }
                break;
            }
            element_patterns.push_back(parsePattern(list[i]));
        }
        
        if (!rest_var.empty()) {
            return PatternBuilder::listWithRest(element_patterns, rest_var);
        }
        return PatternBuilder::list(element_patterns);
    }
    
    // Default to wildcard
    return PatternBuilder::wildcard();
}

//=============================================================================
// PatternMacros implementation
//=============================================================================

void PatternMacros::registerMacros(std::shared_ptr<Environment> env) {
    // Register match macro
    // (match expr
    //   [pattern1 result1]
    //   [pattern2 result2]
    //   [_ default])
    
    // This would be implemented in the evaluator as a special form
    // Here we just prepare the pattern matching infrastructure
}

} // namespace dsl
} // namespace digistar