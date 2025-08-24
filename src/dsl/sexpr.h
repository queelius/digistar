#pragma once

#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <unordered_map>

namespace digistar {
namespace dsl {

// Forward declarations
class SExpr;
class Closure;
using SExprPtr = std::shared_ptr<SExpr>;

// Value types that can be stored in an S-expression
using Atom = std::variant<
    double,                    // Numbers
    std::string,              // Symbols and strings
    bool,                     // Booleans
    std::vector<double>,      // Vectors [x y z]
    std::shared_ptr<Closure>  // Closures (lambdas)
>;

// S-expression node
class SExpr {
public:
    enum Type {
        ATOM,
        LIST,
        NIL
    };
    
private:
    Type type;
    Atom atom_value;
    std::vector<SExprPtr> list_value;
    
public:
    SExpr() : type(NIL) {}
    explicit SExpr(const Atom& a) : type(ATOM), atom_value(a) {}
    explicit SExpr(const std::vector<SExprPtr>& l) : type(LIST), list_value(l) {}
    
    Type getType() const { return type; }
    bool isAtom() const { return type == ATOM; }
    bool isList() const { return type == LIST; }
    bool isNil() const { return type == NIL; }
    
    // Atom accessors
    bool isNumber() const;
    bool isSymbol() const;
    bool isString() const;
    bool isBool() const;
    bool isVector() const;
    bool isClosure() const;
    
    double asNumber() const;
    const std::string& asSymbol() const;
    const std::string& asString() const;
    bool asBool() const;
    const std::vector<double>& asVector() const;
    std::shared_ptr<Closure> asClosure() const;
    
    // List accessors
    size_t length() const;
    SExprPtr car() const;  // First element
    SExprPtr cdr() const;  // Rest of list
    SExprPtr nth(size_t n) const;
    const std::vector<SExprPtr>& asList() const { return list_value; }
    
    // Utilities
    std::string toString() const;
    bool equals(const SExpr& other) const;
    
    // Static constructors for convenience
    static SExprPtr makeNumber(double n);
    static SExprPtr makeSymbol(const std::string& s);
    static SExprPtr makeString(const std::string& s);
    static SExprPtr makeBool(bool b);
    static SExprPtr makeVector(const std::vector<double>& v);
    static SExprPtr makeList(const std::vector<SExprPtr>& l);
    static SExprPtr makeClosure(std::shared_ptr<Closure> c);
    static SExprPtr makeNil();
};

// S-expression parser
class SExprParser {
private:
    std::string input;
    size_t pos;
    
    // Tokenization
    void skipWhitespace();
    bool isDelimiter(char c) const;
    std::string readToken();
    std::string readString();
    std::vector<double> readVector();
    
    // Parsing
    SExprPtr parseAtom(const std::string& token);
    SExprPtr parseList();
    SExprPtr parseQuoted();
    
public:
    explicit SExprParser(const std::string& s) : input(s), pos(0) {}
    
    SExprPtr parse();
    std::vector<SExprPtr> parseAll();  // Parse multiple expressions
    
    // Static parse methods
    static SExprPtr parseString(const std::string& s);
    static std::vector<SExprPtr> parseFile(const std::string& filename);
};

// Environment for variable bindings
class Environment : public std::enable_shared_from_this<Environment> {
private:
    std::unordered_map<std::string, SExprPtr> bindings;
    std::shared_ptr<Environment> parent;
    
public:
    Environment() = default;
    explicit Environment(std::shared_ptr<Environment> p) : parent(p) {}
    
    // Define a new binding in THIS environment (shadowing allowed)
    void define(const std::string& name, SExprPtr value);
    
    // Mutate an existing binding (searches up the chain) - returns false if not found
    bool set(const std::string& name, SExprPtr value);
    
    // Lookup a binding (searches parent envs)
    SExprPtr lookup(const std::string& name) const;
    
    // Check if binding exists (searches parent envs)
    bool contains(const std::string& name) const;
    
    // Create a new child environment
    std::shared_ptr<Environment> extend() const;
    
    // Get parent environment
    std::shared_ptr<Environment> getParent() const { return parent; }
};

// Closure - captures environment with a lambda
class Closure {
private:
    std::vector<std::string> params;  // Parameter names
    SExprPtr body;                    // Function body (can be a list of expressions)
    std::shared_ptr<Environment> env; // Captured environment
    
public:
    Closure(const std::vector<std::string>& params, 
            SExprPtr body,
            std::shared_ptr<Environment> env)
        : params(params), body(body), env(env) {}
    
    const std::vector<std::string>& getParams() const { return params; }
    SExprPtr getBody() const { return body; }
    std::shared_ptr<Environment> getEnv() const { return env; }
    
    // Check if this is a variadic function (last param is "...")
    bool isVariadic() const {
        return !params.empty() && params.back() == "...";
    }
    
    size_t getArity() const {
        if (isVariadic()) {
            return params.size() - 1;  // Minimum arity
        }
        return params.size();
    }
};

// Pretty printer
class SExprPrinter {
private:
    int indent_level;
    int indent_width;
    bool use_colors;
    
    std::string indent() const;
    std::string colorize(const std::string& s, const std::string& color) const;
    
public:
    SExprPrinter(int width = 2, bool colors = false) 
        : indent_level(0), indent_width(width), use_colors(colors) {}
    
    std::string print(SExprPtr expr);
    std::string printPretty(SExprPtr expr);
};

// Utility functions
inline bool isSymbol(SExprPtr expr, const std::string& sym) {
    return expr && expr->isSymbol() && expr->asSymbol() == sym;
}

inline bool isList(SExprPtr expr) {
    return expr && expr->isList();
}

inline bool isNumber(SExprPtr expr) {
    return expr && expr->isNumber();
}

// Helper to extract keyword arguments from a list
class KeywordArgs {
private:
    std::unordered_map<std::string, SExprPtr> args;
    
public:
    explicit KeywordArgs(const std::vector<SExprPtr>& list);
    
    bool has(const std::string& key) const;
    SExprPtr get(const std::string& key) const;
    SExprPtr getOrDefault(const std::string& key, SExprPtr default_val) const;
    
    double getNumber(const std::string& key, double default_val = 0.0) const;
    std::string getString(const std::string& key, const std::string& default_val = "") const;
    bool getBool(const std::string& key, bool default_val = false) const;
    std::vector<double> getVector(const std::string& key, const std::vector<double>& default_val = {}) const;
};

} // namespace dsl
} // namespace digistar