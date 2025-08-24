#include "sexpr.h"
#include <sstream>
#include <fstream>
#include <cctype>
#include <cmath>
#include <algorithm>

namespace digistar {
namespace dsl {

// ============ SExpr Implementation ============

bool SExpr::isNumber() const {
    return isAtom() && std::holds_alternative<double>(atom_value);
}

bool SExpr::isSymbol() const {
    if (!isAtom()) return false;
    if (auto* s = std::get_if<std::string>(&atom_value)) {
        return !s->empty() && (*s)[0] != '"';
    }
    return false;
}

bool SExpr::isString() const {
    if (!isAtom()) return false;
    if (auto* s = std::get_if<std::string>(&atom_value)) {
        return !s->empty() && (*s)[0] == '"';
    }
    return false;
}

bool SExpr::isBool() const {
    return isAtom() && std::holds_alternative<bool>(atom_value);
}

bool SExpr::isVector() const {
    return isAtom() && std::holds_alternative<std::vector<double>>(atom_value);
}

bool SExpr::isClosure() const {
    return isAtom() && std::holds_alternative<std::shared_ptr<Closure>>(atom_value);
}

double SExpr::asNumber() const {
    return std::get<double>(atom_value);
}

const std::string& SExpr::asSymbol() const {
    return std::get<std::string>(atom_value);
}

const std::string& SExpr::asString() const {
    static std::string empty;
    if (auto* s = std::get_if<std::string>(&atom_value)) {
        if (s->length() >= 2 && (*s)[0] == '"' && (*s)[s->length()-1] == '"') {
            static std::string unquoted;
            unquoted = s->substr(1, s->length() - 2);
            return unquoted;
        }
        return *s;
    }
    return empty;
}

bool SExpr::asBool() const {
    return std::get<bool>(atom_value);
}

const std::vector<double>& SExpr::asVector() const {
    return std::get<std::vector<double>>(atom_value);
}

std::shared_ptr<Closure> SExpr::asClosure() const {
    return std::get<std::shared_ptr<Closure>>(atom_value);
}

size_t SExpr::length() const {
    return isList() ? list_value.size() : 0;
}

SExprPtr SExpr::car() const {
    if (isList() && !list_value.empty()) {
        return list_value[0];
    }
    return makeNil();
}

SExprPtr SExpr::cdr() const {
    if (isList() && list_value.size() > 1) {
        std::vector<SExprPtr> rest(list_value.begin() + 1, list_value.end());
        return makeList(rest);
    }
    return makeNil();
}

SExprPtr SExpr::nth(size_t n) const {
    if (isList() && n < list_value.size()) {
        return list_value[n];
    }
    return makeNil();
}

std::string SExpr::toString() const {
    if (isNil()) {
        return "nil";
    }
    
    if (isAtom()) {
        if (isNumber()) {
            std::ostringstream ss;
            ss << asNumber();
            return ss.str();
        }
        if (isSymbol() || isString()) {
            return asSymbol();
        }
        if (isBool()) {
            return asBool() ? "true" : "false";
        }
        if (isVector()) {
            std::ostringstream ss;
            ss << "[";
            auto& v = asVector();
            for (size_t i = 0; i < v.size(); i++) {
                if (i > 0) ss << " ";
                ss << v[i];
            }
            ss << "]";
            return ss.str();
        }
        if (isClosure()) {
            return "<closure>";
        }
    }
    
    if (isList()) {
        std::ostringstream ss;
        ss << "(";
        for (size_t i = 0; i < list_value.size(); i++) {
            if (i > 0) ss << " ";
            ss << list_value[i]->toString();
        }
        ss << ")";
        return ss.str();
    }
    
    return "?";
}

bool SExpr::equals(const SExpr& other) const {
    if (type != other.type) return false;
    
    if (isAtom()) {
        return atom_value == other.atom_value;
    }
    
    if (isList()) {
        if (list_value.size() != other.list_value.size()) return false;
        for (size_t i = 0; i < list_value.size(); i++) {
            if (!list_value[i]->equals(*other.list_value[i])) return false;
        }
        return true;
    }
    
    return true;  // Both nil
}

// Static constructors
SExprPtr SExpr::makeNumber(double n) {
    return std::make_shared<SExpr>(Atom(n));
}

SExprPtr SExpr::makeSymbol(const std::string& s) {
    return std::make_shared<SExpr>(Atom(s));
}

SExprPtr SExpr::makeString(const std::string& s) {
    return std::make_shared<SExpr>(Atom("\"" + s + "\""));
}

SExprPtr SExpr::makeBool(bool b) {
    return std::make_shared<SExpr>(Atom(b));
}

SExprPtr SExpr::makeVector(const std::vector<double>& v) {
    return std::make_shared<SExpr>(Atom(v));
}

SExprPtr SExpr::makeList(const std::vector<SExprPtr>& l) {
    return std::make_shared<SExpr>(l);
}

SExprPtr SExpr::makeClosure(std::shared_ptr<Closure> c) {
    return std::make_shared<SExpr>(Atom(c));
}

SExprPtr SExpr::makeNil() {
    return std::make_shared<SExpr>();
}

// ============ SExprParser Implementation ============

void SExprParser::skipWhitespace() {
    while (pos < input.length() && std::isspace(input[pos])) {
        pos++;
    }
    // Skip comments
    if (pos < input.length() && input[pos] == ';') {
        while (pos < input.length() && input[pos] != '\n') {
            pos++;
        }
        skipWhitespace();
    }
}

bool SExprParser::isDelimiter(char c) const {
    return c == '(' || c == ')' || c == '[' || c == ']' || 
           c == '\'' || c == '"' || std::isspace(c) || c == ';';
}

std::string SExprParser::readToken() {
    std::string token;
    while (pos < input.length() && !isDelimiter(input[pos])) {
        token += input[pos++];
    }
    return token;
}

std::string SExprParser::readString() {
    std::string str = "\"";
    pos++;  // Skip opening quote
    
    while (pos < input.length() && input[pos] != '"') {
        if (input[pos] == '\\' && pos + 1 < input.length()) {
            pos++;
            switch (input[pos]) {
                case 'n': str += '\n'; break;
                case 't': str += '\t'; break;
                case '\\': str += '\\'; break;
                case '"': str += '"'; break;
                default: str += input[pos];
            }
        } else {
            str += input[pos];
        }
        pos++;
    }
    
    if (pos < input.length()) {
        pos++;  // Skip closing quote
    }
    str += '"';
    return str;
}

std::vector<double> SExprParser::readVector() {
    std::vector<double> vec;
    pos++;  // Skip '['
    
    skipWhitespace();
    while (pos < input.length() && input[pos] != ']') {
        // Read number
        std::string num_str;
        if (input[pos] == '-') {
            num_str += input[pos++];
        }
        while (pos < input.length() && 
               (std::isdigit(input[pos]) || input[pos] == '.' || 
                input[pos] == 'e' || input[pos] == 'E')) {
            num_str += input[pos++];
        }
        
        if (!num_str.empty()) {
            vec.push_back(std::stod(num_str));
        }
        
        skipWhitespace();
    }
    
    if (pos < input.length()) {
        pos++;  // Skip ']'
    }
    
    return vec;
}

SExprPtr SExprParser::parseAtom(const std::string& token) {
    // Number
    if (!token.empty() && (std::isdigit(token[0]) || 
        (token[0] == '-' && token.length() > 1 && std::isdigit(token[1])) ||
        (token[0] == '.' && token.length() > 1 && std::isdigit(token[1])))) {
        try {
            return SExpr::makeNumber(std::stod(token));
        } catch (...) {
            // Not a valid number, treat as symbol
        }
    }
    
    // Boolean
    if (token == "true" || token == "#t") {
        return SExpr::makeBool(true);
    }
    if (token == "false" || token == "#f") {
        return SExpr::makeBool(false);
    }
    
    // Nil
    if (token == "nil" || token == "()") {
        return SExpr::makeNil();
    }
    
    // Symbol (including keywords starting with :)
    return SExpr::makeSymbol(token);
}

SExprPtr SExprParser::parseList() {
    std::vector<SExprPtr> elements;
    pos++;  // Skip '('
    
    skipWhitespace();
    while (pos < input.length() && input[pos] != ')') {
        elements.push_back(parse());
        skipWhitespace();
    }
    
    if (pos < input.length()) {
        pos++;  // Skip ')'
    }
    
    // Empty list is nil
    if (elements.empty()) {
        return SExpr::makeNil();
    }
    
    return SExpr::makeList(elements);
}

SExprPtr SExprParser::parseQuoted() {
    pos++;  // Skip quote
    SExprPtr expr = parse();
    
    // 'expr -> (quote expr)
    std::vector<SExprPtr> quoted;
    quoted.push_back(SExpr::makeSymbol("quote"));
    quoted.push_back(expr);
    
    return SExpr::makeList(quoted);
}

SExprPtr SExprParser::parse() {
    skipWhitespace();
    
    if (pos >= input.length()) {
        return SExpr::makeNil();
    }
    
    char c = input[pos];
    
    // List
    if (c == '(') {
        return parseList();
    }
    
    // Quoted expression
    if (c == '\'') {
        return parseQuoted();
    }
    
    // String
    if (c == '"') {
        std::string str = readString();
        return SExpr::makeSymbol(str);  // Store with quotes
    }
    
    // Vector
    if (c == '[') {
        std::vector<double> vec = readVector();
        return SExpr::makeVector(vec);
    }
    
    // Atom (number, symbol, keyword)
    std::string token = readToken();
    return parseAtom(token);
}

std::vector<SExprPtr> SExprParser::parseAll() {
    std::vector<SExprPtr> expressions;
    
    while (pos < input.length()) {
        skipWhitespace();
        if (pos >= input.length()) break;
        
        expressions.push_back(parse());
    }
    
    return expressions;
}

// Static methods
SExprPtr SExprParser::parseString(const std::string& s) {
    SExprParser parser(s);
    return parser.parse();
}

std::vector<SExprPtr> SExprParser::parseFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        return {};
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    
    SExprParser parser(buffer.str());
    return parser.parseAll();
}

// ============ Environment Implementation ============

void Environment::define(const std::string& name, SExprPtr value) {
    // Always define in the current environment
    // This can shadow bindings in parent environments
    bindings[name] = value;
}

bool Environment::set(const std::string& name, SExprPtr value) {
    // set! semantics: search for existing binding and mutate it
    auto it = bindings.find(name);
    if (it != bindings.end()) {
        it->second = value;
        return true;
    }
    
    // Not found locally, search parent environments
    if (parent) {
        return parent->set(name, value);
    }
    
    // Variable doesn't exist anywhere - this is an error
    return false;
}

SExprPtr Environment::lookup(const std::string& name) const {
    auto it = bindings.find(name);
    if (it != bindings.end()) {
        return it->second;
    }
    
    if (parent) {
        return parent->lookup(name);
    }
    
    // Return nullptr to indicate unbound variable
    // Caller should handle this as an error
    return nullptr;
}

bool Environment::contains(const std::string& name) const {
    if (bindings.find(name) != bindings.end()) {
        return true;
    }
    
    if (parent) {
        return parent->contains(name);
    }
    
    return false;
}

std::shared_ptr<Environment> Environment::extend() const {
    return std::make_shared<Environment>(
        std::const_pointer_cast<Environment>(shared_from_this())
    );
}

// ============ KeywordArgs Implementation ============

KeywordArgs::KeywordArgs(const std::vector<SExprPtr>& list) {
    for (size_t i = 0; i < list.size(); i++) {
        if (list[i]->isSymbol()) {
            const std::string& sym = list[i]->asSymbol();
            if (!sym.empty() && sym[0] == ':') {
                // Keyword argument
                std::string key = sym.substr(1);
                if (i + 1 < list.size()) {
                    args[key] = list[i + 1];
                    i++;  // Skip value
                }
            }
        }
    }
}

bool KeywordArgs::has(const std::string& key) const {
    return args.find(key) != args.end();
}

SExprPtr KeywordArgs::get(const std::string& key) const {
    auto it = args.find(key);
    if (it != args.end()) {
        return it->second;
    }
    return SExpr::makeNil();
}

SExprPtr KeywordArgs::getOrDefault(const std::string& key, SExprPtr default_val) const {
    auto it = args.find(key);
    if (it != args.end()) {
        return it->second;
    }
    return default_val;
}

double KeywordArgs::getNumber(const std::string& key, double default_val) const {
    SExprPtr val = get(key);
    if (val && val->isNumber()) {
        return val->asNumber();
    }
    return default_val;
}

std::string KeywordArgs::getString(const std::string& key, const std::string& default_val) const {
    SExprPtr val = get(key);
    if (val && (val->isString() || val->isSymbol())) {
        return val->isString() ? val->asString() : val->asSymbol();
    }
    return default_val;
}

bool KeywordArgs::getBool(const std::string& key, bool default_val) const {
    SExprPtr val = get(key);
    if (val && val->isBool()) {
        return val->asBool();
    }
    return default_val;
}

std::vector<double> KeywordArgs::getVector(const std::string& key, const std::vector<double>& default_val) const {
    SExprPtr val = get(key);
    if (val && val->isVector()) {
        return val->asVector();
    }
    return default_val;
}

} // namespace dsl
} // namespace digistar