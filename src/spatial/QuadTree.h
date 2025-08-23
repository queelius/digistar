#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <limits>
#include <cmath>
#include <iostream>
#include <algorithm>

// Simple 2D vector and bounding box types
struct Vec2 {
    float x, y;
    
    Vec2() : x(0), y(0) {}
    Vec2(float x_, float y_) : x(x_), y(y_) {}
    
    Vec2 operator+(const Vec2& v) const { return Vec2(x + v.x, y + v.y); }
    Vec2 operator-(const Vec2& v) const { return Vec2(x - v.x, y - v.y); }
    Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
    
    float lengthSquared() const { return x * x + y * y; }
    float length() const { return sqrt(lengthSquared()); }
};

struct BoundingBox {
    Vec2 min, max;
    
    BoundingBox() : min(0, 0), max(0, 0) {}
    BoundingBox(const Vec2& min_, const Vec2& max_) : min(min_), max(max_) {}
    BoundingBox(float x, float y, float width, float height) 
        : min(x, y), max(x + width, y + height) {}
    
    Vec2 center() const { return (min + max) * 0.5f; }
    Vec2 size() const { return max - min; }
    float width() const { return max.x - min.x; }
    float height() const { return max.y - min.y; }
    
    bool contains(const Vec2& point) const {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y;
    }
    
    bool intersects(const BoundingBox& other) const {
        return !(other.max.x < min.x || other.min.x > max.x ||
                 other.max.y < min.y || other.min.y > max.y);
    }
    
    bool contains(const BoundingBox& other) const {
        return other.min.x >= min.x && other.max.x <= max.x &&
               other.min.y >= min.y && other.max.y <= max.y;
    }
};

// Generic QuadTree implementation
// T is the type of objects stored (must have .pos member of type Vec2)
template<typename T>
class QuadTree {
public:
    struct Node {
        BoundingBox bounds;
        Vec2 center_of_mass;  // For Barnes-Hut
        float total_mass = 0;
        
        std::vector<T> objects;  // Leaf node stores objects
        std::unique_ptr<Node> children[4];  // NW, NE, SW, SE
        
        bool is_leaf = true;
        int depth = 0;
        
        Node(const BoundingBox& b, int d = 0) : bounds(b), depth(d) {}
    };
    
private:
    std::unique_ptr<Node> root;
    size_t max_objects_per_node;
    size_t max_depth;
    size_t total_objects;
    
    // Statistics
    mutable size_t nodes_visited = 0;
    mutable size_t distance_calculations = 0;
    
public:
    QuadTree(const BoundingBox& bounds, 
             size_t max_objects = 4, 
             size_t max_depth = 12)
        : root(std::make_unique<Node>(bounds)),
          max_objects_per_node(max_objects),
          max_depth(max_depth),
          total_objects(0) {}
    
    // Clear the tree
    void clear() {
        root = std::make_unique<Node>(root->bounds);
        total_objects = 0;
        nodes_visited = 0;
        distance_calculations = 0;
    }
    
    // Insert an object into the tree
    void insert(const T& obj) {
        if (root->bounds.contains(obj.pos)) {
            insertIntoNode(root.get(), obj);
            total_objects++;
        }
    }
    
    // Bulk insert for efficiency
    void insertBulk(const std::vector<T>& objects) {
        for (const auto& obj : objects) {
            insert(obj);
        }
    }
    
    // Query: Find all objects within a bounding box
    std::vector<T> query(const BoundingBox& range) const {
        std::vector<T> found;
        queryNode(root.get(), range, found);
        return found;
    }
    
    // Query: Find all objects within radius of a point
    std::vector<T> queryRadius(const Vec2& center, float radius) const {
        BoundingBox range(
            Vec2(center.x - radius, center.y - radius),
            Vec2(center.x + radius, center.y + radius)
        );
        
        std::vector<T> candidates = query(range);
        std::vector<T> results;
        float radius2 = radius * radius;
        
        for (const auto& obj : candidates) {
            Vec2 diff = obj.pos - center;
            if (diff.lengthSquared() <= radius2) {
                results.push_back(obj);
                distance_calculations++;
            }
        }
        
        return results;
    }
    
    // Query: Find k nearest neighbors
    std::vector<T> queryKNearest(const Vec2& point, size_t k) const {
        std::vector<std::pair<float, T>> candidates;
        float max_dist2 = std::numeric_limits<float>::max();
        
        queryKNearestNode(root.get(), point, k, candidates, max_dist2);
        
        std::vector<T> results;
        for (const auto& [dist, obj] : candidates) {
            results.push_back(obj);
        }
        return results;
    }
    
    // Barnes-Hut: Calculate force on a particle
    Vec2 calculateForce(const T& particle, float theta = 0.5f, float G = 1.0f, float softening = 0.1f) const {
        Vec2 force(0, 0);
        calculateForceNode(root.get(), particle, force, theta * theta, G, softening * softening);
        return force;
    }
    
    // Iterate over all objects
    void forEach(std::function<void(const T&)> callback) const {
        forEachNode(root.get(), callback);
    }
    
    // Get tree statistics
    size_t size() const { return total_objects; }
    size_t getNodeCount() const { return countNodes(root.get()); }
    size_t getHeight() const { return getNodeHeight(root.get()); }
    size_t getNodesVisited() const { return nodes_visited; }
    size_t getDistanceCalculations() const { return distance_calculations; }
    
    void resetStatistics() {
        nodes_visited = 0;
        distance_calculations = 0;
    }
    
    // Debug: Print tree structure
    void print(int max_depth = 3) const {
        printNode(root.get(), 0, max_depth);
    }
    
private:
    void insertIntoNode(Node* node, const T& obj) {
        // Update center of mass for Barnes-Hut
        float new_total_mass = node->total_mass + obj.mass;
        if (new_total_mass > 0) {
            node->center_of_mass.x = (node->center_of_mass.x * node->total_mass + obj.pos.x * obj.mass) / new_total_mass;
            node->center_of_mass.y = (node->center_of_mass.y * node->total_mass + obj.pos.y * obj.mass) / new_total_mass;
        }
        node->total_mass = new_total_mass;
        
        if (node->is_leaf) {
            node->objects.push_back(obj);
            
            // Split if we exceed capacity and haven't reached max depth
            if (node->objects.size() > max_objects_per_node && node->depth < max_depth) {
                subdivide(node);
            }
        } else {
            // Insert into appropriate child
            int index = getChildIndex(node, obj.pos);
            if (index >= 0 && node->children[index]) {
                insertIntoNode(node->children[index].get(), obj);
            }
        }
    }
    
    void subdivide(Node* node) {
        Vec2 center = node->bounds.center();
        Vec2 min = node->bounds.min;
        Vec2 max = node->bounds.max;
        
        // Create four children
        node->children[0] = std::make_unique<Node>(
            BoundingBox(Vec2(min.x, center.y), Vec2(center.x, max.y)), 
            node->depth + 1); // NW
        node->children[1] = std::make_unique<Node>(
            BoundingBox(Vec2(center.x, center.y), max), 
            node->depth + 1); // NE
        node->children[2] = std::make_unique<Node>(
            BoundingBox(min, center), 
            node->depth + 1); // SW
        node->children[3] = std::make_unique<Node>(
            BoundingBox(Vec2(center.x, min.y), Vec2(max.x, center.y)), 
            node->depth + 1); // SE
        
        // Move objects to children
        std::vector<T> objects = std::move(node->objects);
        node->is_leaf = false;
        
        for (const auto& obj : objects) {
            int index = getChildIndex(node, obj.pos);
            if (index >= 0 && node->children[index]) {
                insertIntoNode(node->children[index].get(), obj);
            }
        }
    }
    
    int getChildIndex(const Node* node, const Vec2& pos) const {
        Vec2 center = node->bounds.center();
        
        if (pos.x < center.x) {
            return (pos.y < center.y) ? 2 : 0;  // SW : NW
        } else {
            return (pos.y < center.y) ? 3 : 1;  // SE : NE
        }
    }
    
    void queryNode(const Node* node, const BoundingBox& range, std::vector<T>& found) const {
        if (!node || !node->bounds.intersects(range)) {
            return;
        }
        
        nodes_visited++;
        
        if (node->is_leaf) {
            for (const auto& obj : node->objects) {
                if (range.contains(obj.pos)) {
                    found.push_back(obj);
                }
            }
        } else {
            for (int i = 0; i < 4; i++) {
                if (node->children[i]) {
                    queryNode(node->children[i].get(), range, found);
                }
            }
        }
    }
    
    void queryKNearestNode(const Node* node, const Vec2& point, size_t k,
                           std::vector<std::pair<float, T>>& candidates,
                           float& max_dist2) const {
        if (!node) return;
        
        nodes_visited++;
        
        if (node->is_leaf) {
            for (const auto& obj : node->objects) {
                Vec2 diff = obj.pos - point;
                float dist2 = diff.lengthSquared();
                distance_calculations++;
                
                if (candidates.size() < k || dist2 < max_dist2) {
                    candidates.push_back({dist2, obj});
                    
                    // Keep only k best
                    if (candidates.size() > k) {
                        std::sort(candidates.begin(), candidates.end());
                        candidates.resize(k);
                        max_dist2 = candidates.back().first;
                    }
                }
            }
        } else {
            // Visit children in order of distance to point
            std::vector<std::pair<float, int>> child_dists;
            for (int i = 0; i < 4; i++) {
                if (node->children[i]) {
                    Vec2 child_center = node->children[i]->bounds.center();
                    Vec2 diff = child_center - point;
                    child_dists.push_back({diff.lengthSquared(), i});
                }
            }
            
            std::sort(child_dists.begin(), child_dists.end());
            
            for (const auto& [dist, index] : child_dists) {
                queryKNearestNode(node->children[index].get(), point, k, candidates, max_dist2);
            }
        }
    }
    
    void calculateForceNode(const Node* node, const T& particle, Vec2& force,
                           float theta2, float G, float soft2) const {
        if (!node || node->total_mass == 0) return;
        
        nodes_visited++;
        
        Vec2 diff = node->center_of_mass - particle.pos;
        float dist2 = diff.lengthSquared();
        
        if (node->is_leaf || (node->bounds.size().lengthSquared() / dist2 < theta2)) {
            // Use node as single body
            if (dist2 > 0) {
                distance_calculations++;
                float r2 = dist2 + soft2;
                float r = sqrt(r2);
                float f = G * node->total_mass / (r2 * r);
                force = force + diff * (f * particle.mass);
            }
        } else {
            // Recurse into children
            for (int i = 0; i < 4; i++) {
                if (node->children[i]) {
                    calculateForceNode(node->children[i].get(), particle, force, theta2, G, soft2);
                }
            }
        }
    }
    
    void forEachNode(const Node* node, std::function<void(const T&)> callback) const {
        if (!node) return;
        
        if (node->is_leaf) {
            for (const auto& obj : node->objects) {
                callback(obj);
            }
        } else {
            for (int i = 0; i < 4; i++) {
                if (node->children[i]) {
                    forEachNode(node->children[i].get(), callback);
                }
            }
        }
    }
    
    size_t countNodes(const Node* node) const {
        if (!node) return 0;
        
        size_t count = 1;
        if (!node->is_leaf) {
            for (int i = 0; i < 4; i++) {
                count += countNodes(node->children[i].get());
            }
        }
        return count;
    }
    
    size_t getNodeHeight(const Node* node) const {
        if (!node || node->is_leaf) return 1;
        
        size_t max_height = 0;
        for (int i = 0; i < 4; i++) {
            if (node->children[i]) {
                max_height = std::max(max_height, getNodeHeight(node->children[i].get()));
            }
        }
        return max_height + 1;
    }
    
    void printNode(const Node* node, int depth, int max_depth) const {
        if (!node || depth > max_depth) return;
        
        for (int i = 0; i < depth; i++) std::cout << "  ";
        
        std::cout << "Node [" << node->bounds.min.x << "," << node->bounds.min.y 
                  << " - " << node->bounds.max.x << "," << node->bounds.max.y << "] ";
        
        if (node->is_leaf) {
            std::cout << "Leaf with " << node->objects.size() << " objects";
        } else {
            std::cout << "Internal, mass=" << node->total_mass 
                      << " com=(" << node->center_of_mass.x << "," << node->center_of_mass.y << ")";
        }
        std::cout << std::endl;
        
        if (!node->is_leaf) {
            for (int i = 0; i < 4; i++) {
                if (node->children[i]) {
                    printNode(node->children[i].get(), depth + 1, max_depth);
                }
            }
        }
    }
};