#include "src/spatial/QuadTree.h"
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>

// Simple particle for testing
struct Particle {
    Vec2 pos;
    float mass;
    int id;
    
    Particle() : pos(0, 0), mass(1), id(-1) {}
    Particle(float x, float y, float m = 1, int i = -1) 
        : pos(x, y), mass(m), id(i) {}
    
    // Comparison operator for sorting (needed by queryKNearest)
    bool operator<(const Particle& other) const {
        return id < other.id;
    }
};

// Test QuadTree spatial queries
void testSpatialQueries() {
    std::cout << "\n=== Spatial Query Tests ===" << std::endl;
    
    // Create quadtree covering 100x100 area
    QuadTree<Particle> tree(BoundingBox(Vec2(0, 0), Vec2(100, 100)));
    
    // Insert test particles in known pattern
    std::vector<Particle> particles;
    
    // Grid of particles
    for (int x = 10; x <= 90; x += 10) {
        for (int y = 10; y <= 90; y += 10) {
            particles.push_back(Particle(x, y, 1.0f, particles.size()));
            tree.insert(particles.back());
        }
    }
    
    std::cout << "Inserted " << tree.size() << " particles in grid pattern" << std::endl;
    std::cout << "Tree has " << tree.getNodeCount() << " nodes, height " << tree.getHeight() << std::endl;
    
    // Test 1: Bounding box query
    std::cout << "\n1. Bounding Box Query (20,20) to (40,40):" << std::endl;
    auto found = tree.query(BoundingBox(Vec2(20, 20), Vec2(40, 40)));
    std::cout << "   Found " << found.size() << " particles" << std::endl;
    
    // Test 2: Radius query
    std::cout << "\n2. Radius Query around (50,50) with r=15:" << std::endl;
    tree.resetStatistics();
    auto radius_found = tree.queryRadius(Vec2(50, 50), 15);
    std::cout << "   Found " << radius_found.size() << " particles" << std::endl;
    std::cout << "   Visited " << tree.getNodesVisited() << " nodes" << std::endl;
    std::cout << "   Made " << tree.getDistanceCalculations() << " distance calculations" << std::endl;
    
    // Test 3: K-nearest neighbors
    std::cout << "\n3. K-Nearest Neighbors to (45,55) with k=5:" << std::endl;
    tree.resetStatistics();
    auto nearest = tree.queryKNearest(Vec2(45, 55), 5);
    std::cout << "   Found " << nearest.size() << " nearest neighbors:" << std::endl;
    for (const auto& p : nearest) {
        Vec2 diff = p.pos - Vec2(45, 55);
        std::cout << "     Particle " << p.id << " at (" << p.pos.x << "," << p.pos.y 
                  << ") dist=" << diff.length() << std::endl;
    }
    std::cout << "   Visited " << tree.getNodesVisited() << " nodes" << std::endl;
}

// Test Barnes-Hut force calculation
void testBarnesHut() {
    std::cout << "\n=== Barnes-Hut Force Calculation Test ===" << std::endl;
    
    // Create random particle distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> pos_dist(10, 90);
    std::uniform_real_distribution<> mass_dist(0.5, 2.0);
    
    size_t n = 1000;
    std::vector<Particle> particles;
    for (size_t i = 0; i < n; i++) {
        particles.push_back(Particle(pos_dist(gen), pos_dist(gen), mass_dist(gen), i));
    }
    
    // Build tree
    QuadTree<Particle> tree(BoundingBox(Vec2(0, 0), Vec2(100, 100)));
    for (const auto& p : particles) {
        tree.insert(p);
    }
    
    std::cout << "Created " << n << " random particles" << std::endl;
    std::cout << "Tree has " << tree.getNodeCount() << " nodes" << std::endl;
    
    // Compare brute force vs Barnes-Hut
    Particle test_particle(50, 50, 1.0, -1);
    
    // Brute force
    auto start = std::chrono::high_resolution_clock::now();
    Vec2 brute_force(0, 0);
    for (const auto& p : particles) {
        Vec2 diff = p.pos - test_particle.pos;
        float r2 = diff.lengthSquared() + 0.01f;
        float r = sqrt(r2);
        float f = p.mass / (r2 * r);
        brute_force = brute_force + diff * f;
    }
    auto brute_time = std::chrono::high_resolution_clock::now() - start;
    
    // Barnes-Hut with different theta values
    std::vector<float> thetas = {0.3f, 0.5f, 0.7f, 1.0f};
    
    std::cout << "\nForce calculations on particle at (50,50):" << std::endl;
    std::cout << std::setw(10) << "Method" << std::setw(15) << "Force X" 
              << std::setw(15) << "Force Y" << std::setw(12) << "Time (μs)" 
              << std::setw(12) << "Speedup" << std::setw(15) << "Nodes Visited" << std::endl;
    std::cout << std::string(78, '-') << std::endl;
    
    std::cout << std::setw(10) << "Brute" 
              << std::setw(15) << brute_force.x 
              << std::setw(15) << brute_force.y
              << std::setw(12) << std::chrono::duration_cast<std::chrono::microseconds>(brute_time).count()
              << std::setw(12) << "1.0x"
              << std::setw(15) << n << std::endl;
    
    for (float theta : thetas) {
        tree.resetStatistics();
        start = std::chrono::high_resolution_clock::now();
        Vec2 bh_force = tree.calculateForce(test_particle, theta, 1.0f, 0.1f);
        auto bh_time = std::chrono::high_resolution_clock::now() - start;
        
        float speedup = (float)std::chrono::duration_cast<std::chrono::microseconds>(brute_time).count() / 
                       std::chrono::duration_cast<std::chrono::microseconds>(bh_time).count();
        
        std::cout << std::setw(10) << ("BH θ=" + std::to_string(theta).substr(0, 3))
                  << std::setw(15) << bh_force.x 
                  << std::setw(15) << bh_force.y
                  << std::setw(12) << std::chrono::duration_cast<std::chrono::microseconds>(bh_time).count()
                  << std::setw(12) << (std::to_string(speedup).substr(0, 4) + "x")
                  << std::setw(15) << tree.getNodesVisited() << std::endl;
    }
}

// Benchmark different operations
void benchmark() {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;
    
    std::vector<size_t> sizes = {100, 500, 1000, 5000, 10000};
    
    std::cout << std::setw(10) << "N" << std::setw(15) << "Build (ms)" 
              << std::setw(15) << "Nodes" << std::setw(15) << "Height"
              << std::setw(20) << "Range Query (μs)" << std::setw(20) << "Radius Query (μs)"
              << std::setw(20) << "Barnes-Hut (μs)" << std::endl;
    std::cout << std::string(115, '-') << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 100);
    
    for (size_t n : sizes) {
        // Generate random particles
        std::vector<Particle> particles;
        for (size_t i = 0; i < n; i++) {
            particles.push_back(Particle(dist(gen), dist(gen), 1.0f, i));
        }
        
        // Build tree
        auto start = std::chrono::high_resolution_clock::now();
        QuadTree<Particle> tree(BoundingBox(Vec2(0, 0), Vec2(100, 100)));
        for (const auto& p : particles) {
            tree.insert(p);
        }
        auto build_time = std::chrono::high_resolution_clock::now() - start;
        
        // Range query
        start = std::chrono::high_resolution_clock::now();
        auto found = tree.query(BoundingBox(Vec2(40, 40), Vec2(60, 60)));
        auto range_time = std::chrono::high_resolution_clock::now() - start;
        
        // Radius query
        start = std::chrono::high_resolution_clock::now();
        auto radius_found = tree.queryRadius(Vec2(50, 50), 10);
        auto radius_time = std::chrono::high_resolution_clock::now() - start;
        
        // Barnes-Hut force
        Particle test(50, 50, 1.0);
        start = std::chrono::high_resolution_clock::now();
        Vec2 force = tree.calculateForce(test, 0.5f);
        auto bh_time = std::chrono::high_resolution_clock::now() - start;
        
        std::cout << std::setw(10) << n
                  << std::setw(15) << std::chrono::duration_cast<std::chrono::milliseconds>(build_time).count()
                  << std::setw(15) << tree.getNodeCount()
                  << std::setw(15) << tree.getHeight()
                  << std::setw(20) << std::chrono::duration_cast<std::chrono::microseconds>(range_time).count()
                  << std::setw(20) << std::chrono::duration_cast<std::chrono::microseconds>(radius_time).count()
                  << std::setw(20) << std::chrono::duration_cast<std::chrono::microseconds>(bh_time).count()
                  << std::endl;
    }
}

int main() {
    std::cout << "=== QuadTree Test Suite ===" << std::endl;
    
    // Run tests
    testSpatialQueries();
    testBarnesHut();
    benchmark();
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "QuadTree provides:" << std::endl;
    std::cout << "  - O(log n) spatial queries (range, radius, k-nearest)" << std::endl;
    std::cout << "  - O(n log n) Barnes-Hut force calculation" << std::endl;
    std::cout << "  - Efficient neighbor finding for collision detection" << std::endl;
    std::cout << "  - General purpose spatial indexing" << std::endl;
    
    return 0;
}