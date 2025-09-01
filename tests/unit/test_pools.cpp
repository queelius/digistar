#include "../../src/physics/pools.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <random>

using namespace digistar;

// Test helper macros
#define ASSERT(cond) do { if (!(cond)) { std::cerr << "FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(1); } } while(0)
#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_LT(a, b) ASSERT((a) < (b))
#define ASSERT_GT(a, b) ASSERT((a) > (b))
#define ASSERT_NEAR(a, b, eps) ASSERT(std::abs((a) - (b)) < (eps))

struct TestCase {
    const char* name;
    void (*func)();
};

// Forward declarations
void test_particle_pool_allocate();
void test_particle_pool_create();
void test_particle_pool_destroy_single();
void test_particle_pool_destroy_swap_and_pop();
void test_particle_pool_clear_forces();
void test_particle_pool_apply_boundaries();
void test_particle_pool_stress_test();
void test_particle_pool_particle_ref();
void test_spring_pool_allocate();
void test_spring_pool_create();
void test_spring_pool_deactivate_and_compact();
void test_contact_pool_allocate();
void test_contact_pool_add_and_clear();
void test_composite_pool_allocate();
void test_gravity_field_allocate();
void test_gravity_field_clear();

// ===== PARTICLE POOL TESTS =====

void test_particle_pool_allocate() {
    ParticlePool pool;
    pool.allocate(1000);
    
    ASSERT_EQ(pool.capacity, 1000u);
    ASSERT_EQ(pool.count, 0u);
    ASSERT_NE(pool.pos_x, nullptr);
    ASSERT_NE(pool.pos_y, nullptr);
    ASSERT_NE(pool.vel_x, nullptr);
    ASSERT_NE(pool.vel_y, nullptr);
    ASSERT_NE(pool.force_x, nullptr);
    ASSERT_NE(pool.force_y, nullptr);
    ASSERT_NE(pool.mass, nullptr);
    ASSERT_NE(pool.radius, nullptr);
}

void test_particle_pool_create() {
    ParticlePool pool;
    pool.allocate(10);
    
    uint32_t id1 = pool.create(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
    ASSERT_NE(id1, ParticlePool::INVALID_ID);
    ASSERT_EQ(pool.count, 1u);
    
    // Check data was set correctly
    ASSERT_NEAR(pool.pos_x[0], 1.0f, 1e-6f);
    ASSERT_NEAR(pool.pos_y[0], 2.0f, 1e-6f);
    ASSERT_NEAR(pool.vel_x[0], 3.0f, 1e-6f);
    ASSERT_NEAR(pool.vel_y[0], 4.0f, 1e-6f);
    ASSERT_NEAR(pool.mass[0], 5.0f, 1e-6f);
    ASSERT_NEAR(pool.radius[0], 6.0f, 1e-6f);
    
    // Check ID mapping
    ASSERT_EQ(pool.get_index(id1), 0u);
    ASSERT_EQ(pool.get_id(0), id1);
}

void test_particle_pool_destroy_single() {
    ParticlePool pool;
    pool.allocate(10);
    
    uint32_t id = pool.create(1.0f, 2.0f, 0, 0, 1.0f, 1.0f);
    ASSERT_EQ(pool.count, 1u);
    ASSERT(pool.exists(id));
    
    pool.destroy(id);
    ASSERT_EQ(pool.count, 0u);
    ASSERT(!pool.exists(id));
}

void test_particle_pool_destroy_swap_and_pop() {
    ParticlePool pool;
    pool.allocate(10);
    
    // Create 3 particles
    uint32_t id1 = pool.create(1.0f, 1.0f, 0, 0, 1.0f, 1.0f);
    uint32_t id2 = pool.create(2.0f, 2.0f, 0, 0, 2.0f, 2.0f);
    uint32_t id3 = pool.create(3.0f, 3.0f, 0, 0, 3.0f, 3.0f);
    
    ASSERT_EQ(pool.count, 3u);
    
    // Remove middle particle - should swap with last
    pool.destroy(id2);
    
    ASSERT_EQ(pool.count, 2u);
    ASSERT(pool.exists(id1));
    ASSERT(!pool.exists(id2));
    ASSERT(pool.exists(id3));
    
    // Check that particle 3 was moved to index 1
    uint32_t idx3 = pool.get_index(id3);
    ASSERT_EQ(idx3, 1u);
    ASSERT_NEAR(pool.pos_x[idx3], 3.0f, 1e-6f);
    ASSERT_NEAR(pool.mass[idx3], 3.0f, 1e-6f);
    
    // Check that particle 1 is still at index 0
    uint32_t idx1 = pool.get_index(id1);
    ASSERT_EQ(idx1, 0u);
    ASSERT_NEAR(pool.pos_x[idx1], 1.0f, 1e-6f);
}

void test_particle_pool_clear_forces() {
    ParticlePool pool;
    pool.allocate(100);
    
    // Create some particles
    for (int i = 0; i < 10; i++) {
        pool.create(i, i, 0, 0, 1.0f, 1.0f);
        pool.force_x[i] = i * 10.0f;
        pool.force_y[i] = i * 20.0f;
    }
    
    pool.clear_forces();
    
    // Only active particles should be cleared
    for (size_t i = 0; i < pool.count; i++) {
        ASSERT_EQ(pool.force_x[i], 0.0f);
        ASSERT_EQ(pool.force_y[i], 0.0f);
    }
}

void test_particle_pool_apply_boundaries() {
    ParticlePool pool;
    pool.allocate(10);
    
    // Create particles outside boundaries
    pool.create(-5.0f, -5.0f, 0, 0, 1.0f, 1.0f);
    pool.create(105.0f, 105.0f, 0, 0, 1.0f, 1.0f);
    pool.create(50.0f, -10.0f, 0, 0, 1.0f, 1.0f);
    
    float world_size = 100.0f;
    pool.apply_boundaries(world_size);
    
    // Check all particles are within bounds
    for (size_t i = 0; i < pool.count; i++) {
        ASSERT_GT(pool.pos_x[i], -0.001f);
        ASSERT_LT(pool.pos_x[i], world_size + 0.001f);
        ASSERT_GT(pool.pos_y[i], -0.001f);
        ASSERT_LT(pool.pos_y[i], world_size + 0.001f);
    }
    
    // Check specific wrapping
    ASSERT_NEAR(pool.pos_x[0], 95.0f, 1e-6f);  // -5 -> 95
    ASSERT_NEAR(pool.pos_y[0], 95.0f, 1e-6f);  // -5 -> 95
    ASSERT_NEAR(pool.pos_x[1], 5.0f, 1e-6f);   // 105 -> 5
    ASSERT_NEAR(pool.pos_y[1], 5.0f, 1e-6f);   // 105 -> 5
    ASSERT_NEAR(pool.pos_x[2], 50.0f, 1e-6f);  // 50 -> 50
    ASSERT_NEAR(pool.pos_y[2], 90.0f, 1e-6f);  // -10 -> 90
}

void test_particle_pool_stress_test() {
    ParticlePool pool;
    const size_t N = 10000;
    pool.allocate(N);
    
    std::vector<uint32_t> ids;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0, 100);
    
    // Create N particles
    for (size_t i = 0; i < N; i++) {
        uint32_t id = pool.create(dist(rng), dist(rng), dist(rng), dist(rng), 1.0f, 1.0f);
        ASSERT_NE(id, ParticlePool::INVALID_ID);
        ids.push_back(id);
    }
    
    ASSERT_EQ(pool.count, N);
    
    // Randomly remove half
    std::shuffle(ids.begin(), ids.end(), rng);
    for (size_t i = 0; i < N/2; i++) {
        pool.destroy(ids[i]);
    }
    
    ASSERT_EQ(pool.count, N - N/2);
    
    // Verify remaining particles
    for (size_t i = N/2; i < N; i++) {
        ASSERT(pool.exists(ids[i]));
    }
    
    // Verify removed particles
    for (size_t i = 0; i < N/2; i++) {
        ASSERT(!pool.exists(ids[i]));
    }
    
    // Verify all active particles are contiguous
    for (size_t i = 0; i < pool.count; i++) {
        uint32_t id = pool.get_id(i);
        ASSERT_NE(id, ParticlePool::INVALID_ID);
        ASSERT(pool.exists(id));
    }
}

void test_particle_pool_particle_ref() {
    ParticlePool pool;
    pool.allocate(10);
    
    uint32_t id = pool.create(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
    
    auto ref = pool.get(id);
    ASSERT_NEAR(ref.x(), 1.0f, 1e-6f);
    ASSERT_NEAR(ref.y(), 2.0f, 1e-6f);
    ASSERT_NEAR(ref.vx(), 3.0f, 1e-6f);
    ASSERT_NEAR(ref.vy(), 4.0f, 1e-6f);
    ASSERT_NEAR(ref.mass(), 5.0f, 1e-6f);
    ASSERT_NEAR(ref.radius(), 6.0f, 1e-6f);
    
    // Modify through reference
    ref.x() = 10.0f;
    ref.vy() = 20.0f;
    
    // Check modification worked
    ASSERT_NEAR(pool.pos_x[0], 10.0f, 1e-6f);
    ASSERT_NEAR(pool.vel_y[0], 20.0f, 1e-6f);
}

// ===== SPRING POOL TESTS =====

void test_spring_pool_allocate() {
    SpringPool pool;
    pool.allocate(100);
    
    ASSERT_EQ(pool.capacity, 100u);
    ASSERT_EQ(pool.count, 0u);
    ASSERT_NE(pool.particle1_id, nullptr);
    ASSERT_NE(pool.particle2_id, nullptr);
    ASSERT_NE(pool.rest_length, nullptr);
    ASSERT_NE(pool.stiffness, nullptr);
    ASSERT_NE(pool.damping, nullptr);
}

void test_spring_pool_create() {
    SpringPool pool;
    pool.allocate(10);
    
    uint32_t idx = pool.create(1, 2, 10.0f, 100.0f, 1.0f);
    
    ASSERT_NE(idx, ParticlePool::INVALID_ID);
    ASSERT_EQ(pool.count, 1u);
    ASSERT_EQ(pool.particle1_id[0], 1u);
    ASSERT_EQ(pool.particle2_id[0], 2u);
    ASSERT_NEAR(pool.rest_length[0], 10.0f, 1e-6f);
    ASSERT_NEAR(pool.stiffness[0], 100.0f, 1e-6f);
    ASSERT_NEAR(pool.damping[0], 1.0f, 1e-6f);
    ASSERT(pool.active[0]);
}

void test_spring_pool_deactivate_and_compact() {
    SpringPool pool;
    pool.allocate(10);
    
    // Create 5 springs
    for (int i = 0; i < 5; i++) {
        pool.create(i, i+1, 10.0f, 100.0f, 1.0f);
    }
    
    ASSERT_EQ(pool.count, 5u);
    
    // Deactivate springs 1 and 3
    pool.deactivate(1);
    pool.deactivate(3);
    
    // Still same count before compaction
    ASSERT_EQ(pool.count, 5u);
    
    // Compact
    pool.compact();
    
    // Now only 3 active springs
    ASSERT_EQ(pool.count, 3u);
    
    // Check that remaining springs are correct (0, 2, 4 from original)
    ASSERT_EQ(pool.particle1_id[0], 0u);
    ASSERT_EQ(pool.particle1_id[1], 2u);
    ASSERT_EQ(pool.particle1_id[2], 4u);
}

// ===== CONTACT POOL TESTS =====

void test_contact_pool_allocate() {
    ContactPool pool;
    pool.allocate(100);
    
    ASSERT_EQ(pool.capacity, 100u);
    ASSERT_EQ(pool.count, 0u);
    ASSERT_NE(pool.particle1, nullptr);
    ASSERT_NE(pool.particle2, nullptr);
    ASSERT_NE(pool.overlap, nullptr);
    ASSERT_NE(pool.normal_x, nullptr);
    ASSERT_NE(pool.normal_y, nullptr);
}

void test_contact_pool_add_and_clear() {
    ContactPool pool;
    pool.allocate(10);
    
    // Add some contacts
    ASSERT(pool.add(0, 1, 0.1f, 1.0f, 0.0f));
    ASSERT(pool.add(1, 2, 0.2f, 0.0f, 1.0f));
    ASSERT(pool.add(2, 3, 0.3f, 0.7071f, 0.7071f));
    
    ASSERT_EQ(pool.count, 3u);
    
    // Check data
    ASSERT_EQ(pool.particle1[0], 0u);
    ASSERT_EQ(pool.particle2[0], 1u);
    ASSERT_NEAR(pool.overlap[0], 0.1f, 1e-6f);
    ASSERT_NEAR(pool.normal_x[0], 1.0f, 1e-6f);
    
    // Clear
    pool.clear();
    ASSERT_EQ(pool.count, 0u);
}

// ===== COMPOSITE POOL TESTS =====

void test_composite_pool_allocate() {
    CompositePool pool;
    pool.allocate(10, 100);
    
    ASSERT_EQ(pool.capacity, 10u);
    ASSERT_EQ(pool.particle_capacity, 100u);
    ASSERT_EQ(pool.count, 0u);
    ASSERT_EQ(pool.total_particles, 0u);
    ASSERT_NE(pool.first_particle, nullptr);
    ASSERT_NE(pool.particle_count, nullptr);
    ASSERT_NE(pool.particle_ids, nullptr);
}

// ===== FIELD TESTS =====

void test_gravity_field_allocate() {
    GravityField field;
    field.allocate(64);
    
    ASSERT_EQ(field.grid_size, 64u);
    ASSERT_NE(field.density, nullptr);
    ASSERT_NE(field.potential, nullptr);
    ASSERT_NE(field.force_x, nullptr);
    ASSERT_NE(field.force_y, nullptr);
    
    // Should be zero-initialized
    for (size_t i = 0; i < 64*64; i++) {
        ASSERT_EQ(field.density[i], 0.0f);
        ASSERT_EQ(field.potential[i], 0.0f);
    }
}

void test_gravity_field_clear() {
    GravityField field;
    field.allocate(32);
    
    // Set some values
    field.density[0] = 1.0f;
    field.potential[10] = 2.0f;
    field.force_x[100] = 3.0f;
    field.force_y[200] = 4.0f;
    
    // Clear
    field.clear();
    
    // Check all cleared
    for (size_t i = 0; i < 32*32; i++) {
        ASSERT_EQ(field.density[i], 0.0f);
        ASSERT_EQ(field.potential[i], 0.0f);
        ASSERT_EQ(field.force_x[i], 0.0f);
        ASSERT_EQ(field.force_y[i], 0.0f);
    }
}

// ===== MAIN TEST RUNNER =====

int main() {
    std::cout << "Running Pool Unit Tests\n";
    std::cout << "========================\n\n";
    
    // Initialize test list
    std::vector<TestCase> tests = {
        {"particle_pool_allocate", test_particle_pool_allocate},
        {"particle_pool_create", test_particle_pool_create},
        {"particle_pool_destroy_single", test_particle_pool_destroy_single},
        {"particle_pool_destroy_swap_and_pop", test_particle_pool_destroy_swap_and_pop},
        {"particle_pool_clear_forces", test_particle_pool_clear_forces},
        {"particle_pool_apply_boundaries", test_particle_pool_apply_boundaries},
        {"particle_pool_stress_test", test_particle_pool_stress_test},
        {"particle_pool_particle_ref", test_particle_pool_particle_ref},
        {"spring_pool_allocate", test_spring_pool_allocate},
        {"spring_pool_create", test_spring_pool_create},
        {"spring_pool_deactivate_and_compact", test_spring_pool_deactivate_and_compact},
        {"contact_pool_allocate", test_contact_pool_allocate},
        {"contact_pool_add_and_clear", test_contact_pool_add_and_clear},
        {"composite_pool_allocate", test_composite_pool_allocate},
        {"gravity_field_allocate", test_gravity_field_allocate},
        {"gravity_field_clear", test_gravity_field_clear}
    };
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& test : tests) {
        std::cout << "Running: " << test.name << " ... ";
        std::cout.flush();
        
        try {
            test.func();
            std::cout << "PASSED\n";
            passed++;
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << "\n";
            failed++;
        } catch (...) {
            std::cout << "FAILED: Unknown exception\n";
            failed++;
        }
    }
    
    std::cout << "\n========================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    
    return failed > 0 ? 1 : 0;
}