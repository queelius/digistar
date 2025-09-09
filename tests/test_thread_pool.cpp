#include <gtest/gtest.h>
#include "../src/dsl/thread_pool.h"
#include "../src/dsl/script_manager.h"
#include "../src/core/simulation.h"
#include <chrono>
#include <atomic>

using namespace digistar;
using namespace digistar::dsl;

class ThreadPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure fresh thread pool for each test
        ScriptExecutor::shutdown();
    }
    
    void TearDown() override {
        ScriptExecutor::shutdown();
    }
};

TEST_F(ThreadPoolTest, BasicExecution) {
    ThreadPool pool(2);
    
    std::atomic<int> counter{0};
    
    auto future1 = pool.submit([&counter]() {
        counter++;
        return 42;
    });
    
    auto future2 = pool.submit([&counter]() {
        counter++;
        return 84;
    });
    
    EXPECT_EQ(future1.get(), 42);
    EXPECT_EQ(future2.get(), 84);
    EXPECT_EQ(counter.load(), 2);
}

TEST_F(ThreadPoolTest, ManyTasks) {
    ThreadPool pool(4);
    
    const int num_tasks = 100;
    std::atomic<int> counter{0};
    std::vector<std::future<int>> futures;
    
    for (int i = 0; i < num_tasks; i++) {
        futures.push_back(pool.submit([&counter, i]() {
            counter++;
            return i * 2;
        }));
    }
    
    for (int i = 0; i < num_tasks; i++) {
        EXPECT_EQ(futures[i].get(), i * 2);
    }
    
    EXPECT_EQ(counter.load(), num_tasks);
}

TEST_F(ThreadPoolTest, SingleThread) {
    ThreadPool pool(1);  // Single thread
    
    std::atomic<int> max_concurrent{0};
    std::atomic<int> current{0};
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < 10; i++) {
        futures.push_back(pool.submit([&max_concurrent, &current]() {
            current++;
            int mc = max_concurrent.load();
            while (!max_concurrent.compare_exchange_weak(mc, std::max(mc, current.load())));
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            current--;
        }));
    }
    
    for (auto& f : futures) {
        f.get();
    }
    
    // With single thread, max concurrent should be 1
    EXPECT_EQ(max_concurrent.load(), 1);
}

TEST_F(ThreadPoolTest, Resize) {
    ThreadPool pool(2);
    EXPECT_EQ(pool.size(), 2);
    
    pool.resize(4);
    EXPECT_EQ(pool.size(), 4);
    
    pool.resize(1);
    EXPECT_EQ(pool.size(), 1);
    
    // Test that it still works after resize
    auto future = pool.submit([]() { return 123; });
    EXPECT_EQ(future.get(), 123);
}

TEST_F(ThreadPoolTest, PauseResume) {
    ThreadPool pool(2);
    std::atomic<int> counter{0};
    
    pool.pause();
    
    // Submit tasks while paused
    auto future1 = pool.submit([&counter]() {
        counter++;
        return 1;
    });
    
    auto future2 = pool.submit([&counter]() {
        counter++;
        return 2;
    });
    
    // Give time to (not) execute
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(counter.load(), 0);  // Should not execute while paused
    
    pool.resume();
    
    EXPECT_EQ(future1.get(), 1);
    EXPECT_EQ(future2.get(), 2);
    EXPECT_EQ(counter.load(), 2);
}

TEST_F(ThreadPoolTest, WaitAll) {
    ThreadPool pool(4);
    std::atomic<int> completed{0};
    
    for (int i = 0; i < 10; i++) {
        pool.submit_detached([&completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            completed++;
        });
    }
    
    pool.wait_all();
    EXPECT_EQ(completed.load(), 10);
}

TEST_F(ThreadPoolTest, Statistics) {
    ThreadPool pool(2);
    
    EXPECT_EQ(pool.completed_tasks(), 0);
    EXPECT_EQ(pool.pending_tasks(), 0);
    
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 5; i++) {
        futures.push_back(pool.submit([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }));
    }
    
    // Some tasks should be pending
    EXPECT_GT(pool.pending_tasks() + pool.active_threads(), 0);
    
    for (auto& f : futures) {
        f.get();
    }
    
    EXPECT_EQ(pool.completed_tasks(), 5);
    EXPECT_EQ(pool.pending_tasks(), 0);
}

// ============ ScriptManager Tests ============

class ScriptManagerTest : public ::testing::Test {
protected:
    Simulation sim;
    ScriptManager manager;
    
    void SetUp() override {
        manager.setSimulation(&sim);
    }
};

TEST_F(ScriptManagerTest, LoadAndExecuteScript) {
    std::string code = "(particle :mass 1.0 :pos [10 20] :vel [1 0])";
    
    auto script = manager.loadScript("test", code, false);
    ASSERT_NE(script, nullptr);
    EXPECT_EQ(script->getName(), "test");
    EXPECT_FALSE(script->isPersistent());
    
    auto future = manager.executeScript("test");
    future.wait();
    
    EXPECT_EQ(manager.getPendingCommands(), 1);
    
    manager.applyCommands(sim);
    
    EXPECT_EQ(sim.getParticleCount(), 1);
    EXPECT_EQ(manager.getPendingCommands(), 0);
}

TEST_F(ScriptManagerTest, ExecuteMultipleScripts) {
    manager.setThreadCount(4);
    
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < 10; i++) {
        std::string code = "(particle :mass " + std::to_string(i) + 
                          " :pos [" + std::to_string(i*10) + " 0])";
        futures.push_back(manager.executeCode(code));
    }
    
    for (auto& f : futures) {
        f.wait();
    }
    
    EXPECT_EQ(manager.getPendingCommands(), 10);
    
    manager.applyCommands(sim);
    
    EXPECT_EQ(sim.getParticleCount(), 10);
    EXPECT_EQ(manager.getTotalScriptsRun(), 10);
}

TEST_F(ScriptManagerTest, PersistentScript) {
    std::string code = "(particle :mass 1.0 :pos [0 0])";
    
    auto script = manager.loadScript("persistent", code, true);
    EXPECT_TRUE(script->isPersistent());
    
    // Update should execute persistent scripts
    manager.update(0.016f);  // 16ms frame
    
    // Give thread pool time to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    EXPECT_GT(manager.getPendingCommands(), 0);
    
    manager.applyCommands(sim);
    EXPECT_GT(sim.getParticleCount(), 0);
}

TEST_F(ScriptManagerTest, CloudCreation) {
    std::string code = "(cloud :center [0 0] :radius 100 :n 50)";
    
    auto future = manager.executeCode(code);
    future.wait();
    
    EXPECT_EQ(manager.getPendingCommands(), 1);
    
    manager.applyCommands(sim);
    
    EXPECT_EQ(sim.getParticleCount(), 50);
}

TEST_F(ScriptManagerTest, ScriptControl) {
    auto script1 = manager.loadScript("s1", "(particle :mass 1)", false);
    auto script2 = manager.loadScript("s2", "(particle :mass 2)", true);
    
    EXPECT_EQ(manager.getScriptNames().size(), 2);
    
    script1->pause();
    EXPECT_EQ(script1->getState(), Script::PAUSED);
    
    script1->resume();
    EXPECT_EQ(script1->getState(), Script::IDLE);
    
    manager.stopAll();
    EXPECT_EQ(script1->getState(), Script::COMPLETED);
    EXPECT_EQ(script2->getState(), Script::COMPLETED);
    
    manager.clearAll();
    EXPECT_EQ(manager.getScriptNames().size(), 0);
}

TEST_F(ScriptManagerTest, ThreadConfiguration) {
    EXPECT_GT(manager.getThreadCount(), 0);  // Auto-detected
    
    manager.setThreadCount(1);
    EXPECT_EQ(manager.getThreadCount(), 1);
    
    manager.setThreadCount(4);
    EXPECT_EQ(manager.getThreadCount(), 4);
    
    // Should still work with different thread counts
    auto future = manager.executeCode("(particle :mass 1)");
    future.wait();
    
    manager.applyCommands(sim);
    EXPECT_EQ(sim.getParticleCount(), 1);
}

TEST_F(ScriptManagerTest, SetVelocityCommand) {
    // Create a particle first
    manager.executeCode("(particle :mass 1 :pos [0 0])").wait();
    manager.applyCommands(sim);
    
    auto particle_id = 1;  // First particle has ID 1
    
    // Set its velocity
    std::string code = "(set-velocity " + std::to_string(particle_id) + " [5 10])";
    manager.executeCode(code).wait();
    manager.applyCommands(sim);
    
    auto* particle = sim.getParticle(particle_id);
    ASSERT_NE(particle, nullptr);
    EXPECT_EQ(particle->velocity, Vec2(5, 10));
}

TEST_F(ScriptManagerTest, ApplyForceCommand) {
    // Create a particle
    manager.executeCode("(particle :mass 2 :pos [0 0])").wait();
    manager.applyCommands(sim);
    
    auto particle_id = 1;
    
    // Apply force
    std::string code = "(apply-force " + std::to_string(particle_id) + " [10 0])";
    manager.executeCode(code).wait();
    manager.applyCommands(sim);
    
    // Force should change velocity: dv = F * dt / m = 10 * 0.01 / 2 = 0.05
    auto* particle = sim.getParticle(particle_id);
    ASSERT_NE(particle, nullptr);
    EXPECT_NEAR(particle->velocity.x, 0.05, 1e-6);
}