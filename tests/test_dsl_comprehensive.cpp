/**
 * Comprehensive DSL Testing Suite
 * 
 * This file contains extensive unit tests for the DigiStar DSL system,
 * focusing on achieving 90%+ code coverage through systematic testing of:
 * - Pattern matching edge cases
 * - Procedural generation scenarios
 * - Bytecode compilation and execution
 * - Event system integration
 * - Concurrent execution
 * - Error handling
 * - Performance benchmarks
 */

#include <gtest/gtest.h>
#include "../src/dsl/pattern_matcher.h"
#include "../src/dsl/procedural_generator.h"
#include "../src/dsl/bytecode_compiler.h"
#include "../src/dsl/bytecode_vm.h"
#include "../src/dsl/event_bridge.h"
#include "../src/dsl/dsl_runtime.h"
#include "../src/physics/pools.h"
#include <chrono>
#include <thread>
#include <random>
#include <future>
#include <valgrind/memcheck.h>  // For memory leak detection

using namespace digistar;
using namespace digistar::dsl;

//=============================================================================
// Extended Pattern Matching Tests
//=============================================================================

class ExtendedPatternMatcherTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Test nested pattern matching with multiple levels
TEST_F(ExtendedPatternMatcherTest, DeeplyNestedPatterns) {
    auto pattern = PatternBuilder::list({
        PatternBuilder::literal("outer"),
        PatternBuilder::list({
            PatternBuilder::literal("middle"),
            PatternBuilder::list({
                PatternBuilder::literal("inner"),
                PatternBuilder::var("value")
            })
        })
    });
    
    auto value = SExpr::makeList({
        SExpr::makeSymbol("outer"),
        SExpr::makeList({
            SExpr::makeSymbol("middle"),
            SExpr::makeList({
                SExpr::makeSymbol("inner"),
                SExpr::makeNumber(42)
            })
        })
    });
    
    auto result = PatternMatcher::matchPattern(pattern, value);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.get("value")->asNumber(), 42);
}

// Test pattern matching with wildcards
TEST_F(ExtendedPatternMatcherTest, WildcardPatterns) {
    auto pattern = PatternBuilder::list({
        PatternBuilder::wildcard(),
        PatternBuilder::var("second"),
        PatternBuilder::wildcard()
    });
    
    auto value = SExpr::makeList({
        SExpr::makeNumber(1),
        SExpr::makeSymbol("important"),
        SExpr::makeNumber(3)
    });
    
    auto result = PatternMatcher::matchPattern(pattern, value);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.get("second")->asSymbol(), "important");
}

// Test pattern matching with rest patterns
TEST_F(ExtendedPatternMatcherTest, RestPatterns) {
    auto pattern = PatternBuilder::list({
        PatternBuilder::var("first"),
        PatternBuilder::rest("remainder")
    });
    
    auto value = SExpr::makeList({
        SExpr::makeNumber(1),
        SExpr::makeNumber(2),
        SExpr::makeNumber(3),
        SExpr::makeNumber(4)
    });
    
    auto result = PatternMatcher::matchPattern(pattern, value);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.get("first")->asNumber(), 1);
    auto remainder = result.get("remainder");
    EXPECT_TRUE(remainder->isList());
    EXPECT_EQ(remainder->length(), 3);
}

// Test pattern matching with type predicates
TEST_F(ExtendedPatternMatcherTest, TypePredicatePatterns) {
    auto pattern = PatternBuilder::typed("value", 
        [](SExprPtr val) { return val->isNumber() && val->asNumber() > 0; },
        "positive-number"
    );
    
    auto positive = SExpr::makeNumber(42);
    auto negative = SExpr::makeNumber(-42);
    auto non_number = SExpr::makeSymbol("foo");
    
    EXPECT_TRUE(PatternMatcher::matchPattern(pattern, positive).success);
    EXPECT_FALSE(PatternMatcher::matchPattern(pattern, negative).success);
    EXPECT_FALSE(PatternMatcher::matchPattern(pattern, non_number).success);
}

// Test pattern compilation and caching
TEST_F(ExtendedPatternMatcherTest, PatternCompilationCaching) {
    PatternCompiler compiler;
    
    auto pattern_str = "(collision ?p1 ?p2 (> ?energy 1000))";
    auto compiled1 = compiler.compile(pattern_str);
    auto compiled2 = compiler.compile(pattern_str);
    
    // Should return cached version
    EXPECT_EQ(compiled1, compiled2);
    
    // Test cache stats
    auto stats = compiler.getStats();
    EXPECT_EQ(stats.cache_hits, 1);
    EXPECT_EQ(stats.compilations, 1);
}

//=============================================================================
// Extended Procedural Generation Tests
//=============================================================================

class ExtendedProceduralGeneratorTest : public ::testing::Test {
protected:
    std::unique_ptr<SimulationState> sim_state;
    std::unique_ptr<ProceduralGenerator> generator;
    
    void SetUp() override {
        sim_state = std::make_unique<SimulationState>();
        sim_state->particles.allocate(1000000);
        generator = std::make_unique<ProceduralGenerator>(sim_state.get());
    }
};

// Test distribution statistics
TEST_F(ExtendedProceduralGeneratorTest, DistributionStatistics) {
    auto gaussian = generator->createDistribution("gaussian", {0.0f, 1.0f});
    
    const int samples = 100000;
    std::vector<float> values;
    values.reserve(samples);
    
    for (int i = 0; i < samples; ++i) {
        values.push_back(gaussian->sample());
    }
    
    // Calculate statistics
    float mean = std::accumulate(values.begin(), values.end(), 0.0f) / samples;
    float variance = 0;
    for (float v : values) {
        variance += (v - mean) * (v - mean);
    }
    variance /= samples;
    float stddev = std::sqrt(variance);
    
    // Check within 3-sigma confidence interval
    EXPECT_NEAR(mean, 0.0f, 0.01f);
    EXPECT_NEAR(stddev, 1.0f, 0.01f);
    
    // Check distribution shape (68-95-99.7 rule)
    int within_1_sigma = 0, within_2_sigma = 0, within_3_sigma = 0;
    for (float v : values) {
        float z = std::abs(v - mean) / stddev;
        if (z <= 1.0f) within_1_sigma++;
        if (z <= 2.0f) within_2_sigma++;
        if (z <= 3.0f) within_3_sigma++;
    }
    
    EXPECT_NEAR(within_1_sigma / float(samples), 0.68f, 0.01f);
    EXPECT_NEAR(within_2_sigma / float(samples), 0.95f, 0.01f);
    EXPECT_NEAR(within_3_sigma / float(samples), 0.997f, 0.003f);
}

// Test procedural generation with constraints
TEST_F(ExtendedProceduralGeneratorTest, ConstrainedGeneration) {
    ParticleTemplate tmpl;
    tmpl.withMass(std::make_shared<UniformDistribution>(generator->getRNG(), 1.0f, 10.0f))
        .withConstraint([](const ParticleData& p) {
            // Only accept particles with mass * radius < 5
            return p.mass * p.radius < 5.0f;
        });
    
    auto indices = generator->generateCloud(100, 0, 0, 50, tmpl);
    
    // Verify all particles meet constraint
    for (size_t idx : indices) {
        float mass = sim_state->particles.mass[idx];
        float radius = sim_state->particles.radius[idx];
        EXPECT_LT(mass * radius, 5.0f);
    }
}

// Test large-scale generation performance
TEST_F(ExtendedProceduralGeneratorTest, LargeScaleGenerationPerformance) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Generate 1 million particles
    generator->beginBatch();
    for (int i = 0; i < 1000; ++i) {
        for (int j = 0; j < 1000; ++j) {
            generator->addToBatch(i * 10, j * 10, 0, 0, 1.0f, 0.5f, 300.0f);
        }
    }
    auto indices = generator->commitBatch();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_EQ(indices.size(), 1000000);
    EXPECT_LT(duration.count(), 1000);  // Should complete within 1 second
    
    // Calculate generation rate
    double particles_per_second = 1000000.0 / (duration.count() / 1000.0);
    std::cout << "Generation rate: " << particles_per_second << " particles/second" << std::endl;
    EXPECT_GT(particles_per_second, 1000000);  // Target: 1M particles/sec
}

//=============================================================================
// Extended Bytecode Compilation Tests
//=============================================================================

class ExtendedBytecodeCompilerTest : public ::testing::Test {
protected:
    std::unique_ptr<BytecodeCompiler> compiler;
    
    void SetUp() override {
        compiler = std::make_unique<BytecodeCompiler>();
    }
};

// Test recursive function compilation
TEST_F(ExtendedBytecodeCompilerTest, RecursiveFunctionCompilation) {
    std::string factorial = R"(
        (define (factorial n)
          (if (<= n 1)
              1
              (* n (factorial (- n 1)))))
    )";
    
    auto expr = SExprParser::parseString(factorial);
    auto chunk = compiler->compile(expr);
    
    ASSERT_NE(chunk, nullptr);
    
    // Check for tail call optimization
    bool has_tail_call = false;
    for (const auto& inst : chunk->instructions) {
        if (inst.opcode == OpCode::TAIL_CALL) {
            has_tail_call = true;
            break;
        }
    }
    // Recursive factorial should be optimized to tail call
    EXPECT_TRUE(has_tail_call);
}

// Test closure compilation
TEST_F(ExtendedBytecodeCompilerTest, ClosureCompilation) {
    std::string closure = R"(
        (let ((counter 0))
          (lambda ()
            (set! counter (+ counter 1))
            counter))
    )";
    
    auto expr = SExprParser::parseString(closure);
    auto chunk = compiler->compile(expr);
    
    ASSERT_NE(chunk, nullptr);
    
    // Check for closure creation
    bool has_closure = false;
    for (const auto& inst : chunk->instructions) {
        if (inst.opcode == OpCode::MAKE_CLOSURE) {
            has_closure = true;
            break;
        }
    }
    EXPECT_TRUE(has_closure);
}

// Test optimization passes
TEST_F(ExtendedBytecodeCompilerTest, OptimizationPasses) {
    std::string code = R"(
        (+ 1 2 3)  ; Constant folding opportunity
    )";
    
    auto expr = SExprParser::parseString(code);
    auto unoptimized = compiler->compile(expr);
    auto optimized = compiler->compileOptimized(expr);
    
    // Optimized version should have fewer instructions
    EXPECT_LT(optimized->instructions.size(), unoptimized->instructions.size());
    
    // Should fold to single constant
    EXPECT_EQ(optimized->instructions[0].opcode, OpCode::PUSH_CONST);
    EXPECT_EQ(optimized->constants[0]->asNumber(), 6.0);
}

//=============================================================================
// Extended Bytecode VM Tests
//=============================================================================

class ExtendedBytecodeVMTest : public ::testing::Test {
protected:
    std::unique_ptr<SimulationState> sim_state;
    std::unique_ptr<BytecodeCompiler> compiler;
    std::unique_ptr<BytecodeVM> vm;
    
    void SetUp() override {
        sim_state = std::make_unique<SimulationState>();
        compiler = std::make_unique<BytecodeCompiler>();
        vm = std::make_unique<BytecodeVM>(sim_state.get());
    }
};

// Test stack overflow protection
TEST_F(ExtendedBytecodeVMTest, StackOverflowProtection) {
    // Create deeply recursive code
    std::string code = "(define (recurse) (recurse))";
    auto expr = SExprParser::parseString(code);
    auto chunk = compiler->compile(expr);
    
    // Should not crash, but return error
    auto result = vm->executeWithTimeout(chunk, std::chrono::milliseconds(100));
    
    EXPECT_TRUE(vm->getState().has_error);
    EXPECT_NE(vm->getState().error_message.find("stack"), std::string::npos);
}

// Test concurrent VM execution
TEST_F(ExtendedBytecodeVMTest, ConcurrentExecution) {
    std::string code = "(+ 1 2 3)";
    auto expr = SExprParser::parseString(code);
    auto chunk = compiler->compile(expr);
    
    const int num_threads = 10;
    std::vector<std::future<SExprPtr>> futures;
    
    for (int i = 0; i < num_threads; ++i) {
        futures.push_back(std::async(std::launch::async, [this, chunk]() {
            BytecodeVM local_vm(sim_state.get());
            return local_vm.execute(chunk);
        }));
    }
    
    for (auto& future : futures) {
        auto result = future.get();
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->asNumber(), 6.0);
    }
}

// Test VM instruction tracing
TEST_F(ExtendedBytecodeVMTest, InstructionTracing) {
    vm->enableTracing(true);
    
    std::string code = "(+ 1 2)";
    auto expr = SExprParser::parseString(code);
    auto chunk = compiler->compile(expr);
    
    auto result = vm->execute(chunk);
    
    // Get trace log
    auto trace = vm->getTraceLog();
    EXPECT_FALSE(trace.empty());
    
    // Should see PUSH_CONST instructions
    EXPECT_NE(trace.find("PUSH_CONST"), std::string::npos);
    EXPECT_NE(trace.find("ADD"), std::string::npos);
}

//=============================================================================
// Memory Leak Detection Tests
//=============================================================================

class MemoryLeakTest : public ::testing::Test {
protected:
    void CheckForLeaks(std::function<void()> test_func) {
        // Mark memory checkpoint
        VALGRIND_DO_LEAK_CHECK;
        
        test_func();
        
        // Check for leaks
        VALGRIND_DO_LEAK_CHECK;
    }
};

TEST_F(MemoryLeakTest, NoLeaksInPatternMatching) {
    CheckForLeaks([]() {
        for (int i = 0; i < 10000; ++i) {
            auto pattern = PatternBuilder::var("x");
            auto value = SExpr::makeNumber(42);
            auto result = PatternMatcher::matchPattern(pattern, value);
        }
    });
}

TEST_F(MemoryLeakTest, NoLeaksInBytecodeCompilation) {
    CheckForLeaks([]() {
        BytecodeCompiler compiler;
        for (int i = 0; i < 1000; ++i) {
            auto expr = SExprParser::parseString("(+ 1 2 3)");
            auto chunk = compiler.compile(expr);
        }
    });
}

//=============================================================================
// Fuzz Testing for Bytecode Compiler
//=============================================================================

class FuzzTest : public ::testing::Test {
protected:
    std::unique_ptr<BytecodeCompiler> compiler;
    std::mt19937 rng{std::random_device{}()};
    
    void SetUp() override {
        compiler = std::make_unique<BytecodeCompiler>();
    }
    
    std::string generateRandomSExpr(int depth = 0) {
        if (depth > 5) {
            // Terminal case
            std::uniform_int_distribution<> dist(0, 2);
            switch (dist(rng)) {
                case 0: return std::to_string(rng() % 100);
                case 1: return "'symbol" + std::to_string(rng() % 10);
                default: return "#t";
            }
        }
        
        std::uniform_int_distribution<> dist(0, 4);
        switch (dist(rng)) {
            case 0: // Number
                return std::to_string(rng() % 100);
            case 1: // Symbol
                return "sym" + std::to_string(rng() % 10);
            case 2: // List
                {
                    std::string list = "(";
                    int elements = (rng() % 4) + 1;
                    for (int i = 0; i < elements; ++i) {
                        if (i > 0) list += " ";
                        list += generateRandomSExpr(depth + 1);
                    }
                    list += ")";
                    return list;
                }
            case 3: // Arithmetic
                return "(+ " + generateRandomSExpr(depth + 1) + " " + 
                       generateRandomSExpr(depth + 1) + ")";
            default: // Boolean
                return rng() % 2 ? "#t" : "#f";
        }
    }
};

TEST_F(FuzzTest, CompilerDoesNotCrash) {
    const int iterations = 1000;
    int successful_compilations = 0;
    
    for (int i = 0; i < iterations; ++i) {
        std::string random_code = generateRandomSExpr();
        
        try {
            auto expr = SExprParser::parseString(random_code);
            if (expr) {
                auto chunk = compiler->compile(expr);
                if (chunk) {
                    successful_compilations++;
                }
            }
        } catch (...) {
            // Should not throw exceptions
            FAIL() << "Compiler threw exception on: " << random_code;
        }
    }
    
    // At least some should compile successfully
    EXPECT_GT(successful_compilations, iterations / 10);
}

//=============================================================================
// Stress Tests for Concurrent Execution
//=============================================================================

class StressTest : public ::testing::Test {
protected:
    std::unique_ptr<SimulationState> sim_state;
    std::unique_ptr<DSLRuntime> runtime;
    
    void SetUp() override {
        sim_state = std::make_unique<SimulationState>();
        sim_state->particles.allocate(1000000);
        runtime = std::make_unique<DSLRuntime>(sim_state.get());
    }
};

TEST_F(StressTest, ConcurrentScriptExecution) {
    // Load multiple scripts
    for (int i = 0; i < 100; ++i) {
        std::string script = "(define var" + std::to_string(i) + " " + std::to_string(i) + ")";
        runtime->loadScript("script" + std::to_string(i), script);
    }
    
    // Execute scripts concurrently
    const int num_threads = 20;
    std::vector<std::thread> threads;
    std::atomic<int> total_executions{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, &total_executions]() {
            for (int i = 0; i < 100; ++i) {
                runtime->runScript("script" + std::to_string(i % 100));
                total_executions++;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(total_executions.load(), num_threads * 100);
    
    // Verify no corruption
    auto perf = runtime->getPerformance();
    EXPECT_GT(perf.total_executions, 0);
}

TEST_F(StressTest, HighFrequencyEventProcessing) {
    // Create event bridge
    auto buffer = std::make_unique<EventRingBuffer>();
    auto bridge = std::make_unique<EventBridge>(buffer.get(), sim_state.get());
    
    // Register handlers
    for (int i = 0; i < 10; ++i) {
        auto pattern = PatternBuilder::collision("p1", "p2", "energy");
        auto action = SExprParser::parseString("(print 'handled)");
        bridge->registerHandler(EventType::HARD_COLLISION, pattern, action, i);
    }
    
    // Generate high-frequency events
    const int num_events = 100000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_events; ++i) {
        Event event;
        event.type = EventType::HARD_COLLISION;
        event.primary_id = i % 100;
        event.secondary_id = (i + 1) % 100;
        event.magnitude = i * 10.0f;
        
        bridge->onEvent(event);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should handle at high rate
    double events_per_second = num_events / (duration.count() / 1000.0);
    std::cout << "Event processing rate: " << events_per_second << " events/second" << std::endl;
    EXPECT_GT(events_per_second, 10000);  // Target: 10K events/sec
}

//=============================================================================
// Integration Tests
//=============================================================================

class IntegrationTest : public ::testing::Test {
protected:
    std::unique_ptr<SimulationState> sim_state;
    std::unique_ptr<DSLRuntime> runtime;
    
    void SetUp() override {
        sim_state = std::make_unique<SimulationState>();
        sim_state->particles.allocate(100000);
        runtime = std::make_unique<DSLRuntime>(sim_state.get());
    }
};

TEST_F(IntegrationTest, CompleteWorkflow) {
    // 1. Generate particles
    auto indices = runtime->generateGalaxy(10000, 0, 0, 1000);
    EXPECT_GT(indices.size(), 10000);
    
    // 2. Define event handlers with pattern matching
    std::string handler_code = R"(
        (on-event 'collision
          (match event
            [(collision ?p1 ?p2 (> ?energy 1000))
             (emit-event 'high-energy :particles [p1 p2])]))
    )";
    runtime->execute(handler_code);
    
    // 3. Run simulation script
    std::string sim_code = R"(
        (define (simulate-step)
          (for-each particles
            (lambda (p)
              (update-position p)
              (check-collisions p))))
    )";
    runtime->loadScript("simulation", sim_code);
    
    // 4. Execute simulation steps
    for (int i = 0; i < 10; ++i) {
        runtime->runScript("simulation");
        runtime->update(0.016f);  // 60 FPS
    }
    
    // 5. Verify results
    auto perf = runtime->getPerformance();
    EXPECT_GT(perf.total_executions, 0);
}

//=============================================================================
// Main test runner with coverage
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Enable coverage collection if available
    #ifdef COVERAGE_ENABLED
    __gcov_flush();
    #endif
    
    int result = RUN_ALL_TESTS();
    
    // Generate coverage report
    #ifdef COVERAGE_ENABLED
    system("gcov -b *.cpp");
    system("lcov --capture --directory . --output-file coverage.info");
    system("genhtml coverage.info --output-directory coverage_report");
    std::cout << "Coverage report generated in coverage_report/index.html" << std::endl;
    #endif
    
    return result;
}