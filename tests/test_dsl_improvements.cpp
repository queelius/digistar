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

using namespace digistar;
using namespace digistar::dsl;

//=============================================================================
// Pattern Matching Tests
//=============================================================================

class PatternMatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(PatternMatcherTest, LiteralPatternMatching) {
    auto pattern = PatternBuilder::literal(42.0);
    auto value = SExpr::makeNumber(42.0);
    
    auto result = PatternMatcher::matchPattern(pattern, value);
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.bindings.empty());
    
    auto wrong_value = SExpr::makeNumber(43.0);
    result = PatternMatcher::matchPattern(pattern, wrong_value);
    EXPECT_FALSE(result.success);
}

TEST_F(PatternMatcherTest, VariablePatternCapture) {
    auto pattern = PatternBuilder::var("x");
    auto value = SExpr::makeNumber(42.0);
    
    auto result = PatternMatcher::matchPattern(pattern, value);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.bindings.size(), 1);
    EXPECT_TRUE(result.get("x")->equals(*value));
}

TEST_F(PatternMatcherTest, ListPatternMatching) {
    auto pattern = PatternBuilder::list({
        PatternBuilder::literal("collision"),
        PatternBuilder::var("p1"),
        PatternBuilder::var("p2"),
        PatternBuilder::number("energy")
    });
    
    auto value = SExpr::makeList({
        SExpr::makeSymbol("collision"),
        SExpr::makeNumber(1),
        SExpr::makeNumber(2),
        SExpr::makeNumber(1000)
    });
    
    auto result = PatternMatcher::matchPattern(pattern, value);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.get("p1")->asNumber(), 1);
    EXPECT_EQ(result.get("p2")->asNumber(), 2);
    EXPECT_EQ(result.get("energy")->asNumber(), 1000);
}

TEST_F(PatternMatcherTest, GuardPatternMatching) {
    auto pattern = PatternBuilder::withGuard(
        PatternBuilder::number("x"),
        [](const std::unordered_map<std::string, SExprPtr>& bindings) {
            auto it = bindings.find("x");
            return it != bindings.end() && it->second->asNumber() > 100;
        },
        "x > 100"
    );
    
    auto value1 = SExpr::makeNumber(150);
    auto result1 = PatternMatcher::matchPattern(pattern, value1);
    EXPECT_TRUE(result1.success);
    EXPECT_EQ(result1.get("x")->asNumber(), 150);
    
    auto value2 = SExpr::makeNumber(50);
    auto result2 = PatternMatcher::matchPattern(pattern, value2);
    EXPECT_FALSE(result2.success);
}

TEST_F(PatternMatcherTest, OrPatternMatching) {
    auto pattern = PatternBuilder::anyOf({
        PatternBuilder::literal("foo"),
        PatternBuilder::literal("bar"),
        PatternBuilder::number()
    });
    
    EXPECT_TRUE(PatternMatcher::matchPattern(pattern, SExpr::makeSymbol("foo")).success);
    EXPECT_TRUE(PatternMatcher::matchPattern(pattern, SExpr::makeSymbol("bar")).success);
    EXPECT_TRUE(PatternMatcher::matchPattern(pattern, SExpr::makeNumber(42)).success);
    EXPECT_FALSE(PatternMatcher::matchPattern(pattern, SExpr::makeSymbol("baz")).success);
}

TEST_F(PatternMatcherTest, ComplexNestedPattern) {
    // Pattern: (event (collision ?p1 ?p2 (> ?energy 1000)))
    auto pattern = PatternBuilder::list({
        PatternBuilder::literal("event"),
        PatternBuilder::list({
            PatternBuilder::literal("collision"),
            PatternBuilder::var("p1"),
            PatternBuilder::var("p2"),
            PatternBuilder::withGuard(
                PatternBuilder::number("energy"),
                [](const auto& bindings) {
                    auto it = bindings.find("energy");
                    return it != bindings.end() && it->second->asNumber() > 1000;
                },
                "energy > 1000"
            )
        })
    });
    
    auto value = SExpr::makeList({
        SExpr::makeSymbol("event"),
        SExpr::makeList({
            SExpr::makeSymbol("collision"),
            SExpr::makeNumber(10),
            SExpr::makeNumber(20),
            SExpr::makeNumber(2000)
        })
    });
    
    auto result = PatternMatcher::matchPattern(pattern, value);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.get("p1")->asNumber(), 10);
    EXPECT_EQ(result.get("p2")->asNumber(), 20);
    EXPECT_EQ(result.get("energy")->asNumber(), 2000);
}

//=============================================================================
// Procedural Generation Tests
//=============================================================================

class ProceduralGeneratorTest : public ::testing::Test {
protected:
    std::unique_ptr<SimulationState> sim_state;
    std::unique_ptr<ProceduralGenerator> generator;
    
    void SetUp() override {
        sim_state = std::make_unique<SimulationState>();
        sim_state->particles.allocate(100000);  // Allocate space
        generator = std::make_unique<ProceduralGenerator>(sim_state.get());
    }
};

TEST_F(ProceduralGeneratorTest, UniformDistribution) {
    auto dist = generator->createDistribution("uniform", {0.0f, 10.0f});
    
    float sum = 0;
    const int samples = 10000;
    for (int i = 0; i < samples; ++i) {
        float val = dist->sample();
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 10.0f);
        sum += val;
    }
    
    float mean = sum / samples;
    EXPECT_NEAR(mean, 5.0f, 0.2f);  // Should be close to midpoint
}

TEST_F(ProceduralGeneratorTest, GaussianDistribution) {
    auto dist = generator->createDistribution("gaussian", {100.0f, 10.0f});
    
    float sum = 0;
    float sum_sq = 0;
    const int samples = 10000;
    
    for (int i = 0; i < samples; ++i) {
        float val = dist->sample();
        sum += val;
        sum_sq += val * val;
    }
    
    float mean = sum / samples;
    float variance = (sum_sq / samples) - (mean * mean);
    float stddev = std::sqrt(variance);
    
    EXPECT_NEAR(mean, 100.0f, 1.0f);
    EXPECT_NEAR(stddev, 10.0f, 1.0f);
}

TEST_F(ProceduralGeneratorTest, CloudGeneration) {
    ParticleTemplate tmpl;
    tmpl.withMass(std::make_shared<UniformDistribution>(generator->getRNG(), 1.0f, 2.0f))
        .withRadius(std::make_shared<UniformDistribution>(generator->getRNG(), 0.5f, 1.0f))
        .withTemperature(std::make_shared<UniformDistribution>(generator->getRNG(), 200.0f, 300.0f));
    
    auto indices = generator->generateCloud(100, 0, 0, 50, tmpl);
    
    EXPECT_EQ(indices.size(), 100);
    
    // Check particles are within expected radius
    for (size_t idx : indices) {
        float x = sim_state->particles.pos_x[idx];
        float y = sim_state->particles.pos_y[idx];
        float dist = std::sqrt(x * x + y * y);
        EXPECT_LE(dist, 50.0f);
        
        // Check properties are in expected ranges
        EXPECT_GE(sim_state->particles.mass[idx], 1.0f);
        EXPECT_LE(sim_state->particles.mass[idx], 2.0f);
        EXPECT_GE(sim_state->particles.radius[idx], 0.5f);
        EXPECT_LE(sim_state->particles.radius[idx], 1.0f);
    }
}

TEST_F(ProceduralGeneratorTest, GalaxyGeneration) {
    auto indices = generator->generateGalaxy(1000, 0, 0, 100, 2);
    
    EXPECT_GT(indices.size(), 1000);  // Includes central black hole
    
    // First particle should be the black hole
    EXPECT_GT(sim_state->particles.mass[indices[0]], 100000.0f);
    
    // Check spiral pattern (simplified check)
    size_t in_arms = 0;
    for (size_t i = 1; i < indices.size(); ++i) {
        float x = sim_state->particles.pos_x[indices[i]];
        float y = sim_state->particles.pos_y[indices[i]];
        float angle = std::atan2(y, x);
        float r = std::sqrt(x * x + y * y);
        
        // Very rough check for spiral structure
        float expected_angle = std::log(r / 10.0f) * 0.3f;
        float angle_diff = std::abs(angle - expected_angle);
        if (angle_diff < 0.5f) {
            in_arms++;
        }
    }
    
    // At least some particles should follow spiral pattern
    EXPECT_GT(in_arms, indices.size() / 10);
}

TEST_F(ProceduralGeneratorTest, BatchOperations) {
    generator->beginBatch();
    
    for (int i = 0; i < 100; ++i) {
        generator->addToBatch(i, i, 0, 0, 1.0f, 0.5f, 300.0f);
    }
    
    auto indices = generator->commitBatch();
    
    EXPECT_EQ(indices.size(), 100);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        EXPECT_EQ(sim_state->particles.pos_x[indices[i]], i);
        EXPECT_EQ(sim_state->particles.pos_y[indices[i]], i);
        EXPECT_EQ(sim_state->particles.mass[indices[i]], 1.0f);
    }
}

//=============================================================================
// Bytecode Compiler Tests
//=============================================================================

class BytecodeCompilerTest : public ::testing::Test {
protected:
    std::unique_ptr<BytecodeCompiler> compiler;
    
    void SetUp() override {
        compiler = std::make_unique<BytecodeCompiler>();
    }
};

TEST_F(BytecodeCompilerTest, CompileConstant) {
    auto expr = SExpr::makeNumber(42.0);
    auto chunk = compiler->compile(expr);
    
    ASSERT_NE(chunk, nullptr);
    EXPECT_EQ(chunk->instructions.size(), 2);  // PUSH_CONST, HALT
    EXPECT_EQ(chunk->instructions[0].opcode, OpCode::PUSH_CONST);
    EXPECT_EQ(chunk->constants[0]->asNumber(), 42.0);
}

TEST_F(BytecodeCompilerTest, CompileArithmetic) {
    auto expr = SExprParser::parseString("(+ 1 2 3)");
    auto chunk = compiler->compile(expr);
    
    ASSERT_NE(chunk, nullptr);
    // Should compile to: PUSH 1, PUSH 2, ADD, PUSH 3, ADD, HALT
    EXPECT_GE(chunk->instructions.size(), 6);
}

TEST_F(BytecodeCompilerTest, CompileIfStatement) {
    auto expr = SExprParser::parseString("(if (> x 10) 'big 'small)");
    auto chunk = compiler->compile(expr);
    
    ASSERT_NE(chunk, nullptr);
    // Should include conditional jump
    bool has_jump = false;
    for (const auto& inst : chunk->instructions) {
        if (inst.opcode == OpCode::JMP_UNLESS || inst.opcode == OpCode::JMP_IF) {
            has_jump = true;
            break;
        }
    }
    EXPECT_TRUE(has_jump);
}

TEST_F(BytecodeCompilerTest, CompileLambda) {
    auto expr = SExprParser::parseString("(lambda (x y) (+ x y))");
    auto chunk = compiler->compile(expr);
    
    ASSERT_NE(chunk, nullptr);
    // Should create closure
    bool has_closure = false;
    for (const auto& inst : chunk->instructions) {
        if (inst.opcode == OpCode::MAKE_CLOSURE) {
            has_closure = true;
            break;
        }
    }
    EXPECT_TRUE(has_closure);
}

//=============================================================================
// Bytecode VM Tests
//=============================================================================

class BytecodeVMTest : public ::testing::Test {
protected:
    std::unique_ptr<BytecodeCompiler> compiler;
    std::unique_ptr<BytecodeVM> vm;
    
    void SetUp() override {
        compiler = std::make_unique<BytecodeCompiler>();
        vm = std::make_unique<BytecodeVM>();
    }
};

TEST_F(BytecodeVMTest, ExecuteArithmetic) {
    auto expr = SExprParser::parseString("(+ 10 20 30)");
    auto chunk = compiler->compile(expr);
    auto result = vm->execute(chunk);
    
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->isNumber());
    EXPECT_EQ(result->asNumber(), 60.0);
}

TEST_F(BytecodeVMTest, ExecuteComparison) {
    auto expr1 = SExprParser::parseString("(> 10 5)");
    auto chunk1 = compiler->compile(expr1);
    auto result1 = vm->execute(chunk1);
    
    ASSERT_NE(result1, nullptr);
    EXPECT_TRUE(result1->isBool());
    EXPECT_TRUE(result1->asBool());
    
    auto expr2 = SExprParser::parseString("(< 10 5)");
    auto chunk2 = compiler->compile(expr2);
    auto result2 = vm->execute(chunk2);
    
    ASSERT_NE(result2, nullptr);
    EXPECT_TRUE(result2->isBool());
    EXPECT_FALSE(result2->asBool());
}

TEST_F(BytecodeVMTest, ExecuteConditional) {
    auto expr = SExprParser::parseString("(if (> 10 5) 'yes 'no)");
    auto chunk = compiler->compile(expr);
    auto result = vm->execute(chunk);
    
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->isSymbol());
    EXPECT_EQ(result->asSymbol(), "yes");
}

TEST_F(BytecodeVMTest, ExecuteWithTimeout) {
    // Create an infinite loop
    auto expr = SExprParser::parseString("(while #t 1)");
    auto chunk = compiler->compile(expr);
    
    auto start = std::chrono::steady_clock::now();
    auto result = vm->executeWithTimeout(chunk, std::chrono::milliseconds(100));
    auto end = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_LT(duration.count(), 200);  // Should timeout within reasonable time
}

//=============================================================================
// Event Bridge Tests
//=============================================================================

class EventBridgeTest : public ::testing::Test {
protected:
    std::unique_ptr<SimulationState> sim_state;
    std::unique_ptr<EventRingBuffer> buffer;
    std::unique_ptr<EventBridge> bridge;
    
    void SetUp() override {
        sim_state = std::make_unique<SimulationState>();
        // Create test event buffer
        buffer = std::make_unique<EventRingBuffer>();
        bridge = std::make_unique<EventBridge>(buffer.get(), sim_state.get());
    }
};

TEST_F(EventBridgeTest, EventToSExprConversion) {
    Event event;
    event.type = EventType::HARD_COLLISION;
    event.primary_id = 10;
    event.secondary_id = 20;
    event.magnitude = 1000.0f;
    
    auto sexpr = EventConverter::eventToSExpr(event);
    
    ASSERT_NE(sexpr, nullptr);
    EXPECT_TRUE(sexpr->isList());
    EXPECT_GE(sexpr->length(), 4);
}

TEST_F(EventBridgeTest, PatternMatchingOnEvents) {
    // Create collision pattern
    auto pattern = PatternBuilder::collision("p1", "p2", "energy");
    
    // Create collision event as S-expression
    auto event_expr = SExpr::makeList({
        SExpr::makeSymbol("collision"),
        SExpr::makeNumber(10),
        SExpr::makeNumber(20),
        SExpr::makeNumber(1500)
    });
    
    auto result = PatternMatcher::matchPattern(pattern, event_expr);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.get("p1")->asNumber(), 10);
    EXPECT_EQ(result.get("p2")->asNumber(), 20);
    EXPECT_EQ(result.get("energy")->asNumber(), 1500);
}

TEST_F(EventBridgeTest, EventHandlerRegistration) {
    bool handler_called = false;
    float captured_energy = 0;
    
    auto pattern = PatternBuilder::collision("p1", "p2", "energy");
    auto action = SExprParser::parseString("(print 'collision-handled)");
    
    bridge->registerHandler(EventType::HARD_COLLISION, pattern, action);
    
    // Simulate event
    Event event;
    event.type = EventType::HARD_COLLISION;
    event.primary_id = 1;
    event.secondary_id = 2;
    event.magnitude = 2000.0f;
    
    bridge->onEvent(event);
    
    // Give some time for async processing
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    const auto& stats = bridge->getStats();
    EXPECT_GT(stats.events_received.load(), 0);
}

//=============================================================================
// DSL Runtime Integration Tests
//=============================================================================

class DSLRuntimeTest : public ::testing::Test {
protected:
    std::unique_ptr<SimulationState> sim_state;
    std::unique_ptr<DSLRuntime> runtime;
    
    void SetUp() override {
        sim_state = std::make_unique<SimulationState>();
        sim_state->particles.allocate(100000);
        runtime = std::make_unique<DSLRuntime>(sim_state.get());
    }
};

TEST_F(DSLRuntimeTest, ExecuteSimpleExpression) {
    auto result = runtime->execute("(+ 1 2 3)");
    
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->isNumber());
    EXPECT_EQ(result->asNumber(), 6.0);
}

TEST_F(DSLRuntimeTest, PatternMatchingIntegration) {
    std::string code = R"(
        (match '(collision 10 20 1500)
          [(collision ?p1 ?p2 (> ?energy 1000))
           (list 'high-energy p1 p2 energy)]
          [(collision ?p1 ?p2 _)
           (list 'low-energy p1 p2)])
    )";
    
    auto result = runtime->execute(code);
    
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->isList());
    EXPECT_EQ(result->nth(0)->asSymbol(), "high-energy");
    EXPECT_EQ(result->nth(1)->asNumber(), 10);
    EXPECT_EQ(result->nth(2)->asNumber(), 20);
    EXPECT_EQ(result->nth(3)->asNumber(), 1500);
}

TEST_F(DSLRuntimeTest, ProceduralGenerationIntegration) {
    std::string code = R"(
        (generate 'cloud
          :particles 50
          :center [100 100]
          :radius 20
          :mass [0.5 2.0])
    )";
    
    auto result = runtime->execute(code);
    
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->isList());
    EXPECT_EQ(result->length(), 50);  // Should return list of particle indices
}

TEST_F(DSLRuntimeTest, EventHandlerRegistration) {
    std::string code = R"(
        (on-event 'collision
          (match event
            [(collision ?p1 ?p2 (> ?energy 1000))
             (emit-event 'high-energy-collision 
               :particles [p1 p2]
               :energy energy)]))
    )";
    
    auto result = runtime->execute(code);
    // Should execute without error
    EXPECT_NE(result, nullptr);
}

TEST_F(DSLRuntimeTest, ScriptManagement) {
    std::string script = "(define test-value 42)";
    
    runtime->loadScript("test", script);
    runtime->runScript("test");
    
    auto result = runtime->execute("test-value");
    
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->isNumber());
    EXPECT_EQ(result->asNumber(), 42);
}

TEST_F(DSLRuntimeTest, PerformanceTracking) {
    // Execute multiple operations
    for (int i = 0; i < 100; ++i) {
        runtime->execute("(+ 1 2 3)");
    }
    
    auto perf = runtime->getPerformance();
    EXPECT_EQ(perf.total_executions, 100);
    EXPECT_GT(perf.total_execution_time.count(), 0);
}

//=============================================================================
// Performance Benchmarks
//=============================================================================

class DSLPerformanceTest : public ::testing::Test {
protected:
    std::unique_ptr<SimulationState> sim_state;
    std::unique_ptr<DSLRuntime> runtime;
    
    void SetUp() override {
        sim_state = std::make_unique<SimulationState>();
        sim_state->particles.allocate(1000000);  // Large pool for benchmarks
        runtime = std::make_unique<DSLRuntime>(sim_state.get());
    }
};

TEST_F(DSLPerformanceTest, LargeScaleGeneration) {
    auto start = std::chrono::high_resolution_clock::now();
    
    auto indices = runtime->generateGalaxy(10000, 0, 0, 1000);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_GT(indices.size(), 10000);
    EXPECT_LT(duration.count(), 1000);  // Should complete within 1 second
    
    std::cout << "Generated " << indices.size() << " particles in " 
              << duration.count() << "ms" << std::endl;
}

TEST_F(DSLPerformanceTest, PatternMatchingPerformance) {
    auto pattern = PatternBuilder::collision("p1", "p2", "energy");
    auto event = SExpr::makeList({
        SExpr::makeSymbol("collision"),
        SExpr::makeNumber(1),
        SExpr::makeNumber(2),
        SExpr::makeNumber(1000)
    });
    
    auto start = std::chrono::high_resolution_clock::now();
    
    const int iterations = 100000;
    for (int i = 0; i < iterations; ++i) {
        auto result = runtime->match(event, pattern);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double per_match = static_cast<double>(duration.count()) / iterations;
    EXPECT_LT(per_match, 10.0);  // Should be less than 10 microseconds per match
    
    std::cout << "Pattern matching: " << per_match << " microseconds per match" << std::endl;
}

TEST_F(DSLPerformanceTest, BytecodeExecutionPerformance) {
    std::string code = "(+ (* 2 3) (* 4 5) (* 6 7))";
    runtime->loadScript("benchmark", code);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    const int iterations = 100000;
    for (int i = 0; i < iterations; ++i) {
        runtime->runScript("benchmark");
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double per_execution = static_cast<double>(duration.count()) / iterations;
    EXPECT_LT(per_execution, 5.0);  // Should be less than 5 microseconds per execution
    
    std::cout << "Bytecode execution: " << per_execution 
              << " microseconds per execution" << std::endl;
}

//=============================================================================
// Main test runner
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}