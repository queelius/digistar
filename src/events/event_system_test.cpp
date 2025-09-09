/**
 * Comprehensive test program for the DigiStar event system
 * 
 * This program validates the lock-free ring buffer implementation,
 * event producer/consumer functionality, and shared memory IPC.
 * It tests both single-threaded and multi-threaded scenarios to
 * ensure the event system performs correctly under load.
 */

#include "event_system.h"
#include "event_producer.h"
#include "event_consumer.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <random>
#include <cassert>
#include <iomanip>

using namespace digistar;
using namespace std::chrono;

// Test configuration
struct TestConfig {
    static constexpr size_t WARMUP_EVENTS = 1000;
    static constexpr size_t PERFORMANCE_EVENTS = 100000;
    static constexpr size_t STRESS_EVENTS = 1000000;
    static constexpr int NUM_CONSUMER_THREADS = 4;
    static constexpr int NUM_PRODUCER_THREADS = 2;
    static constexpr float SIMULATION_TIME_STEP = 1.0f / 60.0f;  // 60 FPS
};

// Global test state
std::atomic<bool> test_running{true};
std::atomic<size_t> total_events_produced{0};
std::atomic<size_t> total_events_consumed{0};
std::atomic<size_t> events_with_errors{0};

/**
 * Utility functions for testing
 */

void print_test_header(const std::string& test_name) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << test_name << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void print_test_result(const std::string& test_name, bool passed, const std::string& details = "") {
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << test_name;
    if (!details.empty()) {
        std::cout << " - " << details;
    }
    std::cout << std::endl;
}

Event create_test_event(EventType type, uint32_t tick, float timestamp) {
    Event event{};
    event.type = type;
    event.flags = FLAG_NONE;
    event.tick = tick;
    event.timestamp = timestamp;
    event.x = static_cast<float>(rand() % 1000);
    event.y = static_cast<float>(rand() % 1000);
    event.primary_id = rand() % 100000;
    event.secondary_id = rand() % 100000;
    event.magnitude = static_cast<float>(rand() % 1000);
    
    // Fill type-specific data based on event type
    switch (type) {
        case EventType::SPRING_BROKEN:
            event.data.spring.stress = event.magnitude;
            event.data.spring.break_force = event.magnitude * 1.2f;
            break;
        case EventType::HARD_COLLISION:
            event.data.collision.impulse = event.magnitude;
            event.data.collision.penetration = 0.5f;
            event.data.collision.velocity_x = static_cast<float>(rand() % 100 - 50);
            event.data.collision.velocity_y = static_cast<float>(rand() % 100 - 50);
            break;
        case EventType::PARTICLE_MERGE:
            event.data.merge.mass_before = event.magnitude;
            event.data.merge.mass_after = event.magnitude * 1.1f;
            break;
        default:
            break;
    }
    
    return event;
}

bool validate_event(const Event& original, const Event& received) {
    if (original.type != received.type) return false;
    if (original.tick != received.tick) return false;
    if (std::abs(original.timestamp - received.timestamp) > 1e-6f) return false;
    if (std::abs(original.x - received.x) > 1e-6f) return false;
    if (std::abs(original.y - received.y) > 1e-6f) return false;
    if (original.primary_id != received.primary_id) return false;
    if (original.secondary_id != received.secondary_id) return false;
    if (std::abs(original.magnitude - received.magnitude) > 1e-6f) return false;
    
    return true;
}

/**
 * Test 1: Basic shared memory creation and buffer validation
 */
bool test_shared_memory_creation() {
    print_test_header("Test 1: Shared Memory Creation");
    
    try {
        // Test creating new shared memory
        SharedMemoryEventSystem event_system("/digistar_test_events", true);
        
        bool valid = event_system.is_valid();
        print_test_result("Create shared memory", valid);
        
        if (!valid) return false;
        
        // Test buffer properties
        auto buffer = event_system.get_buffer();
        bool buffer_valid = buffer && buffer->is_valid();
        print_test_result("Buffer validation", buffer_valid);
        
        if (!buffer_valid) return false;
        
        // Test buffer capacity
        bool capacity_correct = (buffer->capacity == EventRingBuffer::CAPACITY);
        print_test_result("Buffer capacity", capacity_correct, 
                         "Expected " + std::to_string(EventRingBuffer::CAPACITY) + 
                         ", got " + std::to_string(buffer->capacity));
        
        // Test statistics
        auto stats = event_system.get_stats();
        bool stats_valid = (stats.total_events == 0 && stats.dropped_events == 0);
        print_test_result("Initial statistics", stats_valid);
        
        return valid && buffer_valid && capacity_correct && stats_valid;
        
    } catch (const std::exception& e) {
        print_test_result("Shared memory creation", false, e.what());
        return false;
    }
}

/**
 * Test 2: Basic producer/consumer functionality
 */
bool test_basic_producer_consumer() {
    print_test_header("Test 2: Basic Producer/Consumer");
    
    try {
        SharedMemoryEventSystem event_system("/digistar_test_basic", true);
        auto buffer = event_system.get_buffer();
        
        EventProducer producer(buffer);
        EventConsumer consumer(buffer, "TestConsumer");
        
        // Test single event
        Event test_event = create_test_event(EventType::SPRING_BROKEN, 42, 1.234f);
        auto result = producer.emit(test_event);
        
        bool emit_success = (result == EventSystemError::SUCCESS);
        print_test_result("Event emission", emit_success);
        
        if (!emit_success) return false;
        
        // Test consumption
        Event received_event;
        bool poll_success = consumer.poll(received_event);
        print_test_result("Event polling", poll_success);
        
        if (!poll_success) return false;
        
        // Test event integrity
        bool event_valid = validate_event(test_event, received_event);
        print_test_result("Event integrity", event_valid);
        
        // Test that no more events are available
        Event extra_event;
        bool no_extra = !consumer.poll(extra_event);
        print_test_result("No extra events", no_extra);
        
        return emit_success && poll_success && event_valid && no_extra;
        
    } catch (const std::exception& e) {
        print_test_result("Basic producer/consumer", false, e.what());
        return false;
    }
}

/**
 * Test 3: Batch operations
 */
bool test_batch_operations() {
    print_test_header("Test 3: Batch Operations");
    
    try {
        SharedMemoryEventSystem event_system("/digistar_test_batch", true);
        auto buffer = event_system.get_buffer();
        
        EventProducer producer(buffer);
        EventConsumer consumer(buffer, "BatchConsumer");
        
        // Create batch of test events
        constexpr size_t BATCH_SIZE = 1000;
        std::vector<Event> test_events;
        test_events.reserve(BATCH_SIZE);
        
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            test_events.push_back(create_test_event(
                static_cast<EventType>(EventType::SPRING_BROKEN + (i % 5)),
                static_cast<uint32_t>(i), 
                static_cast<float>(i) * 0.01f));
        }
        
        // Test batch emission
        size_t emitted = producer.emit_batch(test_events.data(), BATCH_SIZE);
        bool batch_emit = (emitted == BATCH_SIZE);
        print_test_result("Batch emission", batch_emit, 
                         "Emitted " + std::to_string(emitted) + "/" + std::to_string(BATCH_SIZE));
        
        // Test batch consumption
        std::vector<Event> received_events(BATCH_SIZE);
        size_t consumed = consumer.poll_batch(received_events.data(), BATCH_SIZE);
        bool batch_consume = (consumed == BATCH_SIZE);
        print_test_result("Batch consumption", batch_consume,
                         "Consumed " + std::to_string(consumed) + "/" + std::to_string(BATCH_SIZE));
        
        // Validate all events
        size_t valid_events = 0;
        for (size_t i = 0; i < consumed && i < BATCH_SIZE; i++) {
            if (validate_event(test_events[i], received_events[i])) {
                valid_events++;
            }
        }
        
        bool all_valid = (valid_events == BATCH_SIZE);
        print_test_result("Batch validation", all_valid,
                         "Valid " + std::to_string(valid_events) + "/" + std::to_string(BATCH_SIZE));
        
        return batch_emit && batch_consume && all_valid;
        
    } catch (const std::exception& e) {
        print_test_result("Batch operations", false, e.what());
        return false;
    }
}

/**
 * Test 4: Event filtering
 */
bool test_event_filtering() {
    print_test_header("Test 4: Event Filtering");
    
    try {
        SharedMemoryEventSystem event_system("/digistar_test_filter", true);
        auto buffer = event_system.get_buffer();
        
        EventProducer producer(buffer);
        EventConsumer consumer(buffer, "FilterConsumer");
        
        // Set up type filter (only collision events)
        std::unordered_set<EventType> filter_types = {
            EventType::HARD_COLLISION,
            EventType::SOFT_CONTACT
        };
        consumer.set_event_type_filter(filter_types);
        
        // Emit mixed events
        std::vector<EventType> test_types = {
            EventType::HARD_COLLISION,  // Should pass
            EventType::SPRING_BROKEN,   // Should be filtered
            EventType::SOFT_CONTACT,    // Should pass
            EventType::PARTICLE_MERGE,  // Should be filtered
            EventType::HARD_COLLISION   // Should pass
        };
        
        for (size_t i = 0; i < test_types.size(); i++) {
            Event event = create_test_event(test_types[i], static_cast<uint32_t>(i), 
                                           static_cast<float>(i));
            producer.emit(event);
        }
        
        // Count received events
        std::vector<Event> received_events;
        Event event;
        while (consumer.poll(event)) {
            received_events.push_back(event);
        }
        
        // Should have received 3 collision events
        bool correct_count = (received_events.size() == 3);
        print_test_result("Filtered event count", correct_count,
                         "Expected 3, got " + std::to_string(received_events.size()));
        
        // Verify all received events are collision events
        bool all_collisions = true;
        for (const auto& e : received_events) {
            if (e.type != EventType::HARD_COLLISION && e.type != EventType::SOFT_CONTACT) {
                all_collisions = false;
                break;
            }
        }
        print_test_result("Filter correctness", all_collisions);
        
        return correct_count && all_collisions;
        
    } catch (const std::exception& e) {
        print_test_result("Event filtering", false, e.what());
        return false;
    }
}

/**
 * Test 5: Multiple consumers
 */
bool test_multiple_consumers() {
    print_test_header("Test 5: Multiple Consumers");
    
    try {
        SharedMemoryEventSystem event_system("/digistar_test_multi", true);
        auto buffer = event_system.get_buffer();
        
        EventProducer producer(buffer);
        
        // Create multiple consumers
        std::vector<std::unique_ptr<EventConsumer>> consumers;
        for (int i = 0; i < 4; i++) {
            consumers.push_back(std::make_unique<EventConsumer>(
                buffer, "Consumer" + std::to_string(i)));
        }
        
        // Emit test events
        constexpr size_t NUM_EVENTS = 100;
        std::vector<Event> test_events;
        
        for (size_t i = 0; i < NUM_EVENTS; i++) {
            Event event = create_test_event(EventType::SPRING_BROKEN, 
                                           static_cast<uint32_t>(i), 
                                           static_cast<float>(i));
            test_events.push_back(event);
            producer.emit(event);
        }
        
        // Each consumer should receive all events
        bool all_consumers_ok = true;
        for (size_t c = 0; c < consumers.size(); c++) {
            std::vector<Event> received;
            Event event;
            while (consumers[c]->poll(event)) {
                received.push_back(event);
            }
            
            bool consumer_ok = (received.size() == NUM_EVENTS);
            print_test_result("Consumer " + std::to_string(c), consumer_ok,
                             "Received " + std::to_string(received.size()) + "/" + 
                             std::to_string(NUM_EVENTS));
            
            if (!consumer_ok) {
                all_consumers_ok = false;
            }
        }
        
        return all_consumers_ok;
        
    } catch (const std::exception& e) {
        print_test_result("Multiple consumers", false, e.what());
        return false;
    }
}

/**
 * Test 6: Performance benchmark
 */
bool test_performance() {
    print_test_header("Test 6: Performance Benchmark");
    
    try {
        SharedMemoryEventSystem event_system("/digistar_test_perf", true);
        auto buffer = event_system.get_buffer();
        
        EventProducer producer(buffer);
        EventConsumer consumer(buffer, "PerfConsumer");
        
        // Warmup
        for (size_t i = 0; i < TestConfig::WARMUP_EVENTS; i++) {
            Event event = create_test_event(EventType::HARD_COLLISION, 
                                           static_cast<uint32_t>(i), 
                                           static_cast<float>(i));
            producer.emit(event);
        }
        
        // Clear warmup events
        Event dummy;
        while (consumer.poll(dummy)) {}
        
        // Performance test - emission
        auto emit_start = high_resolution_clock::now();
        
        for (size_t i = 0; i < TestConfig::PERFORMANCE_EVENTS; i++) {
            Event event = create_test_event(EventType::HARD_COLLISION, 
                                           static_cast<uint32_t>(i), 
                                           static_cast<float>(i));
            producer.emit(event);
        }
        
        auto emit_end = high_resolution_clock::now();
        auto emit_duration = duration_cast<microseconds>(emit_end - emit_start);
        
        // Performance test - consumption
        auto consume_start = high_resolution_clock::now();
        
        size_t consumed = 0;
        Event event;
        while (consumer.poll(event)) {
            consumed++;
        }
        
        auto consume_end = high_resolution_clock::now();
        auto consume_duration = duration_cast<microseconds>(consume_end - consume_start);
        
        // Calculate rates
        double emit_rate = (TestConfig::PERFORMANCE_EVENTS * 1e6) / emit_duration.count();
        double consume_rate = (consumed * 1e6) / consume_duration.count();
        
        std::cout << "Emission rate: " << std::fixed << std::setprecision(0) 
                  << emit_rate << " events/second" << std::endl;
        std::cout << "Consumption rate: " << std::fixed << std::setprecision(0) 
                  << consume_rate << " events/second" << std::endl;
        
        // Performance requirements (adjust based on target performance)
        bool emit_fast_enough = (emit_rate > 1000000);  // 1M events/sec
        bool consume_fast_enough = (consume_rate > 1000000);  // 1M events/sec
        bool all_consumed = (consumed == TestConfig::PERFORMANCE_EVENTS);
        
        print_test_result("Emission performance", emit_fast_enough,
                         std::to_string(static_cast<int>(emit_rate)) + " events/sec");
        print_test_result("Consumption performance", consume_fast_enough,
                         std::to_string(static_cast<int>(consume_rate)) + " events/sec");
        print_test_result("All events consumed", all_consumed);
        
        return emit_fast_enough && consume_fast_enough && all_consumed;
        
    } catch (const std::exception& e) {
        print_test_result("Performance benchmark", false, e.what());
        return false;
    }
}

/**
 * Producer thread for stress testing
 */
void producer_thread(EventRingBuffer* buffer, int thread_id, size_t num_events) {
    try {
        EventProducer producer(buffer);
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (size_t i = 0; i < num_events && test_running; i++) {
            EventType type = static_cast<EventType>(
                EventType::SPRING_BROKEN + (gen() % 10));
            Event event = create_test_event(type, 
                                           static_cast<uint32_t>(i + thread_id * num_events), 
                                           static_cast<float>(i) * 0.001f);
            
            if (producer.emit(event) == EventSystemError::SUCCESS) {
                total_events_produced++;
            }
            
            // Add small delay occasionally to simulate realistic physics timing
            if (i % 1000 == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Producer thread " << thread_id << " error: " << e.what() << std::endl;
    }
}

/**
 * Consumer thread for stress testing
 */
void consumer_thread(EventRingBuffer* buffer, int thread_id) {
    try {
        EventConsumer consumer(buffer, "StressConsumer" + std::to_string(thread_id));
        
        while (test_running) {
            Event events[100];
            size_t consumed = consumer.poll_batch(events, 100);
            
            if (consumed > 0) {
                total_events_consumed += consumed;
                
                // Basic validation of some events
                for (size_t i = 0; i < consumed; i++) {
                    if (events[i].type < EventType::PARTICLE_MERGE || 
                        events[i].type > EventType::STATS_UPDATE) {
                        events_with_errors++;
                    }
                }
            } else {
                // No events available, sleep briefly
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Consumer thread " << thread_id << " error: " << e.what() << std::endl;
    }
}

/**
 * Test 7: Multi-threaded stress test
 */
bool test_multithreaded_stress() {
    print_test_header("Test 7: Multi-threaded Stress Test");
    
    try {
        SharedMemoryEventSystem event_system("/digistar_test_stress", true);
        auto buffer = event_system.get_buffer();
        
        // Reset counters
        test_running = true;
        total_events_produced = 0;
        total_events_consumed = 0;
        events_with_errors = 0;
        
        // Start consumer threads
        std::vector<std::thread> consumer_threads;
        for (int i = 0; i < TestConfig::NUM_CONSUMER_THREADS; i++) {
            consumer_threads.emplace_back(consumer_thread, buffer, i);
        }
        
        auto test_start = high_resolution_clock::now();
        
        // Start producer threads
        std::vector<std::thread> producer_threads;
        size_t events_per_producer = TestConfig::STRESS_EVENTS / TestConfig::NUM_PRODUCER_THREADS;
        
        for (int i = 0; i < TestConfig::NUM_PRODUCER_THREADS; i++) {
            producer_threads.emplace_back(producer_thread, buffer, i, events_per_producer);
        }
        
        // Wait for producers to finish
        for (auto& thread : producer_threads) {
            thread.join();
        }
        
        // Let consumers catch up
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        auto test_end = high_resolution_clock::now();
        test_running = false;
        
        // Wait for consumers to finish
        for (auto& thread : consumer_threads) {
            thread.join();
        }
        
        auto test_duration = duration_cast<milliseconds>(test_end - test_start);
        
        // Calculate results
        size_t expected_events = TestConfig::STRESS_EVENTS;
        size_t produced = total_events_produced.load();
        size_t consumed = total_events_consumed.load();
        size_t errors = events_with_errors.load();
        
        double production_rate = (produced * 1000.0) / test_duration.count();
        double consumption_rate = (consumed * 1000.0) / test_duration.count();
        
        std::cout << "Test duration: " << test_duration.count() << " ms" << std::endl;
        std::cout << "Events produced: " << produced << "/" << expected_events << std::endl;
        std::cout << "Events consumed: " << consumed << std::endl;
        std::cout << "Events with errors: " << errors << std::endl;
        std::cout << "Production rate: " << std::fixed << std::setprecision(0) 
                  << production_rate << " events/sec" << std::endl;
        std::cout << "Consumption rate: " << std::fixed << std::setprecision(0) 
                  << consumption_rate << " events/sec" << std::endl;
        
        // Success criteria
        bool no_errors = (errors == 0);
        bool reasonable_production = (produced >= expected_events * 0.95);  // At least 95%
        bool fast_enough = (production_rate > 100000);  // At least 100K events/sec
        
        print_test_result("No errors", no_errors);
        print_test_result("Production rate", reasonable_production,
                         std::to_string(produced) + "/" + std::to_string(expected_events));
        print_test_result("Performance", fast_enough,
                         std::to_string(static_cast<int>(production_rate)) + " events/sec");
        
        return no_errors && reasonable_production && fast_enough;
        
    } catch (const std::exception& e) {
        test_running = false;
        print_test_result("Multi-threaded stress", false, e.what());
        return false;
    }
}

/**
 * Test 8: Specialized consumers
 */
bool test_specialized_consumers() {
    print_test_header("Test 8: Specialized Consumers");
    
    try {
        SharedMemoryEventSystem event_system("/digistar_test_special", true);
        auto buffer = event_system.get_buffer();
        
        EventProducer producer(buffer);
        
        // Create specialized consumers
        AudioEventConsumer audio_consumer(buffer, 100.0f, 100.0f, 500.0f);
        NetworkEventConsumer network_consumer(buffer);
        VFXEventConsumer vfx_consumer(buffer, 200.0f, 200.0f, 1000.0f);
        
        // Emit various events
        std::vector<std::pair<EventType, uint16_t>> test_cases = {
            {EventType::SPRING_BROKEN, FLAG_SPATIAL_FILTER},
            {EventType::PARTICLE_MERGE, FLAG_NETWORK_BROADCAST},
            {EventType::BLACK_HOLE_FORMATION, FLAG_NETWORK_BROADCAST | FLAG_SPATIAL_FILTER},
            {EventType::PLAYER_ACTION, FLAG_NETWORK_BROADCAST},
            {EventType::RESONANCE_DETECTED, FLAG_SPATIAL_FILTER},
            {EventType::TICK_COMPLETE, FLAG_NONE}
        };
        
        for (const auto& [type, flags] : test_cases) {
            Event event = create_test_event(type, 1, 1.0f);
            event.flags = flags;
            
            // Position some events near consumers, some far away
            if (flags & FLAG_SPATIAL_FILTER) {
                event.x = 150.0f;  // Near audio consumer
                event.y = 150.0f;
            } else {
                event.x = 2000.0f;  // Far from all consumers
                event.y = 2000.0f;
            }
            
            producer.emit(event);
        }
        
        // Count events received by each consumer
        size_t audio_events = 0;
        size_t network_events = 0;
        size_t vfx_events = 0;
        
        Event event;
        while (audio_consumer.poll(event)) {
            audio_events++;
        }
        
        while (network_consumer.poll(event)) {
            network_events++;
        }
        
        while (vfx_consumer.poll(event)) {
            vfx_events++;
        }
        
        std::cout << "Audio events: " << audio_events << std::endl;
        std::cout << "Network events: " << network_events << std::endl;
        std::cout << "VFX events: " << vfx_events << std::endl;
        
        // Audio consumer should get spatially filtered events
        bool audio_ok = (audio_events >= 2);  // Spring break + resonance (if in range)
        
        // Network consumer should get network broadcast events
        bool network_ok = (network_events >= 3);  // Merge + black hole + player action
        
        // VFX consumer should get visually interesting events
        bool vfx_ok = (vfx_events >= 2);  // Merge + black hole (if in range)
        
        print_test_result("Audio consumer", audio_ok);
        print_test_result("Network consumer", network_ok);
        print_test_result("VFX consumer", vfx_ok);
        
        return audio_ok && network_ok && vfx_ok;
        
    } catch (const std::exception& e) {
        print_test_result("Specialized consumers", false, e.what());
        return false;
    }
}

/**
 * Main test runner
 */
int main() {
    std::cout << "DigiStar Event System Test Suite" << std::endl;
    std::cout << "=================================" << std::endl;
    
    std::vector<std::pair<std::string, std::function<bool()>>> tests = {
        {"Shared Memory Creation", test_shared_memory_creation},
        {"Basic Producer/Consumer", test_basic_producer_consumer},
        {"Batch Operations", test_batch_operations},
        {"Event Filtering", test_event_filtering},
        {"Multiple Consumers", test_multiple_consumers},
        {"Performance Benchmark", test_performance},
        {"Multi-threaded Stress", test_multithreaded_stress},
        {"Specialized Consumers", test_specialized_consumers}
    };
    
    int passed = 0;
    int total = static_cast<int>(tests.size());
    
    for (const auto& [name, test_func] : tests) {
        try {
            if (test_func()) {
                passed++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Test '" << name << "' threw exception: " << e.what() << std::endl;
        }
        
        // Clean up shared memory between tests
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Print final results
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test Results: " << passed << "/" << total << " passed" << std::endl;
    
    if (passed == total) {
        std::cout << "All tests PASSED! Event system is working correctly." << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED. Please check the implementation." << std::endl;
        return 1;
    }
}