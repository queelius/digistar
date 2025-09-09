#include "../src/events/event_system.h"
#include "../src/events/event_producer.h"
#include "../src/events/event_consumer.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <random>
#include <cassert>
#include <cmath>

using namespace digistar;

// Test helpers (matching project style)
#define TEST(name) std::cout << "Testing " << name << "... "; 
#define PASS() std::cout << "PASSED" << std::endl;
#define FAIL(msg) std::cerr << "FAILED: " << msg << std::endl; exit(1);
#define ASSERT(cond) if (!(cond)) { FAIL("Assertion failed at line " << __LINE__); }
#define ASSERT_EQ(a, b) if ((a) != (b)) { FAIL("Expected " << (b) << ", got " << (a) << " at line " << __LINE__); }
#define ASSERT_NE(a, b) if ((a) == (b)) { FAIL("Expected not equal at line " << __LINE__); }
#define ASSERT_GE(a, b) if ((a) < (b)) { FAIL("Expected " << (a) << " >= " << (b) << " at line " << __LINE__); }
#define ASSERT_NEAR(a, b, eps) if (std::abs((a) - (b)) >= (eps)) { FAIL("Expected " << (a) << " ≈ " << (b) << " at line " << __LINE__); }

// Global test counter for unique shared memory names
std::atomic<int> test_counter_{0};

Event createTestEvent(EventType type, uint32_t tick, float timestamp) {
    Event event{};
    event.type = type;
    event.flags = FLAG_NONE;
    event.tick = tick;
    event.timestamp = timestamp;
    event.x = 100.0f + static_cast<float>(tick);
    event.y = 200.0f + static_cast<float>(tick);
    event.primary_id = tick;
    event.secondary_id = tick + 1000;
    event.magnitude = static_cast<float>(tick) * 0.5f;
    
    switch (type) {
        case EventType::SPRING_BROKEN:
            event.data.spring.stress = event.magnitude;
            event.data.spring.break_force = event.magnitude * 1.2f;
            break;
        case EventType::HARD_COLLISION:
            event.data.collision.impulse = event.magnitude;
            event.data.collision.penetration = 0.5f;
            break;
        default:
            break;
    }
    
    return event;
}

bool validateEvent(const Event& original, const Event& received) {
    return original.type == received.type &&
           original.tick == received.tick &&
           std::abs(original.timestamp - received.timestamp) < 1e-6f &&
           std::abs(original.x - received.x) < 1e-6f &&
           std::abs(original.y - received.y) < 1e-6f &&
           original.primary_id == received.primary_id &&
           original.secondary_id == received.secondary_id &&
           std::abs(original.magnitude - received.magnitude) < 1e-6f;
}

// Test 1: Event structure size and alignment  
void test_event_structure_properties() {
    TEST("event structure properties");
    
    ASSERT_EQ(sizeof(Event), 64);  // Event must be exactly 64 bytes for cache alignment
    ASSERT_EQ(alignof(Event), 64); // Event must be 64-byte aligned
    
    // Test that the structure is properly initialized
    Event event{};
    ASSERT_EQ(static_cast<int>(event.type), 0);
    ASSERT_EQ(event.flags, 0);
    ASSERT_EQ(event.tick, 0u);
    ASSERT_EQ(event.timestamp, 0.0f);
    
    PASS();
}

// Test 2: Ring buffer properties
TEST_F(EventSystemTest, RingBufferProperties) {
    EventRingBuffer buffer;
    
    EXPECT_TRUE(buffer.is_valid());
    EXPECT_EQ(buffer.magic, EventRingBuffer::MAGIC);
    EXPECT_EQ(buffer.version, EventRingBuffer::VERSION);
    EXPECT_EQ(buffer.capacity, EventRingBuffer::CAPACITY);
    EXPECT_EQ(buffer.event_size, sizeof(Event));
    EXPECT_EQ(buffer.num_consumers.load(), 0u);
    EXPECT_EQ(buffer.total_events.load(), 0u);
    EXPECT_EQ(buffer.dropped_events.load(), 0u);
}

// Test 3: Shared memory creation
TEST_F(EventSystemTest, SharedMemoryCreation) {
    EXPECT_NO_THROW({
        SharedMemoryEventSystem event_system(test_shm_name_, true);
        EXPECT_TRUE(event_system.is_valid());
        
        auto buffer = event_system.get_buffer();
        ASSERT_NE(buffer, nullptr);
        EXPECT_TRUE(buffer->is_valid());
    });
}

// Test 4: Basic producer/consumer functionality
TEST_F(EventSystemTest, BasicProducerConsumer) {
    SharedMemoryEventSystem event_system(test_shm_name_, true);
    auto buffer = event_system.get_buffer();
    ASSERT_NE(buffer, nullptr);
    
    EventProducer producer(buffer);
    EventConsumer consumer(buffer, "TestConsumer");
    
    // Test single event
    Event test_event = createTestEvent(EventType::SPRING_BROKEN, 42, 1.234f);
    
    EXPECT_EQ(producer.emit(test_event), EventSystemError::SUCCESS);
    
    Event received_event;
    EXPECT_TRUE(consumer.poll(received_event));
    EXPECT_TRUE(validateEvent(test_event, received_event));
    
    // No more events should be available
    Event extra_event;
    EXPECT_FALSE(consumer.poll(extra_event));
}

// Test 5: Batch operations
TEST_F(EventSystemTest, BatchOperations) {
    SharedMemoryEventSystem event_system(test_shm_name_, true);
    auto buffer = event_system.get_buffer();
    
    EventProducer producer(buffer);
    EventConsumer consumer(buffer, "BatchConsumer");
    
    constexpr size_t BATCH_SIZE = 100;
    std::vector<Event> test_events;
    
    // Create test events
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        test_events.push_back(createTestEvent(EventType::HARD_COLLISION, 
                                            static_cast<uint32_t>(i), 
                                            static_cast<float>(i) * 0.01f));
    }
    
    // Test batch emission
    size_t emitted = producer.emit_batch(test_events.data(), BATCH_SIZE);
    EXPECT_EQ(emitted, BATCH_SIZE);
    
    // Test batch consumption
    std::vector<Event> received_events(BATCH_SIZE);
    size_t consumed = consumer.poll_batch(received_events.data(), BATCH_SIZE);
    EXPECT_EQ(consumed, BATCH_SIZE);
    
    // Validate all events
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        EXPECT_TRUE(validateEvent(test_events[i], received_events[i]))
            << "Event " << i << " failed validation";
    }
}

// Test 6: Event type filtering
TEST_F(EventSystemTest, EventTypeFiltering) {
    SharedMemoryEventSystem event_system(test_shm_name_, true);
    auto buffer = event_system.get_buffer();
    
    EventProducer producer(buffer);
    EventConsumer consumer(buffer, "FilterConsumer");
    
    // Set up filter for collision events only
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
        Event event = createTestEvent(test_types[i], static_cast<uint32_t>(i), 
                                     static_cast<float>(i));
        EXPECT_EQ(producer.emit(event), EventSystemError::SUCCESS);
    }
    
    // Count received events - should be 3 collision events
    std::vector<Event> received_events;
    Event event;
    while (consumer.poll(event)) {
        received_events.push_back(event);
    }
    
    EXPECT_EQ(received_events.size(), 3u);
    
    // Verify all received events are collision events
    for (const auto& e : received_events) {
        EXPECT_TRUE(e.type == EventType::HARD_COLLISION || e.type == EventType::SOFT_CONTACT)
            << "Unexpected event type: " << static_cast<int>(e.type);
    }
}

// Test 7: Spatial filtering
TEST_F(EventSystemTest, SpatialFiltering) {
    SharedMemoryEventSystem event_system(test_shm_name_, true);
    auto buffer = event_system.get_buffer();
    
    EventProducer producer(buffer);
    EventConsumer consumer(buffer, "SpatialConsumer");
    
    // Set spatial filter centered at (100, 100) with radius 50
    consumer.set_spatial_filter(100.0f, 100.0f, 50.0f);
    
    // Emit events at different locations
    struct TestCase {
        float x, y;
        bool should_pass;
    } test_cases[] = {
        {100.0f, 100.0f, true},   // At center
        {120.0f, 120.0f, true},   // Within radius
        {200.0f, 200.0f, false},  // Outside radius
        {80.0f, 80.0f, true},     // Within radius
        {50.0f, 50.0f, false}     // Outside radius
    };
    
    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        Event event = createTestEvent(EventType::SPRING_BROKEN, static_cast<uint32_t>(i), 
                                     static_cast<float>(i));
        event.x = test_cases[i].x;
        event.y = test_cases[i].y;
        
        EXPECT_EQ(producer.emit(event), EventSystemError::SUCCESS);
    }
    
    // Count events that pass the spatial filter
    size_t received_count = 0;
    Event event;
    while (consumer.poll(event)) {
        received_count++;
    }
    
    // Should receive 3 events that are within the radius
    EXPECT_EQ(received_count, 3u);
}

// Test 8: Multiple consumers
TEST_F(EventSystemTest, MultipleConsumers) {
    SharedMemoryEventSystem event_system(test_shm_name_, true);
    auto buffer = event_system.get_buffer();
    
    EventProducer producer(buffer);
    
    constexpr int NUM_CONSUMERS = 4;
    std::vector<std::unique_ptr<EventConsumer>> consumers;
    
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        consumers.push_back(std::make_unique<EventConsumer>(
            buffer, "Consumer" + std::to_string(i)));
    }
    
    // Emit test events
    constexpr size_t NUM_EVENTS = 50;
    for (size_t i = 0; i < NUM_EVENTS; i++) {
        Event event = createTestEvent(EventType::SPRING_BROKEN, 
                                     static_cast<uint32_t>(i), 
                                     static_cast<float>(i));
        EXPECT_EQ(producer.emit(event), EventSystemError::SUCCESS);
    }
    
    // Each consumer should receive all events
    for (int c = 0; c < NUM_CONSUMERS; c++) {
        size_t received_count = 0;
        Event event;
        while (consumers[c]->poll(event)) {
            received_count++;
        }
        
        EXPECT_EQ(received_count, NUM_EVENTS)
            << "Consumer " << c << " received wrong number of events";
    }
}

// Test 9: Producer helper methods
TEST_F(EventSystemTest, ProducerHelperMethods) {
    SharedMemoryEventSystem event_system(test_shm_name_, true);
    auto buffer = event_system.get_buffer();
    
    EventProducer producer(buffer);
    EventConsumer consumer(buffer, "HelperConsumer");
    
    producer.set_simulation_state(100, 5.0f);
    
    // Test spring break event
    EXPECT_EQ(producer.emit_spring_break(1, 2, 3, 10.0f, 20.0f, 100.0f, 150.0f), 
              EventSystemError::SUCCESS);
    
    // Test collision event
    EXPECT_EQ(producer.emit_collision(4, 5, 30.0f, 40.0f, 50.0f, 1.0f, 10.0f, 15.0f, true), 
              EventSystemError::SUCCESS);
    
    // Test particle merge event
    EXPECT_EQ(producer.emit_particle_merge(6, 7, 50.0f, 60.0f, 100.0f, 105.0f), 
              EventSystemError::SUCCESS);
    
    // Verify events were emitted correctly
    std::vector<Event> events;
    Event event;
    while (consumer.poll(event)) {
        events.push_back(event);
        
        // Check that simulation state was applied
        EXPECT_EQ(event.tick, 100u);
        EXPECT_FLOAT_EQ(event.timestamp, 5.0f);
    }
    
    EXPECT_EQ(events.size(), 3u);
    
    // Verify event types
    EXPECT_EQ(events[0].type, EventType::SPRING_BROKEN);
    EXPECT_EQ(events[1].type, EventType::HARD_COLLISION);
    EXPECT_EQ(events[2].type, EventType::PARTICLE_MERGE);
}

// Test 10: Buffer overflow handling
TEST_F(EventSystemTest, BufferOverflowHandling) {
    SharedMemoryEventSystem event_system(test_shm_name_, true);
    auto buffer = event_system.get_buffer();
    
    EventProducer producer(buffer);
    // Note: Not creating a consumer, so events will accumulate
    
    // Try to emit more events than buffer capacity
    size_t capacity = buffer->capacity;
    size_t events_to_emit = capacity + 100;
    
    size_t successful_emits = 0;
    for (size_t i = 0; i < events_to_emit; i++) {
        Event event = createTestEvent(EventType::SPRING_BROKEN, 
                                     static_cast<uint32_t>(i), 
                                     static_cast<float>(i));
        
        if (producer.emit(event) == EventSystemError::SUCCESS) {
            successful_emits++;
        }
    }
    
    // Should have successfully emitted exactly the buffer capacity
    EXPECT_EQ(successful_emits, capacity);
    
    // Check that dropped events were tracked
    EXPECT_EQ(buffer->dropped_events.load(), events_to_emit - capacity);
}

// Test 11: Specialized consumers
TEST_F(EventSystemTest, SpecializedConsumers) {
    SharedMemoryEventSystem event_system(test_shm_name_, true);
    auto buffer = event_system.get_buffer();
    
    EventProducer producer(buffer);
    
    AudioEventConsumer audio_consumer(buffer, 100.0f, 100.0f, 200.0f);
    NetworkEventConsumer network_consumer(buffer);
    VFXEventConsumer vfx_consumer(buffer, 150.0f, 150.0f, 300.0f);
    
    // Emit various types of events
    std::vector<std::tuple<EventType, uint16_t, float, float>> test_events = {
        {EventType::SPRING_BROKEN, FLAG_SPATIAL_FILTER, 120.0f, 120.0f},    // Audio + VFX
        {EventType::PARTICLE_MERGE, FLAG_NETWORK_BROADCAST, 0.0f, 0.0f},   // Network + VFX
        {EventType::PLAYER_ACTION, FLAG_NETWORK_BROADCAST, 0.0f, 0.0f},    // Network only
        {EventType::BLACK_HOLE_FORMATION, FLAG_NETWORK_BROADCAST | FLAG_SPATIAL_FILTER, 100.0f, 100.0f}, // All
        {EventType::TICK_COMPLETE, FLAG_NONE, 0.0f, 0.0f}                 // Network only
    };
    
    for (size_t i = 0; i < test_events.size(); i++) {
        auto [type, flags, x, y] = test_events[i];
        Event event = createTestEvent(type, static_cast<uint32_t>(i), static_cast<float>(i));
        event.flags = flags;
        event.x = x;
        event.y = y;
        
        EXPECT_EQ(producer.emit(event), EventSystemError::SUCCESS);
    }
    
    // Count events received by each consumer
    size_t audio_count = 0, network_count = 0, vfx_count = 0;
    
    Event event;
    while (audio_consumer.poll(event)) {
        audio_count++;
    }
    
    while (network_consumer.poll(event)) {
        network_count++;
    }
    
    while (vfx_consumer.poll(event)) {
        vfx_count++;
    }
    
    // Audio consumer should receive audio-relevant events within spatial range
    EXPECT_GE(audio_count, 1u);  // At least spring broken
    
    // Network consumer should receive network broadcast events  
    EXPECT_GE(network_count, 3u);  // Merge, player action, black hole, tick complete
    
    // VFX consumer should receive visually interesting events
    EXPECT_GE(vfx_count, 2u);  // At least merge and black hole
}

// Test 12: Thread safety (basic multi-threading test)
TEST_F(EventSystemTest, ThreadSafety) {
    SharedMemoryEventSystem event_system(test_shm_name_, true);
    auto buffer = event_system.get_buffer();
    
    constexpr int NUM_PRODUCER_THREADS = 2;
    constexpr int NUM_CONSUMER_THREADS = 2;
    constexpr size_t EVENTS_PER_PRODUCER = 1000;
    
    std::atomic<bool> running{true};
    std::atomic<size_t> total_produced{0};
    std::atomic<size_t> total_consumed{0};
    
    // Producer threads
    std::vector<std::thread> producer_threads;
    for (int i = 0; i < NUM_PRODUCER_THREADS; i++) {
        producer_threads.emplace_back([&, i]() {
            EventProducer producer(buffer);
            for (size_t j = 0; j < EVENTS_PER_PRODUCER && running; j++) {
                Event event = createTestEvent(EventType::SPRING_BROKEN,
                                             static_cast<uint32_t>(i * EVENTS_PER_PRODUCER + j),
                                             static_cast<float>(j));
                if (producer.emit(event) == EventSystemError::SUCCESS) {
                    total_produced++;
                }
            }
        });
    }
    
    // Consumer threads
    std::vector<std::thread> consumer_threads;
    for (int i = 0; i < NUM_CONSUMER_THREADS; i++) {
        consumer_threads.emplace_back([&, i]() {
            EventConsumer consumer(buffer, "ThreadConsumer" + std::to_string(i));
            Event event;
            while (running) {
                if (consumer.poll(event)) {
                    total_consumed++;
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            }
        });
    }
    
    // Wait for producers to finish
    for (auto& thread : producer_threads) {
        thread.join();
    }
    
    // Let consumers catch up
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    running = false;
    
    // Wait for consumers to finish
    for (auto& thread : consumer_threads) {
        thread.join();
    }
    
    // Verify reasonable results
    size_t expected_total = NUM_PRODUCER_THREADS * EVENTS_PER_PRODUCER;
    EXPECT_GE(total_produced.load(), expected_total * 0.9)  // At least 90% produced
        << "Produced: " << total_produced.load() << ", Expected: " << expected_total;
    
    // Each consumer should have consumed some events
    // Total consumption = produced * NUM_CONSUMER_THREADS (since each consumer gets all events)
    EXPECT_GE(total_consumed.load(), total_produced.load() * NUM_CONSUMER_THREADS * 0.8)
        << "Consumed: " << total_consumed.load() << ", Produced: " << total_produced.load();
}