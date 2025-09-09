#include "physics_pipeline.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <cmath>

namespace digistar {

// EventAggregator implementation
void EventAggregator::addEvent(const Event& event) {
    std::lock_guard<std::mutex> lock(aggregation_mutex_);
    
    uint64_t key = makeAggregationKey(event.type, event.x, event.y);
    auto& agg = aggregated_events_[key];
    
    if (agg.count == 0) {
        // First event of this type/location
        agg.type = event.type;
        agg.first_time = std::chrono::steady_clock::now();
        agg.center_x = event.x;
        agg.center_y = event.y;
        agg.sum_magnitude = event.magnitude;
        agg.max_magnitude = event.magnitude;
        agg.participant_ids.push_back(event.primary_id);
        if (event.secondary_id != 0xFFFFFFFF) {
            agg.participant_ids.push_back(event.secondary_id);
        }
    } else {
        // Update aggregated data
        agg.sum_magnitude += event.magnitude;
        agg.max_magnitude = std::max(agg.max_magnitude, event.magnitude);
        
        // Update center of mass
        float weight = 1.0f / (agg.count + 1);
        agg.center_x = agg.center_x * (1.0f - weight) + event.x * weight;
        agg.center_y = agg.center_y * (1.0f - weight) + event.y * weight;
        
        // Add participants (avoid duplicates for now - could optimize)
        agg.participant_ids.push_back(event.primary_id);
        if (event.secondary_id != 0xFFFFFFFF) {
            agg.participant_ids.push_back(event.secondary_id);
        }
    }
    
    agg.count++;
    agg.last_time = std::chrono::steady_clock::now();
}

std::vector<Event> EventAggregator::flushAggregated(std::chrono::milliseconds max_age) {
    std::lock_guard<std::mutex> lock(aggregation_mutex_);
    std::vector<Event> events;
    auto now = std::chrono::steady_clock::now();
    
    auto it = aggregated_events_.begin();
    while (it != aggregated_events_.end()) {
        const auto& agg = it->second;
        
        if (now - agg.first_time >= max_age) {
            // Create aggregated event
            Event event{};
            event.type = agg.type;
            event.flags = FLAG_AGGREGATE;
            event.x = agg.center_x;
            event.y = agg.center_y;
            event.magnitude = agg.sum_magnitude;
            event.secondary_value = static_cast<float>(agg.count);
            
            // Use first and last participants
            if (!agg.participant_ids.empty()) {
                event.primary_id = agg.participant_ids.front();
                if (agg.participant_ids.size() > 1) {
                    event.secondary_id = agg.participant_ids.back();
                }
            }
            
            events.push_back(event);
            it = aggregated_events_.erase(it);
        } else {
            ++it;
        }
    }
    
    return events;
}

void EventAggregator::clear() {
    std::lock_guard<std::mutex> lock(aggregation_mutex_);
    aggregated_events_.clear();
}

size_t EventAggregator::getPendingCount() const {
    std::lock_guard<std::mutex> lock(aggregation_mutex_);
    return aggregated_events_.size();
}

uint64_t EventAggregator::makeAggregationKey(EventType type, float x, float y, uint32_t spatial_bucket) {
    // Create spatial buckets for aggregation
    uint32_t bucket_x = static_cast<uint32_t>((x + 10000.0f) / spatial_bucket);  // Offset to handle negative coords
    uint32_t bucket_y = static_cast<uint32_t>((y + 10000.0f) / spatial_bucket);
    
    uint64_t key = static_cast<uint64_t>(static_cast<uint16_t>(type));
    key = (key << 32) | ((static_cast<uint64_t>(bucket_x) << 16) | bucket_y);
    
    return key;
}

// PhysicsPipeline implementation
PhysicsPipeline::PhysicsPipeline(std::shared_ptr<IBackend> backend,
                                std::shared_ptr<EventProducer> event_producer)
    : backend_(backend), 
      event_producer_(event_producer),
      command_queue_(CommandComparator{}),
      last_stats_update_(std::chrono::steady_clock::now()) {
    
    event_aggregator_ = std::make_unique<EventAggregator>();
    
    // Set default event configuration
    event_config_.enabled_events = {
        EventType::PARTICLE_MERGE,
        EventType::PARTICLE_FISSION,
        EventType::SPRING_CREATED,
        EventType::SPRING_BROKEN,
        EventType::SOFT_CONTACT,
        EventType::HARD_COLLISION,
        EventType::COMPOSITE_FORMED,
        EventType::COMPOSITE_BROKEN
    };
}

PhysicsPipeline::~PhysicsPipeline() {
    if (async_processing_enabled_) {
        enableAsyncProcessing(false);
    }
    shutdown();
}

void PhysicsPipeline::initialize(const SimulationConfig& config) {
    if (initialized_) return;
    
    if (!backend_) {
        throw std::runtime_error("Cannot initialize physics pipeline without backend");
    }
    
    // Initialize backend if not already done
    try {
        backend_->initialize(config);
    } catch (const std::exception& e) {
        throw std::runtime_error("Backend initialization failed: " + std::string(e.what()));
    }
    
    // Set up event producer connection
    if (event_producer_ && backend_->supportsEvents()) {
        backend_->setEventProducer(event_producer_);
    }
    
    initialized_ = true;
    logInfo("Physics pipeline initialized with backend: " + backend_->getName());
}

void PhysicsPipeline::shutdown() {
    if (!initialized_) return;
    
    // Stop async processing
    if (async_processing_enabled_) {
        enableAsyncProcessing(false);
    }
    
    // Clear pending commands
    clearQueue();
    
    // Clear pending events
    {
        std::lock_guard<std::mutex> lock(event_mutex_);
        pending_events_.clear();
    }
    
    // Clear aggregated events
    event_aggregator_->clear();
    
    initialized_ = false;
    logInfo("Physics pipeline shut down");
}

void PhysicsPipeline::setEventProcessingConfig(const EventProcessingConfig& config) {
    event_config_ = config;
}

void PhysicsPipeline::enqueueCommand(const PipelineCommand& command) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (command_queue_.size() >= max_queue_size_) {
        handleCommandError("Command queue full", command);
        stats_.commands_failed++;
        return;
    }
    
    command_queue_.push(command);
    stats_.peak_queue_size = std::max(stats_.peak_queue_size, command_queue_.size());
    
    if (async_processing_enabled_) {
        processing_cv_.notify_one();
    }
}

void PhysicsPipeline::enqueueCommands(const std::vector<PipelineCommand>& commands) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    for (const auto& command : commands) {
        if (command_queue_.size() >= max_queue_size_) {
            handleCommandError("Command queue full during batch enqueue", command);
            stats_.commands_failed++;
            break;
        }
        
        command_queue_.push(command);
    }
    
    stats_.peak_queue_size = std::max(stats_.peak_queue_size, command_queue_.size());
    
    if (async_processing_enabled_) {
        processing_cv_.notify_all();
    }
}

size_t PhysicsPipeline::getQueueSize() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return command_queue_.size();
}

void PhysicsPipeline::clearQueue() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Clear priority queue by swapping with empty queue
    std::priority_queue<PipelineCommand, std::vector<PipelineCommand>, 
                       std::function<bool(const PipelineCommand&, const PipelineCommand&)>>
        empty_queue(CommandComparator{});
    command_queue_.swap(empty_queue);
}

PipelineCommand PhysicsPipeline::createParticle(float x, float y, float mass, float radius) {
    PipelineCommand command;
    command.type = PipelineCommandType::CREATE_PARTICLE;
    command.x = x;
    command.y = y;
    command.mass = mass;
    command.radius = radius;
    command.priority = 50;  // Medium priority
    return command;
}

PipelineCommand PhysicsPipeline::destroyParticle(uint32_t particle_id) {
    PipelineCommand command;
    command.type = PipelineCommandType::DESTROY_PARTICLE;
    command.target_id = particle_id;
    command.priority = 30;  // High priority
    return command;
}

PipelineCommand PhysicsPipeline::applyForce(uint32_t particle_id, float fx, float fy) {
    PipelineCommand command;
    command.type = PipelineCommandType::APPLY_FORCE;
    command.target_id = particle_id;
    command.fx = fx;
    command.fy = fy;
    command.priority = 80;  // Lower priority
    return command;
}

PipelineCommand PhysicsPipeline::createSpring(uint32_t p1_id, uint32_t p2_id, float stiffness, float damping) {
    PipelineCommand command;
    command.type = PipelineCommandType::CREATE_SPRING;
    command.target_ids = {p1_id, p2_id};
    command.float_params["stiffness"] = stiffness;
    command.float_params["damping"] = damping;
    command.priority = 60;  // Medium priority
    return command;
}

PipelineCommand PhysicsPipeline::generateGalaxy(float x, float y, size_t count, float radius) {
    PipelineCommand command;
    command.type = PipelineCommandType::GENERATE_OBJECTS;
    command.x = x;
    command.y = y;
    command.int_params["count"] = static_cast<int>(count);
    command.float_params["radius"] = radius;
    command.string_params["type"] = "galaxy";
    command.priority = 10;  // Very high priority for generation
    return command;
}

void PhysicsPipeline::update(SimulationState& state, const PhysicsConfig& physics_config, float dt) {
    if (!initialized_ || !backend_) return;
    
    auto update_start = std::chrono::steady_clock::now();
    
    try {
        // Process pending commands
        if (!async_processing_enabled_) {
            processCommands(state);
        }
        
        // Update physics
        backend_->step(state, physics_config, dt);
        
        // Generate and process events
        generateEvents(state, dt);
        processEvents();
        
        // Update statistics
        updateStatistics();
        
        auto update_end = std::chrono::steady_clock::now();
        stats_.command_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(update_end - update_start);
        
    } catch (const std::exception& e) {
        std::string error = "Physics pipeline update failed: " + std::string(e.what());
        logError(error);
        stats_.last_error = error;
    }
}

void PhysicsPipeline::setEventProducer(std::shared_ptr<EventProducer> producer) {
    event_producer_ = producer;
    
    if (backend_ && backend_->supportsEvents()) {
        backend_->setEventProducer(producer);
    }
}

void PhysicsPipeline::resetStats() {
    stats_ = PipelineStats{};
    last_stats_update_ = std::chrono::steady_clock::now();
}

std::string PhysicsPipeline::getStatsReport() const {
    std::stringstream ss;
    
    ss << "=== Physics Pipeline Statistics ===\n";
    ss << std::fixed << std::setprecision(2);
    
    ss << "Commands:\n";
    ss << "  Total Processed: " << stats_.total_commands_processed << "\n";
    ss << "  Currently Pending: " << stats_.commands_pending << "\n";
    ss << "  Failed: " << stats_.commands_failed << "\n";
    ss << "  Commands/sec: " << stats_.commands_per_second << "\n";
    ss << "  Peak Queue Size: " << stats_.peak_queue_size << "\n";
    
    ss << "\nEvents:\n";
    ss << "  Generated: " << stats_.events_generated << "\n";
    ss << "  Filtered: " << stats_.events_filtered << "\n";
    ss << "  Aggregated: " << stats_.events_aggregated << "\n";
    ss << "  Events/sec: " << stats_.events_per_second << "\n";
    
    ss << "\nPerformance:\n";
    ss << "  Command Processing: " << stats_.command_processing_time.count() << " μs\n";
    ss << "  Event Processing: " << stats_.event_processing_time.count() << " μs\n";
    
    if (!stats_.last_error.empty()) {
        ss << "\nLast Error: " << stats_.last_error << "\n";
    }
    
    return ss.str();
}

void PhysicsPipeline::enableBatchProcessing(bool enable, size_t batch_size) {
    batch_processing_enabled_ = enable;
    batch_size_ = batch_size;
    
    if (enable) {
        command_batch_.reserve(batch_size);
    } else {
        command_batch_.clear();
        command_batch_.shrink_to_fit();
    }
}

void PhysicsPipeline::enableAsyncProcessing(bool enable) {
    if (enable && !async_processing_enabled_) {
        processing_active_.store(true);
        processing_thread_ = std::make_unique<std::thread>(&PhysicsPipeline::asyncProcessingLoop, this);
        async_processing_enabled_ = true;
        logInfo("Enabled async command processing");
    } else if (!enable && async_processing_enabled_) {
        processing_active_.store(false);
        processing_cv_.notify_all();
        
        if (processing_thread_ && processing_thread_->joinable()) {
            processing_thread_->join();
        }
        
        processing_thread_.reset();
        async_processing_enabled_ = false;
        logInfo("Disabled async command processing");
    }
}

void PhysicsPipeline::setMaxQueueSize(size_t max_size) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    max_queue_size_ = max_size;
}

void PhysicsPipeline::processCommands(SimulationState& state) {
    std::vector<PipelineCommand> commands_to_process;
    
    // Extract commands from queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        if (batch_processing_enabled_) {
            // Process in batches
            size_t batch_count = std::min(batch_size_, command_queue_.size());
            commands_to_process.reserve(batch_count);
            
            for (size_t i = 0; i < batch_count && !command_queue_.empty(); ++i) {
                commands_to_process.push_back(command_queue_.top());
                command_queue_.pop();
            }
        } else {
            // Process all commands
            commands_to_process.reserve(command_queue_.size());
            
            while (!command_queue_.empty()) {
                commands_to_process.push_back(command_queue_.top());
                command_queue_.pop();
            }
        }
    }
    
    stats_.commands_pending = getQueueSize();
    
    // Process commands
    if (batch_processing_enabled_ && commands_to_process.size() > 1) {
        processBatch(commands_to_process, state);
    } else {
        for (const auto& command : commands_to_process) {
            processCommand(command, state);
        }
    }
    
    stats_.total_commands_processed += commands_to_process.size();
}

void PhysicsPipeline::processCommand(const PipelineCommand& command, SimulationState& state) {
    try {
        bool success = false;
        
        switch (command.type) {
        case PipelineCommandType::CREATE_PARTICLE:
            success = processCreateParticle(command, state);
            break;
        case PipelineCommandType::DESTROY_PARTICLE:
            success = processDestroyParticle(command, state);
            break;
        case PipelineCommandType::APPLY_FORCE:
            success = processApplyForce(command, state);
            break;
        case PipelineCommandType::CREATE_SPRING:
            success = processCreateSpring(command, state);
            break;
        case PipelineCommandType::GENERATE_OBJECTS:
            success = processGenerateObjects(command, state);
            break;
        default:
            handleCommandError("Unsupported command type", command);
            success = false;
            break;
        }
        
        if (!success) {
            stats_.commands_failed++;
        }
        
        if (command.completion_callback) {
            command.completion_callback(success, success ? "" : stats_.last_error);
        }
        
    } catch (const std::exception& e) {
        handleCommandError("Command processing exception: " + std::string(e.what()), command);
        stats_.commands_failed++;
        
        if (command.completion_callback) {
            command.completion_callback(false, e.what());
        }
    }
}

void PhysicsPipeline::processBatch(const std::vector<PipelineCommand>& batch, SimulationState& state) {
    // Group commands by type for batch processing
    std::unordered_map<PipelineCommandType, std::vector<const PipelineCommand*>> command_groups;
    
    for (const auto& command : batch) {
        command_groups[command.type].push_back(&command);
    }
    
    // Process each group
    for (const auto& [type, commands] : command_groups) {
        for (const auto* command : commands) {
            processCommand(*command, state);
        }
    }
}

bool PhysicsPipeline::processCreateParticle(const PipelineCommand& command, SimulationState& state) {
    if (state.particles.size() >= state.particles.capacity()) {
        handleCommandError("Particle pool full", command);
        return false;
    }
    
    size_t particle_id = state.particles.add();
    
    state.particles.x[particle_id] = command.x;
    state.particles.y[particle_id] = command.y;
    state.particles.mass[particle_id] = command.mass;
    state.particles.radius[particle_id] = command.radius;
    state.particles.temperature[particle_id] = command.temperature;
    state.particles.charge[particle_id] = command.charge;
    state.particles.vx[particle_id] = command.vx;
    state.particles.vy[particle_id] = command.vy;
    
    return true;
}

bool PhysicsPipeline::processDestroyParticle(const PipelineCommand& command, SimulationState& state) {
    if (!state.particles.isValid(command.target_id)) {
        handleCommandError("Invalid particle ID", command);
        return false;
    }
    
    state.particles.remove(command.target_id);
    return true;
}

bool PhysicsPipeline::processApplyForce(const PipelineCommand& command, SimulationState& state) {
    if (!state.particles.isValid(command.target_id)) {
        handleCommandError("Invalid particle ID for force application", command);
        return false;
    }
    
    state.particles.fx[command.target_id] += command.fx;
    state.particles.fy[command.target_id] += command.fy;
    return true;
}

bool PhysicsPipeline::processCreateSpring(const PipelineCommand& command, SimulationState& state) {
    if (command.target_ids.size() != 2) {
        handleCommandError("Spring creation requires exactly 2 particles", command);
        return false;
    }
    
    uint32_t p1 = command.target_ids[0];
    uint32_t p2 = command.target_ids[1];
    
    if (!state.particles.isValid(p1) || !state.particles.isValid(p2)) {
        handleCommandError("Invalid particle IDs for spring creation", command);
        return false;
    }
    
    if (state.springs.size() >= state.springs.capacity()) {
        handleCommandError("Spring pool full", command);
        return false;
    }
    
    size_t spring_id = state.springs.add();
    
    state.springs.particle1[spring_id] = p1;
    state.springs.particle2[spring_id] = p2;
    
    auto stiffness_it = command.float_params.find("stiffness");
    state.springs.stiffness[spring_id] = (stiffness_it != command.float_params.end()) 
        ? stiffness_it->second : 1000.0f;
    
    auto damping_it = command.float_params.find("damping");
    state.springs.damping[spring_id] = (damping_it != command.float_params.end()) 
        ? damping_it->second : 10.0f;
    
    // Calculate equilibrium distance as current distance
    float dx = state.particles.x[p1] - state.particles.x[p2];
    float dy = state.particles.y[p1] - state.particles.y[p2];
    state.springs.equilibrium_distance[spring_id] = std::sqrt(dx * dx + dy * dy);
    
    return true;
}

bool PhysicsPipeline::processGenerateObjects(const PipelineCommand& command, SimulationState& state) {
    auto type_it = command.string_params.find("type");
    if (type_it == command.string_params.end()) {
        handleCommandError("Generate command missing 'type' parameter", command);
        return false;
    }
    
    auto count_it = command.int_params.find("count");
    int count = (count_it != command.int_params.end()) ? count_it->second : 100;
    
    auto radius_it = command.float_params.find("radius");
    float radius = (radius_it != command.float_params.end()) ? radius_it->second : 100.0f;
    
    if (type_it->second == "galaxy") {
        // Generate spiral galaxy
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> mass_dist(1.0f, 0.5f);
        std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
        std::uniform_real_distribution<float> radius_dist(0.1f, radius);
        
        for (int i = 0; i < count && state.particles.size() < state.particles.capacity(); ++i) {
            float r = radius_dist(gen);
            float theta = angle_dist(gen) + 0.1f * r;  // Spiral effect
            
            float x = command.x + r * std::cos(theta);
            float y = command.y + r * std::sin(theta);
            
            // Orbital velocity for circular motion
            float v_orbital = std::sqrt(1000.0f / r);  // Simplified gravity
            float vx = -v_orbital * std::sin(theta);
            float vy = v_orbital * std::cos(theta);
            
            size_t particle_id = state.particles.add();
            state.particles.x[particle_id] = x;
            state.particles.y[particle_id] = y;
            state.particles.vx[particle_id] = vx;
            state.particles.vy[particle_id] = vy;
            state.particles.mass[particle_id] = std::max(0.1f, mass_dist(gen));
            state.particles.radius[particle_id] = 0.5f;
            state.particles.temperature[particle_id] = 300.0f + r * 10.0f;  // Cooler at edges
        }
        
        return true;
    }
    
    handleCommandError("Unsupported generation type: " + type_it->second, command);
    return false;
}

void PhysicsPipeline::generateEvents(const SimulationState& state, float dt) {
    // This is where the physics backend would generate events
    // For now, we'll just track the event generation time
    auto event_start = std::chrono::steady_clock::now();
    
    // Placeholder for event generation logic
    // In a real implementation, this would analyze the simulation state
    // and generate appropriate events
    
    auto event_end = std::chrono::steady_clock::now();
    stats_.event_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(event_end - event_start);
}

void PhysicsPipeline::processEvents() {
    if (!event_producer_) return;
    
    std::vector<Event> events_to_process;
    
    // Get pending events
    {
        std::lock_guard<std::mutex> lock(event_mutex_);
        events_to_process.swap(pending_events_);
    }
    
    // Process aggregated events
    if (event_config_.enable_event_aggregation) {
        auto aggregated = event_aggregator_->flushAggregated(event_config_.aggregation_window);
        events_to_process.insert(events_to_process.end(), aggregated.begin(), aggregated.end());
        stats_.events_aggregated += aggregated.size();
    }
    
    // Apply filtering and emit events
    size_t emitted_count = 0;
    for (const auto& event : events_to_process) {
        if (shouldFilterEvent(event)) {
            stats_.events_filtered++;
            continue;
        }
        
        emitEvent(event);
        emitted_count++;
        
        if (event_config_.enable_rate_limiting && 
            emitted_count >= event_config_.max_events_per_frame) {
            break;
        }
    }
    
    stats_.events_generated += emitted_count;
}

void PhysicsPipeline::emitEvent(const Event& event) {
    if (event_producer_) {
        event_producer_->writeEvent(event);
    }
}

bool PhysicsPipeline::shouldFilterEvent(const Event& event) const {
    // Check if event type is enabled
    if (event_config_.enabled_events.find(event.type) == event_config_.enabled_events.end()) {
        return true;
    }
    
    // Apply spatial filtering if enabled
    if (event_config_.enable_spatial_filtering) {
        float dx = event.x - event_config_.spatial_filter_x;
        float dy = event.y - event_config_.spatial_filter_y;
        float distance_sq = dx * dx + dy * dy;
        float radius_sq = event_config_.spatial_filter_radius * event_config_.spatial_filter_radius;
        
        if (distance_sq > radius_sq) {
            return true;
        }
    }
    
    return false;
}

void PhysicsPipeline::updateStatistics() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_update_);
    
    if (elapsed.count() > 0) {
        stats_.commands_per_second = stats_.total_commands_processed / elapsed.count();
        stats_.events_per_second = stats_.events_generated / elapsed.count();
        last_stats_update_ = now;
    }
}

void PhysicsPipeline::asyncProcessingLoop() {
    while (processing_active_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        processing_cv_.wait(lock, [this] {
            return !processing_active_.load() || !command_queue_.empty();
        });
        
        if (!processing_active_.load()) break;
        
        // Process commands asynchronously
        // Note: This would need careful coordination with the main thread
        // to avoid race conditions with the simulation state
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void PhysicsPipeline::handleCommandError(const std::string& error, const PipelineCommand& command) {
    stats_.last_error = error;
    stats_.failed_commands_by_type[command.type]++;
    
    if (error_handler_) {
        error_handler_(error, command);
    }
    
    logError("Command failed: " + error);
}

void PhysicsPipeline::logInfo(const std::string& message) {
    std::cout << "[PIPELINE INFO] " << message << std::endl;
}

void PhysicsPipeline::logWarning(const std::string& message) {
    std::cout << "[PIPELINE WARN] " << message << std::endl;
}

void PhysicsPipeline::logError(const std::string& message) {
    std::cerr << "[PIPELINE ERROR] " << message << std::endl;
}

} // namespace digistar