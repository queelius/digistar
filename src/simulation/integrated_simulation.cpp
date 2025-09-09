#include "integrated_simulation.h"
#include "../backend/cpu_backend_simple.h"
#include "../backend/cpu_backend_openmp.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace digistar {

IntegratedSimulation::IntegratedSimulation(const IntegratedSimulationConfig& config)
    : config_(config) {
    
    logInfo("Creating integrated simulation with " + 
           BackendFactory::getDescription(config_.backend_type) + " backend");
}

IntegratedSimulation::~IntegratedSimulation() {
    if (initialized_) {
        shutdown();
    }
}

bool IntegratedSimulation::initialize() {
    if (initialized_) {
        logWarning("Simulation already initialized");
        return true;
    }
    
    logInfo("Initializing integrated simulation...");
    
    try {
        // Initialize backend
        if (!initializeBackend()) {
            handleError("Failed to initialize physics backend", "backend");
            return false;
        }
        
        // Initialize event system
        if (config_.enable_events && !initializeEventSystem()) {
            handleError("Failed to initialize event system", "events");
            if (!config_.continue_on_backend_error) return false;
        }
        
        // Initialize DSL runtime
        if (config_.enable_dsl && !initializeDSLRuntime()) {
            handleError("Failed to initialize DSL runtime", "dsl");
            if (!config_.continue_on_dsl_error) return false;
        }
        
        // Load startup scripts
        if (config_.enable_dsl) {
            loadStartupScripts();
        }
        
        // Initialize statistics
        resetStats();
        
        initialized_ = true;
        logInfo("Integrated simulation initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        handleError(std::string("Initialization failed: ") + e.what(), "init");
        return false;
    }
}

void IntegratedSimulation::shutdown() {
    if (!initialized_) return;
    
    logInfo("Shutting down integrated simulation...");
    
    // Stop execution if running
    if (running_.load()) {
        stop();
    }
    
    // Wait for threads to complete
    if (physics_thread_ && physics_thread_->joinable()) {
        physics_thread_->join();
    }
    if (event_thread_ && event_thread_->joinable()) {
        event_thread_->join();
    }
    
    // Shutdown components in reverse order
    dsl_runtime_.reset();
    event_consumer_.reset();
    event_producer_.reset();
    event_system_.reset();
    
    if (backend_) {
        backend_->shutdown();
        backend_.reset();
    }
    
    initialized_ = false;
    logInfo("Integrated simulation shut down successfully");
}

void IntegratedSimulation::start() {
    if (!initialized_) {
        handleError("Cannot start simulation - not initialized", "lifecycle");
        return;
    }
    
    if (running_.load()) {
        logWarning("Simulation already running");
        return;
    }
    
    logInfo("Starting integrated simulation...");
    
    running_.store(true);
    paused_.store(false);
    stop_requested_.store(false);
    
    simulation_start_time_ = std::chrono::steady_clock::now();
    last_frame_time_ = simulation_start_time_;
    stats_.current_tick = 0;
    stats_.simulation_time = 0.0f;
    
    // Call startup handler
    if (event_handlers_.on_start) {
        event_handlers_.on_start();
    }
    
    // Start threads or run single-threaded
    if (config_.use_separate_physics_thread) {
        physics_thread_ = std::make_unique<std::thread>(&IntegratedSimulation::physicsThreadLoop, this);
        
        if (config_.use_separate_event_thread) {
            event_thread_ = std::make_unique<std::thread>(&IntegratedSimulation::eventThreadLoop, this);
        }
        
        mainUpdateLoop();
    } else {
        singleThreadedUpdate();
    }
}

void IntegratedSimulation::stop() {
    if (!running_.load()) return;
    
    logInfo("Stopping integrated simulation...");
    
    stop_requested_.store(true);
    running_.store(false);
    
    // Call stop handler
    if (event_handlers_.on_stop) {
        event_handlers_.on_stop();
    }
    
    // Join threads
    if (physics_thread_ && physics_thread_->joinable()) {
        physics_thread_->join();
        physics_thread_.reset();
    }
    
    if (event_thread_ && event_thread_->joinable()) {
        event_thread_->join();
        event_thread_.reset();
    }
    
    logInfo("Integrated simulation stopped");
}

void IntegratedSimulation::pause() {
    if (!running_.load() || paused_.load()) return;
    
    paused_.store(true);
    stats_.is_paused = true;
    
    if (event_handlers_.on_pause) {
        event_handlers_.on_pause();
    }
    
    logInfo("Simulation paused");
}

void IntegratedSimulation::resume() {
    if (!running_.load() || !paused_.load()) return;
    
    paused_.store(false);
    stats_.is_paused = false;
    last_frame_time_ = std::chrono::steady_clock::now();  // Reset timing
    
    if (event_handlers_.on_resume) {
        event_handlers_.on_resume();
    }
    
    logInfo("Simulation resumed");
}

void IntegratedSimulation::step() {
    if (!initialized_ || !running_.load() || !paused_.load()) return;
    
    float dt = calculateDeltaTime();
    updatePhysics(dt);
    
    if (config_.enable_dsl) {
        updateDSL(dt);
    }
    
    if (config_.enable_events) {
        processEvents();
    }
    
    updateStatistics();
    stats_.current_tick++;
}

bool IntegratedSimulation::initializeBackend() {
    try {
        backend_ = BackendFactory::create(config_.backend_type, config_.simulation_config);
        if (!backend_) {
            return false;
        }
        
        backend_->initialize(config_.simulation_config);
        
        // Set up event producer if events are enabled
        if (config_.enable_events && event_producer_) {
            backend_->setEventProducer(event_producer_);
        }
        
        logInfo("Backend initialized: " + backend_->getName());
        return true;
        
    } catch (const std::exception& e) {
        handleError(std::string("Backend initialization failed: ") + e.what(), "backend");
        return false;
    }
}

bool IntegratedSimulation::initializeEventSystem() {
    try {
        event_system_ = std::make_unique<SharedMemoryEventSystem>(
            config_.event_shm_name, config_.auto_create_event_system);
        
        if (!event_system_->is_valid()) {
            return false;
        }
        
        // Create event producer and consumer
        event_producer_ = std::make_unique<EventProducer>(event_system_->get_buffer());
        event_consumer_ = std::make_unique<EventConsumer>(event_system_->get_buffer(), "integrated_sim");
        
        logInfo("Event system initialized with shared memory: " + config_.event_shm_name);
        return true;
        
    } catch (const std::exception& e) {
        handleError(std::string("Event system initialization failed: ") + e.what(), "events");
        return false;
    }
}

bool IntegratedSimulation::initializeDSLRuntime() {
    try {
        dsl_runtime_ = std::make_unique<dsl::DSLRuntime>(&simulation_state_, config_.event_shm_name);
        
        logInfo("DSL runtime initialized");
        return true;
        
    } catch (const std::exception& e) {
        handleError(std::string("DSL runtime initialization failed: ") + e.what(), "dsl");
        return false;
    }
}

void IntegratedSimulation::loadStartupScripts() {
    for (const auto& script : config_.startup_scripts) {
        try {
            loadScriptFile("startup_" + std::to_string(stats_.current_tick), script);
            runScript("startup_" + std::to_string(stats_.current_tick));
        } catch (const std::exception& e) {
            handleError("Failed to load startup script '" + script + "': " + e.what(), "dsl");
        }
    }
}

void IntegratedSimulation::physicsThreadLoop() {
    while (running_.load() && !stop_requested_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        float dt = calculateDeltaTime();
        updatePhysics(dt);
    }
}

void IntegratedSimulation::eventThreadLoop() {
    while (running_.load() && !stop_requested_.load()) {
        processEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void IntegratedSimulation::mainUpdateLoop() {
    while (running_.load() && !stop_requested_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        auto frame_start = std::chrono::steady_clock::now();
        
        float dt = calculateDeltaTime();
        
        // DSL update (main thread)
        if (config_.enable_dsl) {
            auto dsl_start = std::chrono::steady_clock::now();
            updateDSL(dt);
            auto dsl_end = std::chrono::steady_clock::now();
            stats_.dsl_time = std::chrono::duration_cast<std::chrono::microseconds>(dsl_end - dsl_start);
        }
        
        // Event processing (if not on separate thread)
        if (config_.enable_events && !config_.use_separate_event_thread) {
            auto event_start = std::chrono::steady_clock::now();
            processEvents();
            auto event_end = std::chrono::steady_clock::now();
            stats_.event_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(event_end - event_start);
        }
        
        updateStatistics();
        
        if (event_handlers_.on_frame_complete) {
            event_handlers_.on_frame_complete(dt);
        }
        
        auto frame_end = std::chrono::steady_clock::now();
        stats_.total_frame_time = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
        
        limitFrameRate();
        stats_.current_tick++;
    }
}

void IntegratedSimulation::singleThreadedUpdate() {
    while (running_.load() && !stop_requested_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        auto frame_start = std::chrono::steady_clock::now();
        
        float dt = calculateDeltaTime();
        
        // Physics update
        auto physics_start = std::chrono::steady_clock::now();
        updatePhysics(dt);
        auto physics_end = std::chrono::steady_clock::now();
        stats_.physics_time = std::chrono::duration_cast<std::chrono::microseconds>(physics_end - physics_start);
        
        // DSL update
        if (config_.enable_dsl) {
            auto dsl_start = std::chrono::steady_clock::now();
            updateDSL(dt);
            auto dsl_end = std::chrono::steady_clock::now();
            stats_.dsl_time = std::chrono::duration_cast<std::chrono::microseconds>(dsl_end - dsl_start);
        }
        
        // Event processing
        if (config_.enable_events) {
            auto event_start = std::chrono::steady_clock::now();
            processEvents();
            auto event_end = std::chrono::steady_clock::now();
            stats_.event_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(event_end - event_start);
        }
        
        updateStatistics();
        
        if (event_handlers_.on_frame_complete) {
            event_handlers_.on_frame_complete(dt);
        }
        
        auto frame_end = std::chrono::steady_clock::now();
        stats_.total_frame_time = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
        
        limitFrameRate();
        stats_.current_tick++;
    }
}

void IntegratedSimulation::updatePhysics(float dt) {
    if (!backend_) return;
    
    try {
        // Set simulation context for event timing
        if (backend_->supportsEvents()) {
            backend_->setSimulationContext(stats_.current_tick, stats_.simulation_time);
        }
        
        // Perform physics step
        backend_->step(simulation_state_, config_.physics_config, dt);
        
        // Update backend statistics
        stats_.backend_stats = backend_->getStats();
        stats_.simulation_time += dt;
        
    } catch (const std::exception& e) {
        stats_.backend_errors++;
        handleError("Physics update failed: " + std::string(e.what()), "physics");
    }
}

void IntegratedSimulation::updateDSL(float dt) {
    if (!dsl_runtime_) return;
    
    try {
        dsl_runtime_->update(dt);
        dsl_runtime_->processScheduledScripts();
        
        // Update DSL statistics
        stats_.dsl_stats = dsl_runtime_->getPerformance();
        
    } catch (const std::exception& e) {
        stats_.dsl_errors++;
        handleError("DSL update failed: " + std::string(e.what()), "dsl");
    }
}

void IntegratedSimulation::processEvents() {
    if (!event_consumer_) return;
    
    try {
        Event event;
        while (event_consumer_->tryRead(event)) {
            // Process event (could forward to DSL, logging, etc.)
            // For now, just count processed events
        }
        
        // Update event statistics
        if (event_system_) {
            stats_.event_stats = event_system_->get_stats();
        }
        
    } catch (const std::exception& e) {
        stats_.event_errors++;
        handleError("Event processing failed: " + std::string(e.what()), "events");
    }
}

void IntegratedSimulation::updateStatistics() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_frame_time_);
    
    stats_.total_frames++;
    stats_.is_running = running_.load();
    stats_.is_paused = paused_.load();
    
    // Calculate FPS
    if (elapsed.count() > 0) {
        stats_.current_fps = 1000000.0f / elapsed.count();
        
        // Calculate average FPS
        auto total_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - simulation_start_time_);
        if (total_elapsed.count() > 0) {
            stats_.average_fps = (stats_.total_frames * 1000000.0f) / total_elapsed.count();
        }
    }
    
    last_frame_time_ = now;
    
    // Call statistics handler
    if (event_handlers_.on_stats_update) {
        event_handlers_.on_stats_update(stats_);
    }
}

float IntegratedSimulation::calculateDeltaTime() {
    if (config_.adaptive_timestep) {
        // Use actual elapsed time with limits
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_frame_time_);
        float dt = elapsed.count() / 1000000.0f;  // Convert to seconds
        
        return std::clamp(dt, config_.min_dt, config_.max_dt);
    } else {
        // Fixed timestep
        return 1.0f / config_.target_fps;
    }
}

void IntegratedSimulation::limitFrameRate() {
    if (config_.target_fps <= 0.0f) return;
    
    auto frame_duration = std::chrono::microseconds(static_cast<int64_t>(1000000.0f / config_.target_fps));
    auto sleep_until = last_frame_time_ + frame_duration;
    auto now = std::chrono::steady_clock::now();
    
    if (now < sleep_until) {
        std::this_thread::sleep_until(sleep_until);
    }
}

void IntegratedSimulation::setEventHandlers(const SimulationEventHandlers& handlers) {
    event_handlers_ = handlers;
}

void IntegratedSimulation::loadScript(const std::string& name, const std::string& source) {
    if (!dsl_runtime_) return;
    
    try {
        dsl_runtime_->loadScript(name, source);
        logInfo("Loaded DSL script: " + name);
    } catch (const std::exception& e) {
        handleError("Failed to load script '" + name + "': " + e.what(), "dsl");
    }
}

void IntegratedSimulation::loadScriptFile(const std::string& name, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        handleError("Cannot open script file: " + filename, "dsl");
        return;
    }
    
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    loadScript(name, source);
}

void IntegratedSimulation::runScript(const std::string& name) {
    if (!dsl_runtime_) return;
    
    try {
        dsl_runtime_->runScript(name);
    } catch (const std::exception& e) {
        handleError("Failed to run script '" + name + "': " + e.what(), "dsl");
    }
}

void IntegratedSimulation::scheduleScript(const std::string& name, std::chrono::milliseconds interval) {
    if (!dsl_runtime_) return;
    
    try {
        dsl_runtime_->scheduleScript(name, interval);
    } catch (const std::exception& e) {
        handleError("Failed to schedule script '" + name + "': " + e.what(), "dsl");
    }
}

void IntegratedSimulation::handleError(const std::string& error, const std::string& component) {
    stats_.last_error = error;
    
    logError("[" + component + "] " + error);
    
    if (error_handler_) {
        error_handler_(error);
    }
    
    if (event_handlers_.on_error) {
        event_handlers_.on_error(error);
    }
}

void IntegratedSimulation::resetStats() {
    stats_ = IntegratedSimulationStats{};
    stats_.is_running = running_.load();
    stats_.is_paused = paused_.load();
}

std::string IntegratedSimulation::getPerformanceReport() const {
    std::stringstream ss;
    
    ss << "=== DigiStar Integrated Simulation Performance Report ===\n";
    ss << std::fixed << std::setprecision(2);
    
    // General statistics
    ss << "\nGeneral:\n";
    ss << "  Status: " << (stats_.is_running ? (stats_.is_paused ? "Paused" : "Running") : "Stopped") << "\n";
    ss << "  Total Frames: " << stats_.total_frames << "\n";
    ss << "  Current FPS: " << stats_.current_fps << "\n";
    ss << "  Average FPS: " << stats_.average_fps << "\n";
    ss << "  Simulation Time: " << stats_.simulation_time << "s\n";
    ss << "  Current Tick: " << stats_.current_tick << "\n";
    
    // Timing breakdown
    ss << "\nFrame Timing (μs):\n";
    ss << "  Total Frame: " << stats_.total_frame_time.count() << "\n";
    ss << "  Physics: " << stats_.physics_time.count() << "\n";
    ss << "  DSL: " << stats_.dsl_time.count() << "\n";
    ss << "  Events: " << stats_.event_processing_time.count() << "\n";
    
    // Backend statistics
    if (backend_) {
        ss << "\nBackend (" << backend_->getName() << "):\n";
        ss << "  Active Particles: " << stats_.backend_stats.active_particles << "\n";
        ss << "  Active Springs: " << stats_.backend_stats.active_springs << "\n";
        ss << "  Active Contacts: " << stats_.backend_stats.active_contacts << "\n";
        ss << "  Update Time: " << stats_.backend_stats.update_time_ms << "ms\n";
        ss << "  Total Energy: " << stats_.backend_stats.total_energy << "\n";
    }
    
    // Error statistics
    ss << "\nErrors:\n";
    ss << "  Backend Errors: " << stats_.backend_errors << "\n";
    ss << "  DSL Errors: " << stats_.dsl_errors << "\n";
    ss << "  Event Errors: " << stats_.event_errors << "\n";
    if (!stats_.last_error.empty()) {
        ss << "  Last Error: " << stats_.last_error << "\n";
    }
    
    return ss.str();
}

void IntegratedSimulation::logInfo(const std::string& message) {
    std::cout << "[INFO] " << message << std::endl;
}

void IntegratedSimulation::logWarning(const std::string& message) {
    std::cout << "[WARN] " << message << std::endl;
}

void IntegratedSimulation::logError(const std::string& message) {
    std::cerr << "[ERROR] " << message << std::endl;
}

} // namespace digistar