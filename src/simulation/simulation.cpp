#include "simulation.h"
#include "../backend/cpu_backend_reference.h"
#include "../backend/cpu_backend_openmp.h"
#include <iostream>
#include <thread>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

namespace digistar {

// ============ Simulation Implementation ============

Simulation::Simulation(const Config& cfg) : config(cfg) {
    state = State::UNINITIALIZED;
}

Simulation::~Simulation() {
    shutdown();
}

bool Simulation::initialize() {
    try {
        // Create backend
        initializeBackend();
        
        // Create simulation state
        initializeSimulationState();
        
        // Load scenario
        loadScenario(config.scenario_name);
        
        // Initialize renderer if enabled
        if (config.enable_rendering) {
            initializeRenderer();
        }
        
        // Set initial state
        state = config.pause_on_start ? State::PAUSED : State::READY;
        start_time = std::chrono::high_resolution_clock::now();
        last_frame_time = start_time;
        last_checkpoint = start_time;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize simulation: " << e.what() << std::endl;
        state = State::ERROR;
        return false;
    }
}

void Simulation::initializeBackend() {
    // Create appropriate backend
    SimulationConfig backend_config;
    backend_config.max_particles = config.max_particles;
    backend_config.max_springs = config.max_springs;
    backend_config.max_contacts = config.max_contacts;
    backend_config.world_size = config.world_size;
    backend_config.use_toroidal = config.use_toroidal;
    
    switch (config.backend_type) {
        case BackendFactory::Type::CPU:
            backend = std::make_unique<CpuBackendReference>();
            break;
        case BackendFactory::Type::CPU_SIMD:
            backend = std::make_unique<CpuBackendOpenMP>();
            break;
        default:
            throw std::runtime_error("Unsupported backend type");
    }
    
    backend->initialize(backend_config);
}

void Simulation::initializeSimulationState() {
    sim_state = std::make_unique<SimulationState>();
    
    // Allocate pools
    sim_state->particles.allocate(config.max_particles);
    sim_state->springs.allocate(config.max_springs);
    sim_state->contacts.allocate(config.max_contacts);
    sim_state->composites.allocate(config.max_particles / 10, config.max_particles);
    
    // Create spatial indices
    sim_state->contact_index = std::make_unique<GridSpatialIndex>(config.world_size, 2.0f);
    sim_state->spring_index = std::make_unique<GridSpatialIndex>(config.world_size, 10.0f);
    sim_state->thermal_index = std::make_unique<GridSpatialIndex>(config.world_size, 50.0f);
    sim_state->radiation_index = std::make_unique<GridSpatialIndex>(config.world_size, 200.0f);
    
    // Allocate field grids if needed
    if (config.physics.gravity_mode == PhysicsConfig::PARTICLE_MESH) {
        sim_state->gravity.allocate(512);  // PM grid size
    }
    if (config.physics.enabled_systems & PhysicsConfig::RADIATION) {
        sim_state->radiation.allocate(64);
    }
    if (config.physics.enabled_systems & PhysicsConfig::THERMAL) {
        sim_state->thermal.allocate(64);
    }
}

void Simulation::initializeRenderer() {
    renderer = std::make_unique<AsciiRenderer>(config.renderer_config);
    terminal = std::make_unique<TerminalDisplay>();
    
    if (!config.use_ansi_colors) {
        terminal->setColorMode(false);
    }
}

void Simulation::loadScenario(const std::string& name) {
    if (!sim_state) return;
    
    if (name == "solar_system") {
        ScenarioLoader::loadSolarSystem(*sim_state);
    } else if (name == "galaxy_collision") {
        ScenarioLoader::loadGalaxyCollision(*sim_state);
    } else if (name == "asteroid_field") {
        ScenarioLoader::loadAsteroidField(*sim_state);
    } else if (name == "spring_network") {
        ScenarioLoader::loadSpringNetwork(*sim_state);
    } else if (name == "composite_test") {
        ScenarioLoader::loadCompositeTest(*sim_state);
    } else if (name == "stress_test") {
        ScenarioLoader::loadStressTest(*sim_state, config.max_particles / 2);
    } else {
        ScenarioLoader::loadDefault(*sim_state);
    }
}

void Simulation::shutdown() {
    state = State::STOPPED;
    
    if (backend) {
        backend->shutdown();
    }
    
    backend.reset();
    sim_state.reset();
    renderer.reset();
    terminal.reset();
}

void Simulation::reset() {
    shutdown();
    initialize();
}

void Simulation::run() {
    if (state == State::UNINITIALIZED) {
        if (!initialize()) {
            return;
        }
    }
    
    state = State::RUNNING;
    InputHandler input;
    
    while (state != State::STOPPED && !input.shouldQuit()) {
        // Handle input
        auto cmd = input.pollCommand();
        input.processCommand(cmd, *this);
        
        // Step simulation if running
        if (state == State::RUNNING) {
            step();
        }
        
        // Frame rate limiting
        if (config.fixed_timestep) {
            auto now = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration<float>(now - last_frame_time).count();
            float target_frame_time = 1.0f / config.target_fps;
            
            if (frame_duration < target_frame_time) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(
                        int((target_frame_time - frame_duration) * 1000000)
                    )
                );
            }
        }
        
        updateTiming();
    }
}

void Simulation::step() {
    if (!backend || !sim_state) return;
    
    auto step_start = std::chrono::high_resolution_clock::now();
    
    // Physics step
    float dt = config.timestep * time_scale;
    backend->step(*sim_state, config.physics, dt);
    simulation_time += dt;
    
    auto physics_end = std::chrono::high_resolution_clock::now();
    float physics_time = std::chrono::duration<float, std::milli>(physics_end - step_start).count();
    
    // Rendering
    if (renderer && config.enable_rendering) {
        renderer->render(*sim_state, backend->getStats());
        
        if (terminal) {
            terminal->displayFrame(config.use_ansi_colors ? 
                                  renderer->getFrameWithAnsi() : 
                                  renderer->getFrame());
        }
    }
    
    auto render_end = std::chrono::high_resolution_clock::now();
    float render_time = std::chrono::duration<float, std::milli>(render_end - physics_end).count();
    
    // Update performance metrics (moving average)
    avg_physics_time = avg_physics_time * 0.9f + physics_time * 0.1f;
    avg_render_time = avg_render_time * 0.9f + render_time * 0.1f;
    
    frame_count++;
    
    // Checkpointing
    if (config.enable_checkpoints) {
        handleCheckpointing();
    }
}

void Simulation::updateTiming() {
    auto now = std::chrono::high_resolution_clock::now();
    
    // Update FPS
    float frame_time = std::chrono::duration<float>(now - last_frame_time).count();
    current_fps = 1.0f / frame_time;
    last_frame_time = now;
    
    // Update total elapsed time
    real_time_elapsed = std::chrono::duration<float>(now - start_time).count();
}

void Simulation::handleCheckpointing() {
    auto now = std::chrono::high_resolution_clock::now();
    float time_since_checkpoint = std::chrono::duration<float>(now - last_checkpoint).count();
    
    if (time_since_checkpoint >= config.checkpoint_interval) {
        saveCheckpoint();
        last_checkpoint = now;
    }
}

uint32_t Simulation::addParticle(float x, float y, float vx, float vy, float mass, float radius) {
    if (!sim_state) return UINT32_MAX;
    
    return sim_state->particles.create(x, y, vx, vy, mass, radius);
}

void Simulation::removeParticle(uint32_t id) {
    // TODO: Implement particle removal (need to handle springs, etc.)
}

void Simulation::addSpring(uint32_t p1, uint32_t p2, float stiffness, float damping) {
    if (!sim_state) return;
    
    // Calculate rest length from current positions
    float dx = sim_state->particles.pos_x[p2] - sim_state->particles.pos_x[p1];
    float dy = sim_state->particles.pos_y[p2] - sim_state->particles.pos_y[p1];
    float rest_length = sqrtf(dx * dx + dy * dy);
    
    sim_state->springs.create(p1, p2, rest_length, stiffness, damping);
}

void Simulation::breakSpring(uint32_t spring_id) {
    if (!sim_state || spring_id >= sim_state->springs.count) return;
    
    sim_state->springs.break_spring(spring_id);
}

void Simulation::applyForce(uint32_t particle_id, float fx, float fy) {
    if (!sim_state || particle_id >= sim_state->particles.count) return;
    
    sim_state->particles.force_x[particle_id] += fx;
    sim_state->particles.force_y[particle_id] += fy;
}

void Simulation::applyImpulse(uint32_t particle_id, float ix, float iy) {
    if (!sim_state || particle_id >= sim_state->particles.count) return;
    
    // Impulse = change in momentum
    sim_state->particles.vel_x[particle_id] += ix / sim_state->particles.mass[particle_id];
    sim_state->particles.vel_y[particle_id] += iy / sim_state->particles.mass[particle_id];
}

void Simulation::setCameraCenter(float x, float y) {
    if (renderer) {
        renderer->setViewCenter(x, y);
    }
}

void Simulation::setCameraScale(float scale) {
    if (renderer) {
        renderer->setViewScale(scale);
    }
}

void Simulation::trackParticle(int32_t particle_id) {
    if (renderer) {
        renderer->trackParticle(particle_id);
    }
}

bool Simulation::saveState(const std::string& filename) {
    return ScenarioLoader::saveToFile(*sim_state, filename);
}

bool Simulation::loadState(const std::string& filename) {
    return ScenarioLoader::loadFromFile(*sim_state, filename);
}

bool Simulation::saveCheckpoint() {
    std::stringstream filename;
    filename << config.checkpoint_dir << "/checkpoint_" 
             << std::setfill('0') << std::setw(8) << frame_count << ".dat";
    return saveState(filename.str());
}

// ============ ScenarioLoader Implementation ============

void ScenarioLoader::loadDefault(SimulationState& state) {
    // Simple two-body system
    state.particles.create(0, 0, 0, 0, 1e30, 100);      // Central mass
    state.particles.create(1000, 0, 0, 30, 1e24, 50);   // Orbiting body
}

void ScenarioLoader::loadSolarSystem(SimulationState& state) {
    // Sun
    state.particles.create(0, 0, 0, 0, 1.989e30, 696340);
    
    // Planets (simplified circular orbits)
    struct Planet {
        const char* name;
        float distance;  // AU to meters
        float mass;
        float radius;
        float velocity;
    } planets[] = {
        {"Mercury", 57.9e9f, 3.301e23f, 2439.7f, 47.87e3f},
        {"Venus", 108.2e9f, 4.867e24f, 6051.8f, 35.02e3f},
        {"Earth", 149.6e9f, 5.972e24f, 6371.0f, 29.78e3f},
        {"Mars", 227.9e9f, 6.417e23f, 3389.5f, 24.07e3f},
        {"Jupiter", 778.5e9f, 1.898e27f, 69911.0f, 13.07e3f},
        {"Saturn", 1432e9f, 5.683e26f, 58232.0f, 9.69e3f},
        {"Uranus", 2867e9f, 8.681e25f, 25362.0f, 6.81e3f},
        {"Neptune", 4515e9f, 1.024e26f, 24622.0f, 5.43e3f}
    };
    
    for (const auto& p : planets) {
        // Scale down distances for visualization
        float scaled_dist = p.distance / 1e7f;  // Scale factor
        state.particles.create(scaled_dist, 0, 0, p.velocity / 1000.0f, p.mass, p.radius);
    }
    
    // Add some asteroids
    for (int i = 0; i < 100; i++) {
        float angle = (i / 100.0f) * 2.0f * M_PI;
        float dist = 300.0f + (rand() % 100 - 50);  // Asteroid belt region
        float x = dist * cosf(angle);
        float y = dist * sinf(angle);
        float v = sqrtf(6.67e-11f * 1.989e30f / (dist * 1e7f)) / 1000.0f;
        float vx = -v * sinf(angle);
        float vy = v * cosf(angle);
        
        state.particles.create(x, y, vx, vy, 1e16f, 1.0f);
    }
}

void ScenarioLoader::loadAsteroidField(SimulationState& state) {
    // Create a field of asteroids with some initial velocities
    for (int i = 0; i < 500; i++) {
        float x = (rand() % 10000) - 5000;
        float y = (rand() % 10000) - 5000;
        float vx = (rand() % 20) - 10;
        float vy = (rand() % 20) - 10;
        float mass = 1e15f + rand() % (int)1e16f;
        float radius = cbrtf(mass / 1e15f);
        
        uint32_t id = state.particles.create(x, y, vx, vy, mass, radius);
        state.particles.material_type[id] = MATERIAL_ROCK;
    }
}

void ScenarioLoader::loadSpringNetwork(SimulationState& state) {
    // Create a 10x10 grid of particles connected by springs
    const int grid_size = 10;
    const float spacing = 20.0f;
    const float mass = 1.0f;
    const float radius = 2.0f;
    
    uint32_t particles[grid_size][grid_size];
    
    // Create particles
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            float x = (i - grid_size/2) * spacing;
            float y = (j - grid_size/2) * spacing;
            particles[i][j] = state.particles.create(x, y, 0, 0, mass, radius);
        }
    }
    
    // Connect with springs
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            // Right neighbor
            if (i < grid_size - 1) {
                state.springs.create(particles[i][j], particles[i+1][j], 
                                        spacing, 100.0f, 1.0f);
            }
            // Bottom neighbor
            if (j < grid_size - 1) {
                state.springs.create(particles[i][j], particles[i][j+1], 
                                        spacing, 100.0f, 1.0f);
            }
            // Diagonal
            if (i < grid_size - 1 && j < grid_size - 1) {
                state.springs.create(particles[i][j], particles[i+1][j+1], 
                                        spacing * sqrtf(2), 50.0f, 0.5f);
            }
        }
    }
}

void ScenarioLoader::loadCompositeTest(SimulationState& state) {
    // Create two composite bodies that will collide
    
    // First composite - rigid box
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            float x = -100 + i * 10;
            float y = j * 10;
            uint32_t id = state.particles.create(x, y, 20, 0, 10, 3);
            state.particles.material_type[id] = MATERIAL_METAL;
        }
    }
    
    // Connect with stiff springs
    for (size_t i = 0; i < 25; i++) {
        for (size_t j = i + 1; j < 25; j++) {
            float dx = state.particles.pos_x[j] - state.particles.pos_x[i];
            float dy = state.particles.pos_y[j] - state.particles.pos_y[i];
            float dist = sqrtf(dx * dx + dy * dy);
            
            if (dist < 15) {  // Only nearby particles
                state.springs.create(i, j, dist, 500, 5);
            }
        }
    }
    
    // Second composite - soft blob
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            float x = 100 + i * 10;
            float y = j * 10;
            uint32_t id = state.particles.create(x, y, -20, 0, 10, 3);
            state.particles.material_type[id] = MATERIAL_ORGANIC;
        }
    }
    
    // Connect with soft springs
    for (size_t i = 25; i < 50; i++) {
        for (size_t j = i + 1; j < 50; j++) {
            float dx = state.particles.pos_x[j] - state.particles.pos_x[i];
            float dy = state.particles.pos_y[j] - state.particles.pos_y[i];
            float dist = sqrtf(dx * dx + dy * dy);
            
            if (dist < 15) {
                state.springs.create(i, j, dist, 50, 1);
            }
        }
    }
}

void ScenarioLoader::loadGalaxyCollision(SimulationState& state) {
    // Two spiral galaxies approaching each other
    
    auto createGalaxy = [&state](float cx, float cy, float vx, float vy, int num_stars) {
        // Central black hole
        uint32_t center = state.particles.create(cx, cy, vx, vy, 1e36, 10);
        state.particles.material_type[center] = MATERIAL_PLASMA;
        state.particles.temperature[center] = 1e7;  // Very hot
        
        // Spiral arms
        for (int i = 0; i < num_stars; i++) {
            float angle = (i / (float)num_stars) * 4 * M_PI;  // 2 rotations
            float r = 100 + angle * 20;  // Spiral out
            
            // Add some randomness
            r += (rand() % 40 - 20);
            angle += (rand() % 100 - 50) / 100.0f;
            
            float x = cx + r * cosf(angle);
            float y = cy + r * sinf(angle);
            
            // Orbital velocity
            float v_orbit = sqrtf(6.67e-11f * 1e36f / (r * 1e7f)) / 1000.0f;
            float v_tan_x = -v_orbit * sinf(angle) + vx;
            float v_tan_y = v_orbit * cosf(angle) + vy;
            
            uint32_t star = state.particles.create(x, y, v_tan_x, v_tan_y, 
                                                   1e29 + rand() % (int)1e30, 5);
            uint32_t idx = state.particles.get_index(star);
            state.particles.material_type[idx] = MATERIAL_PLASMA;
            state.particles.temperature[idx] = 5000 + rand() % 5000;
        }
    };
    
    // Create two galaxies moving toward each other
    createGalaxy(-2000, 0, 50, 0, 200);
    createGalaxy(2000, 0, -50, 0, 200);
}

void ScenarioLoader::loadStressTest(SimulationState& state, size_t particle_count) {
    // Random particles for performance testing
    for (size_t i = 0; i < particle_count; i++) {
        float x = (rand() % 20000) - 10000;
        float y = (rand() % 20000) - 10000;
        float vx = (rand() % 100) - 50;
        float vy = (rand() % 100) - 50;
        float mass = 1.0f + rand() % 100;
        float radius = 1.0f + rand() % 5;
        
        state.particles.create(x, y, vx, vy, mass, radius);
    }
}

bool ScenarioLoader::loadFromFile(SimulationState& state, const std::string& filename) {
    // TODO: Implement binary serialization
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    
    // Read particle count, spring count, etc.
    // Read particle data
    // Read spring data
    // etc.
    
    return true;
}

bool ScenarioLoader::saveToFile(const SimulationState& state, const std::string& filename) {
    // TODO: Implement binary serialization
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    
    // Write particle count, spring count, etc.
    // Write particle data
    // Write spring data
    // etc.
    
    return true;
}

// ============ InputHandler Implementation ============

InputHandler::InputHandler() {
    setupTerminal();
}

InputHandler::~InputHandler() {
    restoreTerminal();
}

bool InputHandler::setupTerminal() {
    // Set terminal to non-blocking mode
    struct termios tty;
    tcgetattr(STDIN_FILENO, &tty);
    tty.c_lflag &= ~(ICANON | ECHO);  // Disable canonical mode and echo
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);
    
    // Make stdin non-blocking
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    
    return true;
}

void InputHandler::restoreTerminal() {
    // Restore terminal settings
    struct termios tty;
    tcgetattr(STDIN_FILENO, &tty);
    tty.c_lflag |= (ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);
    
    // Restore blocking mode
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
}

char InputHandler::getCharNonBlocking() {
    char c = 0;
    if (read(STDIN_FILENO, &c, 1) == 1) {
        return c;
    }
    return 0;
}

InputHandler::Command InputHandler::pollCommand() {
    char c = getCharNonBlocking();
    if (c == 0) return Command::NONE;
    
    switch (c) {
        // Control
        case 'q': case 'Q': 
            quit_requested = true;
            return Command::QUIT;
        case ' ': return Command::PAUSE;
        case 's': return Command::STEP;
        
        // Camera
        case '+': case '=': return Command::ZOOM_IN;
        case '-': case '_': return Command::ZOOM_OUT;
        case 'h': return Command::PAN_LEFT;
        case 'l': return Command::PAN_RIGHT;
        case 'j': return Command::PAN_DOWN;
        case 'k': return Command::PAN_UP;
        
        // Time
        case '>': case '.': return Command::SPEED_UP;
        case '<': case ',': return Command::SLOW_DOWN;
        case 'r': return Command::RESET_TIME;
        
        // Display toggles
        case 'g': return Command::TOGGLE_GRID;
        case 'S': return Command::TOGGLE_SPRINGS;
        case 'v': return Command::TOGGLE_VELOCITIES;
        case 'f': return Command::TOGGLE_FORCES;
        case 't': return Command::TOGGLE_TEMPERATURE;
        case 'c': return Command::TOGGLE_COMPOSITES;
        case 'i': return Command::TOGGLE_STATS;
        
        // Tracking
        case 'n': return Command::CYCLE_TRACKING;
        
        // File operations
        case 'F': return Command::SAVE_STATE;
        case 'L': return Command::LOAD_STATE;
        
        default: return Command::NONE;
    }
}

void InputHandler::processCommand(Command cmd, Simulation& sim) {
    switch (cmd) {
        case Command::PAUSE:
            if (sim.isPaused()) {
                sim.resume();
            } else {
                sim.pause();
            }
            break;
            
        case Command::STEP:
            if (sim.isPaused()) {
                sim.step();
            }
            break;
            
        case Command::ZOOM_IN:
            sim.zoomIn();
            break;
            
        case Command::ZOOM_OUT:
            sim.zoomOut();
            break;
            
        case Command::PAN_LEFT:
            if (sim.getRenderer()) {
                auto& cfg = sim.getRenderer()->getConfig();
                sim.getRenderer()->pan(-cfg.view_scale * 0.1f, 0);
            }
            break;
            
        case Command::PAN_RIGHT:
            if (sim.getRenderer()) {
                auto& cfg = sim.getRenderer()->getConfig();
                sim.getRenderer()->pan(cfg.view_scale * 0.1f, 0);
            }
            break;
            
        case Command::PAN_UP:
            if (sim.getRenderer()) {
                auto& cfg = sim.getRenderer()->getConfig();
                sim.getRenderer()->pan(0, cfg.view_scale * 0.1f);
            }
            break;
            
        case Command::PAN_DOWN:
            if (sim.getRenderer()) {
                auto& cfg = sim.getRenderer()->getConfig();
                sim.getRenderer()->pan(0, -cfg.view_scale * 0.1f);
            }
            break;
            
        case Command::SPEED_UP:
            sim.setTimeScale(sim.getTimeScale() * 1.5f);
            break;
            
        case Command::SLOW_DOWN:
            sim.setTimeScale(sim.getTimeScale() / 1.5f);
            break;
            
        case Command::RESET_TIME:
            sim.setTimeScale(1.0f);
            break;
            
        case Command::TOGGLE_GRID:
            if (sim.getRenderer()) {
                sim.getRenderer()->toggleGrid();
            }
            break;
            
        case Command::TOGGLE_SPRINGS:
            if (sim.getRenderer()) {
                sim.getRenderer()->toggleSprings();
            }
            break;
            
        case Command::TOGGLE_VELOCITIES:
            if (sim.getRenderer()) {
                sim.getRenderer()->toggleVelocities();
            }
            break;
            
        case Command::TOGGLE_FORCES:
            if (sim.getRenderer()) {
                sim.getRenderer()->toggleForces();
            }
            break;
            
        case Command::TOGGLE_TEMPERATURE:
            if (sim.getRenderer()) {
                sim.getRenderer()->toggleTemperature();
            }
            break;
            
        case Command::TOGGLE_COMPOSITES:
            if (sim.getRenderer()) {
                sim.getRenderer()->toggleComposites();
            }
            break;
            
        case Command::TOGGLE_STATS:
            if (sim.getRenderer()) {
                sim.getRenderer()->toggleStats();
            }
            break;
            
        case Command::CYCLE_TRACKING:
            // TODO: Implement cycling through particles
            break;
            
        case Command::SAVE_STATE:
            sim.saveState("quicksave.dat");
            break;
            
        case Command::LOAD_STATE:
            sim.loadState("quicksave.dat");
            break;
            
        default:
            break;
    }
}

std::string InputHandler::getHelpText() {
    return R"(
=== DigiStar Simulation Controls ===

Movement:
  h/j/k/l  - Pan camera left/down/up/right
  +/-      - Zoom in/out
  n        - Cycle tracking to next particle

Time Control:
  Space    - Pause/Resume
  s        - Single step (when paused)
  >/<      - Speed up/slow down time
  r        - Reset time scale to 1.0

Display Toggles:
  g        - Toggle grid
  S        - Toggle springs
  v        - Toggle velocity vectors
  f        - Toggle force vectors
  t        - Toggle temperature colors
  c        - Toggle composite outlines
  i        - Toggle stats display

File Operations:
  F        - Quick save state
  L        - Load saved state

Control:
  q        - Quit simulation

)";
}

} // namespace digistar