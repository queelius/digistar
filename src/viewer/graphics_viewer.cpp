#include "graphics_viewer.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>

namespace digistar {

GraphicsViewer::~GraphicsViewer() {
    shutdown();
}

bool GraphicsViewer::initialize(const std::string& title, int width, int height, bool fullscreen) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) < 0) {
        printf("SDL initialization failed: %s\n", SDL_GetError());
        return false;
    }
    
    // Store window properties
    window_width = width;
    window_height = height;
    is_fullscreen = fullscreen;
    
    // Create window
    Uint32 window_flags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
    if (fullscreen) {
        window_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
    }
    
    window = SDL_CreateWindow(title.c_str(),
                             SDL_WINDOWPOS_CENTERED,
                             SDL_WINDOWPOS_CENTERED,
                             width, height, window_flags);
    
    if (!window) {
        printf("Window creation failed: %s\n", SDL_GetError());
        SDL_Quit();
        return false;
    }
    
    // Create renderer with hardware acceleration and vsync
    Uint32 render_flags = SDL_RENDERER_ACCELERATED;
    if (perf_settings.enable_vsync) {
        render_flags |= SDL_RENDERER_PRESENTVSYNC;
    }
    
    renderer = SDL_CreateRenderer(window, -1, render_flags);
    if (!renderer) {
        printf("Renderer creation failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return false;
    }
    
    // Set blend mode for transparency
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    
    // Build color lookup tables
    buildColorLookupTables();
    
    printf("Graphics viewer initialized: %dx%d, %s\n", 
           width, height, fullscreen ? "fullscreen" : "windowed");
    
    return true;
}

void GraphicsViewer::shutdown() {
    if (particle_texture) {
        SDL_DestroyTexture(particle_texture);
        particle_texture = nullptr;
    }
    
    if (renderer) {
        SDL_DestroyRenderer(renderer);
        renderer = nullptr;
    }
    
    if (window) {
        SDL_DestroyWindow(window);
        window = nullptr;
    }
    
    SDL_Quit();
}

bool GraphicsViewer::processEvents() {
    SDL_Event event;
    
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                is_running = false;
                return false;
                
            case SDL_KEYDOWN:
                input_state.keys[event.key.keysym.scancode] = true;
                handleKeyboardInput();
                break;
                
            case SDL_KEYUP:
                input_state.keys[event.key.keysym.scancode] = false;
                break;
                
            case SDL_MOUSEBUTTONDOWN:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    input_state.mouse_left_down = true;
                    input_state.mouse_start_x = event.button.x;
                    input_state.mouse_start_y = event.button.y;
                }
                if (event.button.button == SDL_BUTTON_RIGHT) {
                    input_state.mouse_right_down = true;
                }
                if (event.button.button == SDL_BUTTON_MIDDLE) {
                    input_state.mouse_middle_down = true;
                }
                break;
                
            case SDL_MOUSEBUTTONUP:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    input_state.mouse_left_down = false;
                }
                if (event.button.button == SDL_BUTTON_RIGHT) {
                    input_state.mouse_right_down = false;
                }
                if (event.button.button == SDL_BUTTON_MIDDLE) {
                    input_state.mouse_middle_down = false;
                }
                break;
                
            case SDL_MOUSEMOTION:
                input_state.mouse_x = event.motion.x;
                input_state.mouse_y = event.motion.y;
                handleMouseInput();
                break;
                
            case SDL_MOUSEWHEEL:
                input_state.scroll_delta = event.wheel.y;
                
                // Zoom with mouse wheel
                if (input_state.scroll_delta != 0) {
                    float zoom_factor = 1.0f + (input_state.scroll_delta * 0.1f);
                    setZoom(camera.zoom * zoom_factor);
                }
                break;
                
            case SDL_WINDOWEVENT:
                if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                    window_width = event.window.data1;
                    window_height = event.window.data2;
                }
                break;
        }
    }
    
    return is_running;
}

void GraphicsViewer::beginFrame() {
    // Clear screen with dark background
    SDL_SetRenderDrawColor(renderer, 10, 10, 15, 255);
    SDL_RenderClear(renderer);
}

void GraphicsViewer::renderSimulation(const SimulationState& state) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Update camera if following a particle or auto-centering
    updateCamera(state);
    
    // Render based on current mode
    switch (render_mode) {
        case RenderMode::PARTICLES_ONLY:
            renderParticles(state.particles);
            break;
            
        case RenderMode::PARTICLES_AND_SPRINGS:
            renderSprings(state.particles, state.springs);
            renderParticles(state.particles);
            break;
            
        case RenderMode::COMPOSITE_HIGHLIGHT:
            renderParticles(state.particles);
            renderComposites(state.particles, state.composites);
            break;
            
        case RenderMode::DEBUG_SPATIAL_GRID:
            renderSpatialGrid(state);
            renderParticles(state.particles);
            break;
            
        case RenderMode::HEAT_MAP:
        case RenderMode::VELOCITY_VECTORS:
        case RenderMode::FORCE_VECTORS:
            // TODO: Implement specialized rendering modes
            renderParticles(state.particles);
            break;
    }
    
    // Render visual events
    renderEvents();
    
    // Update performance stats
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    performance_stats.render_time_ms = duration.count() / 1000.0f;
}

void GraphicsViewer::renderUI(const SimulationState& state) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (ui_settings.show_performance_overlay) {
        renderPerformanceOverlay(state);
    }
    
    if (ui_settings.show_simulation_stats) {
        renderSimulationStats(state);
    }
    
    if (ui_settings.show_particle_info) {
        renderParticleInfo(state);
    }
    
    if (ui_settings.show_controls_help) {
        // Render control help overlay
        renderText("Controls:", 10, window_height - 200, {255, 255, 255, 200});
        renderText("  WASD/Arrow Keys: Pan camera", 10, window_height - 180, {200, 200, 200, 200});
        renderText("  Mouse Wheel: Zoom", 10, window_height - 160, {200, 200, 200, 200});
        renderText("  Left Click+Drag: Pan", 10, window_height - 140, {200, 200, 200, 200});
        renderText("  Space: Center on center of mass", 10, window_height - 120, {200, 200, 200, 200});
        renderText("  F: Toggle fullscreen", 10, window_height - 100, {200, 200, 200, 200});
        renderText("  H: Toggle this help", 10, window_height - 80, {200, 200, 200, 200});
        renderText("  ESC: Exit", 10, window_height - 60, {200, 200, 200, 200});
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    performance_stats.ui_time_ms = duration.count() / 1000.0f;
}

void GraphicsViewer::presentFrame() {
    SDL_RenderPresent(renderer);
}

void GraphicsViewer::setCameraPosition(float x, float y) {
    camera.x = x;
    camera.y = y;
    camera.following_particle = false;
}

void GraphicsViewer::setZoom(float zoom) {
    camera.zoom = std::max(camera.min_zoom, std::min(camera.max_zoom, zoom));
}

void GraphicsViewer::followParticle(size_t particle_id) {
    camera.follow_particle_id = particle_id;
    camera.following_particle = true;
    camera.auto_center = false;
}

void GraphicsViewer::stopFollowing() {
    camera.following_particle = false;
}

void GraphicsViewer::centerOnCenterOfMass(const SimulationState& state) {
    if (state.particles.count == 0) return;
    
    // Calculate center of mass
    float total_mass = 0.0f;
    float com_x = 0.0f, com_y = 0.0f;
    
    for (size_t i = 0; i < state.particles.count; i++) {
        float mass = state.particles.mass[i];
        com_x += state.particles.pos_x[i] * mass;
        com_y += state.particles.pos_y[i] * mass;
        total_mass += mass;
    }
    
    if (total_mass > 0) {
        camera.x = com_x / total_mass;
        camera.y = com_y / total_mass;
    }
}

void GraphicsViewer::worldToScreen(float world_x, float world_y, int& screen_x, int& screen_y) const {
    // Transform world coordinates to screen coordinates
    float rel_x = (world_x - camera.x) * camera.zoom;
    float rel_y = (world_y - camera.y) * camera.zoom;
    
    screen_x = static_cast<int>(window_width / 2 + rel_x);
    screen_y = static_cast<int>(window_height / 2 - rel_y);  // Flip Y axis
}

void GraphicsViewer::screenToWorld(int screen_x, int screen_y, float& world_x, float& world_y) const {
    // Transform screen coordinates to world coordinates
    float rel_x = (screen_x - window_width / 2.0f) / camera.zoom;
    float rel_y = (window_height / 2.0f - screen_y) / camera.zoom;  // Flip Y axis
    
    world_x = camera.x + rel_x;
    world_y = camera.y + rel_y;
}

void GraphicsViewer::addVisualEvent(const VisualEvent& event) {
    active_events.push_back(event);
}

bool GraphicsViewer::isKeyPressed(SDL_Scancode key) const {
    return input_state.keys[key];
}

void GraphicsViewer::getMousePosition(int& x, int& y) const {
    x = input_state.mouse_x;
    y = input_state.mouse_y;
}

void GraphicsViewer::getMouseWorldPosition(float& x, float& y) const {
    screenToWorld(input_state.mouse_x, input_state.mouse_y, x, y);
}

bool GraphicsViewer::isMouseButtonPressed(int button) const {
    switch (button) {
        case SDL_BUTTON_LEFT: return input_state.mouse_left_down;
        case SDL_BUTTON_RIGHT: return input_state.mouse_right_down;
        case SDL_BUTTON_MIDDLE: return input_state.mouse_middle_down;
        default: return false;
    }
}

void GraphicsViewer::setWindowTitle(const std::string& title) {
    if (window) {
        SDL_SetWindowTitle(window, title.c_str());
    }
}

void GraphicsViewer::toggleFullscreen() {
    if (window) {
        is_fullscreen = !is_fullscreen;
        if (is_fullscreen) {
            SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN_DESKTOP);
        } else {
            SDL_SetWindowFullscreen(window, 0);
        }
    }
}

// Private implementation methods

void GraphicsViewer::renderParticles(const ParticlePool& particles) {
    performance_stats.particles_rendered = 0;
    
    // Early exit if no particles
    if (particles.count == 0) return;
    
    // Calculate screen bounds for culling
    float min_world_x, min_world_y, max_world_x, max_world_y;
    screenToWorld(0, 0, min_world_x, max_world_y);
    screenToWorld(window_width, window_height, max_world_x, min_world_y);
    
    // Expand bounds slightly to account for particle radius
    float cull_margin = 100.0f / camera.zoom;  // Margin in world units
    min_world_x -= cull_margin;
    min_world_y -= cull_margin;
    max_world_x += cull_margin;
    max_world_y += cull_margin;
    
    // Render particles
    for (size_t i = 0; i < particles.count; i++) {
        float x = particles.pos_x[i];
        float y = particles.pos_y[i];
        float radius = particles.radius[i];
        
        // Frustum culling
        if (x < min_world_x || x > max_world_x || y < min_world_y || y > max_world_y) {
            continue;
        }
        
        // Get screen position
        int screen_x, screen_y;
        worldToScreen(x, y, screen_x, screen_y);
        
        // Skip if off screen
        if (screen_x < -50 || screen_x > window_width + 50 || 
            screen_y < -50 || screen_y > window_height + 50) {
            continue;
        }
        
        // Calculate screen radius
        int screen_radius = std::max(1, static_cast<int>(radius * camera.zoom));
        
        // Get particle color
        SDL_Color color = getParticleColor(i, particles);
        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
        
        // Render based on style
        switch (particle_style) {
            case ParticleStyle::POINTS:
                SDL_RenderDrawPoint(renderer, screen_x, screen_y);
                break;
                
            case ParticleStyle::CIRCLES:
                // Draw filled circle (simplified - could use better circle algorithm)
                for (int dy = -screen_radius; dy <= screen_radius; dy++) {
                    for (int dx = -screen_radius; dx <= screen_radius; dx++) {
                        if (dx*dx + dy*dy <= screen_radius*screen_radius) {
                            SDL_RenderDrawPoint(renderer, screen_x + dx, screen_y + dy);
                        }
                    }
                }
                break;
                
            case ParticleStyle::CIRCLES_OUTLINED:
                // Draw filled circle
                for (int dy = -screen_radius; dy <= screen_radius; dy++) {
                    for (int dx = -screen_radius; dx <= screen_radius; dx++) {
                        if (dx*dx + dy*dy <= screen_radius*screen_radius) {
                            SDL_RenderDrawPoint(renderer, screen_x + dx, screen_y + dy);
                        }
                    }
                }
                // Draw outline in contrasting color
                SDL_SetRenderDrawColor(renderer, 255 - color.r, 255 - color.g, 255 - color.b, 255);
                // Simple circle outline (could be improved)
                for (int angle = 0; angle < 360; angle += 5) {
                    int dx = static_cast<int>(screen_radius * cos(angle * M_PI / 180.0));
                    int dy = static_cast<int>(screen_radius * sin(angle * M_PI / 180.0));
                    SDL_RenderDrawPoint(renderer, screen_x + dx, screen_y + dy);
                }
                break;
                
            case ParticleStyle::SPRITES:
                // TODO: Implement sprite rendering
                SDL_RenderDrawPoint(renderer, screen_x, screen_y);
                break;
        }
        
        performance_stats.particles_rendered++;
        
        // Limit rendering for performance
        if (performance_stats.particles_rendered >= perf_settings.max_rendered_particles) {
            break;
        }
    }
}

void GraphicsViewer::renderSprings(const ParticlePool& particles, const SpringPool& springs) {
    performance_stats.springs_rendered = 0;
    
    if (springs.count == 0) return;
    
    // Set spring color (could be based on stress, etc.)
    SDL_SetRenderDrawColor(renderer, 100, 150, 255, 128);  // Semi-transparent blue
    
    for (size_t i = 0; i < springs.count; i++) {
        size_t p1_id = springs.particle1_id[i];
        size_t p2_id = springs.particle2_id[i];
        
        // Bounds check
        if (p1_id >= particles.count || p2_id >= particles.count) {
            continue;
        }
        
        // Get world positions
        float x1 = particles.pos_x[p1_id];
        float y1 = particles.pos_y[p1_id];
        float x2 = particles.pos_x[p2_id];
        float y2 = particles.pos_y[p2_id];
        
        // Convert to screen coordinates
        int screen_x1, screen_y1, screen_x2, screen_y2;
        worldToScreen(x1, y1, screen_x1, screen_y1);
        worldToScreen(x2, y2, screen_x2, screen_y2);
        
        // Cull springs that are completely off screen
        if ((screen_x1 < 0 && screen_x2 < 0) || 
            (screen_x1 > window_width && screen_x2 > window_width) ||
            (screen_y1 < 0 && screen_y2 < 0) ||
            (screen_y1 > window_height && screen_y2 > window_height)) {
            continue;
        }
        
        // Draw line
        SDL_RenderDrawLine(renderer, screen_x1, screen_y1, screen_x2, screen_y2);
        
        performance_stats.springs_rendered++;
    }
}

void GraphicsViewer::renderComposites(const ParticlePool& particles, const CompositePool& composites) {
    // TODO: Implement composite body highlighting
    // Could draw bounding boxes, convex hulls, or special particle colors
}

void GraphicsViewer::renderSpatialGrid(const SimulationState& state) {
    // TODO: Implement spatial grid visualization
    // Show the different resolution grids used for contact, spring, thermal, radiation
}

void GraphicsViewer::renderEvents() {
    for (auto& event : active_events) {
        // Calculate fade factor based on remaining time
        float fade = event.time_remaining / event.duration;
        
        // Get screen position
        int screen_x, screen_y;
        worldToScreen(event.x, event.y, screen_x, screen_y);
        
        // Skip if off screen
        if (screen_x < -100 || screen_x > window_width + 100 || 
            screen_y < -100 || screen_y > window_height + 100) {
            continue;
        }
        
        // Calculate screen radius
        float progress = 1.0f - fade;
        int radius = static_cast<int>((event.max_radius * progress * camera.zoom));
        
        // Set color with fade
        SDL_Color color = event.color;
        color.a = static_cast<Uint8>(255 * fade * event.intensity);
        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
        
        // Render based on event type
        switch (event.type) {
            case VisualEvent::EXPLOSION:
            case VisualEvent::COLLISION:
                // Draw expanding circle
                for (int angle = 0; angle < 360; angle += 10) {
                    int dx = static_cast<int>(radius * cos(angle * M_PI / 180.0));
                    int dy = static_cast<int>(radius * sin(angle * M_PI / 180.0));
                    SDL_RenderDrawPoint(renderer, screen_x + dx, screen_y + dy);
                }
                break;
                
            case VisualEvent::SPRING_BREAK:
                // Draw X pattern
                SDL_RenderDrawLine(renderer, screen_x - radius, screen_y - radius,
                                 screen_x + radius, screen_y + radius);
                SDL_RenderDrawLine(renderer, screen_x + radius, screen_y - radius,
                                 screen_x - radius, screen_y + radius);
                break;
                
            case VisualEvent::COMPOSITE_FORMATION:
                // Draw plus pattern
                SDL_RenderDrawLine(renderer, screen_x - radius, screen_y,
                                 screen_x + radius, screen_y);
                SDL_RenderDrawLine(renderer, screen_x, screen_y - radius,
                                 screen_x, screen_y + radius);
                break;
                
            default:
                SDL_RenderDrawPoint(renderer, screen_x, screen_y);
                break;
        }
    }
    
    // Update events and remove expired ones
    updateEvents(1.0f / 60.0f);  // Assume 60 FPS for simplicity
}

void GraphicsViewer::renderPerformanceOverlay(const SimulationState& state) {
    int y_offset = 10;
    int x_offset = (ui_settings.overlay_position % 2 == 0) ? 10 : window_width - 200;
    
    SDL_Color text_color = {255, 255, 255, static_cast<Uint8>(255 * ui_settings.overlay_alpha)};
    
    renderTextf(x_offset, y_offset, text_color, "FPS: %.1f", performance_stats.fps);
    y_offset += 20;
    
    renderTextf(x_offset, y_offset, text_color, "Frame: %.2f ms", performance_stats.frame_time_ms);
    y_offset += 20;
    
    renderTextf(x_offset, y_offset, text_color, "Render: %.2f ms", performance_stats.render_time_ms);
    y_offset += 20;
    
    renderTextf(x_offset, y_offset, text_color, "UI: %.2f ms", performance_stats.ui_time_ms);
    y_offset += 20;
    
    renderTextf(x_offset, y_offset, text_color, "Particles: %zu", performance_stats.particles_rendered);
    y_offset += 20;
    
    renderTextf(x_offset, y_offset, text_color, "Springs: %zu", performance_stats.springs_rendered);
    y_offset += 20;
    
    renderTextf(x_offset, y_offset, text_color, "Zoom: %.3f", camera.zoom);
}

void GraphicsViewer::renderSimulationStats(const SimulationState& state) {
    int y_offset = window_height - 150;
    int x_offset = 10;
    
    SDL_Color text_color = {200, 255, 200, static_cast<Uint8>(255 * ui_settings.overlay_alpha)};
    
    renderTextf(x_offset, y_offset, text_color, "Total Particles: %zu", state.particles.count);
    y_offset += 20;
    
    renderTextf(x_offset, y_offset, text_color, "Active Springs: %zu", state.springs.count);
    y_offset += 20;
    
    renderTextf(x_offset, y_offset, text_color, "Active Contacts: %zu", state.contacts.count);
    y_offset += 20;
    
    renderTextf(x_offset, y_offset, text_color, "Total Energy: %.2e", state.stats.total_energy);
    y_offset += 20;
    
    renderTextf(x_offset, y_offset, text_color, "Max Velocity: %.2f", state.stats.max_velocity);
}

void GraphicsViewer::renderParticleInfo(const SimulationState& state) {
    // TODO: Show detailed information about clicked/selected particle
}

void GraphicsViewer::buildColorLookupTables() {
    // Build color lookup tables for different schemes
    mass_colors.clear();
    velocity_colors.clear(); 
    temperature_colors.clear();
    
    // Mass-based colors (blue to red)
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        mass_colors.push_back({
            static_cast<Uint8>(255 * t),     // Red increases
            static_cast<Uint8>(100),         // Green constant
            static_cast<Uint8>(255 * (1-t)), // Blue decreases
            255
        });
    }
    
    // Velocity-based colors (green to yellow to red)
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        if (t < 0.5f) {
            // Green to yellow
            float local_t = t * 2.0f;
            velocity_colors.push_back({
                static_cast<Uint8>(255 * local_t), // Red increases
                255,                               // Green constant
                0,                                // Blue zero
                255
            });
        } else {
            // Yellow to red
            float local_t = (t - 0.5f) * 2.0f;
            velocity_colors.push_back({
                255,                                    // Red constant
                static_cast<Uint8>(255 * (1-local_t)), // Green decreases
                0,                                     // Blue zero
                255
            });
        }
    }
    
    // Temperature-based colors (black to white hot)
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        Uint8 intensity = static_cast<Uint8>(255 * t);
        temperature_colors.push_back({intensity, intensity, intensity, 255});
    }
}

SDL_Color GraphicsViewer::getParticleColor(size_t particle_id, const ParticlePool& particles) const {
    switch (color_scheme) {
        case ColorScheme::DEFAULT:
            return {255, 255, 255, 255};  // White
            
        case ColorScheme::MASS_BASED:
            {
                float mass = particles.mass[particle_id];
                // Normalize mass to 0-255 range (could use better normalization)
                int index = std::min(255, static_cast<int>(mass * 10));
                return mass_colors[index];
            }
            
        case ColorScheme::VELOCITY_BASED:
            {
                float vx = particles.vel_x[particle_id];
                float vy = particles.vel_y[particle_id];
                float speed = sqrt(vx*vx + vy*vy);
                int index = std::min(255, static_cast<int>(speed * 5));
                return velocity_colors[index];
            }
            
        case ColorScheme::TEMPERATURE_BASED:
            {
                float temp = particles.temperature[particle_id];
                int index = std::min(255, static_cast<int>(temp * 50));
                return temperature_colors[index];
            }
            
        case ColorScheme::COMPOSITE_BASED:
            // TODO: Color by composite membership
            return {128, 192, 255, 255};
            
        case ColorScheme::TYPE_BASED:
            // TODO: Color by particle type
            return {255, 192, 128, 255};
            
        default:
            return {255, 255, 255, 255};
    }
}

void GraphicsViewer::updateCamera(const SimulationState& state) {
    if (camera.following_particle && state.particles.count > camera.follow_particle_id) {
        camera.x = state.particles.pos_x[camera.follow_particle_id];
        camera.y = state.particles.pos_y[camera.follow_particle_id];
    }
    
    if (camera.auto_center) {
        centerOnCenterOfMass(state);
    }
}

void GraphicsViewer::updatePerformanceStats(float frame_time_ms) {
    performance_stats.frame_time_ms = frame_time_ms;
    performance_stats.fps = (frame_time_ms > 0) ? (1000.0f / frame_time_ms) : 0.0f;
}

void GraphicsViewer::updateEvents(float dt) {
    // Update event timers and remove expired events
    for (auto it = active_events.begin(); it != active_events.end();) {
        it->time_remaining -= dt;
        if (it->time_remaining <= 0) {
            it = active_events.erase(it);
        } else {
            ++it;
        }
    }
}

void GraphicsViewer::handleKeyboardInput() {
    const float pan_speed = 50.0f / camera.zoom;  // Adjust for zoom level
    
    // Camera panning
    if (isKeyPressed(SDL_SCANCODE_W) || isKeyPressed(SDL_SCANCODE_UP)) {
        camera.y += pan_speed;
        camera.following_particle = false;
        camera.auto_center = false;
    }
    if (isKeyPressed(SDL_SCANCODE_S) || isKeyPressed(SDL_SCANCODE_DOWN)) {
        camera.y -= pan_speed;
        camera.following_particle = false;
        camera.auto_center = false;
    }
    if (isKeyPressed(SDL_SCANCODE_A) || isKeyPressed(SDL_SCANCODE_LEFT)) {
        camera.x -= pan_speed;
        camera.following_particle = false;
        camera.auto_center = false;
    }
    if (isKeyPressed(SDL_SCANCODE_D) || isKeyPressed(SDL_SCANCODE_RIGHT)) {
        camera.x += pan_speed;
        camera.following_particle = false;
        camera.auto_center = false;
    }
    
    // Toggle options
    if (isKeyPressed(SDL_SCANCODE_F)) {
        toggleFullscreen();
    }
    
    if (isKeyPressed(SDL_SCANCODE_H)) {
        ui_settings.show_controls_help = !ui_settings.show_controls_help;
    }
    
    if (isKeyPressed(SDL_SCANCODE_SPACE)) {
        camera.auto_center = !camera.auto_center;
    }
    
    if (isKeyPressed(SDL_SCANCODE_ESCAPE)) {
        is_running = false;
    }
}

void GraphicsViewer::handleMouseInput() {
    if (input_state.mouse_left_down) {
        // Pan with left mouse button drag
        float dx = (input_state.mouse_x - input_state.mouse_start_x) / camera.zoom;
        float dy = (input_state.mouse_y - input_state.mouse_start_y) / camera.zoom;
        
        camera.x -= dx;
        camera.y += dy;  // Flip Y axis
        
        input_state.mouse_start_x = input_state.mouse_x;
        input_state.mouse_start_y = input_state.mouse_y;
        
        camera.following_particle = false;
        camera.auto_center = false;
    }
}

void GraphicsViewer::renderText(const std::string& text, int x, int y, SDL_Color color) {
    // Simplified text rendering - just draw character-sized rectangles
    // In a real implementation, you'd use SDL_ttf or similar
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    
    int char_width = 8;
    int char_height = 12;
    
    for (size_t i = 0; i < text.length(); i++) {
        if (text[i] != ' ') {
            SDL_Rect rect = {x + static_cast<int>(i) * char_width, y, char_width, char_height};
            SDL_RenderDrawRect(renderer, &rect);
        }
    }
}

void GraphicsViewer::renderTextf(int x, int y, SDL_Color color, const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    renderText(buffer, x, y, color);
}

// Factory implementations
std::unique_ptr<GraphicsViewer> ViewerFactory::createPerformanceViewer() {
    auto viewer = std::make_unique<GraphicsViewer>();
    
    auto& perf = viewer->getPerformanceSettings();
    perf.max_rendered_particles = 500000;
    perf.use_instanced_rendering = true;
    perf.enable_vsync = false;
    perf.adaptive_quality = true;
    
    auto& ui = viewer->getUISettings();
    ui.show_performance_overlay = true;
    ui.show_simulation_stats = false;
    ui.show_controls_help = false;
    ui.show_particle_info = false;
    
    viewer->setParticleStyle(GraphicsViewer::ParticleStyle::POINTS);
    viewer->setRenderMode(GraphicsViewer::RenderMode::PARTICLES_ONLY);
    
    return viewer;
}

std::unique_ptr<GraphicsViewer> ViewerFactory::createDebugViewer() {
    auto viewer = std::make_unique<GraphicsViewer>();
    
    auto& ui = viewer->getUISettings();
    ui.show_performance_overlay = true;
    ui.show_simulation_stats = true;
    ui.show_controls_help = true;
    ui.show_particle_info = true;
    
    viewer->setRenderMode(GraphicsViewer::RenderMode::DEBUG_SPATIAL_GRID);
    
    return viewer;
}

std::unique_ptr<GraphicsViewer> ViewerFactory::createPresentationViewer() {
    auto viewer = std::make_unique<GraphicsViewer>();
    
    auto& perf = viewer->getPerformanceSettings();
    perf.enable_vsync = true;
    perf.adaptive_quality = false;
    
    auto& ui = viewer->getUISettings();
    ui.show_performance_overlay = false;
    ui.show_simulation_stats = false;
    ui.show_controls_help = false;
    ui.show_particle_info = false;
    ui.overlay_alpha = 0.6f;
    
    viewer->setParticleStyle(GraphicsViewer::ParticleStyle::CIRCLES);
    viewer->setRenderMode(GraphicsViewer::RenderMode::PARTICLES_AND_SPRINGS);
    viewer->setColorScheme(GraphicsViewer::ColorScheme::MASS_BASED);
    
    return viewer;
}

std::unique_ptr<GraphicsViewer> ViewerFactory::fromConfig(const std::string& config_json) {
    // TODO: Parse JSON configuration and create viewer
    return createPerformanceViewer();
}

} // namespace digistar