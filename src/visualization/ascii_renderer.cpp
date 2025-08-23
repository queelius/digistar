#include "ascii_renderer.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstring>

namespace digistar {

// Character set definitions
constexpr char AsciiRenderer::CharacterSets::PARTICLE_TINY;
constexpr char AsciiRenderer::CharacterSets::PARTICLE_SMALL;
constexpr char AsciiRenderer::CharacterSets::PARTICLE_MEDIUM;
constexpr char AsciiRenderer::CharacterSets::PARTICLE_LARGE;
constexpr char AsciiRenderer::CharacterSets::PARTICLE_HUGE;

AsciiRenderer::AsciiRenderer(const Config& cfg) : config(cfg) {
    resize();
}

void AsciiRenderer::resize() {
    resize(config.width, config.height);
}

void AsciiRenderer::resize(size_t width, size_t height) {
    config.width = width;
    config.height = height;
    
    size_t buffer_size = width * height;
    front_buffer.resize(buffer_size, ' ');
    back_buffer.resize(buffer_size, ' ');
    depth_buffer.resize(buffer_size, 0);
    color_buffer.resize(buffer_size, 7);  // Default white
}

void AsciiRenderer::render(const SimulationState& state, const SimulationStats& stats) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Clear back buffer
    clearBuffer();
    
    // Update viewport if tracking
    if (config.track_particle >= 0 && config.track_particle < (int32_t)state.particles.count) {
        config.view_x = state.particles.pos_x[config.track_particle];
        config.view_y = state.particles.pos_y[config.track_particle];
    } else if (config.auto_scale) {
        updateViewport(state.particles);
    }
    
    // Draw grid if enabled
    if (config.show_grid) {
        drawGrid();
    }
    
    // Draw springs (lower layer)
    if (config.show_springs) {
        size_t springs_to_draw = std::min(state.springs.count, config.max_springs_render);
        for (size_t i = 0; i < springs_to_draw; i++) {
            if (!state.springs.is_broken[i]) {
                drawSpring(state.particles, state.springs, i);
                springs_rendered++;
            }
        }
    }
    
    // Draw composites (bounding boxes)
    if (config.show_composites) {
        for (size_t i = 0; i < state.composites.count; i++) {
            drawComposite(state.particles, state.springs, state.composites, i);
        }
    }
    
    // Draw particles (top layer)
    size_t particles_to_draw = std::min(state.particles.count, config.max_particles_render);
    
    // Sort particles by mass for better visibility (bigger on top)
    std::vector<size_t> particle_indices(particles_to_draw);
    for (size_t i = 0; i < particles_to_draw; i++) {
        particle_indices[i] = i;
    }
    std::sort(particle_indices.begin(), particle_indices.end(),
              [&state](size_t a, size_t b) {
                  return state.particles.mass[a] < state.particles.mass[b];
              });
    
    for (size_t idx : particle_indices) {
        drawParticle(state.particles, idx);
        particles_rendered++;
    }
    
    // Draw border
    drawBorder();
    
    // Draw stats overlay
    if (config.show_stats) {
        drawStats(stats);
    }
    
    // Draw legend
    if (config.show_legend) {
        drawLegend();
    }
    
    // Swap buffers
    swapBuffers();
    
    // Calculate render time
    auto end_time = std::chrono::high_resolution_clock::now();
    render_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    // Update FPS (simple moving average)
    float frame_time = render_time_ms / 1000.0f;
    current_fps = current_fps * 0.9f + (1.0f / frame_time) * 0.1f;
}

void AsciiRenderer::clearBuffer() {
    std::fill(back_buffer.begin(), back_buffer.end(), ' ');
    std::fill(depth_buffer.begin(), depth_buffer.end(), 0);
    std::fill(color_buffer.begin(), color_buffer.end(), 7);
    particles_rendered = 0;
    springs_rendered = 0;
}

void AsciiRenderer::worldToScreen(float wx, float wy, int& sx, int& sy) const {
    // Transform world coordinates to screen coordinates
    float dx = wx - config.view_x;
    float dy = wy - config.view_y;
    
    // Scale to screen space
    float scale = config.width / config.view_scale;
    sx = (int)(config.width / 2 + dx * scale);
    sy = (int)(config.height / 2 - dy * scale);  // Flip Y axis
}

float AsciiRenderer::getScreenScale() const {
    return config.width / config.view_scale;
}

void AsciiRenderer::setPixel(int x, int y, char c, uint8_t depth, uint8_t color) {
    if (x < 0 || x >= (int)config.width || y < 0 || y >= (int)config.height) {
        return;
    }
    
    size_t idx = y * config.width + x;
    
    // Z-buffer check
    if (depth >= depth_buffer[idx]) {
        back_buffer[idx] = c;
        depth_buffer[idx] = depth;
        color_buffer[idx] = color;
    }
}

void AsciiRenderer::drawLine(int x1, int y1, int x2, int y2, char c) {
    // Bresenham's line algorithm
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;
    
    while (true) {
        setPixel(x1, y1, c);
        
        if (x1 == x2 && y1 == y2) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
}

void AsciiRenderer::drawText(int x, int y, const std::string& text) {
    for (size_t i = 0; i < text.length(); i++) {
        setPixel(x + i, y, text[i], 255);  // Max depth for UI elements
    }
}

void AsciiRenderer::drawBox(int x1, int y1, int x2, int y2) {
    // Top and bottom
    for (int x = x1; x <= x2; x++) {
        setPixel(x, y1, CharacterSets::COMPOSITE_HORIZONTAL, 100);
        setPixel(x, y2, CharacterSets::COMPOSITE_HORIZONTAL, 100);
    }
    
    // Left and right
    for (int y = y1; y <= y2; y++) {
        setPixel(x1, y, CharacterSets::COMPOSITE_VERTICAL, 100);
        setPixel(x2, y, CharacterSets::COMPOSITE_VERTICAL, 100);
    }
    
    // Corners
    setPixel(x1, y1, CharacterSets::COMPOSITE_CORNER_TL, 100);
    setPixel(x2, y1, CharacterSets::COMPOSITE_CORNER_TR, 100);
    setPixel(x1, y2, CharacterSets::COMPOSITE_CORNER_BL, 100);
    setPixel(x2, y2, CharacterSets::COMPOSITE_CORNER_BR, 100);
}

void AsciiRenderer::drawParticle(const ParticlePool& particles, size_t idx) {
    int sx, sy;
    worldToScreen(particles.pos_x[idx], particles.pos_y[idx], sx, sy);
    
    char c = getParticleChar(particles, idx);
    uint8_t color = 7;  // Default white
    
    // Color based on temperature if enabled
    if (config.show_temperature) {
        color = getTemperatureColor(particles.temp_internal[idx]);
    }
    
    // Draw particle with high depth priority
    setPixel(sx, sy, c, 200, color);
    
    // Draw velocity vector if enabled
    if (config.show_velocities && (particles.vel_x[idx] != 0 || particles.vel_y[idx] != 0)) {
        float vx = particles.vel_x[idx];
        float vy = particles.vel_y[idx];
        float v_mag = sqrtf(vx * vx + vy * vy);
        
        if (v_mag > 0.1f) {
            // Normalize and scale velocity vector
            vx /= v_mag;
            vy /= v_mag;
            
            int vx_end, vy_end;
            float scale = getScreenScale();
            worldToScreen(particles.pos_x[idx] + vx * 5, 
                         particles.pos_y[idx] + vy * 5, 
                         vx_end, vy_end);
            
            drawLine(sx, sy, vx_end, vy_end, '-');
        }
    }
}

char AsciiRenderer::getParticleChar(const ParticlePool& particles, size_t idx) const {
    // Try emergent classification first
    char emergent = getEmergentParticleChar(particles, idx);
    
    // If no special emergent property, use size-based
    if (emergent == '\0') {
        float r = particles.radius[idx];
        float scale = getScreenScale();
        float screen_radius = r * scale;
        
        if (screen_radius < 0.5f) return CharacterSets::PARTICLE_TINY;
        if (screen_radius < 1.0f) return CharacterSets::PARTICLE_SMALL;
        if (screen_radius < 2.0f) return CharacterSets::PARTICLE_MEDIUM;
        if (screen_radius < 4.0f) return CharacterSets::PARTICLE_LARGE;
        return CharacterSets::PARTICLE_HUGE;
    }
    
    return emergent;
}

char AsciiRenderer::getEmergentParticleChar(const ParticlePool& particles, size_t idx) const {
    float mass = particles.mass[idx];
    float temp = particles.temp_internal[idx];
    float radius = particles.radius[idx];
    uint8_t material = particles.material_type[idx];
    
    // Calculate density to detect extreme objects
    float volume = (4.0f/3.0f) * M_PI * radius * radius * radius;  // Assuming 3D for density
    float density = mass / volume;
    
    // Black hole: extreme density (collapsed matter)
    if (density > 1e18f) {
        return CharacterSets::BLACK_HOLE;  // 'X'
    }
    
    // Star: hot, massive, mostly plasma
    if (temp > 5000.0f && mass > 1e29f && material == MATERIAL_PLASMA) {
        return CharacterSets::STAR;  // '*'
    }
    
    // Planet: massive, solid/liquid, moderate temperature
    if (mass > 1e23f && temp < 1000.0f && 
        (material == MATERIAL_ROCK || material == MATERIAL_ICE || material == MATERIAL_METAL)) {
        return CharacterSets::PLANET;  // 'P'
    }
    
    // Comet: icy, relatively small
    if (material == MATERIAL_ICE && temp < 273.0f && mass < 1e20f) {
        return CharacterSets::COMET;  // 'c'
    }
    
    // Asteroid: rocky, small
    if (material == MATERIAL_ROCK && mass < 1e21f) {
        return CharacterSets::ASTEROID;  // 'a'
    }
    
    // Plasma blob: hot gas
    if (material == MATERIAL_PLASMA || material == MATERIAL_GAS) {
        if (temp > 1000.0f) {
            return '!';  // Hot gas/plasma
        }
    }
    
    // No special classification
    return '\0';
}

char AsciiRenderer::getCompositeChar(const CompositePool& composites, const SpringPool& springs,
                                    const ParticlePool& particles, size_t composite_idx) const {
    // Analyze the composite's spring network to determine its nature
    float total_stiffness = 0;
    float total_strain = 0;
    float max_strain = 0;
    int spring_count = 0;
    int broken_count = 0;
    
    // Material composition
    int material_counts[8] = {0};  // Assuming 8 material types
    float total_temp = 0;
    
    size_t start = composites.member_start[composite_idx];
    size_t count = composites.member_count[composite_idx];
    
    // First, analyze particle properties
    for (size_t i = 0; i < count; i++) {
        uint32_t pid = composites.member_particles[start + i];
        material_counts[particles.material_type[pid] % 8]++;
        total_temp += particles.temp_internal[pid];
    }
    float avg_temp = total_temp / count;
    
    // Find dominant material
    uint8_t dominant_material = 0;
    int max_material_count = 0;
    for (int i = 0; i < 8; i++) {
        if (material_counts[i] > max_material_count) {
            max_material_count = material_counts[i];
            dominant_material = i;
        }
    }
    
    // Now analyze springs within the composite
    for (size_t s = 0; s < springs.count; s++) {
        uint32_t p1 = springs.particle1[s];
        uint32_t p2 = springs.particle2[s];
        
        // Check if both particles belong to this composite
        bool p1_in_composite = false;
        bool p2_in_composite = false;
        
        for (size_t i = 0; i < count; i++) {
            uint32_t member = composites.member_particles[start + i];
            if (member == p1) p1_in_composite = true;
            if (member == p2) p2_in_composite = true;
            if (p1_in_composite && p2_in_composite) break;
        }
        
        if (p1_in_composite && p2_in_composite) {
            spring_count++;
            
            if (springs.is_broken[s]) {
                broken_count++;
            } else {
                total_stiffness += springs.stiffness[s];
                float strain = fabsf(springs.current_strain[s]);
                total_strain += strain;
                max_strain = std::max(max_strain, strain);
            }
        }
    }
    
    // Classify based on properties
    if (spring_count == 0) {
        return '?';  // No springs? Shouldn't happen for a composite
    }
    
    float avg_stiffness = total_stiffness / (spring_count - broken_count + 1);
    float avg_strain = total_strain / (spring_count - broken_count + 1);
    float broken_ratio = (float)broken_count / spring_count;
    
    // Breaking apart
    if (broken_ratio > 0.5f) {
        return '!';  // Fragmenting
    }
    
    // Under high stress
    if (max_strain > 0.4f || avg_strain > 0.2f) {
        return '=';  // Stressed
    }
    
    // Material-based classification
    if (dominant_material == MATERIAL_METAL && avg_stiffness > 300.0f) {
        return 'S';  // Structure/spacecraft (rigid metal)
    }
    
    if (dominant_material == MATERIAL_ROCK) {
        if (avg_stiffness < 100.0f) {
            return 'A';  // Asteroid rubble pile (loose rocks)
        } else {
            return 'R';  // Solid rock formation
        }
    }
    
    if (dominant_material == MATERIAL_ICE) {
        if (avg_temp > 273.0f) {
            return '~';  // Melting ice
        } else {
            return 'I';  // Ice formation
        }
    }
    
    if (dominant_material == MATERIAL_GAS || dominant_material == MATERIAL_PLASMA) {
        return '~';  // Fluid/gas cloud
    }
    
    // Stiffness-based classification
    if (avg_stiffness > 500.0f) {
        return 'H';  // Hard/rigid body
    }
    if (avg_stiffness < 50.0f) {
        return '~';  // Soft/fluid body
    }
    
    // Default composite
    return '#';
}

void AsciiRenderer::drawSpring(const ParticlePool& particles, const SpringPool& springs, size_t idx) {
    uint32_t i = springs.particle1[idx];
    uint32_t j = springs.particle2[idx];
    
    int x1, y1, x2, y2;
    worldToScreen(particles.pos_x[i], particles.pos_y[i], x1, y1);
    worldToScreen(particles.pos_x[j], particles.pos_y[j], x2, y2);
    
    // Don't draw if both endpoints are off-screen
    if ((x1 < 0 || x1 >= (int)config.width || y1 < 0 || y1 >= (int)config.height) &&
        (x2 < 0 || x2 >= (int)config.width || y2 < 0 || y2 >= (int)config.height)) {
        return;
    }
    
    float dx = particles.pos_x[j] - particles.pos_x[i];
    float dy = particles.pos_y[j] - particles.pos_y[i];
    char spring_char = getSpringChar(dx, dy, springs.current_strain[idx]);
    
    drawLine(x1, y1, x2, y2, spring_char);
}

char AsciiRenderer::getSpringChar(float dx, float dy, float strain) const {
    // Choose character based on strain
    if (fabsf(strain) > 0.4f) {
        return CharacterSets::SPRING_BREAKING;
    }
    if (fabsf(strain) > 0.2f) {
        return CharacterSets::SPRING_STRESSED;
    }
    
    // Choose character based on direction
    float angle = atan2f(dy, dx);
    float angle_deg = angle * 180.0f / M_PI;
    
    if (angle_deg < -157.5f || angle_deg > 157.5f) return CharacterSets::SPRING_HORIZONTAL;
    if (angle_deg > -22.5f && angle_deg < 22.5f) return CharacterSets::SPRING_HORIZONTAL;
    if (angle_deg > 67.5f && angle_deg < 112.5f) return CharacterSets::SPRING_VERTICAL;
    if (angle_deg > -112.5f && angle_deg < -67.5f) return CharacterSets::SPRING_VERTICAL;
    if (angle_deg > 22.5f && angle_deg < 67.5f) return CharacterSets::SPRING_DIAGONAL_1;
    if (angle_deg > -157.5f && angle_deg < -112.5f) return CharacterSets::SPRING_DIAGONAL_1;
    return CharacterSets::SPRING_DIAGONAL_2;
}

void AsciiRenderer::drawComposite(const ParticlePool& particles, const SpringPool& springs,
                                 const CompositePool& composites, size_t idx) {
    // Get composite classification character
    char composite_char = getCompositeChar(composites, springs, particles, idx);
    
    // Draw bounding box around composite
    float min_x = 1e9, max_x = -1e9;
    float min_y = 1e9, max_y = -1e9;
    
    size_t start = composites.member_start[idx];
    size_t count = composites.member_count[idx];
    
    for (size_t i = 0; i < count; i++) {
        uint32_t pid = composites.member_particles[start + i];
        float x = particles.pos_x[pid];
        float y = particles.pos_y[pid];
        float r = particles.radius[pid];
        
        min_x = std::min(min_x, x - r);
        max_x = std::max(max_x, x + r);
        min_y = std::min(min_y, y - r);
        max_y = std::max(max_y, y + r);
    }
    
    int sx1, sy1, sx2, sy2;
    worldToScreen(min_x, max_y, sx1, sy1);
    worldToScreen(max_x, min_y, sx2, sy2);
    
    // Draw bounding box
    drawBox(sx1, sy1, sx2, sy2);
    
    // Draw classification character at center of composite
    int cx = (sx1 + sx2) / 2;
    int cy = (sy1 + sy2) / 2;
    setPixel(cx, cy, composite_char, 150);  // Medium-high depth priority
}

void AsciiRenderer::drawGrid() {
    float grid_spacing = config.view_scale / 10;  // 10 grid lines across screen
    
    // Snap to grid
    float start_x = floorf(config.view_x / grid_spacing) * grid_spacing - config.view_scale / 2;
    float start_y = floorf(config.view_y / grid_spacing) * grid_spacing - config.view_scale / 2;
    
    for (float x = start_x; x < start_x + config.view_scale; x += grid_spacing) {
        for (float y = start_y; y < start_y + config.view_scale; y += grid_spacing) {
            int sx, sy;
            worldToScreen(x, y, sx, sy);
            
            // Major grid lines every 5 spaces
            bool major = (int(x / grid_spacing) % 5 == 0) && (int(y / grid_spacing) % 5 == 0);
            setPixel(sx, sy, major ? CharacterSets::GRID_HEAVY : CharacterSets::GRID_LIGHT, 1);
        }
    }
}

void AsciiRenderer::drawBorder() {
    // Simple border around display
    for (size_t x = 0; x < config.width; x++) {
        setPixel(x, 0, '═', 255);
        setPixel(x, config.height - 1, '═', 255);
    }
    for (size_t y = 0; y < config.height; y++) {
        setPixel(0, y, '║', 255);
        setPixel(config.width - 1, y, '║', 255);
    }
    
    // Corners
    setPixel(0, 0, '╔', 255);
    setPixel(config.width - 1, 0, '╗', 255);
    setPixel(0, config.height - 1, '╚', 255);
    setPixel(config.width - 1, config.height - 1, '╝', 255);
}

void AsciiRenderer::drawStats(const SimulationStats& stats) {
    // Stats box in top-left corner
    std::stringstream ss;
    
    ss << "FPS: " << std::fixed << std::setprecision(1) << current_fps;
    drawText(2, 1, ss.str());
    ss.str("");
    
    ss << "Particles: " << stats.active_particles;
    drawText(2, 2, ss.str());
    ss.str("");
    
    ss << "Springs: " << stats.active_springs;
    drawText(2, 3, ss.str());
    ss.str("");
    
    ss << "Contacts: " << stats.active_contacts;
    drawText(2, 4, ss.str());
    ss.str("");
    
    ss << "Composites: " << stats.active_composites;
    drawText(2, 5, ss.str());
    ss.str("");
    
    ss << "Energy: " << std::scientific << std::setprecision(2) << stats.total_energy;
    drawText(2, 6, ss.str());
    ss.str("");
    
    ss << "Max Vel: " << std::fixed << std::setprecision(1) << stats.max_velocity;
    drawText(2, 7, ss.str());
    
    // Performance stats in top-right
    ss.str("");
    ss << "Update: " << std::fixed << std::setprecision(1) << stats.update_time << "ms";
    drawText(config.width - 20, 1, ss.str());
    
    ss.str("");
    ss << "Render: " << std::fixed << std::setprecision(1) << render_time_ms << "ms";
    drawText(config.width - 20, 2, ss.str());
}

void AsciiRenderer::drawLegend() {
    // Legend in bottom-left corner
    size_t y = config.height - 8;
    
    drawText(2, y++, "═══ Legend ═══");
    drawText(2, y++, ". Small particle");
    drawText(2, y++, "O Large particle");
    drawText(2, y++, "* Star");
    drawText(2, y++, "─ Spring");
    drawText(2, y++, "~ Breaking spring");
    drawText(2, y++, "□ Composite body");
}

void AsciiRenderer::swapBuffers() {
    std::swap(front_buffer, back_buffer);
}

std::string AsciiRenderer::getFrame() const {
    std::stringstream ss;
    
    for (size_t y = 0; y < config.height; y++) {
        for (size_t x = 0; x < config.width; x++) {
            ss << front_buffer[y * config.width + x];
        }
        if (y < config.height - 1) {
            ss << '\n';
        }
    }
    
    return ss.str();
}

std::string AsciiRenderer::getFrameWithAnsi() const {
    std::stringstream ss;
    
    // ANSI color codes
    const char* color_codes[] = {
        "\033[30m",  // 0: Black
        "\033[31m",  // 1: Red
        "\033[32m",  // 2: Green
        "\033[33m",  // 3: Yellow
        "\033[34m",  // 4: Blue
        "\033[35m",  // 5: Magenta
        "\033[36m",  // 6: Cyan
        "\033[37m",  // 7: White
        "\033[91m",  // 8: Bright Red
        "\033[92m",  // 9: Bright Green
        "\033[93m",  // 10: Bright Yellow
        "\033[94m",  // 11: Bright Blue
        "\033[95m",  // 12: Bright Magenta
        "\033[96m",  // 13: Bright Cyan
        "\033[97m",  // 14: Bright White
    };
    
    uint8_t last_color = 255;
    
    for (size_t y = 0; y < config.height; y++) {
        for (size_t x = 0; x < config.width; x++) {
            size_t idx = y * config.width + x;
            
            // Change color if needed
            if (color_buffer[idx] != last_color) {
                ss << color_codes[color_buffer[idx] % 15];
                last_color = color_buffer[idx];
            }
            
            ss << front_buffer[idx];
        }
        if (y < config.height - 1) {
            ss << '\n';
        }
    }
    
    // Reset color at end
    ss << "\033[0m";
    
    return ss.str();
}

uint8_t AsciiRenderer::getTemperatureColor(float temp) const {
    // Map temperature to color
    // Cold (< 273K) = Blue
    // Room temp (273-373K) = White
    // Hot (373-1000K) = Yellow
    // Very hot (1000-3000K) = Red
    // Plasma (> 3000K) = Magenta
    
    if (temp < 273) return 4;       // Blue
    if (temp < 373) return 7;       // White
    if (temp < 1000) return 3;      // Yellow
    if (temp < 3000) return 1;      // Red
    return 5;                        // Magenta
}

void AsciiRenderer::updateViewport(const ParticlePool& particles) {
    if (particles.count == 0) return;
    
    // Find bounding box of all particles
    float min_x = particles.pos_x[0];
    float max_x = particles.pos_x[0];
    float min_y = particles.pos_y[0];
    float max_y = particles.pos_y[0];
    
    for (size_t i = 1; i < particles.count; i++) {
        min_x = std::min(min_x, particles.pos_x[i]);
        max_x = std::max(max_x, particles.pos_x[i]);
        min_y = std::min(min_y, particles.pos_y[i]);
        max_y = std::max(max_y, particles.pos_y[i]);
    }
    
    // Center view on bounding box
    config.view_x = (min_x + max_x) / 2;
    config.view_y = (min_y + max_y) / 2;
    
    // Adjust scale to fit all particles with 10% margin
    float width = max_x - min_x;
    float height = max_y - min_y;
    config.view_scale = std::max(width, height) * 1.1f;
}

void AsciiRenderer::setViewCenter(float x, float y) {
    config.view_x = x;
    config.view_y = y;
}

void AsciiRenderer::setViewScale(float scale) {
    config.view_scale = scale;
}

void AsciiRenderer::trackParticle(int32_t particle_id) {
    config.track_particle = particle_id;
}

void AsciiRenderer::zoomIn(float factor) {
    config.view_scale /= factor;
}

void AsciiRenderer::zoomOut(float factor) {
    config.view_scale *= factor;
}

void AsciiRenderer::pan(float dx, float dy) {
    config.view_x += dx;
    config.view_y += dy;
}

// ============ AsciiMiniMap Implementation ============

AsciiMiniMap::AsciiMiniMap(size_t w, size_t h) : width(w), height(h) {
    buffer.resize(width * height, ' ');
}

void AsciiMiniMap::render(const SimulationState& state, const AsciiRenderer::Config& main_view) {
    std::fill(buffer.begin(), buffer.end(), ' ');
    
    // Draw particles as dots
    for (size_t i = 0; i < state.particles.count; i++) {
        // Map to minimap coordinates
        int x = (int)((state.particles.pos_x[i] / 10000.0f + 0.5f) * width);
        int y = (int)((state.particles.pos_y[i] / 10000.0f + 0.5f) * height);
        
        if (x >= 0 && x < (int)width && y >= 0 && y < (int)height) {
            buffer[y * width + x] = '.';
        }
    }
    
    // Draw viewport rectangle
    int vx1 = (int)(((main_view.view_x - main_view.view_scale/2) / 10000.0f + 0.5f) * width);
    int vx2 = (int)(((main_view.view_x + main_view.view_scale/2) / 10000.0f + 0.5f) * width);
    int vy1 = (int)(((main_view.view_y - main_view.view_scale/2) / 10000.0f + 0.5f) * height);
    int vy2 = (int)(((main_view.view_y + main_view.view_scale/2) / 10000.0f + 0.5f) * height);
    
    // Draw viewport borders
    for (int x = vx1; x <= vx2; x++) {
        if (x >= 0 && x < (int)width) {
            if (vy1 >= 0 && vy1 < (int)height) buffer[vy1 * width + x] = '-';
            if (vy2 >= 0 && vy2 < (int)height) buffer[vy2 * width + x] = '-';
        }
    }
    for (int y = vy1; y <= vy2; y++) {
        if (y >= 0 && y < (int)height) {
            if (vx1 >= 0 && vx1 < (int)width) buffer[y * width + vx1] = '|';
            if (vx2 >= 0 && vx2 < (int)width) buffer[y * width + vx2] = '|';
        }
    }
}

std::string AsciiMiniMap::getFrame() const {
    std::stringstream ss;
    
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            ss << buffer[y * width + x];
        }
        if (y < height - 1) ss << '\n';
    }
    
    return ss.str();
}

// ============ AsciiGraphRenderer Implementation ============

AsciiGraphRenderer::AsciiGraphRenderer(size_t w, size_t h, size_t hist) 
    : width(w), height(h), history_size(hist) {
    history.reserve(history_size);
}

void AsciiGraphRenderer::addDataPoint(float value) {
    history.push_back(value);
    if (history.size() > history_size) {
        history.erase(history.begin());
    }
}

void AsciiGraphRenderer::render(const std::string& title, float min_val, float max_val) {
    // TODO: Implement graph rendering
}

// ============ TerminalDisplay Implementation ============

TerminalDisplay::TerminalDisplay() {
    // Detect terminal capabilities
    const char* term = std::getenv("TERM");
    if (term) {
        std::string term_str(term);
        use_ansi = (term_str != "dumb");
        use_colors = (term_str.find("color") != std::string::npos || 
                     term_str.find("256") != std::string::npos);
    }
}

void TerminalDisplay::clearScreen() const {
    if (use_ansi) {
        std::cout << "\033[2J\033[H";
    }
}

void TerminalDisplay::moveCursor(int x, int y) const {
    if (use_ansi) {
        std::cout << "\033[" << y << ";" << x << "H";
    }
}

void TerminalDisplay::hideCursor() const {
    if (use_ansi) {
        std::cout << "\033[?25l";
    }
}

void TerminalDisplay::showCursor() const {
    if (use_ansi) {
        std::cout << "\033[?25h";
    }
}

void TerminalDisplay::setColor(uint8_t fg, uint8_t bg) const {
    if (use_ansi && use_colors) {
        std::cout << "\033[" << (30 + fg) << ";" << (40 + bg) << "m";
    }
}

void TerminalDisplay::resetColor() const {
    if (use_ansi) {
        std::cout << "\033[0m";
    }
}

void TerminalDisplay::displayFrame(const std::string& frame) const {
    if (clear_screen) {
        clearScreen();
    }
    std::cout << frame << std::flush;
}

void TerminalDisplay::displayFrameAt(int x, int y, const std::string& frame) const {
    moveCursor(x, y);
    std::cout << frame << std::flush;
}

} // namespace digistar