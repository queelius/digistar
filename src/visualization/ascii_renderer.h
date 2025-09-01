#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include "../physics/pools.h"
#include "../backend/backend_interface.h"

namespace digistar {

// ASCII rendering for terminal display and monitoring
class AsciiRenderer {
public:
    struct Config {
        // Display dimensions
        size_t width;        // Terminal columns
        size_t height;       // Terminal rows
        
        // Viewport (world coordinates)
        float view_x;          // Center X
        float view_y;          // Center Y
        float view_scale;      // World units per screen width
        
        // Display options
        bool show_grid;
        bool show_springs;
        bool show_velocities;
        bool show_forces;
        bool show_temperature;
        bool show_composites;
        bool show_stats;
        bool show_legend;
        
        // Default constructor
        Config() :
            width(120),
            height(40),
            view_x(0),
            view_y(0),
            view_scale(100.0f),
            show_grid(false),
            show_springs(true),
            show_velocities(false),
            show_forces(false),
            show_temperature(false),
            show_composites(true),
            show_stats(true),
            show_legend(true) {}
        
        // Tracking
        int32_t track_particle = -1;  // Auto-center on this particle
        bool auto_scale = false;       // Auto-adjust scale to fit all particles
        
        // Performance
        size_t max_particles_render = 10000;  // Limit for performance
        size_t max_springs_render = 5000;
        
        // Update rate
        float fps_target = 30.0f;      // Target frame rate
    };
    
    // Character sets for different particle types and states
    struct CharacterSets {
        // Particle size gradations
        static constexpr char PARTICLE_TINY = '.';
        static constexpr char PARTICLE_SMALL = 'o';
        static constexpr char PARTICLE_MEDIUM = 'O';
        static constexpr char PARTICLE_LARGE = '@';
        static constexpr char PARTICLE_HUGE = '#';
        
        // Special particles
        static constexpr char STAR = '*';
        static constexpr char PLANET = 'P';
        static constexpr char ASTEROID = 'a';
        static constexpr char COMET = 'c';
        static constexpr char BLACK_HOLE = 'X';
        
        // Temperature gradients (cold to hot)
        static constexpr const char* TEMP_GRADIENT = " .-+*%#@";
        
        // Velocity indicators
        static constexpr const char* VELOCITY_ARROWS = "←↖↑↗→↘↓↙";
        
        // Spring characters
        static constexpr char SPRING_HORIZONTAL = '-';
        static constexpr char SPRING_VERTICAL = '|';
        static constexpr char SPRING_DIAGONAL_1 = '/';
        static constexpr char SPRING_DIAGONAL_2 = '\\';
        static constexpr char SPRING_CROSS = '+';
        static constexpr char SPRING_STRESSED = '=';
        static constexpr char SPRING_BREAKING = '~';
        
        // Composite body outline
        static constexpr char COMPOSITE_CORNER_TL = '┌';
        static constexpr char COMPOSITE_CORNER_TR = '┐';
        static constexpr char COMPOSITE_CORNER_BL = '└';
        static constexpr char COMPOSITE_CORNER_BR = '┘';
        static constexpr char COMPOSITE_HORIZONTAL = '─';
        static constexpr char COMPOSITE_VERTICAL = '│';
        
        // Grid characters
        static constexpr char GRID_LIGHT = '·';
        static constexpr char GRID_HEAVY = '▪';
    };
    
private:
    Config config;
    
    // Display buffers (double buffering)
    std::vector<char> front_buffer;
    std::vector<char> back_buffer;
    std::vector<uint8_t> depth_buffer;  // Z-order for overlapping particles
    
    // Color codes (if terminal supports it)
    std::vector<uint8_t> color_buffer;
    
    // Stats tracking
    float current_fps = 0;
    float render_time_ms = 0;
    size_t particles_rendered = 0;
    size_t springs_rendered = 0;
    
    // Viewport management
    void updateViewport(const ParticlePool& particles);
    void worldToScreen(float wx, float wy, int& sx, int& sy) const;
    float getScreenScale() const;
    
    // Rendering helpers
    void clearBuffer();
    void drawParticle(const ParticlePool& particles, size_t idx);
    void drawSpring(const ParticlePool& particles, const SpringPool& springs, size_t idx);
    void drawComposite(const ParticlePool& particles, const SpringPool& springs, 
                      const CompositePool& composites, size_t idx);
    void drawGrid();
    void drawStats(const SimulationStats& stats);
    void drawLegend();
    void drawBorder();
    
    // Buffer management
    void setPixel(int x, int y, char c, uint8_t depth = 0, uint8_t color = 7);
    void drawLine(int x1, int y1, int x2, int y2, char c);
    void drawBox(int x1, int y1, int x2, int y2);
    void drawText(int x, int y, const std::string& text);
    void swapBuffers();
    
    // Character selection based on emergent properties
    char getParticleChar(const ParticlePool& particles, size_t idx) const;
    char getEmergentParticleChar(const ParticlePool& particles, size_t idx) const;
    char getCompositeChar(const CompositePool& composites, const SpringPool& springs, 
                         const ParticlePool& particles, size_t composite_idx) const;
    char getSpringChar(float dx, float dy, float strain) const;
    uint8_t getTemperatureColor(float temp) const;
    
public:
    AsciiRenderer(const Config& cfg = Config());
    ~AsciiRenderer() = default;
    
    // Main rendering
    void render(const SimulationState& state, const SimulationStats& stats);
    
    // Get rendered frame as string
    std::string getFrame() const;
    std::string getFrameWithAnsi() const;  // With ANSI color codes
    
    // Camera control
    void setViewCenter(float x, float y);
    void setViewScale(float scale);
    void trackParticle(int32_t particle_id);
    void zoomIn(float factor = 1.5f);
    void zoomOut(float factor = 1.5f);
    void pan(float dx, float dy);
    
    // Display options
    void toggleGrid() { config.show_grid = !config.show_grid; }
    void toggleSprings() { config.show_springs = !config.show_springs; }
    void toggleVelocities() { config.show_velocities = !config.show_velocities; }
    void toggleForces() { config.show_forces = !config.show_forces; }
    void toggleTemperature() { config.show_temperature = !config.show_temperature; }
    void toggleComposites() { config.show_composites = !config.show_composites; }
    void toggleStats() { config.show_stats = !config.show_stats; }
    
    // Configuration
    void setConfig(const Config& cfg) { config = cfg; resize(); }
    Config& getConfig() { return config; }
    
    // Resize display
    void resize(size_t width, size_t height);
    void resize();  // Use config dimensions
    
    // Performance stats
    float getFPS() const { return current_fps; }
    float getRenderTime() const { return render_time_ms; }
};

// Mini-map renderer for overview
class AsciiMiniMap {
    size_t width = 20;
    size_t height = 10;
    std::vector<char> buffer;
    
public:
    AsciiMiniMap(size_t w = 20, size_t h = 10);
    void render(const SimulationState& state, const AsciiRenderer::Config& main_view);
    std::string getFrame() const;
};

// Specialized renderers for different views
class AsciiGraphRenderer {
    size_t width;
    size_t height;
    size_t history_size;
    std::vector<float> history;
    
public:
    AsciiGraphRenderer(size_t w = 40, size_t h = 10, size_t history = 100);
    void addDataPoint(float value);
    void render(const std::string& title, float min_val, float max_val);
    std::string getFrame() const;
};

// Terminal output manager (handles ANSI codes, clearing, etc.)
class TerminalDisplay {
    bool use_ansi = true;
    bool clear_screen = true;
    bool use_colors = true;
    
public:
    TerminalDisplay();
    
    // Output control
    void clearScreen() const;
    void moveCursor(int x, int y) const;
    void hideCursor() const;
    void showCursor() const;
    
    // Color control
    void setColor(uint8_t fg, uint8_t bg = 0) const;
    void resetColor() const;
    
    // Full frame output
    void displayFrame(const std::string& frame) const;
    void displayFrameAt(int x, int y, const std::string& frame) const;
    
    // Utility
    bool supportsAnsi() const { return use_ansi; }
    bool supportsColor() const { return use_colors; }
    void setAnsiMode(bool enabled) { use_ansi = enabled; }
    void setColorMode(bool enabled) { use_colors = enabled; }
};

} // namespace digistar