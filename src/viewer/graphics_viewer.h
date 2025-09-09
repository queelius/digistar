#pragma once

#include <SDL2/SDL.h>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <unordered_map>
#include "../physics/types.h"
#include "../physics/pools.h"
#include "../backend/backend_interface.h"

namespace digistar {

/**
 * High-performance 2D graphics viewer for DigiStar simulations
 * 
 * Features:
 * - Real-time particle rendering with size/color based on properties
 * - Spring visualization
 * - Composite body highlighting  
 * - Multiple layers for pseudo-3D effect
 * - Pan, zoom, follow capabilities
 * - Performance overlay (FPS, particle count, etc.)
 * - Event visualization (explosions, collisions, etc.)
 * - Modular design - viewer as optional component
 */
class GraphicsViewer {
public:
    // Rendering modes
    enum class RenderMode {
        PARTICLES_ONLY,      // Just particles as circles/points
        PARTICLES_AND_SPRINGS,  // Particles + spring connections
        COMPOSITE_HIGHLIGHT,    // Highlight composite bodies
        HEAT_MAP,              // Color by temperature
        VELOCITY_VECTORS,      // Show velocity vectors
        FORCE_VECTORS,         // Show force vectors
        DEBUG_SPATIAL_GRID     // Show spatial grid overlay
    };
    
    // Particle rendering styles
    enum class ParticleStyle {
        POINTS,              // Single pixels (fastest)
        CIRCLES,             // Filled circles
        CIRCLES_OUTLINED,    // Circles with outlines
        SPRITES              // Custom sprite textures
    };
    
    // Color schemes
    enum class ColorScheme {
        DEFAULT,             // Simple color coding
        MASS_BASED,          // Color by mass
        VELOCITY_BASED,      // Color by velocity magnitude
        TEMPERATURE_BASED,   // Color by temperature
        COMPOSITE_BASED,     // Color by composite membership
        TYPE_BASED           // Color by particle type
    };
    
    // Camera controls
    struct Camera {
        float x = 0.0f, y = 0.0f;    // Center position
        float zoom = 1.0f;            // Zoom level (1.0 = normal)
        float min_zoom = 0.001f;      // Minimum zoom level
        float max_zoom = 100.0f;      // Maximum zoom level
        bool following_particle = false;  // Following a specific particle
        size_t follow_particle_id = 0;     // ID of particle to follow
        bool auto_center = false;          // Auto-center on center of mass
    };
    
    // Performance settings
    struct PerformanceSettings {
        size_t max_rendered_particles = 1000000;  // Limit for performance
        float lod_distance = 100.0f;              // Level-of-detail distance
        bool use_instanced_rendering = true;      // Use GPU instancing
        bool enable_vsync = true;                 // Enable V-sync
        int target_fps = 60;                      // Target FPS
        bool adaptive_quality = true;             // Reduce quality when FPS drops
    };
    
    // Event visualization
    struct EventVisualization {
        bool show_collisions = true;       // Show collision events
        bool show_spring_breaks = true;    // Show spring breaking events
        bool show_explosions = true;       // Show explosion/fission events
        bool show_formations = true;       // Show composite formation events
        float event_fade_time = 2.0f;      // How long events stay visible
        float event_max_distance = 500.0f; // Max distance to show events
    };
    
    // UI overlay settings
    struct UISettings {
        bool show_performance_overlay = true;     // Show FPS, particle count
        bool show_simulation_stats = true;       // Show energy, momentum
        bool show_controls_help = false;         // Show control help
        bool show_particle_info = false;         // Show clicked particle info
        float overlay_alpha = 0.8f;              // UI transparency
        int overlay_position = 0;                // 0=top-left, 1=top-right, etc.
    };

private:
    // SDL resources
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* particle_texture = nullptr;
    
    // Window properties
    int window_width = 1920;
    int window_height = 1080;
    bool is_fullscreen = false;
    bool is_running = true;
    
    // Rendering state
    RenderMode render_mode = RenderMode::PARTICLES_AND_SPRINGS;
    ParticleStyle particle_style = ParticleStyle::CIRCLES;
    ColorScheme color_scheme = ColorScheme::DEFAULT;
    Camera camera;
    PerformanceSettings perf_settings;
    EventVisualization event_viz;
    UISettings ui_settings;
    
    // Performance tracking
    struct PerformanceStats {
        float fps = 0.0f;
        float frame_time_ms = 0.0f;
        size_t particles_rendered = 0;
        size_t springs_rendered = 0;
        float render_time_ms = 0.0f;
        float ui_time_ms = 0.0f;
    } performance_stats;
    
    // Input state
    struct InputState {
        bool mouse_left_down = false;
        bool mouse_right_down = false;
        bool mouse_middle_down = false;
        int mouse_x = 0, mouse_y = 0;
        int mouse_start_x = 0, mouse_start_y = 0;
        bool keys[SDL_NUM_SCANCODES] = {false};
        float scroll_delta = 0.0f;
    } input_state;
    
    // Color lookup tables (for performance)
    std::vector<SDL_Color> mass_colors;
    std::vector<SDL_Color> velocity_colors;
    std::vector<SDL_Color> temperature_colors;
    
    // Event system integration
    std::vector<VisualEvent> active_events;
    
    // UI font and text rendering (simplified - could use TTF library)
    int font_size = 16;
    
public:
    GraphicsViewer() = default;
    ~GraphicsViewer();
    
    // === Lifecycle ===
    
    /**
     * Initialize the graphics viewer
     * @param title Window title
     * @param width Window width
     * @param height Window height
     * @param fullscreen Start in fullscreen mode
     * @return true on success
     */
    bool initialize(const std::string& title = "DigiStar Simulation", 
                   int width = 1920, int height = 1080, bool fullscreen = false);
    
    /**
     * Shutdown and cleanup resources
     */
    void shutdown();
    
    // === Main Rendering Loop ===
    
    /**
     * Process SDL events and update input state
     * @return false if should exit
     */
    bool processEvents();
    
    /**
     * Clear the screen and prepare for rendering
     */
    void beginFrame();
    
    /**
     * Render the simulation state
     * @param state Current simulation state
     */
    void renderSimulation(const SimulationState& state);
    
    /**
     * Render UI overlay
     * @param state Current simulation state for stats
     */
    void renderUI(const SimulationState& state);
    
    /**
     * Present the frame to screen
     */
    void presentFrame();
    
    /**
     * Check if viewer is still running
     */
    bool isRunning() const { return is_running; }
    
    // === Camera Control ===
    
    /**
     * Set camera position
     */
    void setCameraPosition(float x, float y);
    
    /**
     * Set zoom level
     */
    void setZoom(float zoom);
    
    /**
     * Follow a specific particle
     */
    void followParticle(size_t particle_id);
    
    /**
     * Stop following particle
     */
    void stopFollowing();
    
    /**
     * Center camera on center of mass
     */
    void centerOnCenterOfMass(const SimulationState& state);
    
    /**
     * Convert world coordinates to screen coordinates
     */
    void worldToScreen(float world_x, float world_y, int& screen_x, int& screen_y) const;
    
    /**
     * Convert screen coordinates to world coordinates
     */
    void screenToWorld(int screen_x, int screen_y, float& world_x, float& world_y) const;
    
    // === Rendering Configuration ===
    
    /**
     * Set rendering mode
     */
    void setRenderMode(RenderMode mode) { render_mode = mode; }
    RenderMode getRenderMode() const { return render_mode; }
    
    /**
     * Set particle rendering style
     */
    void setParticleStyle(ParticleStyle style) { particle_style = style; }
    ParticleStyle getParticleStyle() const { return particle_style; }
    
    /**
     * Set color scheme
     */
    void setColorScheme(ColorScheme scheme) { 
        color_scheme = scheme;
        buildColorLookupTables();
    }
    ColorScheme getColorScheme() const { return color_scheme; }
    
    /**
     * Get camera reference for direct manipulation
     */
    Camera& getCamera() { return camera; }
    const Camera& getCamera() const { return camera; }
    
    /**
     * Get performance settings reference
     */
    PerformanceSettings& getPerformanceSettings() { return perf_settings; }
    const PerformanceSettings& getPerformanceSettings() const { return perf_settings; }
    
    /**
     * Get UI settings reference
     */
    UISettings& getUISettings() { return ui_settings; }
    const UISettings& getUISettings() const { return ui_settings; }
    
    // === Event Visualization ===
    
    /**
     * Add visual event (explosion, collision, etc.)
     */
    void addVisualEvent(const VisualEvent& event);
    
    /**
     * Get event visualization settings
     */
    EventVisualization& getEventVisualization() { return event_viz; }
    const EventVisualization& getEventVisualization() const { return event_viz; }
    
    // === Input Handling ===
    
    /**
     * Check if key is currently pressed
     */
    bool isKeyPressed(SDL_Scancode key) const;
    
    /**
     * Get mouse position in screen coordinates
     */
    void getMousePosition(int& x, int& y) const;
    
    /**
     * Get mouse position in world coordinates
     */
    void getMouseWorldPosition(float& x, float& y) const;
    
    /**
     * Check if mouse button is pressed
     */
    bool isMouseButtonPressed(int button) const;
    
    // === Performance and Debugging ===
    
    /**
     * Get performance statistics
     */
    const PerformanceStats& getPerformanceStats() const { return performance_stats; }
    
    /**
     * Take screenshot and save to file
     */
    bool saveScreenshot(const std::string& filename) const;
    
    /**
     * Set window title (useful for showing simulation info)
     */
    void setWindowTitle(const std::string& title);
    
    /**
     * Toggle fullscreen mode
     */
    void toggleFullscreen();
    
private:
    // Internal rendering methods
    void renderParticles(const ParticlePool& particles);
    void renderSprings(const ParticlePool& particles, const SpringPool& springs);
    void renderComposites(const ParticlePool& particles, const CompositePool& composites);
    void renderSpatialGrid(const SimulationState& state);
    void renderEvents();
    void renderPerformanceOverlay(const SimulationState& state);
    void renderSimulationStats(const SimulationState& state);
    void renderParticleInfo(const SimulationState& state);
    
    // Helper methods
    void buildColorLookupTables();
    SDL_Color getParticleColor(size_t particle_id, const ParticlePool& particles) const;
    void updateCamera(const SimulationState& state);
    void updatePerformanceStats(float frame_time_ms);
    void updateEvents(float dt);
    void handleKeyboardInput();
    void handleMouseInput();
    size_t findParticleAtPosition(float world_x, float world_y, const ParticlePool& particles) const;
    
    // Text rendering (simplified)
    void renderText(const std::string& text, int x, int y, SDL_Color color = {255, 255, 255, 255});
    void renderTextf(int x, int y, SDL_Color color, const char* format, ...);
};

/**
 * Visual event for rendering explosions, collisions, etc.
 */
struct VisualEvent {
    enum Type {
        COLLISION,
        EXPLOSION,
        SPRING_BREAK,
        COMPOSITE_FORMATION,
        PARTICLE_DEATH,
        PARTICLE_BIRTH
    };
    
    Type type;
    float x, y;                    // World position
    float intensity = 1.0f;        // Event intensity (0-1)
    float max_radius = 50.0f;      // Maximum effect radius
    float duration = 2.0f;         // Total duration
    float time_remaining = 2.0f;   // Time remaining
    SDL_Color color = {255, 255, 255, 255};  // Event color
    
    VisualEvent(Type t, float pos_x, float pos_y, float inten = 1.0f) 
        : type(t), x(pos_x), y(pos_y), intensity(inten), time_remaining(duration) {}
};

/**
 * Factory for creating graphics viewer with preset configurations
 */
class ViewerFactory {
public:
    /**
     * Create viewer optimized for performance (minimal UI, fast rendering)
     */
    static std::unique_ptr<GraphicsViewer> createPerformanceViewer();
    
    /**
     * Create viewer for debugging (all overlays, detailed info)
     */
    static std::unique_ptr<GraphicsViewer> createDebugViewer();
    
    /**
     * Create viewer for presentations (clean UI, smooth rendering)
     */
    static std::unique_ptr<GraphicsViewer> createPresentationViewer();
    
    /**
     * Create viewer from configuration
     */
    static std::unique_ptr<GraphicsViewer> fromConfig(const std::string& config_json);
};

} // namespace digistar