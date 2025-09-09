# DigiStar Graphics Viewer

The DigiStar Graphics Viewer provides real-time 2D visualization of particle simulations with high performance and rich visual features.

## Features

### Core Rendering
- **Real-time particle rendering** with up to 1M+ particles
- **Spring network visualization** showing connections between particles
- **Composite body highlighting** for emergent structures
- **Multiple rendering modes**: particles-only, with springs, heat maps, etc.
- **Particle styles**: points, circles, outlined circles, custom sprites
- **Color schemes**: mass-based, velocity-based, temperature-based, composite-based

### Interactive Controls
- **Pan and zoom** with mouse and keyboard
- **Follow specific particles** for detailed tracking
- **Auto-center on center of mass** for system-wide view
- **Multiple camera presets** for common viewing scenarios
- **Real-time switching** between rendering modes and color schemes

### Visual Events
- **Physics event visualization**: explosions, collisions, spring breaks
- **Event types**: particle mergers, fission, star formation, black hole formation
- **Customizable effects**: duration, intensity, colors, radius
- **Spatial filtering** to show only relevant events
- **Event fade-out** for smooth visual transitions

### Performance Features
- **Adaptive quality**: reduces detail when frame rate drops
- **Level-of-detail (LOD)**: simplified rendering at distance
- **Frustum culling**: only renders visible particles
- **GPU acceleration**: hardware-accelerated rendering via SDL2
- **Configurable limits**: particle count, render distance, etc.

### UI and Monitoring
- **Performance overlay**: FPS, particle count, render time
- **Simulation statistics**: energy, momentum, temperature
- **Interactive particle info**: click particles for detailed data
- **Control help overlay**: keyboard and mouse reference
- **Customizable UI positioning** and transparency

## Quick Start

### Basic Usage

```cpp
#include "src/viewer/graphics_viewer.h"
using namespace digistar;

// Create and initialize viewer
auto viewer = std::make_unique<GraphicsViewer>();
if (!viewer->initialize("My Simulation", 1920, 1080)) {
    // Handle error
}

// Main render loop
while (viewer->isRunning()) {
    // Process input events
    if (!viewer->processEvents()) break;
    
    // Update your simulation state
    updateSimulation(sim_state, dt);
    
    // Render
    viewer->beginFrame();
    viewer->renderSimulation(sim_state);
    viewer->renderUI(sim_state);
    viewer->presentFrame();
}

viewer->shutdown();
```

### Factory Presets

```cpp
// Performance-optimized viewer (minimal UI, fast rendering)
auto viewer = ViewerFactory::createPerformanceViewer();

// Debug viewer (all overlays, detailed info)
auto viewer = ViewerFactory::createDebugViewer();

// Presentation viewer (clean UI, smooth rendering)
auto viewer = ViewerFactory::createPresentationViewer();
```

### Event System Integration

```cpp
#include "src/viewer/viewer_event_bridge.h"

// Create event bridge to connect physics events to visual effects
auto event_bridge = EventBridgeFactory::createPresentationBridge();
event_bridge->initialize(viewer, config);
event_bridge->start();  // Starts background thread for event processing
```

## Configuration

### Camera Settings

```cpp
auto& camera = viewer->getCamera();
camera.zoom = 2.0f;                    // Zoom level
camera.following_particle = true;      // Follow specific particle
camera.follow_particle_id = 42;        // ID of particle to follow
camera.auto_center = false;            // Auto-center on center of mass
```

### Performance Settings

```cpp
auto& perf = viewer->getPerformanceSettings();
perf.max_rendered_particles = 500000;  // Limit for performance
perf.enable_vsync = false;             // Disable V-sync for speed
perf.adaptive_quality = true;          // Reduce quality when needed
perf.target_fps = 60;                  // Target frame rate
```

### Rendering Configuration

```cpp
// Set rendering mode
viewer->setRenderMode(GraphicsViewer::RenderMode::PARTICLES_AND_SPRINGS);

// Set particle style
viewer->setParticleStyle(GraphicsViewer::ParticleStyle::CIRCLES);

// Set color scheme
viewer->setColorScheme(GraphicsViewer::ColorScheme::VELOCITY_BASED);
```

### Event Visualization

```cpp
auto& events = viewer->getEventVisualization();
events.show_collisions = true;         // Show collision events
events.show_explosions = true;         // Show explosion events
events.event_fade_time = 3.0f;         // How long events stay visible
events.event_max_distance = 1000.0f;   // Max distance to show events
```

## Controls

### Keyboard Controls
- **WASD / Arrow Keys**: Pan camera
- **Space**: Center camera on center of mass
- **F**: Toggle fullscreen
- **H**: Toggle help overlay
- **P**: Pause/resume (in applications that support it)
- **1-9**: Switch between rendering modes (application-specific)
- **ESC**: Exit application

### Mouse Controls
- **Left Click + Drag**: Pan camera
- **Mouse Wheel**: Zoom in/out
- **Right Click**: Context menu (application-specific)
- **Middle Click**: Reset zoom and position

## Rendering Modes

### Particle-Only Mode
- Renders only particles as points, circles, or sprites
- Fastest rendering mode for maximum performance
- Color coding based on selected scheme (mass, velocity, etc.)

### Particles and Springs Mode
- Shows particles connected by spring networks
- Visualizes emergent composite structures
- Springs colored by stress/strain levels

### Composite Highlight Mode
- Highlights composite bodies formed by spring networks
- Shows bounding boxes or convex hulls
- Different colors for different composite types

### Heat Map Mode
- Colors particles by temperature
- Shows thermal gradients and hot spots
- Useful for thermal physics analysis

### Debug Modes
- **Spatial Grid**: Shows spatial indexing grids
- **Force Vectors**: Displays force vectors on particles
- **Velocity Vectors**: Shows velocity vectors
- **Collision Detection**: Highlights collision boundaries

## Event Types and Visualization

### Physics Events
- **Particle Merge**: Orange explosion effect
- **Particle Fission**: Red explosion with multiple fragments
- **Star Formation**: Bright white/yellow burst, long duration
- **Black Hole Formation**: Purple implosion effect
- **Black Hole Absorption**: Dark purple fade effect

### Spring Events
- **Spring Creation**: Green cross pattern
- **Spring Break**: Orange X pattern
- **Critical Stress**: Yellow warning flash

### Collision Events
- **Soft Contact**: Not visualized (too frequent)
- **Hard Collision**: Bright yellow impact flash
- **Composite Collision**: Pink impact with larger radius

### Thermal Events
- **Phase Transition**: Light blue expansion
- **Fusion Ignition**: Bright white explosion, very long duration
- **Thermal Explosion**: Bright red explosion, medium duration

## Performance Optimization

### For Large Simulations (1M+ particles)
```cpp
auto& perf = viewer->getPerformanceSettings();
perf.max_rendered_particles = 500000;  // Limit rendered particles
perf.use_instanced_rendering = true;   // Use GPU instancing
perf.adaptive_quality = true;          // Auto-reduce quality
perf.enable_vsync = false;             // Disable V-sync

viewer->setParticleStyle(GraphicsViewer::ParticleStyle::POINTS);
viewer->setRenderMode(GraphicsViewer::RenderMode::PARTICLES_ONLY);
```

### For Visual Quality
```cpp
auto& perf = viewer->getPerformanceSettings();
perf.enable_vsync = true;              // Smooth frame rate
perf.adaptive_quality = false;         // Maintain quality
perf.target_fps = 60;                  // Standard frame rate

viewer->setParticleStyle(GraphicsViewer::ParticleStyle::CIRCLES);
viewer->setRenderMode(GraphicsViewer::RenderMode::PARTICLES_AND_SPRINGS);
```

## Building with Graphics Viewer

### Prerequisites
- **SDL2**: Graphics and input handling library
- **C++17**: Required language standard
- **OpenGL**: Hardware acceleration (via SDL2)

### Ubuntu/Debian
```bash
sudo apt-get install libsdl2-dev
```

### CentOS/RHEL/Fedora
```bash
sudo dnf install SDL2-devel  # Fedora
sudo yum install SDL2-devel  # CentOS/RHEL
```

### macOS
```bash
brew install sdl2
```

### Build Commands

#### Using CMake
```bash
mkdir build && cd build
cmake -DBUILD_EXAMPLES=ON ..
make -j4
./simple_graphics_test        # Run basic graphics test
./graphics_viewer_demo        # Run full demo
```

#### Using Makefile
```bash
make examples                 # Build all examples including graphics
./build/bin/simple_graphics_test
```

### Build Without Graphics Viewer
If SDL2 is not available, the build system will automatically disable the graphics viewer and continue with ASCII-only rendering:
```bash
# Graphics viewer will be automatically disabled if SDL2 not found
make all
```

## Integration with Existing Applications

### Adding Graphics to Existing DigiStar Applications

1. **Include headers**:
```cpp
#include "src/viewer/graphics_viewer.h"
#ifdef DIGISTAR_HAS_GRAPHICS_VIEWER
#include "src/viewer/viewer_event_bridge.h"
#endif
```

2. **Initialize viewer**:
```cpp
#ifdef DIGISTAR_HAS_GRAPHICS_VIEWER
std::unique_ptr<GraphicsViewer> viewer;
viewer = ViewerFactory::createPerformanceViewer();
if (viewer->initialize("DigiStar", 1280, 720)) {
    // Graphics viewer available
    use_graphics = true;
} else {
    // Fall back to ASCII renderer
    use_graphics = false;
}
#endif
```

3. **Render loop**:
```cpp
if (use_graphics) {
#ifdef DIGISTAR_HAS_GRAPHICS_VIEWER
    viewer->beginFrame();
    viewer->renderSimulation(sim_state);
    viewer->renderUI(sim_state);
    viewer->presentFrame();
#endif
} else {
    // ASCII rendering
    ascii_renderer.render(sim_state);
}
```

## Troubleshooting

### Common Issues

**Graphics viewer fails to initialize**
- Check SDL2 installation
- Verify graphics drivers are installed
- Try running with software rendering: `SDL_VIDEODRIVER=software ./program`

**Poor performance with many particles**
- Reduce `max_rendered_particles` setting
- Use POINTS particle style instead of CIRCLES
- Disable springs rendering in dense simulations
- Enable adaptive quality

**Events not showing**
- Check event bridge connection to shared memory
- Verify event buffer is being written by physics engine
- Adjust event distance filtering settings
- Check event type filters

**Controls not responding**
- Ensure `processEvents()` is called every frame
- Check for conflicting keyboard/mouse handlers
- Verify SDL2 event handling is not blocked

### Debug Information

Enable debug output:
```cpp
// Show detailed performance information
viewer->getUISettings().show_performance_overlay = true;

// Show event processing statistics
auto stats = event_bridge->getStats();
printf("Events processed: %zu\n", stats.events_processed);
printf("Events filtered: %zu\n", stats.events_filtered);
```

## Future Enhancements

### Planned Features
- **3D rendering mode** with OpenGL/Vulkan
- **VR support** for immersive visualization
- **Network streaming** for remote visualization
- **Recording and playback** of simulation sessions
- **Custom particle shaders** for advanced effects
- **Multi-viewport support** for comparing different views
- **Plugin system** for custom rendering modes

### Extension Points
The graphics viewer is designed to be extensible:
- Custom particle rendering styles
- Additional event types and visualizations
- New color schemes and mapping functions
- Custom UI overlays and information panels
- Integration with external analysis tools