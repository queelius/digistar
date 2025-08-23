// Interactive Solar System Simulation
// Multi-body gravity with realistic orbital mechanics
// ASCII visualization with aspect ratio correction

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <algorithm>
#include <map>
#include <complex>

// Terminal characters are typically 2:1 (height:width)
constexpr float ASPECT_RATIO = 2.0f;
// Simulation Constants
namespace Constants {
    // Physics
    constexpr float GRAVITATIONAL_CONSTANT = 1.0f;  // Scaled G
    constexpr float SOFTENING_DISTANCE = 1.0f;      // Prevent singularities
    constexpr float TIME_STEP = 0.01f;            // Integration dt
    
    // Solar System (scaled for visibility)
    constexpr float SUN_MASS = 100000.0f;        // Arbitrary units
    constexpr float SUN_RADIUS = 3.0f;
    
    // Orbital distances (scaled much larger)
    constexpr float MERCURY_DISTANCE = 150.0f;
    constexpr float VENUS_DISTANCE = 250.0f;
    constexpr float EARTH_DISTANCE = 400.0f;
    constexpr float MARS_DISTANCE = 600.0f;
    constexpr float ASTEROID_BELT_INNER = 750.0f;
    constexpr float ASTEROID_BELT_OUTER = 950.0f;
    constexpr float JUPITER_DISTANCE = 1200.0f;
    constexpr float SATURN_DISTANCE = 1800.0f;
    constexpr float MOON_DISTANCE = 10.0f;  // From Earth
    
    // Planet masses (scaled down for stability)sti
    constexpr float MERCURY_MASS = 0.055f;
    constexpr float VENUS_MASS = 0.815f;
    constexpr float EARTH_MASS = 1.0f;
    constexpr float MARS_MASS = 0.107f;
    constexpr float JUPITER_MASS = 3.0f;    // Real: 317.8
    constexpr float SATURN_MASS = 1.0f;     // Real: 95.2
    constexpr float MOON_MASS = 0.012f;
    
    // Visual sizes
    constexpr float SMALL_PLANET_RADIUS = 0.5f;
    constexpr float MEDIUM_PLANET_RADIUS = 1.0f;
    constexpr float LARGE_PLANET_RADIUS = 2.5f;
    
    // Display
    constexpr int TRAIL_LENGTH = 500;
    constexpr int TRAIL_FADE_START = 50;
}

// 2D Vector
struct float2 {
    float x, y;
    
    float2() : x(0), y(0) {}
    float2(float x_, float y_) : x(x_), y(y_) {}
    
    float2 operator+(const float2& o) const { return {x + o.x, y + o.y}; }
    float2 operator-(const float2& o) const { return {x - o.x, y - o.y}; }
    float2 operator*(float s) const { return {x * s, y * s}; }
    float2 operator/(float s) const { return {x / s, y / s}; }
    float2& operator+=(const float2& o) { x += o.x; y += o.y; return *this; }
    float2& operator-=(const float2& o) { x -= o.x; y -= o.y; return *this; }
    
    float length() const { return std::sqrt(x * x + y * y); }
    float2 normalized() const { 
        float len = length();
        return len > 0 ? float2(x/len, y/len) : float2(0, 0);
    }
};

// Celestial body
struct Body {
    std::string name;
    float2 pos;
    float2 vel;
    float2 force;
    float mass;
    float radius;
    char symbol;
    std::string color;  // ANSI color code
    bool show_trail;
    std::vector<float2> trail;  // Position history for orbit trail
    
    Body(const std::string& n, float2 p, float2 v, float m, float r, char s, 
         const std::string& c, bool show_t = true) 
        : name(n), pos(p), vel(v), mass(m), radius(r), symbol(s), 
          color(c), show_trail(show_t) {
        trail.reserve(1000);
    }
    
    void update_trail() {
        trail.push_back(pos);
        if (trail.size() > Constants::TRAIL_LENGTH) {
            trail.erase(trail.begin());
        }
    }
};


// Particle-Mesh Gravity using FFT
class ParticleMeshGravity {
    static constexpr int GRID_SIZE = 2048;  // Much higher resolution
    static constexpr float WORLD_SIZE = 100000.0f;  // 100k units - still huge but more reasonable
    static constexpr float CELL_SIZE = WORLD_SIZE / GRID_SIZE;  // ~49 units per cell
    
    std::vector<std::vector<float>> density;
    std::vector<std::vector<float>> potential;
    std::vector<std::vector<float2>> field;
    
    // FFT arrays
    std::vector<std::vector<std::complex<float>>> density_fft;
    std::vector<std::vector<std::complex<float>>> potential_fft;
    std::vector<std::vector<float>> greens_function;
    
    // Twiddle factors for FFT
    std::vector<std::complex<float>> twiddle_factors;
    std::vector<int> bit_reverse_table;
    
public:
    ParticleMeshGravity() {
        density.resize(GRID_SIZE, std::vector<float>(GRID_SIZE, 0));
        potential.resize(GRID_SIZE, std::vector<float>(GRID_SIZE, 0));
        field.resize(GRID_SIZE, std::vector<float2>(GRID_SIZE, float2(0, 0)));
        
        // FFT arrays
        density_fft.resize(GRID_SIZE, std::vector<std::complex<float>>(GRID_SIZE));
        potential_fft.resize(GRID_SIZE, std::vector<std::complex<float>>(GRID_SIZE));
        greens_function.resize(GRID_SIZE, std::vector<float>(GRID_SIZE));
        
        // Initialize FFT tables
        init_fft_tables();
        init_greens_function();
    }
    
    void compute_forces(std::vector<Body>& bodies) {
        // Clear grids and forces
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                density[i][j] = 0;
                potential[i][j] = 0;
                field[i][j] = float2(0, 0);
            }
        }
        
        // Clear all particle forces first
        for (auto& b : bodies) {
            b.force = float2(0, 0);
        }
        
        // 1. Deposit mass onto grid using CIC (Cloud-In-Cell)
        for (const auto& b : bodies) {
            // Map position to grid (no wraparound)
            float gx = (b.pos.x + WORLD_SIZE/2) / CELL_SIZE;
            float gy = (b.pos.y + WORLD_SIZE/2) / CELL_SIZE;
            
            // Skip if outside grid (need space for CIC interpolation)
            if (gx < 0 || gx >= GRID_SIZE-1 || gy < 0 || gy >= GRID_SIZE-1) {
                continue;
            }
            
            int ix = std::max(0, std::min(GRID_SIZE-2, (int)gx));
            int iy = std::max(0, std::min(GRID_SIZE-2, (int)gy));
            float fx = std::max(0.0f, std::min(1.0f, gx - ix));
            float fy = std::max(0.0f, std::min(1.0f, gy - iy));
            
            // CIC interpolation with bounds checking
            // Normalize by cell volume
            float mass_per_cell = b.mass / (CELL_SIZE * CELL_SIZE);
            
            if (iy >= 0 && iy < GRID_SIZE && ix >= 0 && ix < GRID_SIZE)
                density[iy][ix] += mass_per_cell * (1-fx) * (1-fy);
            if (iy >= 0 && iy < GRID_SIZE && ix+1 >= 0 && ix+1 < GRID_SIZE)
                density[iy][ix+1] += mass_per_cell * fx * (1-fy);
            if (iy+1 >= 0 && iy+1 < GRID_SIZE && ix >= 0 && ix < GRID_SIZE)
                density[iy+1][ix] += mass_per_cell * (1-fx) * fy;
            if (iy+1 >= 0 && iy+1 < GRID_SIZE && ix+1 >= 0 && ix+1 < GRID_SIZE)
                density[iy+1][ix+1] += mass_per_cell * fx * fy;
        }
        
        // 2. Solve Poisson equation (simplified - use direct convolution)
        // In a real implementation, this would use FFT
        solve_poisson();
        
        // 3. Calculate force field from potential gradient
        calculate_field();
        
        // 4. Interpolate forces back to particles
        for (auto& b : bodies) {
            float gx = (b.pos.x + WORLD_SIZE/2) / CELL_SIZE;
            float gy = (b.pos.y + WORLD_SIZE/2) / CELL_SIZE;
            
            // Skip if outside grid
            if (gx < 0 || gx >= GRID_SIZE-1 || gy < 0 || gy >= GRID_SIZE-1) {
                b.force = float2(0, 0);
                continue;
            }
            
            int ix = (int)gx;
            int iy = (int)gy;
            float fx = gx - ix;
            float fy = gy - iy;
            
            // CIC interpolation of force (no wraparound)
            float2 f00 = field[iy][ix];
            float2 f10 = field[iy][ix+1];
            float2 f01 = field[iy+1][ix];
            float2 f11 = field[iy+1][ix+1];
            
            b.force = f00 * ((1-fx) * (1-fy)) +
                     f10 * (fx * (1-fy)) +
                     f01 * ((1-fx) * fy) +
                     f11 * (fx * fy);
            
            b.force = b.force * b.mass;  // F = ma, so multiply by mass
        }
    }
    
private:
    void init_fft_tables() {
        // Initialize twiddle factors
        twiddle_factors.resize(GRID_SIZE);
        for (int i = 0; i < GRID_SIZE; i++) {
            float angle = -2.0f * M_PI * i / GRID_SIZE;
            twiddle_factors[i] = std::complex<float>(cos(angle), sin(angle));
        }
        
        // Initialize bit reversal table
        bit_reverse_table.resize(GRID_SIZE);
        int log2_size = 0;
        int n = GRID_SIZE;
        while (n > 1) { n >>= 1; log2_size++; }
        
        for (int i = 0; i < GRID_SIZE; i++) {
            int rev = 0;
            int n = i;
            for (int j = 0; j < log2_size; j++) {
                rev = (rev << 1) | (n & 1);
                n >>= 1;
            }
            bit_reverse_table[i] = rev;
        }
    }
    
    void init_greens_function() {
        // Initialize Green's function for Poisson solver in Fourier space
        for (int ky = 0; ky < GRID_SIZE; ky++) {
            for (int kx = 0; kx < GRID_SIZE; kx++) {
                // Wave numbers (accounting for Nyquist)
                float kx_val = (kx <= GRID_SIZE/2) ? kx : kx - GRID_SIZE;
                float ky_val = (ky <= GRID_SIZE/2) ? ky : ky - GRID_SIZE;
                
                kx_val *= 2.0f * M_PI / WORLD_SIZE;
                ky_val *= 2.0f * M_PI / WORLD_SIZE;
                
                float k2 = kx_val * kx_val + ky_val * ky_val;
                
                if (k2 > 0) {
                    // Green's function: G(k) = -4πG / (k² + softening²)
                    float softening = Constants::SOFTENING_DISTANCE / WORLD_SIZE;
                    greens_function[ky][kx] = 
                        -4.0f * M_PI * Constants::GRAVITATIONAL_CONSTANT / (k2 + softening * softening);
                } else {
                    greens_function[ky][kx] = 0;  // DC component
                }
            }
        }
    }
    
    // 1D FFT using Cooley-Tukey algorithm
    void fft1d(std::complex<float>* data, bool inverse) {
        // Bit reversal
        for (int i = 0; i < GRID_SIZE; i++) {
            int j = bit_reverse_table[i];
            if (i < j) {
                std::swap(data[i], data[j]);
            }
        }
        
        // Cooley-Tukey FFT
        int log2_size = 0;
        int n = GRID_SIZE;
        while (n > 1) { n >>= 1; log2_size++; }
        
        for (int stage = 1; stage <= log2_size; stage++) {
            int m = 1 << stage;  // 2^stage
            int m2 = m >> 1;      // m/2
            
            for (int k = 0; k < GRID_SIZE; k += m) {
                for (int j = 0; j < m2; j++) {
                    int idx1 = k + j;
                    int idx2 = idx1 + m2;
                    
                    // Twiddle factor
                    int twiddle_idx = (j * GRID_SIZE) / m;
                    std::complex<float> w = twiddle_factors[twiddle_idx];
                    if (inverse) w = std::conj(w);
                    
                    // Butterfly operation
                    std::complex<float> t = w * data[idx2];
                    data[idx2] = data[idx1] - t;
                    data[idx1] = data[idx1] + t;
                }
            }
        }
        
        // Normalize for inverse transform
        if (inverse) {
            float norm = 1.0f / GRID_SIZE;
            for (int i = 0; i < GRID_SIZE; i++) {
                data[i] *= norm;
            }
        }
    }
    
    // 2D FFT
    void fft2d(std::vector<std::vector<std::complex<float>>>& data, bool inverse) {
        // FFT along rows
        for (int y = 0; y < GRID_SIZE; y++) {
            fft1d(data[y].data(), inverse);
        }
        
        // FFT along columns
        std::vector<std::complex<float>> col(GRID_SIZE);
        for (int x = 0; x < GRID_SIZE; x++) {
            // Copy column
            for (int y = 0; y < GRID_SIZE; y++) {
                col[y] = data[y][x];
            }
            
            // FFT
            fft1d(col.data(), inverse);
            
            // Copy back
            for (int y = 0; y < GRID_SIZE; y++) {
                data[y][x] = col[y];
            }
        }
        
        // Note: 1D FFT already normalizes, so 2D gets normalized twice which is correct
    }
    
    void solve_poisson() {
        // Convert density to complex for FFT
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                density_fft[i][j] = std::complex<float>(density[i][j], 0);
            }
        }
        
        // Forward FFT of density
        fft2d(density_fft, false);
        
        // Solve Poisson equation in Fourier space: Φ(k) = G(k) * ρ(k)
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                potential_fft[i][j] = density_fft[i][j] * greens_function[i][j];
            }
        }
        
        // Inverse FFT to get potential
        fft2d(potential_fft, true);
        
        // Extract real part
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                potential[i][j] = potential_fft[i][j].real();
            }
        }
    }
    
    void calculate_field() {
        // Calculate force field as negative gradient of potential
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                // Use forward/backward difference at boundaries
                float fx = 0, fy = 0;
                
                // X gradient
                if (j > 0 && j < GRID_SIZE-1) {
                    fx = -(potential[i][j+1] - potential[i][j-1]) / (2 * CELL_SIZE);
                } else if (j == 0 && j < GRID_SIZE-1) {
                    fx = -(potential[i][j+1] - potential[i][j]) / CELL_SIZE;
                } else if (j == GRID_SIZE-1 && j > 0) {
                    fx = -(potential[i][j] - potential[i][j-1]) / CELL_SIZE;
                }
                
                // Y gradient
                if (i > 0 && i < GRID_SIZE-1) {
                    fy = -(potential[i+1][j] - potential[i-1][j]) / (2 * CELL_SIZE);
                } else if (i == 0 && i < GRID_SIZE-1) {
                    fy = -(potential[i+1][j] - potential[i][j]) / CELL_SIZE;
                } else if (i == GRID_SIZE-1 && i > 0) {
                    fy = -(potential[i][j] - potential[i-1][j]) / CELL_SIZE;
                }
                
                field[i][j] = float2(fx, fy);
            }
        }
    }
};

// ASCII Visualization with aspect ratio correction
class ASCIIRenderer {
    int width = 120;
    int height = 40;
    float2 camera_pos;
    float zoom = 1.0f;
    std::vector<std::vector<char>> buffer;
    std::vector<std::vector<std::string>> color_buffer;
    bool show_trails = true;
    bool show_names = true;
    int focus_body = -1;  // -1 for no focus, otherwise body index
    
public:
    ASCIIRenderer() : camera_pos(0, 0) {
        buffer.resize(height, std::vector<char>(width, ' '));
        color_buffer.resize(height, std::vector<std::string>(width, ""));
        zoom = 0.05f;  // Start very zoomed out to see the whole system
    }
    
    void pan(float dx, float dy) {
        camera_pos.x += dx / zoom;
        camera_pos.y += dy / zoom;
    }
    
    void zoom_in() { zoom *= 1.2f; }
    void zoom_out() { zoom /= 1.2f; }
    void reset_view() { camera_pos = float2(0, 0); zoom = 1.0f; }
    void toggle_trails() { show_trails = !show_trails; }
    void toggle_names() { show_names = !show_names; }
    
    void focus_on(int body_idx, const std::vector<Body>& bodies) {
        focus_body = body_idx;
        if (body_idx >= 0 && body_idx < bodies.size()) {
            camera_pos = bodies[body_idx].pos;
        }
    }
    
    void render(const std::vector<Body>& bodies, float sim_time, bool physics_stalled = false) {
        // Clear buffers
        for (auto& row : buffer) {
            std::fill(row.begin(), row.end(), ' ');
        }
        for (auto& row : color_buffer) {
            std::fill(row.begin(), row.end(), "");
        }
        
        // Update camera for focused body
        if (focus_body >= 0 && focus_body < bodies.size()) {
            camera_pos = bodies[focus_body].pos;
        }
        
        // Draw trails first (behind bodies)
        if (show_trails) {
            for (const auto& body : bodies) {
                if (body.show_trail) {
                    draw_trail(body);
                }
            }
        }
        
        // Draw bodies
        for (size_t i = 0; i < bodies.size(); i++) {
            draw_body(bodies[i], i == focus_body);
        }
        
        // Draw UI
        draw_ui(bodies, sim_time, physics_stalled);
        
        // Output to terminal
        display();
    }
    
private:
    void draw_trail(const Body& body) {
        for (size_t i = 1; i < body.trail.size(); i++) {
            int sx = world_to_screen_x(body.trail[i].x);
            int sy = world_to_screen_y(body.trail[i].y);
            
            if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                if (buffer[sy][sx] == ' ') {
                    // Fade trail based on age
                    if (i > body.trail.size() - Constants::TRAIL_FADE_START) {
                        buffer[sy][sx] = '.';
                    } else {
                        buffer[sy][sx] = '\'';
                    }
                    color_buffer[sy][sx] = "\033[0;90m";  // Dark gray
                }
            }
        }
    }
    
    void draw_body(const Body& body, bool is_focused) {
        int sx = world_to_screen_x(body.pos.x);
        int sy = world_to_screen_y(body.pos.y);
        
        // Draw body radius (with aspect correction)
        int screen_radius_x = std::max(1, (int)(body.radius * zoom));
        int screen_radius_y = std::max(1, (int)(body.radius * zoom / ASPECT_RATIO));
        
        // Draw filled circle
        for (int dy = -screen_radius_y; dy <= screen_radius_y; dy++) {
            for (int dx = -screen_radius_x; dx <= screen_radius_x; dx++) {
                // Ellipse equation with aspect ratio
                float norm_x = (float)dx / screen_radius_x;
                float norm_y = (float)dy / screen_radius_y;
                
                if (norm_x * norm_x + norm_y * norm_y <= 1.0f) {
                    int px = sx + dx;
                    int py = sy + dy;
                    
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        // Center gets special symbol
                        if (dx == 0 && dy == 0) {
                            buffer[py][px] = body.symbol;
                            color_buffer[py][px] = body.color;
                        } else {
                            // Fill based on distance from center
                            float dist = std::sqrt(norm_x * norm_x + norm_y * norm_y);
                            if (dist > 0.8f) {
                                if (buffer[py][px] == ' ') {
                                    buffer[py][px] = '.';
                                    color_buffer[py][px] = body.color;
                                }
                            } else {
                                buffer[py][px] = body.symbol;
                                color_buffer[py][px] = body.color;
                            }
                        }
                    }
                }
            }
        }
        
        // Draw name label
        if (show_names && body.radius * zoom > 2) {
            int label_y = sy - screen_radius_y - 1;
            int label_x = sx - body.name.length() / 2;
            
            if (label_y >= 0 && label_y < height) {
                for (size_t i = 0; i < body.name.length(); i++) {
                    int px = label_x + i;
                    if (px >= 0 && px < width) {
                        buffer[label_y][px] = body.name[i];
                        color_buffer[label_y][px] = is_focused ? "\033[1;32m" : "\033[0;37m";
                    }
                }
            }
        }
    }
    
    void draw_ui(const std::vector<Body>& bodies, float sim_time, bool physics_stalled = false) {
        // Top info bar
        std::string info = "Time: " + format_time(sim_time) + 
                          " | Zoom: " + std::to_string((int)(zoom * 100)) + "% | ";
        
        if (physics_stalled) {
            info += "PHYSICS STALLED!";
        } else if (focus_body >= 0 && focus_body < bodies.size()) {
            info += "Focus: " + bodies[focus_body].name;
        } else {
            info += "Free camera";
        }
        
        for (size_t i = 0; i < info.length() && i < width; i++) {
            buffer[0][i] = info[i];
            color_buffer[0][i] = "\033[1;37m";  // Bright white
        }
        
        // Body list (right side)
        int list_x = width - 20;
        for (size_t i = 0; i < bodies.size() && i < 10; i++) {
            std::string num = std::to_string(i) + ": " + bodies[i].name;
            for (size_t j = 0; j < num.length() && list_x + j < width; j++) {
                buffer[2 + i][list_x + j] = num[j];
                color_buffer[2 + i][list_x + j] = bodies[i].color;
            }
        }
        
        // Controls help (bottom)
        std::string controls = "WASD:pan +-:zoom R:reset T:trails N:names 0-9:focus Q:quit";
        for (size_t i = 0; i < controls.length() && i < width; i++) {
            buffer[height-1][i] = controls[i];
            color_buffer[height-1][i] = "\033[0;36m";  // Cyan
        }
    }
    
    std::string format_time(float sim_units) {
        // Convert simulation units to days
        // Assuming 1 sim unit = 1 hour for reasonable time scale
        float hours = sim_units;
        float days = hours / 24.0f;
        
        if (days < 365) {
            return std::to_string((int)days) + " days";
        }
        float years = days / 365.25f;
        return std::to_string((int)years) + " years";
    }
    
    int world_to_screen_x(float wx) {
        // Apply aspect ratio correction to X coordinate
        return width/2 + (int)((wx - camera_pos.x) * zoom);
    }
    
    int world_to_screen_y(float wy) {
        // Y coordinate doesn't need aspect correction (already taller)
        return height/2 - (int)((wy - camera_pos.y) * zoom / ASPECT_RATIO);
    }
    
    void display() {
        // Clear screen
        std::cout << "\033[2J\033[H";
        
        // Draw buffer WITHOUT colors
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                std::cout << buffer[y][x];
            }
            std::cout << '\n';
        }
        std::cout << std::flush;
    }
};

// Terminal input handling
class InputHandler {
    struct termios old_tio, new_tio;
    
public:
    InputHandler() {
        tcgetattr(STDIN_FILENO, &old_tio);
        new_tio = old_tio;
        new_tio.c_lflag &= (~ICANON & ~ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
    }
    
    ~InputHandler() {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
    }
    
    char get_input() {
        char c = 0;
        read(STDIN_FILENO, &c, 1);
        return c;
    }
};

// Solar System Simulation
class SolarSystemSimulation {
    std::vector<Body> bodies;
    ParticleMeshGravity gravity;  // PM method
    ASCIIRenderer renderer;
    InputHandler input;
    
    float sim_time = 0;
    float dt = Constants::TIME_STEP;
    float time_scale = 1.0f;  // Time multiplier
    bool paused = false;
    bool running = true;
    bool physics_stalled = false;  // Track if physics stopped due to numerical issues
    bool use_pm_gravity = false;  // Use PM instead of direct N-body
    
public:
    void set_gravity_method(bool use_pm) {
        use_pm_gravity = use_pm;
    }
    
    void initialize() {
        using namespace Constants;
        
        // Sun at center
        bodies.emplace_back("Sun", float2(0, 0), float2(0, 0), 
                           SUN_MASS, SUN_RADIUS, '@', "\033[1;33m", false);
        
        // Mercury
        float mercury_vel = std::sqrt(SUN_MASS / MERCURY_DISTANCE);
        bodies.emplace_back("Mercury", float2(MERCURY_DISTANCE, 0), float2(0, mercury_vel),
                           MERCURY_MASS, SMALL_PLANET_RADIUS, 'o', "\033[0;37m");
        
        // Venus
        float venus_vel = std::sqrt(SUN_MASS / VENUS_DISTANCE);
        bodies.emplace_back("Venus", float2(VENUS_DISTANCE * 0.7f, VENUS_DISTANCE * 0.7f), 
                           float2(-venus_vel * 0.7f, venus_vel * 0.7f),
                           VENUS_MASS, MEDIUM_PLANET_RADIUS * 0.9f, 'o', "\033[1;33m");
        
        // Earth
        float earth_vel = std::sqrt(SUN_MASS / EARTH_DISTANCE);
        bodies.emplace_back("Earth", float2(0, EARTH_DISTANCE), float2(-earth_vel, 0),
                           EARTH_MASS, MEDIUM_PLANET_RADIUS, 'o', "\033[0;34m");
        
        // Mars
        float mars_vel = std::sqrt(SUN_MASS / MARS_DISTANCE);
        bodies.emplace_back("Mars", float2(-MARS_DISTANCE, 0), float2(0, -mars_vel),
                           MARS_MASS, MEDIUM_PLANET_RADIUS * 0.7f, 'o', "\033[0;31m");
        
        // Jupiter
        float jupiter_vel = std::sqrt(SUN_MASS / JUPITER_DISTANCE);
        bodies.emplace_back("Jupiter", float2(JUPITER_DISTANCE * 0.5f, -JUPITER_DISTANCE * 0.866f),
                           float2(jupiter_vel * 0.866f, jupiter_vel * 0.5f),
                           JUPITER_MASS, LARGE_PLANET_RADIUS, 'O', "\033[0;33m");
        
        // Saturn
        float saturn_vel = std::sqrt(SUN_MASS / SATURN_DISTANCE);
        bodies.emplace_back("Saturn", float2(0, -SATURN_DISTANCE), float2(saturn_vel, 0),
                           SATURN_MASS, LARGE_PLANET_RADIUS * 0.8f, 'O', "\033[1;33m");
        
        // Moon (orbiting Earth)
        float moon_vel = std::sqrt(EARTH_MASS / MOON_DISTANCE) + earth_vel;
        bodies.emplace_back("Moon", float2(MOON_DISTANCE, EARTH_DISTANCE), 
                           float2(-earth_vel, moon_vel),
                           MOON_MASS, SMALL_PLANET_RADIUS * 0.6f, '.', "\033[0;37m");
        
        // Add asteroid belt (10000 asteroids)
        std::srand(42);  // Fixed seed for reproducibility
        int num_asteroids = 10000;
        
        for (int i = 0; i < num_asteroids; i++) {
            // Random radius within belt
            float r = ASTEROID_BELT_INNER + 
                     (ASTEROID_BELT_OUTER - ASTEROID_BELT_INNER) * (rand() / (float)RAND_MAX);
            
            // Random angle
            float theta = 2 * M_PI * (rand() / (float)RAND_MAX);
            
            // Position
            float x = r * cos(theta);
            float y = r * sin(theta);
            
            // Circular orbital velocity
            float v_orbital = std::sqrt(SUN_MASS / r);
            
            // Add some random variation to velocity (for eccentricity)
            float v_variation = 0.1f * v_orbital * ((rand() / (float)RAND_MAX) - 0.5f);
            
            // Velocity perpendicular to radius
            float vx = -v_orbital * sin(theta) + v_variation * cos(theta);
            float vy = v_orbital * cos(theta) + v_variation * sin(theta);
            
            // Very small mass for asteroids
            float asteroid_mass = 0.00001f * ((rand() / (float)RAND_MAX) + 0.5f);
            
            // Create asteroid (no trail to save memory)
            bodies.emplace_back("Asteroid" + std::to_string(i), 
                               float2(x, y), float2(vx, vy),
                               asteroid_mass, 0.1f, '.', "\033[0;90m", false);
        }
        
        std::cout << "Created " << num_asteroids << " asteroids in the belt\n";
        std::cout << "Total bodies: " << bodies.size() << "\n";
        
        // Don't adjust for center of mass - Sun is so massive it barely moves
        // This prevents giving the Sun an initial velocity
    }
    
    void compute_nbody_forces() {
        // Direct O(N^2) gravity calculation
        for (auto& b : bodies) {
            b.force = float2(0, 0);
        }
        
        for (size_t i = 0; i < bodies.size(); i++) {
            for (size_t j = i + 1; j < bodies.size(); j++) {
                float2 delta = bodies[j].pos - bodies[i].pos;
                float dist_sq = delta.x * delta.x + delta.y * delta.y;
                
                // Apply softening to prevent singularities
                dist_sq = std::max(dist_sq, Constants::SOFTENING_DISTANCE * Constants::SOFTENING_DISTANCE);
                
                float dist = std::sqrt(dist_sq);
                float force_mag = Constants::GRAVITATIONAL_CONSTANT * bodies[i].mass * bodies[j].mass / dist_sq;
                
                float2 force = delta.normalized() * force_mag;
                
                bodies[i].force += force;
                bodies[j].force -= force;  // Newton's third law
            }
        }
    }
    
    void update() {
        if (paused) return;
        
        // Velocity Verlet integration for stable orbits
        float scaled_dt = dt * time_scale;
        
        // Update positions
        for (auto& b : bodies) {
            float2 acc = b.force / b.mass;
            
            // Check for NaN or extremely large accelerations
            if (std::isnan(acc.x) || std::isnan(acc.y) || 
                std::abs(acc.x) > 1e6f || std::abs(acc.y) > 1e6f) {
                // Skip this update to prevent corruption
                physics_stalled = true;
                return;
            }
            
            b.pos += b.vel * scaled_dt + acc * (0.5f * scaled_dt * scaled_dt);
            b.update_trail();
        }
        
        // Save old forces
        std::vector<float2> old_forces;
        for (const auto& b : bodies) {
            old_forces.push_back(b.force);
        }
        
        // Compute new forces
        if (use_pm_gravity) {
            gravity.compute_forces(bodies);
        } else {
            compute_nbody_forces();
        }
        
        // Update velocities
        for (size_t i = 0; i < bodies.size(); i++) {
            float2 avg_force = (old_forces[i] + bodies[i].force) * 0.5f;
            float2 acc = avg_force / bodies[i].mass;
            
            // Check for NaN or extremely large accelerations
            if (std::isnan(acc.x) || std::isnan(acc.y) || 
                std::abs(acc.x) > 1e6f || std::abs(acc.y) > 1e6f) {
                // Skip velocity update to prevent corruption
                continue;
            }
            
            bodies[i].vel += acc * scaled_dt;
        }
        
        sim_time += scaled_dt;
    }
    
    void handle_input() {
        char c = input.get_input();
        
        switch(c) {
            case 'q': case 'Q': running = false; break;
            case ' ': paused = !paused; break;
            case 'r': case 'R': renderer.reset_view(); break;
            case 't': case 'T': renderer.toggle_trails(); break;
            case 'n': case 'N': renderer.toggle_names(); break;
            
            // Camera controls
            case 'w': case 'W': renderer.pan(0, 10); break;
            case 's': case 'S': renderer.pan(0, -10); break;
            case 'a': case 'A': renderer.pan(-10, 0); break;
            case 'd': case 'D': renderer.pan(10, 0); break;
            
            // Zoom
            case '+': case '=': renderer.zoom_in(); break;
            case '-': case '_': renderer.zoom_out(); break;
            
            // Time control - allow much slower speeds
            case ',': time_scale = std::max(0.1f, time_scale * 0.5f); break;  // Can go down to 0.01x
            case '.': time_scale = std::min(100.0f, time_scale * 2.0f); break;  // Can go up to 100x
            
            // Focus on bodies
            case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8': case '9':
                renderer.focus_on(c - '0', bodies);
                break;
            
            case 'f': case 'F': renderer.focus_on(-1, bodies); break;  // Free camera
        }
    }
    
    void run() {
        initialize();
        
        auto last_time = std::chrono::steady_clock::now();
        const auto frame_duration = std::chrono::milliseconds(50);  // 20 FPS
        
        while (running) {
            auto current_time = std::chrono::steady_clock::now();
            
            // Physics update (multiple substeps for stability)
            for (int i = 0; i < 10; i++) {
                update();
            }
            
            // Handle input
            handle_input();
            
            // Render at fixed framerate
            if (current_time - last_time >= frame_duration) {
                renderer.render(bodies, sim_time, physics_stalled);
                last_time = current_time;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "\033[2J\033[H";
    // Parse command line arguments
    bool use_pm = false;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--pm" || arg == "-p") {
            use_pm = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --pm, -p     Use Particle-Mesh gravity (O(N))\n";
            std::cout << "  --nbody, -n  Use direct N-body gravity (O(N²)) [default]\n";
            std::cout << "  --help, -h   Show this help\n";
            return 0;
        }
    }
    
    std::cout << "=== Solar System Simulation ===\n";
    std::cout << (use_pm ? "Particle-Mesh (PM) Gravity" : "Direct N-Body Gravity") << " with Realistic Orbits\n\n";
    
    if (use_pm) {
        std::cout << "WARNING: PM method assumes periodic boundaries and works best with\n";
        std::cout << "         large numbers of uniformly distributed particles (10,000+).\n";
        std::cout << "         For accurate solar system dynamics, use default N-body mode.\n\n";
    }
    std::cout << "Controls:\n";
    std::cout << "  WASD    - Pan camera\n";
    std::cout << "  +/-     - Zoom in/out\n";
    std::cout << "  0-9     - Focus on body\n";
    std::cout << "  F       - Free camera\n";
    std::cout << "  T       - Toggle trails\n";
    std::cout << "  N       - Toggle names\n";
    std::cout << "  ,/.     - Slow/speed time\n";
    std::cout << "  Space   - Pause\n";
    std::cout << "  R       - Reset view\n";
    std::cout << "  Q       - Quit\n\n";
    std::cout << "Terminal aspect ratio correction: " << ASPECT_RATIO << ":1\n\n";
    std::cout << "Press any key to start...";
    std::cin.get();
    
    try {
        SolarSystemSimulation sim;
        sim.set_gravity_method(use_pm);
        sim.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\033[2J\033[H";
    std::cout << "Simulation ended.\n";
    
    return 0;
}