// Solar System Simulation with FFTW3-based Particle Mesh
// Uses the battle-tested FFTW library for accurate PM gravity

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
#include <complex>
#include <fftw3.h>

// Terminal aspect ratio
constexpr float ASPECT_RATIO = 2.0f;

// Simulation Constants
namespace Constants {
    constexpr float GRAVITATIONAL_CONSTANT = 1.0f;
    constexpr float SOFTENING_DISTANCE = 0.5f;
    constexpr float TIME_STEP = 0.01f;
    
    // Solar System
    constexpr float SUN_MASS = 100000.0f;
    constexpr float SUN_RADIUS = 3.0f;
    
    // Orbital distances
    constexpr float MERCURY_DISTANCE = 150.0f;
    constexpr float VENUS_DISTANCE = 250.0f;
    constexpr float EARTH_DISTANCE = 400.0f;
    constexpr float MARS_DISTANCE = 600.0f;
    constexpr float JUPITER_DISTANCE = 1200.0f;
    constexpr float SATURN_DISTANCE = 1800.0f;
    
    // Planet masses
    constexpr float MERCURY_MASS = 0.055f;
    constexpr float VENUS_MASS = 0.815f;
    constexpr float EARTH_MASS = 1.0f;
    constexpr float MARS_MASS = 0.107f;
    constexpr float JUPITER_MASS = 3.0f;
    constexpr float SATURN_MASS = 1.0f;
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
    
    Body(const std::string& n, float2 p, float2 v, float m, float r, char s) 
        : name(n), pos(p), vel(v), mass(m), radius(r), symbol(s) {
        force = float2(0, 0);
    }
};

// FFTW-based Particle Mesh gravity solver
class FFTWParticleMesh {
private:
    static constexpr int GRID_SIZE = 512;  // Power of 2
    static constexpr float WORLD_SIZE = 20000.0f;  // Large enough to minimize boundary effects
    static constexpr float CELL_SIZE = WORLD_SIZE / GRID_SIZE;
    
    // Real-space grids
    float* density;
    float* potential;
    float* force_x;
    float* force_y;
    
    // Fourier-space grids
    fftwf_complex* density_fft;
    fftwf_complex* potential_fft;
    float* greens_function;
    
    // FFTW plans
    fftwf_plan plan_forward;
    fftwf_plan plan_backward;
    
public:
    FFTWParticleMesh() {
        int n = GRID_SIZE;
        int n2 = n * n;
        
        // Allocate aligned memory for FFTW
        density = fftwf_alloc_real(n2);
        potential = fftwf_alloc_real(n2);
        force_x = fftwf_alloc_real(n2);
        force_y = fftwf_alloc_real(n2);
        greens_function = fftwf_alloc_real(n2);
        
        // Complex arrays (n * (n/2 + 1) for r2c transforms)
        int nc = n * (n/2 + 1);
        density_fft = fftwf_alloc_complex(nc);
        potential_fft = fftwf_alloc_complex(nc);
        
        // Create FFTW plans for 2D real-to-complex transforms
        plan_forward = fftwf_plan_dft_r2c_2d(n, n, density, density_fft, FFTW_ESTIMATE);
        plan_backward = fftwf_plan_dft_c2r_2d(n, n, potential_fft, potential, FFTW_ESTIMATE);
        
        // Initialize Green's function
        init_greens_function();
    }
    
    ~FFTWParticleMesh() {
        fftwf_destroy_plan(plan_forward);
        fftwf_destroy_plan(plan_backward);
        
        fftwf_free(density);
        fftwf_free(potential);
        fftwf_free(force_x);
        fftwf_free(force_y);
        fftwf_free(greens_function);
        fftwf_free(density_fft);
        fftwf_free(potential_fft);
    }
    
    void init_greens_function() {
        int n = GRID_SIZE;
        int n2 = n * n;
        
        // Initialize in real space then transform
        for (int i = 0; i < n2; i++) {
            greens_function[i] = 0;
        }
        
        // Create Green's function in Fourier space
        // For isolated boundaries, we use -4πG/k²
        for (int ky = 0; ky < n; ky++) {
            for (int kx = 0; kx <= n/2; kx++) {  // Only need half due to symmetry
                // Wave numbers
                float kx_val = (kx <= n/2) ? kx : kx - n;
                float ky_val = (ky <= n/2) ? ky : ky - n;
                
                kx_val *= 2.0f * M_PI / WORLD_SIZE;
                ky_val *= 2.0f * M_PI / WORLD_SIZE;
                
                float k2 = kx_val * kx_val + ky_val * ky_val;
                
                int idx = ky * (n/2 + 1) + kx;
                
                if (k2 > 0) {
                    float softening = Constants::SOFTENING_DISTANCE / WORLD_SIZE;
                    greens_function[idx] = -4.0f * M_PI * Constants::GRAVITATIONAL_CONSTANT / 
                                          (k2 + softening * softening);
                } else {
                    greens_function[idx] = 0;  // DC component
                }
            }
        }
    }
    
    void compute_forces(std::vector<Body>& bodies) {
        int n = GRID_SIZE;
        int n2 = n * n;
        
        // Clear grids
        std::memset(density, 0, n2 * sizeof(float));
        std::memset(potential, 0, n2 * sizeof(float));
        std::memset(force_x, 0, n2 * sizeof(float));
        std::memset(force_y, 0, n2 * sizeof(float));
        
        // Clear forces
        for (auto& b : bodies) {
            b.force = float2(0, 0);
        }
        
        // 1. Assign particles to mesh using CIC
        for (const auto& b : bodies) {
            // Map to grid coordinates
            float gx = (b.pos.x + WORLD_SIZE/2) / CELL_SIZE;
            float gy = (b.pos.y + WORLD_SIZE/2) / CELL_SIZE;
            
            // Skip if outside grid
            if (gx < 0 || gx >= n-1 || gy < 0 || gy >= n-1) continue;
            
            int ix = (int)gx;
            int iy = (int)gy;
            float fx = gx - ix;
            float fy = gy - iy;
            
            // CIC weights
            float w00 = (1-fx) * (1-fy);
            float w10 = fx * (1-fy);
            float w01 = (1-fx) * fy;
            float w11 = fx * fy;
            
            // Deposit mass (normalized by cell area)
            float mass_density = b.mass / (CELL_SIZE * CELL_SIZE);
            
            density[iy * n + ix] += mass_density * w00;
            density[iy * n + (ix+1)] += mass_density * w10;
            density[(iy+1) * n + ix] += mass_density * w01;
            density[(iy+1) * n + (ix+1)] += mass_density * w11;
        }
        
        // 2. FFT the density field
        fftwf_execute(plan_forward);
        
        // 3. Solve Poisson equation in Fourier space
        for (int ky = 0; ky < n; ky++) {
            for (int kx = 0; kx <= n/2; kx++) {
                int idx = ky * (n/2 + 1) + kx;
                
                // Multiply by Green's function
                float real = density_fft[idx][0] * greens_function[idx];
                float imag = density_fft[idx][1] * greens_function[idx];
                
                potential_fft[idx][0] = real;
                potential_fft[idx][1] = imag;
            }
        }
        
        // 4. Inverse FFT to get potential
        fftwf_execute(plan_backward);
        
        // Normalize (FFTW doesn't normalize inverse transform)
        float norm = 1.0f / (n * n);
        for (int i = 0; i < n2; i++) {
            potential[i] *= norm;
        }
        
        // 5. Calculate force field from potential gradient
        for (int iy = 0; iy < n; iy++) {
            for (int ix = 0; ix < n; ix++) {
                int idx = iy * n + ix;
                
                // Central differences for gradient
                int ixp = (ix + 1) % n;
                int ixm = (ix - 1 + n) % n;
                int iyp = (iy + 1) % n;
                int iym = (iy - 1 + n) % n;
                
                force_x[idx] = -(potential[iy * n + ixp] - potential[iy * n + ixm]) / (2 * CELL_SIZE);
                force_y[idx] = -(potential[iyp * n + ix] - potential[iym * n + ix]) / (2 * CELL_SIZE);
            }
        }
        
        // 6. Interpolate forces back to particles
        for (auto& b : bodies) {
            float gx = (b.pos.x + WORLD_SIZE/2) / CELL_SIZE;
            float gy = (b.pos.y + WORLD_SIZE/2) / CELL_SIZE;
            
            if (gx < 0 || gx >= n-1 || gy < 0 || gy >= n-1) continue;
            
            int ix = (int)gx;
            int iy = (int)gy;
            float fx = gx - ix;
            float fy = gy - iy;
            
            // CIC interpolation
            float w00 = (1-fx) * (1-fy);
            float w10 = fx * (1-fy);
            float w01 = (1-fx) * fy;
            float w11 = fx * fy;
            
            int idx00 = iy * n + ix;
            int idx10 = iy * n + (ix+1);
            int idx01 = (iy+1) * n + ix;
            int idx11 = (iy+1) * n + (ix+1);
            
            float fx_total = force_x[idx00] * w00 + force_x[idx10] * w10 +
                           force_x[idx01] * w01 + force_x[idx11] * w11;
            float fy_total = force_y[idx00] * w00 + force_y[idx10] * w10 +
                           force_y[idx01] * w01 + force_y[idx11] * w11;
            
            b.force = float2(fx_total, fy_total) * b.mass;
        }
    }
};

// Direct N-body gravity
void compute_nbody_forces(std::vector<Body>& bodies) {
    for (auto& b : bodies) {
        b.force = float2(0, 0);
    }
    
    for (size_t i = 0; i < bodies.size(); i++) {
        for (size_t j = i + 1; j < bodies.size(); j++) {
            float2 delta = bodies[j].pos - bodies[i].pos;
            float dist_sq = delta.x * delta.x + delta.y * delta.y;
            dist_sq = std::max(dist_sq, Constants::SOFTENING_DISTANCE * Constants::SOFTENING_DISTANCE);
            
            float dist = std::sqrt(dist_sq);
            float force_mag = Constants::GRAVITATIONAL_CONSTANT * bodies[i].mass * bodies[j].mass / dist_sq;
            
            float2 force = delta.normalized() * force_mag;
            
            bodies[i].force += force;
            bodies[j].force -= force;
        }
    }
}

// Simple display
void display(const std::vector<Body>& bodies, float time, bool use_pm) {
    std::cout << "\033[2J\033[H";  // Clear screen
    
    std::cout << "Solar System Simulation - " << (use_pm ? "FFTW Particle Mesh" : "Direct N-Body") << "\n";
    std::cout << "Time: " << (int)(time) << " days | Bodies: " << bodies.size() << "\n\n";
    
    // Simple ASCII view (80x25)
    const int WIDTH = 80;
    const int HEIGHT = 25;
    std::vector<std::vector<char>> screen(HEIGHT, std::vector<char>(WIDTH, ' '));
    
    // Draw bodies
    for (const auto& b : bodies) {
        // Map to screen (simple projection)
        int sx = WIDTH/2 + (int)(b.pos.x * 0.02f);
        int sy = HEIGHT/2 - (int)(b.pos.y * 0.01f);
        
        if (sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT) {
            screen[sy][sx] = b.symbol;
        }
    }
    
    // Display
    for (const auto& row : screen) {
        for (char c : row) {
            std::cout << c;
        }
        std::cout << '\n';
    }
    
    // Show positions of key bodies
    std::cout << "\nPositions:\n";
    for (size_t i = 0; i < std::min(size_t(5), bodies.size()); i++) {
        std::cout << bodies[i].name << ": (" 
                  << std::fixed << std::setprecision(1) 
                  << bodies[i].pos.x << ", " << bodies[i].pos.y << ")\n";
    }
}

int main(int argc, char* argv[]) {
    bool use_pm = false;
    
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--pm" || arg == "-p") {
            use_pm = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "  --pm     Use FFTW Particle Mesh\n";
            std::cout << "  --nbody  Use direct N-body [default]\n";
            return 0;
        }
    }
    
    // Initialize bodies
    std::vector<Body> bodies;
    
    // Sun
    bodies.emplace_back("Sun", float2(0, 0), float2(0, 0), 
                       Constants::SUN_MASS, Constants::SUN_RADIUS, '@');
    
    // Add planets with circular orbits
    auto add_planet = [&](const std::string& name, float dist, float mass, char symbol) {
        float v_orbital = std::sqrt(Constants::GRAVITATIONAL_CONSTANT * Constants::SUN_MASS / dist);
        bodies.emplace_back(name, float2(dist, 0), float2(0, v_orbital), mass, 1.0f, symbol);
    };
    
    add_planet("Mercury", Constants::MERCURY_DISTANCE, Constants::MERCURY_MASS, 'm');
    add_planet("Venus", Constants::VENUS_DISTANCE, Constants::VENUS_MASS, 'v');
    add_planet("Earth", Constants::EARTH_DISTANCE, Constants::EARTH_MASS, 'E');
    add_planet("Mars", Constants::MARS_DISTANCE, Constants::MARS_MASS, 'M');
    add_planet("Jupiter", Constants::JUPITER_DISTANCE, Constants::JUPITER_MASS, 'J');
    add_planet("Saturn", Constants::SATURN_DISTANCE, Constants::SATURN_MASS, 'S');
    
    // PM solver
    FFTWParticleMesh pm_solver;
    
    // Simulation loop
    float time = 0;
    float dt = Constants::TIME_STEP;
    
    for (int step = 0; step < 10000; step++) {
        // Velocity Verlet integration
        
        // Update positions
        for (auto& b : bodies) {
            float2 acc = b.force / b.mass;
            b.pos += b.vel * dt + acc * (0.5f * dt * dt);
        }
        
        // Save old forces
        std::vector<float2> old_forces;
        for (const auto& b : bodies) {
            old_forces.push_back(b.force);
        }
        
        // Compute new forces
        if (use_pm) {
            pm_solver.compute_forces(bodies);
        } else {
            compute_nbody_forces(bodies);
        }
        
        // Update velocities
        for (size_t i = 0; i < bodies.size(); i++) {
            float2 avg_force = (old_forces[i] + bodies[i].force) * 0.5f;
            float2 acc = avg_force / bodies[i].mass;
            bodies[i].vel += acc * dt;
        }
        
        time += dt;
        
        // Display every 100 steps
        if (step % 100 == 0) {
            display(bodies, time, use_pm);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    return 0;
}