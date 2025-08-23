// Million-Particle Solar System Simulation
// Optimized for Particle-Mesh with real astronomical units
// Uses OpenMP for 12-core parallelization

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <cstring>
#include <algorithm>
#include <complex>
#include <fftw3.h>
#include <omp.h>
#include <random>

// Physical Constants (SI units)
namespace Units {
    constexpr double AU = 1.496e11;              // Astronomical Unit in meters
    constexpr double SOLAR_MASS = 1.989e30;      // kg
    constexpr double EARTH_MASS = 5.972e24;      // kg
    constexpr double JUPITER_MASS = 1.898e27;    // kg
    constexpr double G = 6.67430e-11;            // m³/kg·s²
    constexpr double YEAR = 365.25 * 24 * 3600;  // seconds
    constexpr double DAY = 24 * 3600;            // seconds
}

// Simulation units (normalized for numerical stability)
// Distance: AU, Mass: Solar masses, Time: years
namespace SimUnits {
    constexpr double G = 4.0 * M_PI * M_PI;  // G in AU³/M☉·year²
    constexpr double TIME_STEP = 0.001;      // years (~8.76 hours)
}

// 2D Vector (we'll keep 2D for visualization simplicity)
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

// Particle structure (SoA would be better for cache, but this is clearer)
struct Particle {
    float2 pos;
    float2 vel;
    float2 force;
    float mass;
    uint8_t type;  // 0=star, 1=planet, 2=moon, 3=asteroid, 4=KBO
    
    Particle() : pos(0, 0), vel(0, 0), force(0, 0), mass(0), type(3) {}
    Particle(float2 p, float2 v, float m, uint8_t t) 
        : pos(p), vel(v), force(0, 0), mass(m), type(t) {}
};

// Solar System Data (real values)
struct SolarSystemData {
    struct PlanetData {
        const char* name;
        double semi_major_au;  // AU
        double mass_ratio;     // Relative to solar mass
        double eccentricity;
        int num_major_moons;
    };
    
    static constexpr PlanetData planets[] = {
        {"Mercury", 0.387, 1.66e-7, 0.206, 0},
        {"Venus", 0.723, 2.45e-6, 0.007, 0},
        {"Earth", 1.000, 3.00e-6, 0.017, 1},
        {"Mars", 1.524, 3.23e-7, 0.093, 2},
        {"Jupiter", 5.203, 9.55e-4, 0.048, 79},  // We'll add 4 major moons
        {"Saturn", 9.537, 2.86e-4, 0.054, 82},   // We'll add 5 major moons
        {"Uranus", 19.191, 4.37e-5, 0.047, 27},  // We'll add 2 major moons
        {"Neptune", 30.069, 5.15e-5, 0.009, 14}  // We'll add 1 major moon
    };
    
    static constexpr int NUM_PLANETS = 8;
};

constexpr SolarSystemData::PlanetData SolarSystemData::planets[];

// FFTW-based Particle Mesh solver optimized for millions of particles
class ParticleMeshSolver {
private:
    static constexpr int GRID_SIZE = 1024;  // High resolution for accuracy
    static constexpr float WORLD_SIZE = 200.0f;  // 200 AU to include Kuiper belt
    static constexpr float CELL_SIZE = WORLD_SIZE / GRID_SIZE;
    static constexpr float SOFTENING = 0.001f;  // 0.001 AU softening
    
    // Grids (using fftwf for single precision)
    float* density;
    float* potential;
    float* force_x;
    float* force_y;
    fftwf_complex* density_fft;
    fftwf_complex* potential_fft;
    
    // FFTW plans
    fftwf_plan plan_forward;
    fftwf_plan plan_backward;
    
    // Green's function (precomputed)
    float* greens_function;
    
public:
    ParticleMeshSolver() {
        // Set number of threads for FFTW
        fftwf_init_threads();
        fftwf_plan_with_nthreads(omp_get_max_threads());
        
        int n = GRID_SIZE;
        int n2 = n * n;
        int nc = n * (n/2 + 1);  // Complex array size for r2c transform
        
        // Allocate aligned memory
        density = fftwf_alloc_real(n2);
        potential = fftwf_alloc_real(n2);
        force_x = fftwf_alloc_real(n2);
        force_y = fftwf_alloc_real(n2);
        greens_function = fftwf_alloc_real(nc);
        density_fft = fftwf_alloc_complex(nc);
        potential_fft = fftwf_alloc_complex(nc);
        
        // Create FFTW plans (FFTW_MEASURE for optimization)
        plan_forward = fftwf_plan_dft_r2c_2d(n, n, density, density_fft, FFTW_MEASURE);
        plan_backward = fftwf_plan_dft_c2r_2d(n, n, potential_fft, potential, FFTW_MEASURE);
        
        // Initialize Green's function
        init_greens_function();
        
        std::cout << "PM Solver initialized: " << GRID_SIZE << "x" << GRID_SIZE 
                  << " grid, " << omp_get_max_threads() << " threads\n";
    }
    
    ~ParticleMeshSolver() {
        fftwf_destroy_plan(plan_forward);
        fftwf_destroy_plan(plan_backward);
        fftwf_free(density);
        fftwf_free(potential);
        fftwf_free(force_x);
        fftwf_free(force_y);
        fftwf_free(greens_function);
        fftwf_free(density_fft);
        fftwf_free(potential_fft);
        fftwf_cleanup_threads();
    }
    
    void init_greens_function() {
        int n = GRID_SIZE;
        
        #pragma omp parallel for
        for (int ky = 0; ky < n; ky++) {
            for (int kx = 0; kx <= n/2; kx++) {
                // Wave numbers (accounting for FFT layout)
                float kx_val = (kx <= n/2) ? kx : kx - n;
                float ky_val = (ky <= n/2) ? ky : ky - n;
                
                kx_val *= 2.0f * M_PI / WORLD_SIZE;
                ky_val *= 2.0f * M_PI / WORLD_SIZE;
                
                float k2 = kx_val * kx_val + ky_val * ky_val;
                
                int idx = ky * (n/2 + 1) + kx;
                
                if (k2 > 0) {
                    float soft2 = (SOFTENING * 2.0f * M_PI / WORLD_SIZE);
                    soft2 *= soft2;
                    greens_function[idx] = -SimUnits::G / (k2 + soft2);
                } else {
                    greens_function[idx] = 0;  // DC component
                }
            }
        }
    }
    
    void compute_forces(std::vector<Particle>& particles) {
        auto start = std::chrono::high_resolution_clock::now();
        
        int n = GRID_SIZE;
        int n2 = n * n;
        
        // Clear grids
        std::memset(density, 0, n2 * sizeof(float));
        
        // 1. Particle assignment to mesh (CIC) - parallelized with atomic operations
        #pragma omp parallel for
        for (size_t i = 0; i < particles.size(); i++) {
            const auto& p = particles[i];
            
            // Map to grid coordinates (centered at origin)
            float gx = (p.pos.x + WORLD_SIZE/2) / CELL_SIZE;
            float gy = (p.pos.y + WORLD_SIZE/2) / CELL_SIZE;
            
            // Skip if outside grid
            if (gx < 0 || gx >= n-1 || gy < 0 || gy >= n-1) continue;
            
            int ix = (int)gx;
            int iy = (int)gy;
            float fx = gx - ix;
            float fy = gy - iy;
            
            // CIC weights
            float w00 = (1-fx) * (1-fy) * p.mass / (CELL_SIZE * CELL_SIZE);
            float w10 = fx * (1-fy) * p.mass / (CELL_SIZE * CELL_SIZE);
            float w01 = (1-fx) * fy * p.mass / (CELL_SIZE * CELL_SIZE);
            float w11 = fx * fy * p.mass / (CELL_SIZE * CELL_SIZE);
            
            // Atomic add for thread safety
            #pragma omp atomic
            density[iy * n + ix] += w00;
            #pragma omp atomic
            density[iy * n + (ix+1)] += w10;
            #pragma omp atomic
            density[(iy+1) * n + ix] += w01;
            #pragma omp atomic
            density[(iy+1) * n + (ix+1)] += w11;
        }
        
        // 2. FFT forward transform
        fftwf_execute(plan_forward);
        
        // 3. Solve Poisson equation in Fourier space
        int nc = n * (n/2 + 1);
        #pragma omp parallel for
        for (int i = 0; i < nc; i++) {
            potential_fft[i][0] = density_fft[i][0] * greens_function[i];
            potential_fft[i][1] = density_fft[i][1] * greens_function[i];
        }
        
        // 4. Inverse FFT
        fftwf_execute(plan_backward);
        
        // Normalize (FFTW doesn't normalize inverse)
        float norm = 1.0f / (n * n);
        #pragma omp parallel for
        for (int i = 0; i < n2; i++) {
            potential[i] *= norm;
        }
        
        // 5. Compute force field (gradient of potential)
        #pragma omp parallel for
        for (int iy = 0; iy < n; iy++) {
            for (int ix = 0; ix < n; ix++) {
                int idx = iy * n + ix;
                
                // Periodic boundary for gradient
                int ixp = (ix + 1) % n;
                int ixm = (ix - 1 + n) % n;
                int iyp = (iy + 1) % n;
                int iym = (iy - 1 + n) % n;
                
                force_x[idx] = -(potential[iy * n + ixp] - potential[iy * n + ixm]) / (2 * CELL_SIZE);
                force_y[idx] = -(potential[iyp * n + ix] - potential[iym * n + ix]) / (2 * CELL_SIZE);
            }
        }
        
        // 6. Interpolate forces back to particles
        #pragma omp parallel for
        for (size_t i = 0; i < particles.size(); i++) {
            auto& p = particles[i];
            
            float gx = (p.pos.x + WORLD_SIZE/2) / CELL_SIZE;
            float gy = (p.pos.y + WORLD_SIZE/2) / CELL_SIZE;
            
            if (gx < 0 || gx >= n-1 || gy < 0 || gy >= n-1) {
                p.force = float2(0, 0);
                continue;
            }
            
            int ix = (int)gx;
            int iy = (int)gy;
            float fx = gx - ix;
            float fy = gy - iy;
            
            // CIC interpolation
            int idx00 = iy * n + ix;
            int idx10 = iy * n + (ix+1);
            int idx01 = (iy+1) * n + ix;
            int idx11 = (iy+1) * n + (ix+1);
            
            float fx_interp = force_x[idx00] * (1-fx) * (1-fy) +
                             force_x[idx10] * fx * (1-fy) +
                             force_x[idx01] * (1-fx) * fy +
                             force_x[idx11] * fx * fy;
            
            float fy_interp = force_y[idx00] * (1-fx) * (1-fy) +
                             force_y[idx10] * fx * (1-fy) +
                             force_y[idx01] * (1-fx) * fy +
                             force_y[idx11] * fx * fy;
            
            p.force = float2(fx_interp * p.mass, fy_interp * p.mass);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        static int count = 0;
        if (++count % 100 == 0) {
            std::cout << "PM force calculation: " << duration.count() << " ms\n";
        }
    }
};

// Initialize solar system with realistic data
class SolarSystemBuilder {
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniform;
    
public:
    SolarSystemBuilder() : rng(42), uniform(0.0f, 1.0f) {}
    
    void build(std::vector<Particle>& particles) {
        particles.clear();
        particles.reserve(1100000);  // Reserve space for ~1M particles
        
        // Sun
        particles.emplace_back(float2(0, 0), float2(0, 0), 1.0f, 0);
        std::cout << "Added Sun\n";
        
        // Planets with realistic orbits
        int moon_count = 0;
        for (int i = 0; i < SolarSystemData::NUM_PLANETS; i++) {
            const auto& planet = SolarSystemData::planets[i];
            
            // Kepler orbit
            float r = planet.semi_major_au;
            float v = std::sqrt(SimUnits::G * 1.0f / r);  // Circular velocity
            
            // Add some eccentricity
            float theta = uniform(rng) * 2 * M_PI;
            float2 pos(r * cos(theta), r * sin(theta));
            float2 vel(-v * sin(theta), v * cos(theta));
            
            particles.emplace_back(pos, vel, planet.mass_ratio, 1);
            
            // Add major moons
            if (i == 2) {  // Earth - add Moon
                float moon_dist = 0.00257f;  // AU
                float moon_v = std::sqrt(SimUnits::G * planet.mass_ratio / moon_dist);
                particles.emplace_back(
                    pos + float2(moon_dist, 0),
                    vel + float2(0, moon_v),
                    7.34e-8f,  // Moon mass / solar mass
                    2
                );
                moon_count++;
            } else if (i == 4) {  // Jupiter - add Galilean moons
                const float moon_dists[] = {0.00282f, 0.00449f, 0.00716f, 0.01256f};  // Io, Europa, Ganymede, Callisto
                const float moon_masses[] = {4.5e-8f, 2.4e-8f, 7.5e-8f, 5.4e-8f};
                for (int m = 0; m < 4; m++) {
                    float moon_v = std::sqrt(SimUnits::G * planet.mass_ratio / moon_dists[m]);
                    float angle = uniform(rng) * 2 * M_PI;
                    particles.emplace_back(
                        pos + float2(moon_dists[m] * cos(angle), moon_dists[m] * sin(angle)),
                        vel + float2(-moon_v * sin(angle), moon_v * cos(angle)),
                        moon_masses[m],
                        2
                    );
                    moon_count++;
                }
            } else if (i == 5) {  // Saturn - add Titan + 4 others
                const float moon_dists[] = {0.00195f, 0.00252f, 0.00394f, 0.00816f, 0.02575f};
                const float moon_masses[] = {1.9e-9f, 5.5e-9f, 5.6e-8f, 1.1e-7f, 6.8e-8f};  // Enceladus, Tethys, Dione, Rhea, Titan
                for (int m = 0; m < 5; m++) {
                    float moon_v = std::sqrt(SimUnits::G * planet.mass_ratio / moon_dists[m]);
                    float angle = uniform(rng) * 2 * M_PI;
                    particles.emplace_back(
                        pos + float2(moon_dists[m] * cos(angle), moon_dists[m] * sin(angle)),
                        vel + float2(-moon_v * sin(angle), moon_v * cos(angle)),
                        moon_masses[m],
                        2
                    );
                    moon_count++;
                }
            }
        }
        std::cout << "Added " << SolarSystemData::NUM_PLANETS << " planets and " 
                  << moon_count << " moons\n";
        
        // Main asteroid belt (2.2 - 3.3 AU)
        int asteroid_count = 100000;  // 100k asteroids
        for (int i = 0; i < asteroid_count; i++) {
            float r = 2.2f + 1.1f * uniform(rng);
            float theta = uniform(rng) * 2 * M_PI;
            float inclination = (uniform(rng) - 0.5f) * 0.2f;  // ±0.1 AU vertical
            
            float v = std::sqrt(SimUnits::G / r);
            float2 pos(r * cos(theta), r * sin(theta) + inclination);
            float2 vel(-v * sin(theta), v * cos(theta));
            
            // Power law mass distribution
            float mass = 1e-12f * std::pow(uniform(rng), -2.5f);
            
            particles.emplace_back(pos, vel, mass, 3);
        }
        std::cout << "Added " << asteroid_count << " asteroids\n";
        
        // Kuiper belt (30 - 50 AU)
        int kbo_count = 900000;  // 900k KBOs to reach ~1M total
        for (int i = 0; i < kbo_count; i++) {
            float r = 30.0f + 20.0f * uniform(rng);
            float theta = uniform(rng) * 2 * M_PI;
            float inclination = (uniform(rng) - 0.5f) * 2.0f;  // ±1 AU vertical
            
            float v = std::sqrt(SimUnits::G / r);
            float2 pos(r * cos(theta), r * sin(theta) + inclination);
            float2 vel(-v * sin(theta), v * cos(theta));
            
            // Larger masses for KBOs
            float mass = 1e-10f * std::pow(uniform(rng), -2.0f);
            
            particles.emplace_back(pos, vel, mass, 4);
        }
        std::cout << "Added " << kbo_count << " Kuiper belt objects\n";
        
        // Add a few dwarf planets
        const struct { const char* name; float a; float mass; } dwarfs[] = {
            {"Ceres", 2.77f, 4.7e-10f},
            {"Pluto", 39.5f, 6.6e-9f},
            {"Eris", 67.7f, 8.3e-9f},
            {"Makemake", 45.8f, 2.0e-9f},
            {"Haumea", 43.3f, 2.0e-9f}
        };
        
        for (const auto& dwarf : dwarfs) {
            float v = std::sqrt(SimUnits::G / dwarf.a);
            particles.emplace_back(
                float2(dwarf.a, 0),
                float2(0, v),
                dwarf.mass,
                1  // Treat as planet type
            );
        }
        std::cout << "Added 5 dwarf planets\n";
        
        std::cout << "\nTotal particles: " << particles.size() << "\n\n";
    }
};

// Simple visualization for monitoring
class Visualizer {
    int frame_count = 0;
    
public:
    void display(const std::vector<Particle>& particles, float time) {
        if (++frame_count % 10 != 0) return;  // Only display every 10th frame
        
        std::cout << "\033[2J\033[H";  // Clear screen
        
        // Statistics
        std::cout << "=== Million Particle Solar System ===\n";
        std::cout << "Time: " << std::fixed << std::setprecision(2) << time << " years\n";
        std::cout << "Particles: " << particles.size() << "\n";
        std::cout << "Using " << omp_get_max_threads() << " CPU cores\n\n";
        
        // Count by type
        int counts[5] = {0};
        for (const auto& p : particles) {
            counts[p.type]++;
        }
        
        std::cout << "Stars: " << counts[0] << "\n";
        std::cout << "Planets: " << counts[1] << "\n";
        std::cout << "Moons: " << counts[2] << "\n";
        std::cout << "Asteroids: " << counts[3] << "\n";
        std::cout << "KBOs: " << counts[4] << "\n\n";
        
        // Simple 2D projection (inner system)
        const int WIDTH = 80;
        const int HEIGHT = 30;
        std::vector<std::vector<char>> screen(HEIGHT, std::vector<char>(WIDTH, ' '));
        
        // Draw density map
        for (const auto& p : particles) {
            if (std::abs(p.pos.x) > 10 || std::abs(p.pos.y) > 10) continue;  // Only inner system
            
            int sx = WIDTH/2 + (int)(p.pos.x * 3);
            int sy = HEIGHT/2 - (int)(p.pos.y * 2);
            
            if (sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT) {
                if (p.type == 0) screen[sy][sx] = '@';  // Sun
                else if (p.type == 1) screen[sy][sx] = 'O';  // Planet
                else if (p.type == 2) screen[sy][sx] = 'o';  // Moon
                else if (screen[sy][sx] == ' ') screen[sy][sx] = '.';  // Asteroid/debris
            }
        }
        
        // Display
        std::cout << "Inner System View (±10 AU):\n";
        for (const auto& row : screen) {
            for (char c : row) std::cout << c;
            std::cout << '\n';
        }
        
        // Show some planet positions
        std::cout << "\nPlanet positions (AU):\n";
        int planet_idx = 0;
        for (size_t i = 0; i < particles.size() && planet_idx < 4; i++) {
            if (particles[i].type == 1) {
                std::cout << "  Planet " << planet_idx++ << ": (" 
                          << std::setprecision(2) << particles[i].pos.x 
                          << ", " << particles[i].pos.y << ")\n";
            }
        }
    }
};

// Main simulation loop
void run_simulation() {
    // Set OpenMP threads
    omp_set_num_threads(12);
    std::cout << "OpenMP using " << omp_get_max_threads() << " threads\n\n";
    
    // Initialize
    std::vector<Particle> particles;
    SolarSystemBuilder builder;
    builder.build(particles);
    
    ParticleMeshSolver pm_solver;
    Visualizer viz;
    
    // Simulation parameters
    float time = 0;
    float dt = SimUnits::TIME_STEP;
    const int max_steps = 10000;
    
    std::cout << "Starting simulation...\n";
    std::cout << "Time step: " << dt << " years (" << dt * 365.25 << " days)\n\n";
    
    auto sim_start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < max_steps; step++) {
        auto step_start = std::chrono::high_resolution_clock::now();
        
        // Leapfrog integration (symplectic, good for long-term stability)
        
        // Update positions (parallel)
        #pragma omp parallel for
        for (size_t i = 0; i < particles.size(); i++) {
            auto& p = particles[i];
            p.pos += p.vel * dt;
        }
        
        // Compute forces
        pm_solver.compute_forces(particles);
        
        // Update velocities (parallel)
        #pragma omp parallel for
        for (size_t i = 0; i < particles.size(); i++) {
            auto& p = particles[i];
            if (p.mass > 0) {
                float2 acc = p.force / p.mass;
                
                // Clamp extreme accelerations
                float acc_mag = acc.length();
                if (acc_mag > 1000.0f) {
                    acc = acc * (1000.0f / acc_mag);
                }
                
                p.vel += acc * dt;
            }
        }
        
        time += dt;
        
        // Display
        if (step % 10 == 0) {
            viz.display(particles, time);
            
            auto step_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);
            std::cout << "\nStep time: " << duration.count() << " ms\n";
            std::cout << "Simulated " << particles.size() / (duration.count() / 1000.0) 
                      << " particles/second\n";
        }
        
        // Sleep briefly to see output
        if (step % 100 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    auto sim_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(sim_end - sim_start);
    
    std::cout << "\n=== Simulation Complete ===\n";
    std::cout << "Total time: " << total_duration.count() << " seconds\n";
    std::cout << "Simulated " << time << " years\n";
    std::cout << "Average: " << (particles.size() * max_steps) / total_duration.count() 
              << " particle-updates/second\n";
}

int main() {
    try {
        run_simulation();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}