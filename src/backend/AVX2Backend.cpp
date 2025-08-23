// Explicit AVX2 SIMD Backend
// Uses AVX2 intrinsics to process 8 particles at once for forces
// This gives us guaranteed vectorization vs hoping the compiler does it

#include "ISimulationBackend.h"
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <iostream>

class AVX2Backend : public ISimulationBackend {
private:
    std::vector<Particle> particles;
    std::vector<float2> forces;
    SimulationParams params;
    
public:
    void initialize(size_t num_particles, const SimulationParams& p) override {
        params = p;
        particles.resize(num_particles);
        forces.resize(num_particles);
    }
    
    void setParticles(const std::vector<Particle>& p) override {
        particles = p;
    }
    
    void getParticles(std::vector<Particle>& p) override {
        p = particles;
    }
    
    void computeForces() override {
        size_t n = particles.size();
        
        // Clear forces
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            forces[i].x = 0;
            forces[i].y = 0;
        }
        
        const float G = params.gravity_constant;
        const float soft2 = params.softening * params.softening;
        
        // Process forces with explicit AVX2
        #pragma omp parallel for schedule(dynamic, 8)
        for (size_t i = 0; i < n; i++) {
            __m256 xi = _mm256_set1_ps(particles[i].pos.x);
            __m256 yi = _mm256_set1_ps(particles[i].pos.y);
            __m256 mi = _mm256_set1_ps(particles[i].mass);
            
            __m256 fx_acc = _mm256_setzero_ps();
            __m256 fy_acc = _mm256_setzero_ps();
            
            // Process 8 particles at a time with AVX2
            size_t j = 0;
            for (; j + 7 < n; j += 8) {
                // Skip self-interaction
                bool skip_all = (j <= i && i <= j + 7);
                if (skip_all) {
                    // Handle self-interaction case particle by particle
                    for (size_t k = j; k < j + 8 && k < n; k++) {
                        if (k == i) continue;
                        
                        float dx = particles[k].pos.x - particles[i].pos.x;
                        float dy = particles[k].pos.y - particles[i].pos.y;
                        float r2 = dx*dx + dy*dy + soft2;
                        float r = sqrt(r2);
                        float f = G * particles[k].mass / (r2 * r);
                        
                        forces[i].x += f * dx * particles[i].mass;
                        forces[i].y += f * dy * particles[i].mass;
                    }
                    continue;
                }
                
                // Load 8 particles' data
                __m256 xj = _mm256_set_ps(
                    particles[j+7].pos.x, particles[j+6].pos.x,
                    particles[j+5].pos.x, particles[j+4].pos.x,
                    particles[j+3].pos.x, particles[j+2].pos.x,
                    particles[j+1].pos.x, particles[j+0].pos.x
                );
                
                __m256 yj = _mm256_set_ps(
                    particles[j+7].pos.y, particles[j+6].pos.y,
                    particles[j+5].pos.y, particles[j+4].pos.y,
                    particles[j+3].pos.y, particles[j+2].pos.y,
                    particles[j+1].pos.y, particles[j+0].pos.y
                );
                
                __m256 mj = _mm256_set_ps(
                    particles[j+7].mass, particles[j+6].mass,
                    particles[j+5].mass, particles[j+4].mass,
                    particles[j+3].mass, particles[j+2].mass,
                    particles[j+1].mass, particles[j+0].mass
                );
                
                // Compute distances
                __m256 dx = _mm256_sub_ps(xj, xi);
                __m256 dy = _mm256_sub_ps(yj, yi);
                
                // r2 = dx*dx + dy*dy + softening
                __m256 r2 = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
                    _mm256_set1_ps(soft2)
                );
                
                // r = sqrt(r2)
                __m256 r = _mm256_sqrt_ps(r2);
                
                // f = G * mj / (r2 * r)
                __m256 f = _mm256_div_ps(
                    _mm256_mul_ps(_mm256_set1_ps(G), mj),
                    _mm256_mul_ps(r2, r)
                );
                
                // Accumulate forces
                fx_acc = _mm256_add_ps(fx_acc, _mm256_mul_ps(_mm256_mul_ps(f, dx), mi));
                fy_acc = _mm256_add_ps(fy_acc, _mm256_mul_ps(_mm256_mul_ps(f, dy), mi));
            }
            
            // Sum the 8 components of the accumulator
            float fx_array[8], fy_array[8];
            _mm256_storeu_ps(fx_array, fx_acc);
            _mm256_storeu_ps(fy_array, fy_acc);
            
            float fx_total = 0, fy_total = 0;
            for (int k = 0; k < 8; k++) {
                fx_total += fx_array[k];
                fy_total += fy_array[k];
            }
            
            forces[i].x += fx_total;
            forces[i].y += fy_total;
            
            // Handle remaining particles (less than 8)
            for (; j < n; j++) {
                if (i == j) continue;
                
                float dx = particles[j].pos.x - particles[i].pos.x;
                float dy = particles[j].pos.y - particles[i].pos.y;
                float r2 = dx*dx + dy*dy + soft2;
                float r = sqrt(r2);
                float f = G * particles[j].mass / (r2 * r);
                
                forces[i].x += f * dx * particles[i].mass;
                forces[i].y += f * dy * particles[i].mass;
            }
        }
    }
    
    void integrate(float dt) override {
        size_t n = particles.size();
        
        // Use AVX2 for integration as well
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            // Load 8 particles
            __m256 vx = _mm256_set_ps(
                particles[i+7].vel.x, particles[i+6].vel.x,
                particles[i+5].vel.x, particles[i+4].vel.x,
                particles[i+3].vel.x, particles[i+2].vel.x,
                particles[i+1].vel.x, particles[i+0].vel.x
            );
            
            __m256 vy = _mm256_set_ps(
                particles[i+7].vel.y, particles[i+6].vel.y,
                particles[i+5].vel.y, particles[i+4].vel.y,
                particles[i+3].vel.y, particles[i+2].vel.y,
                particles[i+1].vel.y, particles[i+0].vel.y
            );
            
            __m256 fx = _mm256_set_ps(
                forces[i+7].x, forces[i+6].x,
                forces[i+5].x, forces[i+4].x,
                forces[i+3].x, forces[i+2].x,
                forces[i+1].x, forces[i+0].x
            );
            
            __m256 fy = _mm256_set_ps(
                forces[i+7].y, forces[i+6].y,
                forces[i+5].y, forces[i+4].y,
                forces[i+3].y, forces[i+2].y,
                forces[i+1].y, forces[i+0].y
            );
            
            __m256 m = _mm256_set_ps(
                particles[i+7].mass, particles[i+6].mass,
                particles[i+5].mass, particles[i+4].mass,
                particles[i+3].mass, particles[i+2].mass,
                particles[i+1].mass, particles[i+0].mass
            );
            
            __m256 dt_vec = _mm256_set1_ps(dt);
            
            // Update velocities: v += (f/m) * dt
            vx = _mm256_add_ps(vx, _mm256_mul_ps(_mm256_div_ps(fx, m), dt_vec));
            vy = _mm256_add_ps(vy, _mm256_mul_ps(_mm256_div_ps(fy, m), dt_vec));
            
            // Store back velocities
            float vx_array[8], vy_array[8];
            _mm256_storeu_ps(vx_array, vx);
            _mm256_storeu_ps(vy_array, vy);
            
            for (int k = 0; k < 8; k++) {
                particles[i+k].vel.x = vx_array[7-k];
                particles[i+k].vel.y = vy_array[7-k];
                
                // Update positions
                particles[i+k].pos.x += particles[i+k].vel.x * dt;
                particles[i+k].pos.y += particles[i+k].vel.y * dt;
                
                // Boundary wrapping
                if (particles[i+k].pos.x < 0) particles[i+k].pos.x += params.box_size;
                if (particles[i+k].pos.x >= params.box_size) particles[i+k].pos.x -= params.box_size;
                if (particles[i+k].pos.y < 0) particles[i+k].pos.y += params.box_size;
                if (particles[i+k].pos.y >= params.box_size) particles[i+k].pos.y -= params.box_size;
            }
        }
        
        // Handle remaining particles
        for (; i < n; i++) {
            particles[i].vel.x += forces[i].x / particles[i].mass * dt;
            particles[i].vel.y += forces[i].y / particles[i].mass * dt;
            
            particles[i].pos.x += particles[i].vel.x * dt;
            particles[i].pos.y += particles[i].vel.y * dt;
            
            if (particles[i].pos.x < 0) particles[i].pos.x += params.box_size;
            if (particles[i].pos.x >= params.box_size) particles[i].pos.x -= params.box_size;
            if (particles[i].pos.y < 0) particles[i].pos.y += params.box_size;
            if (particles[i].pos.y >= params.box_size) particles[i].pos.y -= params.box_size;
        }
    }
    
    size_t getMaxParticles() const override { return 100000; }
    std::string getBackendName() const override { 
        return "AVX2 Explicit SIMD (" + std::to_string(omp_get_max_threads()) + " threads, 8-wide)";
    }
    bool isGPU() const override { return false; }
    size_t getMemoryUsage() const override { 
        return particles.size() * (sizeof(Particle) + sizeof(float2));
    }
    void cleanup() override {
        particles.clear();
        forces.clear();
    }
};