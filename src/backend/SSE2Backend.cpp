// Explicit SSE2 SIMD Backend
// Uses SSE2 intrinsics to process 4 particles at once
// SSE2 is available on all x86-64 CPUs since ~2003

#include "ISimulationBackend.h"
#include <emmintrin.h>  // SSE2
#include <omp.h>
#include <cmath>
#include <iostream>

class SSE2Backend : public ISimulationBackend {
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
        
        // Process forces with explicit SSE2 (4-wide)
        #pragma omp parallel for schedule(dynamic, 4)
        for (size_t i = 0; i < n; i++) {
            __m128 xi = _mm_set1_ps(particles[i].pos.x);
            __m128 yi = _mm_set1_ps(particles[i].pos.y);
            __m128 mi = _mm_set1_ps(particles[i].mass);
            
            __m128 fx_acc = _mm_setzero_ps();
            __m128 fy_acc = _mm_setzero_ps();
            
            // Process 4 particles at a time with SSE2
            size_t j = 0;
            for (; j + 3 < n; j += 4) {
                // Skip self-interaction
                bool skip_all = (j <= i && i <= j + 3);
                if (skip_all) {
                    for (size_t k = j; k < j + 4 && k < n; k++) {
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
                
                // Load 4 particles' data
                __m128 xj = _mm_set_ps(
                    particles[j+3].pos.x, particles[j+2].pos.x,
                    particles[j+1].pos.x, particles[j+0].pos.x
                );
                
                __m128 yj = _mm_set_ps(
                    particles[j+3].pos.y, particles[j+2].pos.y,
                    particles[j+1].pos.y, particles[j+0].pos.y
                );
                
                __m128 mj = _mm_set_ps(
                    particles[j+3].mass, particles[j+2].mass,
                    particles[j+1].mass, particles[j+0].mass
                );
                
                // Compute distances
                __m128 dx = _mm_sub_ps(xj, xi);
                __m128 dy = _mm_sub_ps(yj, yi);
                
                // r2 = dx*dx + dy*dy + softening
                __m128 r2 = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)),
                    _mm_set1_ps(soft2)
                );
                
                // r = sqrt(r2)
                __m128 r = _mm_sqrt_ps(r2);
                
                // f = G * mj / (r2 * r)
                __m128 f = _mm_div_ps(
                    _mm_mul_ps(_mm_set1_ps(G), mj),
                    _mm_mul_ps(r2, r)
                );
                
                // Accumulate forces
                fx_acc = _mm_add_ps(fx_acc, _mm_mul_ps(_mm_mul_ps(f, dx), mi));
                fy_acc = _mm_add_ps(fy_acc, _mm_mul_ps(_mm_mul_ps(f, dy), mi));
            }
            
            // Sum the 4 components of the accumulator
            float fx_array[4], fy_array[4];
            _mm_storeu_ps(fx_array, fx_acc);
            _mm_storeu_ps(fy_array, fy_acc);
            
            float fx_total = fx_array[0] + fx_array[1] + fx_array[2] + fx_array[3];
            float fy_total = fy_array[0] + fy_array[1] + fy_array[2] + fy_array[3];
            
            forces[i].x += fx_total;
            forces[i].y += fy_total;
            
            // Handle remaining particles (less than 4)
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
        
        // Use SSE2 for integration
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            // Load 4 particles
            __m128 vx = _mm_set_ps(
                particles[i+3].vel.x, particles[i+2].vel.x,
                particles[i+1].vel.x, particles[i+0].vel.x
            );
            
            __m128 vy = _mm_set_ps(
                particles[i+3].vel.y, particles[i+2].vel.y,
                particles[i+1].vel.y, particles[i+0].vel.y
            );
            
            __m128 fx = _mm_set_ps(
                forces[i+3].x, forces[i+2].x,
                forces[i+1].x, forces[i+0].x
            );
            
            __m128 fy = _mm_set_ps(
                forces[i+3].y, forces[i+2].y,
                forces[i+1].y, forces[i+0].y
            );
            
            __m128 m = _mm_set_ps(
                particles[i+3].mass, particles[i+2].mass,
                particles[i+1].mass, particles[i+0].mass
            );
            
            __m128 dt_vec = _mm_set1_ps(dt);
            
            // Update velocities: v += (f/m) * dt
            vx = _mm_add_ps(vx, _mm_mul_ps(_mm_div_ps(fx, m), dt_vec));
            vy = _mm_add_ps(vy, _mm_mul_ps(_mm_div_ps(fy, m), dt_vec));
            
            // Store back velocities
            float vx_array[4], vy_array[4];
            _mm_storeu_ps(vx_array, vx);
            _mm_storeu_ps(vy_array, vy);
            
            for (int k = 0; k < 4; k++) {
                particles[i+k].vel.x = vx_array[3-k];
                particles[i+k].vel.y = vy_array[3-k];
                
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
        return "SSE2 Explicit SIMD (" + std::to_string(omp_get_max_threads()) + " threads, 4-wide)";
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