#pragma once

#include <vector>
#include <cmath>
#include "../backend/ISimulationBackend.h"

// Constants for thermal radiation
const float STEFAN_BOLTZMANN = 5.67e-8f;  // W/(m²·K⁴) 
const float SPEED_OF_LIGHT = 3e8f;        // m/s (scaled for simulation)
const float BOLTZMANN = 1.38e-23f;        // J/K
const float MIN_TEMPERATURE = 2.7f;       // Cosmic background temperature

// Thermal particle with radiation properties
struct ThermalParticle : public Particle {
    // Temperature
    float temp_internal = 300.0f;     // Kelvin
    
    // Material properties
    float emissivity = 0.9f;          // How well it radiates (0-1)
    float absorptivity = 0.9f;        // How well it absorbs (0-1)
    float specific_heat = 1000.0f;    // J/(kg·K)
    
    // Radiation state
    float luminosity = 0.0f;          // Current power output (W)
    float incident_radiation = 0.0f;  // Radiation hitting us (W)
    
    // For composite objects (solar sails)
    float2 orientation = {1, 0};      // Direction for elongated objects
    float aspect_ratio = 1.0f;        // Length/width ratio
    
    // Computed thermal mass
    float getThermalMass() const {
        return mass * specific_heat;
    }
    
    // Calculate luminosity from temperature
    void updateLuminosity() {
        // In 2D: radiate from circumference
        float circumference = 2.0f * M_PI * radius;
        // Simplified Stefan-Boltzmann (scaled for demo)
        luminosity = emissivity * STEFAN_BOLTZMANN * 
                    pow(temp_internal / 1000.0f, 4) * circumference * 1e6f;
    }
};

class ThermalDynamics {
private:
    float radiation_scale = 1.0f;     // Scale factor for radiation effects
    float cooling_rate = 1.0f;        // Scale factor for cooling
    
public:
    // Calculate cross-section for radiation interception
    float calculateCrossSection(const ThermalParticle& particle, 
                                const float2& ray_direction) {
        if (particle.aspect_ratio == 1.0f) {
            // Circular particle: constant cross-section
            return 2.0f * particle.radius;
        } else {
            // Elongated object: orientation-dependent
            float cos_theta = fabs(particle.orientation.x * ray_direction.x + 
                                  particle.orientation.y * ray_direction.y);
            float width = 2.0f * particle.radius;
            float length = width * particle.aspect_ratio;
            
            // Projected cross-section (simplified)
            return width * (1.0f - cos_theta) + length * cos_theta;
        }
    }
    
    // Apply radiation pressure and heating
    void applyRadiation(std::vector<ThermalParticle>& particles, float dt) {
        // Step 1: Update luminosity for all particles
        for (auto& p : particles) {
            p.updateLuminosity();
        }
        
        // Step 2: Calculate radiation effects on each particle
        for (size_t i = 0; i < particles.size(); i++) {
            float2 total_force = {0, 0};
            float total_incident_power = 0;
            
            // Receive radiation from all other particles
            for (size_t j = 0; j < particles.size(); j++) {
                if (i == j) continue;
                if (particles[j].luminosity < 1e-6f) continue;  // Skip cold particles
                
                // Vector from source to receiver
                float2 r = particles[i].pos - particles[j].pos;
                float dist = sqrt(r.x * r.x + r.y * r.y);
                
                if (dist < 0.1f) continue;  // Too close, handle with contact forces
                
                // Normalize direction
                float2 ray_dir = {r.x / dist, r.y / dist};
                
                // Radiation intensity at this distance (2D: 1/r falloff)
                float intensity = particles[j].luminosity / (2.0f * M_PI * dist);
                
                // Cross-section of receiving particle
                float cross_section = calculateCrossSection(particles[i], ray_dir);
                
                // Power intercepted
                float power_intercepted = intensity * cross_section * 
                                         particles[i].absorptivity;
                
                // Radiation pressure force
                float pressure = (intensity * radiation_scale) / SPEED_OF_LIGHT;
                float force_magnitude = pressure * cross_section * 
                                       particles[i].absorptivity;
                
                total_force.x += force_magnitude * ray_dir.x;
                total_force.y += force_magnitude * ray_dir.y;
                
                total_incident_power += power_intercepted;
            }
            
            // Apply radiation pressure force
            particles[i].vel.x += total_force.x * dt / particles[i].mass;
            particles[i].vel.y += total_force.y * dt / particles[i].mass;
            
            // Apply heating from absorbed radiation
            particles[i].incident_radiation = total_incident_power;
            float energy_absorbed = total_incident_power * dt;
            particles[i].temp_internal += energy_absorbed / particles[i].getThermalMass();
        }
        
        // Step 3: Radiative cooling
        for (auto& p : particles) {
            float energy_radiated = p.luminosity * cooling_rate * dt;
            p.temp_internal -= energy_radiated / p.getThermalMass();
            
            // Minimum temperature (cosmic background)
            p.temp_internal = fmax(p.temp_internal, MIN_TEMPERATURE);
        }
    }
    
    // Thermal pressure for close particles (ideal gas law)
    float2 calculateThermalPressure(const ThermalParticle& p1, 
                                    const ThermalParticle& p2) {
        float2 r = p2.pos - p1.pos;
        float dist = sqrt(r.x * r.x + r.y * r.y);
        
        // Only apply at very close range
        float contact_dist = p1.radius + p2.radius;
        if (dist > contact_dist * 1.5f) return {0, 0};
        
        // Simplified thermal pressure
        float avg_temp = (p1.temp_internal + p2.temp_internal) / 2.0f;
        float pressure = BOLTZMANN * avg_temp * 1e20f;  // Scaled for effect
        
        // Soft repulsion based on temperature
        float overlap = fmax(0, contact_dist - dist);
        float force_mag = pressure * overlap / contact_dist;
        
        float2 force;
        if (dist > 0.001f) {
            force.x = force_mag * r.x / dist;
            force.y = force_mag * r.y / dist;
        } else {
            force = {0, 0};
        }
        
        return force;
    }
    
    // Apply all thermal forces
    void step(std::vector<ThermalParticle>& particles, float dt) {
        // Radiation effects
        applyRadiation(particles, dt);
        
        // Close-range thermal pressure
        for (size_t i = 0; i < particles.size(); i++) {
            for (size_t j = i + 1; j < particles.size(); j++) {
                float2 pressure = calculateThermalPressure(particles[i], particles[j]);
                
                // Newton's third law
                particles[i].vel.x += pressure.x * dt / particles[i].mass;
                particles[i].vel.y += pressure.y * dt / particles[i].mass;
                particles[j].vel.x -= pressure.x * dt / particles[j].mass;
                particles[j].vel.y -= pressure.y * dt / particles[j].mass;
            }
        }
    }
    
    // Configuration
    void setRadiationScale(float scale) { radiation_scale = scale; }
    void setCoolingRate(float rate) { cooling_rate = rate; }
};