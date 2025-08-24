#include "src/backend/SimpleBackend.cpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <thread>
#include <unistd.h>

// Physical constants (using scaled units for numerical stability)
// Distance unit: 1 AU = 1.496e11 m -> 1.0
// Time unit: 1 day = 86400 s -> 1.0
// Mass unit: Solar mass = 1.989e30 kg -> 1.0

const double G_REAL = 6.67430e-11;  // m^3 kg^-1 s^-2
const double AU = 1.496e11;         // meters
const double DAY = 86400;           // seconds
const double SOLAR_MASS = 1.989e30; // kg
const double EARTH_MASS = 5.972e24; // kg
const double MOON_MASS = 7.342e22;  // kg

// Convert to simulation units
const double G_SIM = G_REAL * SOLAR_MASS * DAY * DAY / (AU * AU * AU);

// Real-time ASCII visualization
class RealTimeViewer {
public:  // Make these public for easy access
    float center_x, center_y;
    float scale;
    int width, height;
    std::vector<std::vector<char>> buffer;
    RealTimeViewer(int w = 80, int h = 40, float s = 2.0f) 
        : width(w), height(h), scale(s) {
        buffer.resize(height, std::vector<char>(width, ' '));
    }
    
    void clear() {
        for (auto& row : buffer) {
            std::fill(row.begin(), row.end(), ' ');
        }
    }
    
    void setCenter(float x, float y) {
        center_x = x;
        center_y = y;
    }
    
    void plotPoint(float x, float y, char symbol, const char* color = nullptr) {
        // Transform to screen coordinates
        int sx = (int)((x - center_x) * scale + width/2);
        int sy = (int)((y - center_y) * scale + height/2);
        
        if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
            buffer[sy][sx] = symbol;
        }
    }
    
    void plotOrbit(float cx, float cy, float radius, char symbol = '.') {
        // Draw circle using parametric equation
        int steps = 100;
        for (int i = 0; i < steps; i++) {
            float angle = 2 * M_PI * i / steps;
            float x = cx + radius * cos(angle);
            float y = cy + radius * sin(angle);
            plotPoint(x, y, symbol);
        }
    }
    
    void render(const std::string& title, float sim_time) {
        // Clear screen and reset cursor
        std::cout << "\033[2J\033[H" << std::flush;
        
        // Title
        std::cout << "+" << std::string(width-2, '=') << "+\n";
        std::cout << "| " << title;
        std::cout << std::string(width - title.length() - 3, ' ') << "|\n";
        std::cout << "+" << std::string(width-2, '-') << "+\n";
        
        // Render buffer with colors
        for (const auto& row : buffer) {
            std::cout << "|";
            for (char c : row) {
                switch(c) {
                    case '*': std::cout << "\033[33;1m" << c << "\033[0m"; break;  // Sun - bright yellow
                    case 'E': std::cout << "\033[34;1m" << c << "\033[0m"; break;  // Earth - bright blue
                    case 'M': std::cout << "\033[37m" << c << "\033[0m"; break;    // Moon - white
                    case '.': std::cout << "\033[32m" << c << "\033[0m"; break;    // Orbit - green
                    case ':': std::cout << "\033[33m" << c << "\033[0m"; break;    // Sun orbit - yellow
                    default: std::cout << c;
                }
            }
            std::cout << "|\n";
        }
        
        // Info bar
        std::cout << "+" << std::string(width-2, '-') << "+\n";
        std::cout << "| Time: " << std::fixed << std::setprecision(2) << sim_time << " days";
        std::cout << " | Year progress: " << (sim_time/365.25*100) << "%";
        std::cout << std::string(width - 50, ' ') << "|\n";
        std::cout << "+" << std::string(width-2, '=') << "+\n";
    }
};

// Calculate orbital elements
struct OrbitalElements {
    double semi_major_axis;
    double eccentricity;
    double period;
    double aphelion;
    double perihelion;
    double current_radius;
    double current_velocity;
    
    static OrbitalElements calculate(float x, float y, float vx, float vy, 
                                    float cx, float cy, float central_mass, float G) {
        OrbitalElements elem;
        
        // Position and velocity relative to central body
        float rx = x - cx;
        float ry = y - cy;
        elem.current_radius = sqrt(rx*rx + ry*ry);
        elem.current_velocity = sqrt(vx*vx + vy*vy);
        
        // Specific orbital energy
        float v2 = vx*vx + vy*vy;
        float mu = G * central_mass;
        float E = v2/2 - mu/elem.current_radius;
        
        // Semi-major axis from vis-viva equation
        elem.semi_major_axis = -mu / (2*E);
        
        // Angular momentum
        float h = rx*vy - ry*vx;
        
        // Eccentricity
        elem.eccentricity = sqrt(1 + 2*E*h*h/(mu*mu));
        
        // Orbital period (Kepler's third law)
        elem.period = 2 * M_PI * sqrt(elem.semi_major_axis*elem.semi_major_axis*elem.semi_major_axis / mu);
        
        // Aphelion and perihelion
        elem.aphelion = elem.semi_major_axis * (1 + elem.eccentricity);
        elem.perihelion = elem.semi_major_axis * (1 - elem.eccentricity);
        
        return elem;
    }
};

int main() {
    std::cout << "=== Sun-Earth-Moon System with Real Units ===\n\n";
    
    // Simulation parameters
    SimulationParams params;
    params.box_size = 4.0f;  // 4 AU box
    params.gravity_constant = G_SIM;  // Converted gravitational constant
    params.softening = 1e-6f;  // Very small softening (1000 km in AU)
    params.dt = 0.1f;  // 0.1 days timestep
    params.grid_size = 128;
    
    std::cout << "Scaled units:\n";
    std::cout << "  Distance: 1 unit = 1 AU = " << AU << " m\n";
    std::cout << "  Time: 1 unit = 1 day = " << DAY << " s\n";
    std::cout << "  Mass: 1 unit = 1 solar mass = " << SOLAR_MASS << " kg\n";
    std::cout << "  G_sim = " << G_SIM << "\n\n";
    
    // Create bodies with real values
    std::vector<Particle> bodies(3);
    
    // Sun at origin
    bodies[0].pos = {2.0f, 2.0f};  // Center of box
    bodies[0].vel = {0.0f, 0.0f};
    bodies[0].mass = 1.0f;  // 1 solar mass
    bodies[0].radius = 0.00465f;  // Solar radius in AU
    
    // Earth at 1 AU
    bodies[1].pos.x = 2.0f + 1.0f;  // 1 AU from Sun
    bodies[1].pos.y = 2.0f;
    // Earth orbital velocity: sqrt(GM/r) = 29.78 km/s
    float earth_v = sqrt(G_SIM * bodies[0].mass / 1.0f);
    bodies[1].vel.x = 0.0f;
    bodies[1].vel.y = earth_v;
    bodies[1].mass = EARTH_MASS / SOLAR_MASS;  // Earth mass in solar masses
    bodies[1].radius = 4.26e-5f;  // Earth radius in AU
    
    // Moon at 384,400 km from Earth (0.00257 AU)
    float moon_dist = 384400e3 / AU;  // Convert to AU
    bodies[2].pos.x = bodies[1].pos.x + moon_dist;
    bodies[2].pos.y = bodies[1].pos.y;
    // Moon orbital velocity relative to Earth: 1.022 km/s
    // But it also needs Earth's velocity to orbit the Sun
    float moon_v_rel = sqrt(G_SIM * bodies[1].mass / moon_dist);
    bodies[2].vel.x = bodies[1].vel.x;
    bodies[2].vel.y = bodies[1].vel.y + moon_v_rel;
    bodies[2].mass = MOON_MASS / SOLAR_MASS;
    bodies[2].radius = 1.16e-5f;  // Moon radius in AU
    
    std::cout << "Initial conditions:\n";
    std::cout << "  Sun: pos=(0,0) AU, mass=1.0 M☉\n";
    std::cout << "  Earth: pos=(1,0) AU, vel=" << earth_v*AU/DAY/1000 << " km/s, mass=" 
              << bodies[1].mass << " M☉\n";
    std::cout << "  Moon: pos=(" << moon_dist << ",0) AU from Earth, vel=" 
              << moon_v_rel*AU/DAY/1000 << " km/s rel to Earth\n\n";
    
    // Expected values
    std::cout << "Expected orbital periods:\n";
    std::cout << "  Earth: 365.25 days\n";
    std::cout << "  Moon: 27.32 days (sidereal)\n\n";
    
    // Create backend
    auto backend = std::make_unique<SimpleBackend>();
    backend->setAlgorithm(ForceAlgorithm::BRUTE_FORCE);  // Use exact for 3 bodies
    backend->initialize(bodies.size(), params);
    backend->setParticles(bodies);
    
    // Real-time viewer - start zoomed on full system, then zoom to Earth
    RealTimeViewer viewer(80, 40, 30.0f);  // 30x zoom initially
    
    // Simulation loop
    float sim_time = 0;
    int steps_per_day = 10;  // 10 steps = 1 day
    int days_to_simulate = 400;  // > 1 year
    int frame_skip = 10;  // Update display every 10 days
    
    // Track orbital elements
    std::vector<float> earth_distances;
    std::vector<float> moon_distances;
    float moon_period_sum = 0;
    int moon_orbits = 0;
    float last_moon_angle = 0;
    
    std::cout << "Starting real-time simulation...\n\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    for (int day = 0; day < days_to_simulate; day++) {
        // Run physics steps for one day
        for (int s = 0; s < steps_per_day; s++) {
            backend->step(params.dt);
        }
        sim_time += 1.0f;  // 1 day
        
        // Get current positions
        backend->getParticles(bodies);
        
        // Track Earth distance from Sun
        float earth_dx = bodies[1].pos.x - bodies[0].pos.x;
        float earth_dy = bodies[1].pos.y - bodies[0].pos.y;
        float earth_r = sqrt(earth_dx*earth_dx + earth_dy*earth_dy);
        earth_distances.push_back(earth_r);
        
        // Track Moon distance from Earth
        float moon_dx = bodies[2].pos.x - bodies[1].pos.x;
        float moon_dy = bodies[2].pos.y - bodies[1].pos.y;
        float moon_r = sqrt(moon_dx*moon_dx + moon_dy*moon_dy);
        moon_distances.push_back(moon_r);
        
        // Detect Moon orbits
        float moon_angle = atan2(moon_dy, moon_dx);
        if (last_moon_angle < -M_PI/2 && moon_angle > M_PI/2) {
            moon_orbits++;
            if (moon_orbits > 1) {
                moon_period_sum = sim_time / moon_orbits;
            }
        }
        last_moon_angle = moon_angle;
        
        // Update display
        if (day % frame_skip == 0) {
            viewer.clear();
            
            // After day 100, zoom in on Earth to see Moon orbit
            if (sim_time > 100) {
                // Center on Earth, zoom way in to see Moon
                viewer.setCenter(bodies[1].pos.x, bodies[1].pos.y);
                viewer.scale = 10000.0f;  // Less zoom to see full Moon orbit
                
                // Draw Earth at center
                viewer.plotPoint(bodies[1].pos.x, bodies[1].pos.y, 'E');  // Earth
                viewer.plotPoint(bodies[2].pos.x, bodies[2].pos.y, 'M');  // Moon
                
                // Draw line from Earth to Moon to show connection
                float dx = bodies[2].pos.x - bodies[1].pos.x;
                float dy = bodies[2].pos.y - bodies[1].pos.y;
                for (int i = 1; i < 5; i++) {
                    float t = i / 5.0f;
                    float x = bodies[1].pos.x + t * dx;
                    float y = bodies[1].pos.y + t * dy;
                    viewer.plotPoint(x, y, '.');
                }
            } else {
                // Full system view for first 100 days
                viewer.setCenter(2.0f, 2.0f);  // Center on Sun
                viewer.scale = 30.0f;
                
                // Draw all bodies
                viewer.plotPoint(bodies[0].pos.x, bodies[0].pos.y, '*');  // Sun
                viewer.plotPoint(bodies[1].pos.x, bodies[1].pos.y, 'E');  // Earth
                viewer.plotPoint(bodies[2].pos.x, bodies[2].pos.y, 'M');  // Moon
            }
            
            if (sim_time > 100) {
                viewer.render("Earth-Moon System (Zoomed View)", sim_time);
            } else {
                viewer.render("Sun-Earth-Moon System (Full View)", sim_time);
            }
            
            // Calculate and display orbital elements
            auto earth_orbit = OrbitalElements::calculate(
                bodies[1].pos.x, bodies[1].pos.y, 
                bodies[1].vel.x, bodies[1].vel.y,
                bodies[0].pos.x, bodies[0].pos.y,
                bodies[0].mass, G_SIM
            );
            
            std::cout << "\nEarth Orbital Elements:\n";
            std::cout << "  Current distance: " << earth_r << " AU (expected: ~1.0)\n";
            std::cout << "  Semi-major axis: " << earth_orbit.semi_major_axis << " AU\n";
            std::cout << "  Eccentricity: " << earth_orbit.eccentricity << " (expected: 0.0167)\n";
            std::cout << "  Period: " << earth_orbit.period << " days (expected: 365.25)\n";
            
            std::cout << "\nMoon Orbital Elements:\n";
            std::cout << "  Distance from Earth: " << moon_r*AU/1000 << " km (expected: 384,400)\n";
            if (moon_orbits > 0) {
                std::cout << "  Measured period: " << moon_period_sum << " days (expected: 27.32)\n";
            }
            std::cout << "  Completed orbits: " << moon_orbits << "\n";
            
            // Brief pause for animation
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    // Final analysis
    std::cout << "\n\n" << std::string(60, '=') << "\n";
    std::cout << "SIMULATION COMPLETE - FINAL ANALYSIS\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    // Calculate statistics
    float earth_min = *std::min_element(earth_distances.begin(), earth_distances.end());
    float earth_max = *std::max_element(earth_distances.begin(), earth_distances.end());
    float earth_avg = 0;
    for (float d : earth_distances) earth_avg += d;
    earth_avg /= earth_distances.size();
    
    float moon_min = *std::min_element(moon_distances.begin(), moon_distances.end());
    float moon_max = *std::max_element(moon_distances.begin(), moon_distances.end());
    float moon_avg = 0;
    for (float d : moon_distances) moon_avg += d;
    moon_avg /= moon_distances.size();
    
    std::cout << "Earth Orbit:\n";
    std::cout << "  Average distance: " << earth_avg << " AU (expected: 1.0)\n";
    std::cout << "  Perihelion: " << earth_min << " AU (expected: 0.983)\n";
    std::cout << "  Aphelion: " << earth_max << " AU (expected: 1.017)\n";
    std::cout << "  Error: " << fabs(earth_avg - 1.0) * 100 << "%\n\n";
    
    std::cout << "Moon Orbit:\n";
    std::cout << "  Average distance: " << moon_avg*AU/1000 << " km (expected: 384,400)\n";
    std::cout << "  Perigee: " << moon_min*AU/1000 << " km (expected: 356,500)\n";
    std::cout << "  Apogee: " << moon_max*AU/1000 << " km (expected: 406,700)\n";
    std::cout << "  Measured period: " << moon_period_sum << " days (expected: 27.32)\n";
    std::cout << "  Period error: " << fabs(moon_period_sum - 27.32) / 27.32 * 100 << "%\n";
    
    return 0;
}