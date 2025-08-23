#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <deque>
#include <functional>
#include <map>
#include "../../backend/ISimulationBackend.h"

// Terminal-based monitoring and admin interface for the simulation backend
// Provides visualization, stats, and eventually REPL integration
class TerminalMonitor {
public:
    enum class PanelMode {
        VISUALIZATION,    // Main particle view
        STATS_EXTENDED,   // Detailed statistics
        CONSOLE,         // REPL interface
        SPLIT_VIEW       // Viz + console
    };
    
private:
    // Display configuration
    int width;
    int height;
    int viz_height;      // Height of visualization panel
    int console_height;  // Height of console panel
    PanelMode mode;
    
    // View configuration
    float view_x, view_y;  // Center of view
    float zoom;
    bool auto_center;      // Auto-center on center of mass
    bool show_trails;      // Show particle trails
    
    // Buffers
    std::vector<std::vector<char>> viz_buffer;
    std::vector<std::vector<char>> color_buffer;  // Store color codes
    std::deque<std::string> console_history;
    std::string console_input;
    
    // Performance tracking
    struct Stats {
        float fps;
        float simulation_fps;
        size_t particle_count;
        float total_energy;
        float total_momentum;
        float center_of_mass_x;
        float center_of_mass_y;
        float avg_velocity;
        float max_velocity;
        float simulation_time;
        size_t step_count;
        
        // Backend info
        std::string backend_name;
        std::string algorithm;
        size_t memory_usage_mb;
        
        // Performance history
        std::deque<float> fps_history;
        std::deque<float> energy_history;
    } stats;
    
    // REPL command handlers
    std::map<std::string, std::function<std::string(const std::vector<std::string>&)>> commands;
    
    void initializeCommands() {
        commands["help"] = [this](const auto& args) {
            return "Available commands:\n"
                   "  view <x> <y> [zoom] - Set view position\n"
                   "  zoom <factor>       - Set zoom level\n"
                   "  mode <viz|stats|console|split> - Change display mode\n"
                   "  pause/resume        - Control simulation\n"
                   "  step [n]           - Step simulation n times\n"
                   "  spawn <type> [params] - Add particles\n"
                   "  clear [x y radius] - Clear region or all\n"
                   "  stats              - Show statistics\n"
                   "  perf               - Show performance info\n"
                   "  algo <brute|barnes|pm> - Switch algorithm\n"
                   "  set <param> <value> - Set simulation parameter\n"
                   "  save <filename>    - Save simulation state\n"
                   "  load <filename>    - Load simulation state\n"
                   "  quit               - Exit monitor";
        };
        
        commands["view"] = [this](const auto& args) {
            if (args.size() >= 2) {
                view_x = std::stof(args[0]);
                view_y = std::stof(args[1]);
                if (args.size() >= 3) zoom = std::stof(args[2]);
                return "View updated";
            }
            return "Usage: view <x> <y> [zoom]";
        };
        
        commands["zoom"] = [this](const auto& args) {
            if (args.size() >= 1) {
                zoom = std::stof(args[0]);
                return "Zoom set to " + std::to_string(zoom);
            }
            return "Usage: zoom <factor>";
        };
        
        commands["mode"] = [this](const auto& args) {
            if (args.size() >= 1) {
                if (args[0] == "viz") mode = PanelMode::VISUALIZATION;
                else if (args[0] == "stats") mode = PanelMode::STATS_EXTENDED;
                else if (args[0] == "console") mode = PanelMode::CONSOLE;
                else if (args[0] == "split") mode = PanelMode::SPLIT_VIEW;
                else return "Unknown mode: " + args[0];
                updateLayout();
                return "Mode changed to " + args[0];
            }
            return "Usage: mode <viz|stats|console|split>";
        };
        
        commands["stats"] = [this](const auto& args) {
            return formatStats();
        };
        
        commands["perf"] = [this](const auto& args) {
            return formatPerformance();
        };
    }
    
    void updateLayout() {
        switch (mode) {
            case PanelMode::VISUALIZATION:
                viz_height = height - 8;  // Leave room for stats bar
                console_height = 0;
                break;
            case PanelMode::CONSOLE:
                viz_height = 0;
                console_height = height - 4;
                break;
            case PanelMode::SPLIT_VIEW:
                viz_height = (height - 8) * 2/3;
                console_height = (height - 8) * 1/3;
                break;
            case PanelMode::STATS_EXTENDED:
                viz_height = height - 20;  // More room for stats
                console_height = 0;
                break;
        }
        
        // Resize buffers
        viz_buffer.resize(viz_height, std::vector<char>(width, ' '));
        color_buffer.resize(viz_height, std::vector<char>(width, 0));
    }
    
    char getColorCode(char particle_char) {
        switch (particle_char) {
            case '@': return 5;  // Magenta for black holes
            case 'O': return 3;  // Yellow for large
            case 'o': return 6;  // Cyan for medium
            case '#': return 1;  // Red for dense
            case ':': return 2;  // Green for clusters
            default: return 7;   // White for normal
        }
    }
    
    std::string formatStats() {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "Particles: " << stats.particle_count
           << " | Energy: " << stats.total_energy
           << " | CoM: (" << stats.center_of_mass_x << ", " << stats.center_of_mass_y << ")"
           << " | Avg Vel: " << stats.avg_velocity;
        return ss.str();
    }
    
    std::string formatPerformance() {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1);
        ss << "FPS: " << stats.simulation_fps
           << " | Backend: " << stats.backend_name
           << " | Algorithm: " << stats.algorithm
           << " | Memory: " << stats.memory_usage_mb << " MB";
        return ss.str();
    }
    
public:
    TerminalMonitor(int w = 120, int h = 40) 
        : width(w), height(h), view_x(500), view_y(500), zoom(1.0f),
          auto_center(false), show_trails(false), mode(PanelMode::SPLIT_VIEW) {
        
        initializeCommands();
        updateLayout();
        stats = {0};
        console_history.push_back("Digital Star Terminal Monitor v1.0");
        console_history.push_back("Type 'help' for commands");
    }
    
    void setView(float x, float y, float z) {
        view_x = x;
        view_y = y;
        zoom = z;
    }
    
    void clearVisualization() {
        for (auto& row : viz_buffer) {
            std::fill(row.begin(), row.end(), ' ');
        }
        for (auto& row : color_buffer) {
            std::fill(row.begin(), row.end(), 0);
        }
    }
    
    void renderParticles(const std::vector<Particle>& particles, float box_size) {
        if (viz_height == 0) return;
        
        clearVisualization();
        
        // Auto-center if enabled
        if (auto_center && stats.particle_count > 0) {
            view_x = stats.center_of_mass_x;
            view_y = stats.center_of_mass_y;
        }
        
        for (const auto& p : particles) {
            // Transform to screen coordinates
            float x = (p.pos.x - view_x) * zoom + width/2;
            float y = (p.pos.y - view_y) * zoom + viz_height/2;
            
            int sx = (int)x;
            int sy = (int)y;
            
            if (sx >= 0 && sx < width && sy >= 0 && sy < viz_height) {
                // Choose character based on mass/type
                char c = '.';
                if (p.mass > 100) c = '@';  // Black hole
                else if (p.mass > 10) c = 'O';  // Large mass
                else if (p.mass > 1) c = 'o';   // Medium mass
                else if (viz_buffer[sy][sx] == ' ') c = '.';  // Small mass
                else if (viz_buffer[sy][sx] == '.') c = ':';  // Multiple particles
                else if (viz_buffer[sy][sx] == ':') c = '#';  // Dense region
                
                viz_buffer[sy][sx] = c;
                color_buffer[sy][sx] = getColorCode(c);
            }
        }
        
        // Add grid lines if zoomed out enough
        if (zoom < 0.5f) {
            int grid_spacing = 100 * zoom;
            if (grid_spacing > 5) {
                for (int gx = 0; gx < width; gx += grid_spacing) {
                    for (int gy = 0; gy < viz_height; gy += grid_spacing) {
                        if (viz_buffer[gy][gx] == ' ') {
                            viz_buffer[gy][gx] = '+';
                            color_buffer[gy][gx] = 8;  // Dark gray
                        }
                    }
                }
            }
        }
    }
    
    void updateStats(const std::vector<Particle>& particles, 
                    ISimulationBackend* backend,
                    float fps, size_t steps) {
        stats.simulation_fps = fps;
        stats.particle_count = particles.size();
        stats.step_count = steps;
        
        if (backend) {
            stats.backend_name = backend->getBackendName();
            switch (backend->getAlgorithm()) {
                case ForceAlgorithm::BRUTE_FORCE: stats.algorithm = "Brute"; break;
                case ForceAlgorithm::BARNES_HUT: stats.algorithm = "Barnes-Hut"; break;
                case ForceAlgorithm::PARTICLE_MESH: stats.algorithm = "PM"; break;
                case ForceAlgorithm::HYBRID: stats.algorithm = "Hybrid"; break;
            }
            stats.memory_usage_mb = backend->getMemoryUsage() / (1024 * 1024);
        }
        
        // Calculate physics stats
        stats.total_energy = 0;
        stats.total_momentum = 0;
        stats.avg_velocity = 0;
        stats.max_velocity = 0;
        float total_mass = 0;
        stats.center_of_mass_x = 0;
        stats.center_of_mass_y = 0;
        
        for (const auto& p : particles) {
            float v2 = p.vel.x * p.vel.x + p.vel.y * p.vel.y;
            float v = sqrt(v2);
            stats.total_energy += 0.5f * p.mass * v2;
            stats.avg_velocity += v;
            stats.max_velocity = std::max(stats.max_velocity, v);
            
            stats.center_of_mass_x += p.mass * p.pos.x;
            stats.center_of_mass_y += p.mass * p.pos.y;
            total_mass += p.mass;
        }
        
        if (particles.size() > 0) {
            stats.avg_velocity /= particles.size();
            stats.center_of_mass_x /= total_mass;
            stats.center_of_mass_y /= total_mass;
        }
        
        // Update history
        stats.fps_history.push_back(fps);
        if (stats.fps_history.size() > 60) stats.fps_history.pop_front();
        
        stats.energy_history.push_back(stats.total_energy);
        if (stats.energy_history.size() > 100) stats.energy_history.pop_front();
    }
    
    void display() {
        // Clear screen
        std::cout << "\033[2J\033[H";
        
        // Title bar
        std::cout << "╔" << std::string(width-2, '═') << "╗\n";
        std::cout << "║ Digital Star - Terminal Monitor";
        std::cout << std::string(width-34, ' ') << "║\n";
        std::cout << "╠" << std::string(width-2, '═') << "╣\n";
        
        // Visualization panel
        if (viz_height > 0) {
            for (int y = 0; y < viz_height; y++) {
                std::cout << "║";
                for (int x = 0; x < width; x++) {
                    char c = viz_buffer[y][x];
                    int color = color_buffer[y][x];
                    
                    if (color > 0 && c != ' ') {
                        std::cout << "\033[3" << color << "m" << c << "\033[0m";
                    } else {
                        std::cout << c;
                    }
                }
                std::cout << "║\n";
            }
            
            if (console_height > 0) {
                std::cout << "╠" << std::string(width-2, '─') << "╣\n";
            }
        }
        
        // Console panel
        if (console_height > 0) {
            // Show recent history
            int history_start = std::max(0, (int)console_history.size() - console_height + 1);
            for (int i = history_start; i < console_history.size(); i++) {
                std::cout << "║ " << console_history[i];
                int padding = width - console_history[i].length() - 3;
                std::cout << std::string(padding, ' ') << "║\n";
            }
            
            // Input line
            std::cout << "║ > " << console_input;
            int padding = width - console_input.length() - 4;
            std::cout << std::string(padding, ' ') << "║\n";
        }
        
        // Stats bar
        std::cout << "╠" << std::string(width-2, '═') << "╣\n";
        
        // Line 1: Basic stats
        std::string stats_line = formatStats();
        std::cout << "║ " << stats_line;
        std::cout << std::string(width - stats_line.length() - 3, ' ') << "║\n";
        
        // Line 2: Performance
        std::string perf_line = formatPerformance();
        std::cout << "║ " << perf_line;
        std::cout << std::string(width - perf_line.length() - 3, ' ') << "║\n";
        
        // Controls hint
        std::cout << "║ [WASD: pan] [+/-: zoom] [M: mode] [?: help] [Q: quit]";
        std::cout << std::string(width - 57, ' ') << "║\n";
        
        std::cout << "╚" << std::string(width-2, '═') << "╝\n";
    }
    
    // Process REPL command
    std::string processCommand(const std::string& input) {
        if (input.empty()) return "";
        
        // Add to history
        console_history.push_back("> " + input);
        
        // Parse command and arguments
        std::vector<std::string> tokens;
        std::stringstream ss(input);
        std::string token;
        while (ss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.empty()) return "";
        
        std::string cmd = tokens[0];
        tokens.erase(tokens.begin());
        
        // Execute command
        auto it = commands.find(cmd);
        if (it != commands.end()) {
            std::string result = it->second(tokens);
            console_history.push_back(result);
            return result;
        } else {
            std::string error = "Unknown command: " + cmd + " (type 'help' for commands)";
            console_history.push_back(error);
            return error;
        }
    }
    
    // Getters for state
    PanelMode getMode() const { return mode; }
    float getViewX() const { return view_x; }
    float getViewY() const { return view_y; }
    float getZoom() const { return zoom; }
};