class SimulationAPI {
public:
    // Constructor
    SimulationAPI(Simulation* sim);

    // Body management functions
    void addOrUpdateBody(const char* name, const BodyParams& params);

    // JSON loading functions
    void loadBodiesFromJson(const std::string& filename, double3 position_offset = {0.0, 0.0, 0.0}, double3 velocity_offset = {0.0, 0.0, 0.0});
    void loadBodiesFromJsonDir(const std::string& dirPath);

private:
    Simulation* simulation;

    // Helper functions
    void parseAndAddBody(const json& body, double3 position_offset, double3 velocity_offset, int parentIndex);
};
