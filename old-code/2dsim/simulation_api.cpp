SimulationAPI::SimulationAPI(Simulation* sim) : simulation(sim) {}

void SimulationAPI::addOrUpdateBody(const char* name, const BodyParams& params) {
    int index = simulation->getBodyIndexByName(name);
    if (index != -1) {
        simulation->updateBody(index, params);
    } else {
        Body newBody;
        strcpy(newBody.name, name);
        newBody.position = params.position.value_or(make_double3(0.0, 0.0, 0.0));
        newBody.velocity = params.velocity.value_or(make_double3(0.0, 0.0, 0.0));
        newBody.mass = params.mass.value_or(0.0);
        newBody.radius = params.radius.value_or(0.0);
        newBody.color = params.color.value_or(make_float3(1.0, 1.0, 1.0));
        newBody.parentIndex = params.parentIndex;
        simulation->addBody(newBody);
    }
}

void SimulationAPI::loadBodiesFromJson(const std::string& filename, double3 position_offset, double3 velocity_offset) {
    std::ifstream file(filename);
    json j;
    file >> j;

    double3 parPos = position_offset;
    double3 parVel = velocity_offset;
    int parIdx = -1;

    if (j.contains("parent")) {
        const auto& par = j["parent"];
        const char* parentName = par["name"].get<std::string>().c_str();
        parIdx = simulation->getBodyIndexByName(parentName);
        if (parIdx != -1) {
            parPos = simulation->getBodies()[parIdx].position;
            parVel = simulation->getBodies()[parIdx].velocity;
        } else {
            double3 relPos = make_double3(par["position"][0].get<double>(), par["position"][1].get<double>(), par["position"][2].get<double>());
            double3 relVel = make_double3(par["velocity"][0].get<double>(), par["velocity"][1].get<double>(), par["velocity"][2].get<double>());
            BodyParams parentParams;
            parentParams.position = make_double3(parPos.x + relPos.x, parPos.y + relPos.y, parPos.z + relPos.z);
            parentParams.velocity = make_double3(parVel.x + relVel.x, parVel.y + relVel.y, parVel.z + relVel.z);
            parentParams.mass = par["mass"].get<double>();
            parentParams.radius = par["radius"].get<double>();
            parentParams.color = make_float3(par["color"][0].get<float>(), par["color"][1].get<float>(), par["color"][2].get<float>());
            addOrUpdateBody(parentName, parentParams);
            parIdx = simulation->getActiveBodyCount() - 1;
        }
    }

    for (const auto& body : j["children"]) {
        parseAndAddBody(body, parPos, parVel, parIdx);
    }
}

void SimulationAPI::loadBodiesFromJsonDir(const std::string& dirPath) {
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(dirPath.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.find(".json") != std::string::npos) {
                std::string filepath = dirPath + "/" + filename;
                loadBodiesFromJson(filepath);
            }
        }
        closedir(dir);
    } else {
        perror("opendir");
    }
}

void SimulationAPI::parseAndAddBody(const json& body, double3 position_offset, double3 velocity_offset, int parentIndex) {
    const char* childName = body["name"].get<std::string>().c_str();
    int childIdx = simulation->getBodyIndexByName(childName);
    if (childIdx != -1) {
        printf("Warning: Body with name %s already exists. Ignoring new body.\n", childName);
        return;
    }
    BodyParams childParams;
    childParams.position = make_double3(position_offset.x + body["position"][0].get<double>(),
                                        position_offset.y + body["position"][1].get<double>(),
                                        position_offset.z + body["position"][2].get<double>());
    childParams.velocity = make_double3(velocity_offset.x + body["velocity"][0].get<double>(),
                                        velocity_offset.y + body["velocity"][1].get<double>(),
                                        velocity_offset.z + body["velocity"][2].get<double>());
    childParams.mass = body["mass"].get<double>();
    childParams.radius = body["radius"].get<double>();
    childParams.color = make_float3(body["color"][0].get<float>(), body["color"][1].get<float>(), body["color"][2].get<float>());
    childParams.parentIndex = parentIndex;
    addOrUpdateBody(childName, childParams);
}
