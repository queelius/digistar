#ifndef GAS_LAW_H
#define GAS_LAW_H

#include <list>
using namespace std;

namespace {
    const double GAS_CONSTANT = 8.314472f;

    enum PartType { TankType, ConnectorType, CannonType, SpringType, BarrelType };
};

class Part {
public:
    virtual PartType what() const;
    virtual void update();

    double getPressure() const {
        return (getMoles() * GAS_CONSTANT * getTemperature() / getVolume());
    };

    unsigned int getMoles() const {
        return moles;
    };

    double getVolume() const {
        return volume;
    };

    double getTemperature() const {
        return temperature;
    };

protected:
    unsigned int moles;
    double temperature;
    double volume;    
};

class Connector: public Part {
public:
    Connector(double volume) {
        this->volume = volume;
        this->moles = 0;
        this->temperature = 0;
    };

    PartType what() const {
        return ConnectorType;
    };
};

class Tank: public Part {
public:
    Tank(double volume) {
        this->volume = volume;
        this->moles = 0;
        this->temperature = 0;
    };

    PartType what() const {
        return TankType;
    };

    bool isOpen() const {
        return open;
    };

    void open() {
        open = true;
    };

    void close() {
        open = false;
    };

    void update() {
    };

private:
    bool open;
    list<Connector> connections;
};

#endif