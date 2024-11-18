#include "PhysicalAtomicEntity.h"

class PointMass: public PhysicalAtomicEntity
{
public:
    PointMass(const double s[2], const double mass, bool fixed = false);
    PointMass(const double s[2], const double v[2], const double mass, bool fixed = false);
    void addImpulse(const double force[2], double duration);
    void tick();
    double getMass() const;
    double getCharge() const;
    const double *getPositionOfCenterOfMass() const;
    const double *getVelocityOfCenterOfMass() const;
    double getVelocityOfCenterOfMassMagnitude() const;

private:
    double impulse[2];  // net force vector
    double s[2];        // displacement vector
    double v[2];        // velocity vector
    double mass;        // mass scalar
    double charge;      // charge scalar; most will be 0, but some things can have positive or negative

    bool fixed;
};
