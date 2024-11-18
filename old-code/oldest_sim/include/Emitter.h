#ifndef EMITTER_H
#define EMITTER_H

#include "Entity.h"
#include "Vector2D.h"
#include "ParticleCreator.h"
#include "EmitRateCreator.h"

class Emitter: public Entity
{
public:
                    Emitter(EmitRateCreator *emitRate = NULL);
    void            tick();

protected:
    EmitRateCreator *emitRate;
    EntityCreator   *entityCreator;

    // specify a normal distribution with center as mean
    // and specified variance (radius = 3*variance?)

    // make "straighter" shots normally faster?
    double          velocityMagnitude;

    Vector2D        dir;
    Vector2D        dirNormal;
    
    double          focalLength;
    Vector2D        imagePlane;
    Vector2D        imagePlaneVariance;
};

#endif