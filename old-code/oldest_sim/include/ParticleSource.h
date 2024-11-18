#ifndef EMITTER_H
#define EMITTER_H

#include "object.h"
#include "globals.h"
#include "objectFactory.h"
#include <random>
#include <ctime>

// make jet engine, with tank
// make emitter an object class
// generalize object class -- not all objects need to have draw(), etc.
// make a "physical" subclass or something for that stuff
class Emitter {
public:
    Emitter() {};

    // add angle + mag constructor

    // pass function pointers to determine all variances
    // and rates?
    Emitter(
        ObjectType type,
        const GLfloat emitPos[2],
        const GLfloat emitPosVariance[2],
        const GLfloat emitVector[2],
        const GLfloat emitVectorVariance[2],
        const GLfloat emitColor[3],
        const GLfloat emitColorVariance[3],
        GLfloat emitRate,
        GLfloat emitRateVariance,
        GLfloat emitRateChange,
        GLfloat emitRateChangeVariance,
        GLfloat meanLifetime,
        GLfloat lifetimeVariance,
        Object *relativeTo = NULL) {

        set(type,
            emitPos, emitPosVariance,
            emitVector, emitVectorVariance,
            emitColor, emitColorVariance,
            emitRate, emitRateVariance,
            emitRateChange, emitRateChangeVariance,
            meanLifetime, lifetimeVariance,
            relativeTo);
    };

    void set(
        ObjectType type,
        const GLfloat emitPos[2],
        const GLfloat emitPosVariance[2],
        const GLfloat emitVector[2],
        const GLfloat emitVectorVariance[2],
        const GLfloat emitColor[3],
        const GLfloat emitColorVariance[3],
        GLfloat emitRate,
        GLfloat emitRateVariance,
        GLfloat emitRateChange,
        GLfloat emitRateChangeVariance,
        GLfloat meanLifetime,
        GLfloat lifetimeVariance,
        Object *relativeTo) {

        this->type = type;

        memcpy(this->emitPos, emitPos, 8);
        memcpy(this->emitPosVariance, emitPosVariance, 8);

        memcpy(this->emitVector, emitVector, 8);
        memcpy(this->emitVectorVariance, emitVectorVariance, 8);

        memcpy(this->emitColor, emitColor, 12);
        memcpy(this->emitColorVariance, emitColorVariance, 12);

        this->meanLifetime = meanLifetime;
        this->lifetimeVariance = lifetimeVariance;

        this->emitRate = emitRate;
        this->emitRateVariance = emitRateVariance;

        this->emitRateChange = emitRateChange;
        this->emitRateChangeVariance = emitRateChangeVariance;

        this->accumulator = 0;
        this->emitAmount = 0;

        this->relativeTo = relativeTo;

        this->eng.seed(time(0));
    };

    void setEmitVector(const GLfloat emitVector[2]) {
        memcpy(this->emitVector, emitVector, 8);
    };

    void setEmitVectorVariance(const GLfloat emitVectorVariance[2]) {
        memcpy(this->emitVectorVariance, emitVectorVariance, 8);
    };

    void setEmitPosition(const GLfloat emitPos[2]) {
        memcpy(this->emitPos, emitPos, 8);
    };

    void setEmitPosVariance(const GLfloat emitPosVariance[2]) {
        memcpy(this->emitPosVariance, emitPosVariance, 8);
    };

    void setLifetime(GLfloat meanLifetime, GLfloat lifetimeVariance) {
        this->meanLifetime = meanLifetime;
        this->lifetimeVariance = lifetimeVariance;
    };

    void setType(ObjectType type) {
        this->type = type;
    };

    void update() {
        emitRate += emitRateChange + emitRateChangeVariance * dist(eng);
        accumulator += emitRate + emitRateVariance * dist(eng);

        GLuint num = (GLuint)accumulator;
        emitAmount += num;

        GLfloat pos[] = {emitPos[0] + relativeTo->getPosition()[0], emitPos[1] + relativeTo->getPosition()[1]};
        GLfloat vel[] = {emitVector[0] + relativeTo->getVelocity()[0], emitVector[1] + relativeTo->getVelocity()[1]};

        for (unsigned int i = 0; i < num; ++i) {
            makeObject(type, 0.5, 1.0, emitColor, pos, vel);
        }

        accumulator -= num;
    };

protected:
    Object *relativeTo;

    ObjectType type;

    GLfloat emitColor[3];
    GLfloat emitColorVariance[3];
    
    GLfloat emitPos[2];
    GLfloat emitPosVariance[2];

    GLfloat emitVector[2];
    GLfloat emitVectorVariance[2];

    GLfloat accumulator; // keeps track of how many emits needed to keep pace with emitRate
    GLfloat emitAmount; // number of particles emitted
    GLfloat emitRate; // change in emit amount (emit amount / dt)
    GLfloat emitRateChange; // change in emit rate
    GLfloat emitRateVariance;
    GLfloat emitRateChangeVariance;

    GLfloat meanLifetime;
    GLfloat lifetimeVariance;

    std::mt19937 eng;
    std::normal_distribution<GLfloat> dist;
};

class JetEngine: public Object {
public:
    JetEngine() {
        GLfloat pos[] = {3, 3};
        GLfloat vel[] = {10, 10};
        GLfloat tmp[] = {0, 0};
        GLfloat clr[] = {1, 1, 1};

        jet.set(ProjectileType, pos, tmp, vel, tmp, clr, tmp, 0, 0, 0, 0, 20, 0, this);
    }

protected:
    GLfloat fuelCapacity;
    GLfloat fuel;

    Emitter jet;
};


#endif