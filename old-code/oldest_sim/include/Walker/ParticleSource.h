#ifndef PARTICLE_SOURCE_H
#define PARTICLE_SOURCE_H

#include <list>
#include <cmath>
#include "glut.h"
#include "Globals.h"
#include "Common.h"

enum ParticleType { FLAME, SMOKE };

class Particle {
public:
    virtual ParticleType getType() =0; // returns the type of the particle; see ParticleType {FLAME, SMOKE} above
    virtual bool isExpired()       =0; // returns the life-time remaining for the particle
    virtual void draw()            =0; // draws the particle
    virtual void update()          =0; // updates the state of the particle
};

class SmokeParticle: virtual public Particle {
public:
    SmokeParticle(GLfloat initPos[3], GLfloat theta, GLfloat phi, GLfloat initVel, bool jitter = true);
    ParticleType getType();
    bool isExpired();
    void update();
    void draw();      // draw the smoke particle

    GLfloat pos[3];   // particle position
    GLfloat v[3];     // velocity vector
    GLfloat a[3];     // acceleration vector

    GLfloat size;     // each particle is a square; size is the length of a side
    GLuint  lifeSpan; // the life-span of a particle
    GLuint  age;      // the age of a praticle

    GLfloat gravity;  // accelerate in the direction of -y
};

class ParticleSource {
public:
    void init(const GLfloat pos[], const GLfloat theta, const GLfloat phi, const GLuint texID,
              const GLfloat emitMag, const GLfloat emitRadius, const GLuint emitRate);
    void incrRate();
    void decrRate();
    void incrMag();
    void decrMag();
    void setPos(GLfloat pos[3]);
    void setPos(GLfloat x, GLfloat y, GLfloat z);
    void emit();
    void update();

    GLfloat pos[3];
    GLfloat theta, phi;
    GLfloat emitMag, emitRadius;
    GLuint  emitRate;

    GLuint  texID;

    std::list<SmokeParticle> particles;
};

#endif