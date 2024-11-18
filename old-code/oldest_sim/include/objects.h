///////////////////////////////////////////////////////
//    Filename: objects.h                            //
//        Name: Alex Towell                          //
//    Due Date: 11-3-2009                            //
//      Course: CS 482 (Computer Graphics)           //
// Description: This interface / implementation file //
// stores all the structures and classes used in     //
// the driver, driver.cpp.                           //
///////////////////////////////////////////////////////

#ifndef OBJECTS_H
#define OBJECTS_H

#include <list>
#include "globals.h"
#include "utils.h"
#include <cmath>

enum ParticleType { FLAME, SMOKE };

class Particle {
public:
    virtual ParticleType getType() =0; // returns the type of the particle; see ParticleType {FLAME, SMOKE} above
    virtual bool isExpired() =0;       // returns the life-time remaining for the particle
    virtual void draw() =0;            // draws the particle
    virtual void update() =0;          // updates the state of the particle
};

class SmokeParticle: virtual public Particle {
public:
    SmokeParticle(GLfloat initPos[3], GLfloat theta, GLfloat phi, GLfloat initVel, bool jitter = true) {
        p[0] = initPos[0];
        p[1] = initPos[1];
        p[2] = initPos[2];

        if (jitter) {
            p[0] += getRand(-0.05, 0.05);
            p[1] += getRand(-0.05, 0.05);
            p[2] += getRand(-0.05, 0.05);

            phi   += getRand(-PI/8, PI/8);
            theta += getRand(-PI/8, PI/8);
            initVel *= getRand(MIN_VARIANCE, MAX_VARIANCE);
        }

        v[0] = initVel * cos(theta) * cos(phi);
        v[1] = initVel * sin(phi);
        v[2] = initVel * sin(theta) * cos(phi);

        lifeSpan = getRandInt(SMOKE_MIN_LIFE_SPAN, SMOKE_MAX_LIFE_SPAN);
        size     = SMOKE_SIZE * getRand(MIN_VARIANCE, MAX_VARIANCE);
        gravity  = GRAVITY * getRand(MIN_VARIANCE, MAX_VARIANCE);
    };

    ParticleType getType() { return SMOKE; };
    bool isExpired() { return lifeSpan < age; };

    void update() {
        ++age;

        a[0] = -v[0] * DRAG_COEFF;
        a[1] = -v[1] * DRAG_COEFF - gravity;
        a[2] = -v[2] * DRAG_COEFF;

        v[0] += a[0];
        v[1] += a[1];
        v[2] += a[2];

        p[0] += v[0];
        p[1] += v[1];
        p[2] += v[2];
    };

    // draw the smoke particle
    void draw() {
        glPushMatrix();
            glBegin(GL_QUADS);
                glTexCoord2f(0.0, 0.0); glVertex3f(p[0]-size/2, p[1]-size/2, p[3]);
                glTexCoord2f(1.0, 0.0); glVertex3f(p[0]+size/2, p[1]-size/2, p[3]);
                glTexCoord2f(1.0, 1.0); glVertex3f(p[0]+size/2, p[1]+size/2, p[3]);
                glTexCoord2f(0.0, 1.0); glVertex3f(p[0]-size/2, p[1]+size/2, p[3]);
            glEnd();
        glPopMatrix();
    }

    GLfloat p[3];      // particle position
    GLfloat v[3];      // velocity vector
    GLfloat a[3];      // acceleration vector

    GLfloat size;      // each particle is a square; size is the length of a side
    GLuint  lifeSpan;  // the life-span of a particle
    GLuint  age;       // the age of a praticle

    GLfloat gravity;   // "negative" acceleration in the direction of -y
};

class ParticleSource {
public:
    void init(const GLfloat pos[], const GLfloat theta, const GLfloat phi,
              const GLfloat emitMagnitude, const GLfloat emitRadius, const GLuint emitRate) {
        this->pos[0]        = pos[0];
        this->pos[1]        = pos[1];
        this->pos[2]        = pos[2];

        this->phi           = phi;
        this->theta         = theta;

        this->emitMagnitude = emitMagnitude;
        this->emitRadius    = emitRadius;
        this->emitRate      = emitRate;
    };

    void incrRate() { ++emitRate; };
    void decrRate() { --emitRate; };
    void incrMagnitude() { ++emitMagnitude; };
    void decrMagnitude() { --emitMagnitude; };

    void emit() {
        for (unsigned int i = 0; i < emitRate; ++i)
            particles.push_back(SmokeParticle(pos, theta, phi, emitMagnitude));

        glPushMatrix();
            glEnable(GL_TEXTURE_2D);
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
            glBindTexture(GL_TEXTURE_2D, TexID::TorchSmoke);
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE_MINUS_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR);

            list<SmokeParticle>::iterator i = particles.begin(), tmp;
            while (i != particles.end()) {
                i->update();
                if (i->isExpired()) { tmp = i; ++i; particles.erase(tmp); }
                else                { i->draw(); ++i; }
            }
        glPopMatrix();
    };

private:
    GLfloat pos[3];
    GLfloat dir[3];

    GLfloat theta, phi;
    GLfloat emitMagnitude, emitRadius;
    GLuint  emitRate;

    list<SmokeParticle> particles;
};

// Torch
class Torch {
public:
    void init(const GLfloat base[3], const GLfloat radius, const GLfloat height, const GLfloat theta, const GLfloat phi) {
        this->base[0] = base[0];
        this->base[1] = base[1];
        this->base[2] = base[2];

        this->radius  = radius;
        this->height  = height;
        this->theta   = theta;
        this->phi     = phi;

        this->mode    = SMOKE;

        GLfloat srcPos[3];
        srcPos[0] = base[0] + height * cos(theta) * cos(phi);
        srcPos[1] = base[1] + height * sin(phi);
        srcPos[2] = base[2] + height * sin(theta) * cos(phi);

        this->src.init(srcPos, theta, phi, 0.75, radius, 5);
    };

    void changeMode(ParticleType type) {
        this->mode = type;
    };

    void update() {
    };

    void draw() {
	    glPushMatrix();
	        GLUquadricObj* q = gluNewQuadric();
	        gluQuadricNormals(q, GLU_SMOOTH);		
	        gluQuadricTexture(q, GL_TRUE);
	        glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, TexID::TorchWood);

            glTranslatef(base[0], base[1], base[2]);
            glRotatef(90.0f, 0.0, 1.0, 0.0);
            glRotatef(-phi * 180/PI, 1.0, 0.0, 0.0);
            glRotatef(-theta * 180/PI, 0.0, 1.0, 0.0);
            gluCylinder(q, 0.025, radius, height, 24, 12);
            glTranslatef(0.0f, 0.0f, height);
            gluDisk(q, 0.0f, radius, 24, 12);
            gluDeleteQuadric(q);
	    glPopMatrix();

        glPushMatrix();
            src.emit();
        glPopMatrix();
    };

    ParticleType mode; // current torch mode: flame or smoke
    GLfloat base[3]; // base point location of torch
    GLfloat radius;  // top radius of torch
    GLfloat height;  // height of torch
    GLfloat theta, phi;

    ParticleSource src;
};

#endif