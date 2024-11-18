#ifndef OBJECT_H
#define OBJECT_H

#include <cmath>
#include <limits>
#include "globals.h"
#include "utils.h"
#include "glColor.h"

enum ObjectType { SpaceshipType, GravityWellType, AsteroidType, MissileType, ProjectileType, JetExhaust };

enum State { Resting, Expire };

/*
class Relationship {
public:
    ObjectRelation(Object *o1, Object *o2, function pointer?) {
        this->o1 = o1;
        this->o2 = o2;
    };

protected:
    Object *o1;
    Object *o2;
};
*/

// change ang GLfloat to vector? (object.h)

class Object {
public:
    virtual ObjectType what() const  =0;
    virtual void update()            =0;
    virtual void draw() const        =0;
    //virtual const std::string &id() const =0;

    bool isFixed() const;

    GLfloat getRadius() const;
    GLfloat getAngle() const;
    GLfloat getMass() const;
    GLfloat getMomentOfInertia() const;

    GLfloat getMomentum() const {
        return mass * getSpeed();
    };

    void setPosition(GLfloat pos[2]);
    void setAngularVelocity(GLfloat velocity);
    void setVelocity(GLfloat vel[2]);
    void setRadius(GLfloat radius);
    void setMass(GLfloat mass);
    void setFixed(bool fixed);

    GLfloat getSpeed() const;

    void transForce(GLfloat theta, GLfloat mag);
    void torqueForce(GLfloat F);
    void transForce(const GLfloat F[2]);

    GLfloat getAngularVelocity() const;
    const GLfloat *getVelocity() const;
    const GLfloat *getPosition() const;
    const glColor &getColor() const;

    bool isExpired() const;
    State getState() const;

    // in 10 millisecond units
    GLuint getAge() const;

    //GLfloat getVelocityMagnitude() const;
    //Vector getVelocity() const;
    //Point getPosition() const;

    //// only makes sense after force has been applied
    //// and before the force vector has been reset
    //Vector getAcceleration() const;
    //Vector getForce() const;

    //SimpleTime getAge() const;
    
//protected:
    bool fixed;
    GLuint t0;

    GLfloat pos[2];
    GLfloat ang;

    GLfloat vel[2];
    GLfloat angVel;

    GLfloat netForce[2];
    GLfloat netTorque;

    GLfloat mass;
    GLfloat moment;
    GLfloat radius;

    State state;
    glColor color;

    //Vector velocity;
    //Vector netForce();
    //Point position;

    //SimpleTime age;
    //GLuint age;
};

#endif