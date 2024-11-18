#ifndef SOFTBODY_H
#define SOFTBODY_H

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include "stdlib.h"
#include "glut.h"
using namespace std;

class Spring;
class Node;
class SpringBody;

const double TICK_DURATION = 20.0/1000.0; // 20 msec

/********************************************************
*                                                       *
* Note: provide a few different "z" layers for things   *
*       like rotation axes, etc.                        *
*                                                       *
*       splice in a virtual point mass (node) when one  *
*       point mass intersects another's "surface"       *
*       boundary                                        *
*                                                       *
*       use the elastic limit on surface point masses   *
*       (nodes) to provide rough estimate of "boundary" *
*       since the given body's extent can't go beyond   *
*       that by definition. precompute and use for      *
*       quick checks.                                   *
*                                                       *
*       allow one to "overlay" an arbitrary graphic     *
*       over a spring-connected system. do not use      *
*       a node surface boundary for visual, but this    *
*       boundary may still exist for collision          *
*       detection, etc.                                 *
*                                                       *
*                                                       *
*                                                       *
*                                                       *
*********************************************************/

// elastic connection

// todo:
//  1) be able to determine center of mass position (in this point mass case,
//     center of mass is just location) and velocity of center of mass (in this
//     point mass case, just velocity)
//  2) be able to "attach" forces, like a propulsive force. these impulses should be
//     able to use parameters like velocity of point mass to determine force vector
//
// point mass
// what about point charges? point masses with charges? or point charges with point masses?


/********************************************************
*                                                       *
* Note: provide a few different "z" layers for things   *
*       like rotation axes, etc.                        *
*                                                       *
*********************************************************/

// elastic connection

// todo:
//  1) be able to determine center of mass position (in this point mass case,
//     center of mass is just location) and velocity of center of mass (in this
//     point mass case, just velocity)
//  2) be able to "attach" forces, like a propulsive force. these impulses should be
//     able to use parameters like velocity of point mass to determine force vector
//
// point mass
// what about point charges? point masses with charges? or point charges with point masses?
class Node
{
public:
    Node(const double s[2], const double mass, bool fixed = false);
    Node(const double s[2], const double v[2], const double mass, bool fixed = false);
    void dump() const;
    void addForce(const double force[2]);
    void tick();
    double getMass() const;
    const double *getPosition() const;
    const double *getCenterOfMass() const;
    const double *getVelocityOfCenterOfMass() const;
    double getVelocityOfCenterOfMassMagnitude() const;

//private:
    double force[2]; // net force vector
    double s[2];     // displacement vector
    double v[2];     // velocity vector
    double mass;     // mass scalar
    double charge;   // charge scalar; most will be 0, but some things can have positive or negative

    bool fixed;
};

// have "ideal spring" class inherit from elastic connection, providing it
// with appropriate parameters

// allow the force applied to the end point masses of an elastic connection
// be the result of a function object passed to it

// elastic connection
class Spring {
public:
    // go above elastic limit:
    //  a) change spring constant (elastic modulus?)
    //  b) go far enoug above limit, break elastic connection
    //     * note that sometimes this may cause a composite body
    //       to break into two elastically disconnected pieces if
    //       there are no other elastic connections between these
    //       two bodies

    // elastic modulus
    // elastic limit
    // stiffness
    // linear elasticity
    // non-linear elasticity

    Spring(double springConstant, double equilibriumDist, Node *end1, Node *end2);
    void tick();

//private:
    double springConstant;
    double equilibriumDist;
    Node *end1;
    Node *end2;
};

// big reasons for a composite class even though each node (point mass) is its own thing
//  1) convex hull calculations; one composite object has one visual boundary for aesthetics
//  2) able to easily create point masses where frame of reference for velocity and position
//     is relative to the center of mass position and center of mass velocity of a composite object
//  3) add a point mass such that, when indicated point mass is added, composite object has
//     specified center of mass position / velocity
//  4) force connections, like gravity connections or elastic connections, can be applied to the
//     composite body rather than each node (point mass) of the composite body. this can reduce
//     computational complexity quite a lot i believe as N (total number of independent masses) grows large
//
//  NOTE: i need to generalize this. every thing is a "body" (some things are composite bodies, and some things
//        are atomic bodies. a composite body can contain other bodies -- composite bodies and atomic bodies -- and
//        force connections can be specified at any level of the hierarchy.
//
//        for instance, you can have a galaxy composite object that consists of other composite objects, like star systems
//        and even non-star system bound "rogue" planets. every object at this level of the heirarchy, only one level deep,
//        has a gravity connection to all the other objects at this depth level. going deeper into the heirarchy, let's examine
//        a star system composite body. a star system composite body consists of composite bodies of stars, planetary systems, asteroids,
//        and so on. every object in this star system at this level of the heirarchy has a gravity connection to every other
//        object in this star system at this level of the heirarchy, unless otherwise indicated, e.g., asteroids orbiting a star
//        could be specified to only have gravity connections to said star, ignoring the other bodies, like other asteroids and planets,
//        or ignoring other bodies that have a mass less than some lower bound. going deeper into the heirarchy of this star system,
//        let's examine a planetary system composite body. a planetary system consists of one or more planets or planetoids (e.g., moons)
//        that have gravity connections to each other, and asteroids orbiting around them (with the same options as previously mentioned).
//        going deeper into the heirarchy of this planetary system, let's examine a planet. a planet is a composite body consisting of
//        smaller things, and so on.
//
// to do:
//  1) implement methods for center of mass for position and velocity
//  2) call this a CompositeBody? body held together by springs -- can this be specified in terms of atomic charges?
class SpringBody
{
public:
    SpringBody();
    void tick();

    // equilibrium distance for two nodes defaults to their distance apart that existed
    // during their initial connection
    void joinNodes(double springConstant, unsigned int endNode1, unsigned int endNode2);

    // equilibrium distance for two nodes defaults to their distance apart that existed
    // during their initial connection
    void joinNodes(double springConstant, double equilibriumDist, unsigned int endNode1, unsigned int endNode2);
    void addNode(double nodeMass, double x, double y, bool fixed = false);

    //double *getDipoleMoment()
    //double momentum(SpringBody ref);
    //double momentOfInertia(double axis[2]);
    //void connectNode(Node *node);

    // investigate ability to have negative mass (like a negative charge)
    void addNode(double nodeMass, const double nodePos[2], bool fixed = false);
    
    // put in a bit of checking so that if not needed,
    // don't recompute velocity center of mass every time you
    // call the method
    const double *getVelocityOfCenterOfMass();
    double getVelocityOfCenterOfMassMagnitude();
    const double *getCenterOfMass();

    // a = v_i^2 / r_i
    // use this to force rotation about a fixed axis;
    // create a radius force (pointing towards the center of mass) that applies this to every particle in the composite body

    double getKineticEnergy();
    double getAngularKineticEnergy();
    double getLinearKineticEnergy();
    const double *getMomentum();
    double getLinearMomentum(); // INCORRECT
    //double getAngularMomentum();

    // give impulse to bodies (that make up this object) in a direction
    // perpendicular to axis. add impulse to each body such that
    // change in w, dw, is the same for every body with respect to
    // specified axis?
    void addAngularImpulse(double force[2], double duration, double axis[2]);

    // give impulse to bodies (that make up this object) in a direction
    // perpendicular to center of mass (as an axis of rotation)?
    // or maybe not.
    void addAngularImpulse(double force[2], double duration);

    // add impulse to bodies (that make up this object) such that
    // every body gets same amount of change in momentum?
    // or maybe not.
    void addImpulse(double force[2], double duration);

    // // so, can add a radial force? -1 time duration for infinity
    // void addContinuousForce(function object for force, interval of time)
    // void addContinuousForce(function object for force, start time, end time)

    void addConstantForce(double force[2], double duration);
    void addConstantForce(double force[2], double startTime, double endTime);

    //void addForce(function object whose path you want the force to cause the object to move on)

    // actually, this is addImpulse; specify time of force contact also
    // a few notes:
    //  1) gravity provides the same impulse to every body in this composite
    //     body, so it'll be convenient to have the same impulse that can be
    //     applied to each body in the composition.
    //  2) it will also be nice to have a way to add a force
    void addForce(double force[2]);
    unsigned int getNodeCount();
    double getMass();

    // double getCharge() { return totalCharge; };

//private:

    // should not be only point masses; should be objects, and an object can be a point
    // mass but can also be any other object, composite or otherwise.
    vector<Node*> inodes;
    vector<Spring*> isprings;

    double massMoment[2];
    double centerOfMass[2];
    double vcm[2];
    double momentum[2];
    double totalMass;
    // double totalCharge;
};

//// make this a join by specified force, e.g., an elastic connection, gravity connection, etc.
//// pass function objects to do all of these calculations, then create helper functions that
//// call these general connectors with the appropriate function object.
void joinBodies(double springConstant, SpringBody body1, unsigned int body1Node, SpringBody body2, unsigned int body2Node);

void joinBodies(double springConstant, double equilibriumDist, SpringBody body1, unsigned int body1Node, SpringBody body2, unsigned int body2Node);

#endif