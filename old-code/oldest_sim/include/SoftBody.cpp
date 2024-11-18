#include "SoftBody.h"
using namespace std;

extern vector<Spring> springs;
extern vector<Node> nodes;

// #######################
// # Node implementation #
// #######################

Node::Node(const double s[2], const double mass, bool fixed)
{
    this->fixed = fixed;

    this->s[0] = s[0];
    this->s[1] = s[1];

    this->v[0] = 0.0;
    this->v[1] = 0.0;

    this->force[0] = 0.0;
    this->force[1] = 0.0;

    this->mass = mass;
    //this->charge = 0;
}

Node::Node(const double s[2], const double v[2], const double mass, bool fixed)
{
    this->fixed = fixed;

    this->s[0] = s[0];
    this->s[1] = s[1];

    this->v[0] = v[0];
    this->v[1] = v[1];

    this->force[0] = 0.0;
    this->force[1] = 0.0;

    this->mass = mass;
    //this->charge = 0;
}

void Node::dump() const
{
    cout << "Position: " << s[0] << ", " << s[1] << endl;
    cout << "Velocity: " << v[0] << ", " << v[1] << endl;
    cout << "Force: " << force[0] << ", " << force[1] << endl;
    cout << "Mass: " << mass << endl;
    cout << "Fixed: " << fixed << endl;
    system("pause");
}

void Node::addForce(const double force[2])
{
    this->force[0] += force[0];
    this->force[1] += force[1];
}

void Node::tick()
{
    if (!fixed)
    {
        // cout << "(" << force[0] << ", " << force[1] << ")" << endl;

        v[0] += force[0] / mass * TICK_DURATION;
        v[1] += force[1] / mass * TICK_DURATION;

        s[0] += v[0] * TICK_DURATION;
        s[1] += v[1] * TICK_DURATION;

        force[0] = force[1] = 0.0; // clear net force after using
    }
}

double Node::getMass() const
{
    return mass;
}

const double *Node::getPosition() const
{
    return s;
}

const double *Node::getCenterOfMass() const
{
    return s;
}

const double *Node::getVelocityOfCenterOfMass() const
{
    return v;
}

double Node::getVelocityOfCenterOfMassMagnitude() const
{
    return sqrt(v[0] * v[0] + v[1] * v[1]);
}

// ###############################
// # Ideal Spring implementation #
// ###############################

Spring::Spring(double springConstant, double equilibriumDist, Node *end1, Node *end2)
{
    this->springConstant = springConstant;
    this->equilibriumDist = equilibriumDist;
    this->end1 = end1;
    this->end2 = end2;
}

void Spring::tick()
{
    if (this->end1 == NULL || this->end2 == NULL)
        return;

    const double *p1 = end1->getCenterOfMass();
    const double *p2 = end2->getCenterOfMass();
    const double *v1 = end1->getVelocityOfCenterOfMass();
    const double *v2 = end2->getVelocityOfCenterOfMass();

    double rightEndPos[2], leftEndPos[2];
    double rightEndVel[2], leftEndVel[2];

    bool end1IsOnRight;
    if (p1[0] < p2[0])
    {
        end1IsOnRight = false;
        leftEndPos[0] = p1[0];
        leftEndPos[1] = p1[1];
        rightEndPos[0] = p2[0];
        rightEndPos[1] = p2[1];

        leftEndVel[0] = v1[0];
        leftEndVel[1] = v1[1];
        rightEndVel[0] = v2[0];
        rightEndVel[1] = v2[1];
    }
    else
    {
        end1IsOnRight = true;
        leftEndPos[0] = p2[0];
        leftEndPos[1] = p2[1];
        rightEndPos[0] = p1[0];
        rightEndPos[1] = p1[1];

        leftEndVel[0] = v2[0];
        leftEndVel[1] = v2[1];
        rightEndVel[0] = v1[0];
        rightEndVel[1] = v1[1];
    }

    double Xdisplacement = rightEndPos[0] - leftEndPos[0]; // x
    double Ydisplacement = rightEndPos[1] - leftEndPos[1]; // y
    double displacement = sqrt(Xdisplacement*Xdisplacement + Ydisplacement*Ydisplacement); // r
    double Fspring = springConstant * equilibriumDist - springConstant * displacement;
    double vx = rightEndVel[0] - leftEndVel[0];
    double vy = rightEndVel[1] - leftEndVel[1];
    double dragCoefficient = 10.0;

    double dragX = dragCoefficient * vx;
    double dragY = dragCoefficient * vy;

    double forceOnRight[] =
    {
         Fspring * (Xdisplacement / displacement),
         Fspring * (Ydisplacement / displacement)
    };

    forceOnRight[0] -= dragX;
    forceOnRight[1] -= dragY;

    double forceOnLeft[] =
    {
        -forceOnRight[0],
        -forceOnRight[1]
    };

    if (end1IsOnRight)
    {
        end1->addForce(forceOnRight);
        end2->addForce(forceOnLeft);
    }
    else
    {
        end1->addForce(forceOnLeft);
        end2->addForce(forceOnRight);
    }

    double forceMax = 1E+5;
    double forceCheck = sqrt(forceOnRight[0] * forceOnRight[0] + forceOnRight[1] * forceOnRight[1]);

    if (forceCheck > forceMax)
    {
        this->end1 = this->end2 = NULL;
    }
}


// #############################
// # SpringBody implementation #
// #############################

SpringBody::SpringBody()
{
    massMoment[0] = massMoment[1] = totalMass = 0;
    inodes.reserve(100);
    isprings.reserve(100);
}

void SpringBody::tick()
{
    // find out which nodes are still interconnected
    // elastically; decompose into disjoint sets where
    // the elements are nodes (point masses) and each
    // set represents a separate composite body
}

// equilibrium distance for two nodes defaults to their distance apart that existed
// during their initial connection
void SpringBody::joinNodes(double springConstant, unsigned int endNode1, unsigned int endNode2)
{
    if (endNode1 < 0 || endNode1 >= inodes.size() ||
        endNode2 < 0 || endNode2 >= inodes.size() ||
        endNode1 == endNode2)
    {
        throw exception("Invalid node(s) specified.");
    }
    const double *p1 = inodes[endNode1]->getPosition();
    const double *p2 = inodes[endNode2]->getPosition();
    double dist = sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]));

    springs.push_back(Spring(springConstant, dist, inodes[endNode1], inodes[endNode2]));
    isprings.push_back(&springs[springs.size() - 1]);
}

// equilibrium distance for two nodes defaults to their distance apart that existed
// during their initial connection
void SpringBody::joinNodes(double springConstant, double equilibriumDist, unsigned int endNode1, unsigned int endNode2)
{
    if (endNode1 < 0 || endNode1 >= nodes.size() ||
        endNode2 < 0 || endNode2 >= nodes.size())
    {
        throw exception("Invalid node(s) specified.");
    }

    springs.push_back(Spring(springConstant, equilibriumDist, inodes[endNode1], inodes[endNode2]));
    isprings.push_back(&springs[springs.size() - 1]);
}

void SpringBody::addNode(double nodeMass, double x, double y, bool fixed)
{
    double s[] = {x, y};
    addNode(nodeMass, s, fixed);
}

// investigate ability to have negative mass (like a negative charge)
void SpringBody::addNode(double nodeMass, const double nodePos[2], bool fixed)
{
    if (nodeMass < 0.0)
        throw exception("Cannot have negative mass.");

    totalMass += nodeMass;
    massMoment[0] += nodeMass * nodePos[0];
    massMoment[1] += nodeMass * nodePos[1];

    nodes.push_back(Node(nodePos, nodeMass, fixed));
    inodes.push_back(&nodes[nodes.size() - 1]);
}

// put in a bit of checking so that if not needed,
// don't recompute velocity center of mass every time you
// call the method
const double *SpringBody::getVelocityOfCenterOfMass()
{
    if (totalMass == 0.0)
        throw exception("No center of mass");

    vcm[0] = vcm[1] = 0.0;
    for (unsigned int i = 0; i < inodes.size(); ++i)
    {
        const double *v = inodes[i]->getVelocityOfCenterOfMass();
        vcm[0] += v[0] * inodes[i]->getMass();
        vcm[1] += v[1] * inodes[i]->getMass();
    }

    vcm[0] = vcm[0] / getMass();
    vcm[1] = vcm[1] / getMass();

    return vcm;
}

double SpringBody::getVelocityOfCenterOfMassMagnitude()
{
    const double *vcm = getVelocityOfCenterOfMass();
    return sqrt(vcm[0] * vcm[0] + vcm[1] * vcm[1]);
}

const double *SpringBody::getCenterOfMass()
{
    if (totalMass == 0.0)
        throw exception("No center of mass");

    centerOfMass[0] = massMoment[0] / totalMass;
    centerOfMass[1] = massMoment[1] / totalMass;

    return centerOfMass;
}

// a = v_i^2 / r_i
// use this to force rotation about a fixed axis;
// create a radius force (pointing towards the center of mass) that applies this to every particle in the composite body

double SpringBody::getKineticEnergy()
{
    double KE = 0.0;
    for (unsigned int i = 0; i < inodes.size(); ++i)
    {
        double vcm = inodes[i]->getVelocityOfCenterOfMassMagnitude();
        KE += inodes[i]->getMass() * vcm * vcm;
    }
    return 0.5 * KE;
}

double SpringBody::getAngularKineticEnergy()
{
    return getKineticEnergy() - getLinearKineticEnergy();
}

double SpringBody::getLinearKineticEnergy()
{
    double vcm = getVelocityOfCenterOfMassMagnitude();
    return 0.5 * getMass() *  vcm * vcm;
}

const double *SpringBody::getMomentum()
{
    momentum[0] = momentum[1] = 0.0;
    for (unsigned int i = 0; i < inodes.size(); ++i)
    {
        const double *vcm = inodes[i]->getVelocityOfCenterOfMass();
        momentum[0] += inodes[i]->getMass() * vcm[0];
        momentum[1] += inodes[i]->getMass() * vcm[1];
    }
    return momentum;
}

// incorrect
double SpringBody::getLinearMomentum()
{
    return getMass() * getVelocityOfCenterOfMassMagnitude();
}

//double getAngularMomentum()
//{
//};

// give impulse to bodies (that make up this object) in a direction
// perpendicular to axis. add impulse to each body such that
// change in w, dw, is the same for every body with respect to
// specified axis?
void SpringBody::addAngularImpulse(double force[2], double duration, double axis[2])
{
};

// give impulse to bodies (that make up this object) in a direction
// perpendicular to center of mass (as an axis of rotation)?
// or maybe not.
void SpringBody::addAngularImpulse(double force[2], double duration)
{
}

// add impulse to bodies (that make up this object) such that
// every body gets same amount of change in momentum?
// or maybe not.
void SpringBody::addImpulse(double force[2], double duration)
{
}

// // so, can add a radial force? -1 time duration for infinity
// void addContinuousForce(function object for force, interval of time)
// void addContinuousForce(function object for force, start time, end time)

void SpringBody::addConstantForce(double force[2], double duration)
{
}

void SpringBody::addConstantForce(double force[2], double startTime, double endTime)
{
}

//void addForce(function object whose path you want the force to cause the object to move on)

// actually, this is addImpulse; specify time of force contact also
// a few notes:
//  1) gravity provides the same impulse to every body in this composite
//     body, so it'll be convenient to have the same impulse that can be
//     applied to each body in the composition.
//  2) it will also be nice to have a way to add a force
void SpringBody::addForce(double force[2])
{
    for (unsigned int i = 0; i < inodes.size(); ++i)
        inodes[i]->addForce(force);
}

unsigned int SpringBody::getNodeCount()
{
    return (unsigned int)inodes.size();
}

double SpringBody::getMass()
{
    return totalMass;
}

// double getCharge() { return totalCharge; };

// ####################
// # Helper functions #
// ####################

//// make this a join by specified force, e.g., an elastic connection, gravity connection, etc.
//// pass function objects to do all of these calculations, then create helper functions that
//// call these general connectors with the appropriate function object.
void joinBodies(double springConstant, SpringBody body1, unsigned int body1Node, SpringBody body2, unsigned int body2Node)
{
    const double *p1 = body1.inodes[body1Node]->getPosition();
    const double *p2 = body2.inodes[body2Node]->getPosition();
    double dist = sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]));
    springs.push_back(Spring(springConstant, dist, body1.inodes[body1Node], body2.inodes[body2Node]));
}

void joinBodies(double springConstant, double equilibriumDist, SpringBody body1, unsigned int body1Node, SpringBody body2, unsigned int body2Node)
{
    springs.push_back(Spring(springConstant, equilibriumDist, body1.inodes[body1Node], body2.inodes[body2Node]));
}