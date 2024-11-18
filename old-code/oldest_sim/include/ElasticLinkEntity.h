#ifndef ELASTIC_LINK
#define ELASTIC_LINK

#include "Link.h"

// have "ideal spring" class inherit from ElasticLink, providing it
// with appropriate parameters by default.

// allow the force applied to the end point masses of an elastic connection
// be the result of a function object passed to it

class ElasticLink: public Link
{
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

    ElasticLink(double stiffness, double equilibrium, double elasticLimit, PhysicalObject *obj1, PhysicalObject *obj2);

    void tick();

protected:
    double stiffness;
    double equilibrium;
    Object *obj1;
    Object *obj2;
};

#endif