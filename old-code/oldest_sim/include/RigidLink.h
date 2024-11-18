#ifndef RIGID_LINK
#define RIGID_LINK

#include "Force.h"

class RigidLink: public Force
{
public:
            RigidLink(Entity *ent1, Entity *ent2);
    void    tick();

protected:
    double  distance;
    Entity  *ent1;
    Entity  *ent2;
};

#endif

RigidLink::RigidLink(Entity *ent1, Entity *ent2)
{
    this->ent1 = ent1;
    this->ent2 = ent2;

    // calculate distance here (ent2->position - ent1->position).getMag() ???
}

void RigidLink::tick()
{
    if (this->ent1 == NULL || this->ent2 == NULL)
    {
        // delete ?
        return;
    }

    //// have to do this calculation after "net" force has been calculated
    //// to make force towards link equal the "net" force opposite T (vector
    //// pointing from point under consideration to point it is connected to
    //// rigidly
    //
    //double dx = end1->s[0] - end2->s[0];
    //double dy = end1->s[1] - end2->s[1];

    //double theta = atan2(dy, dx);

    //double Fnet = sqrt(end2->force[0] * end2->force[0] + end2->force[1] * end2->force[1]);
    ////double Fnet = sqrt(end1->force[0] * end1->force[0] + end1->force[1] * end1->force[1]);

    //double Tnet = 0;

    ////cout << "theta: " << theta << " (degees -> " << theta * 180 / 3.14 << ")" << endl;
    ////cout << "tension: " << Tnet << endl;


    ////system("pause");

    //double T[2] = { Tnet * sin(theta), Tnet * cos(theta) };
    ////end1->addForce(T);

    ////double T2[2] = { -Tnet * sin(theta), -Tnet * cos(theta) };
    //end2->addForce(T);
}
