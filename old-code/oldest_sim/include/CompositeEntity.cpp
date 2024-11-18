#include "CompositeEntity.h"

CompositeEntity::CompositeEntity()
{
    totalMass = 0;
    totalCharge = 0;
}

void CompositeEntity::tick()
{
    // find out which nodes are still interconnected;
    // decompose into disjoint sets where the elements
    // are node entities and each set represents a
    // separate composite body
}