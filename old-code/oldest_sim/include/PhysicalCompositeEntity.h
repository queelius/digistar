#ifndef PHYSICAL_COMPOSITE_ENTITY_H
#define PHYSICAL_COMPOSITE_ENTITY_H

// note: a composite entity consists of 2 or more entities

// do not add a mass attribute and such
// since a composite object depends upon the
// mass of its constituent parts (e.g., point masses)

class PhysicalCompositeEntity: public PhysicalEntity
{
};

#endif