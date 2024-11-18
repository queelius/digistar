#ifndef TYPES_H
#define TYPES_H

class Types
{
public:
    static const unsigned OBJECT                = 0;
    static const unsigned PARTICLE              = 1;
    static const unsigned POINT_PARTICLE        = 2;
    static const unsigned COMPOSITE_PARTICLE    = 4;
    static const unsigned FORCE                 = 8;
    static const unsigned IDEAL_SPRING_FORCE    = 16;
    static const unsigned REPULSIVE_FORCE       = 32;
    static const unsigned UNIFORM_FORCE         = 64;
    static const unsigned TICKER                = 128;
    static const unsigned GRAVITY_FORCE         = 256;
    static const unsigned GAS                   = 512;
};

#endif