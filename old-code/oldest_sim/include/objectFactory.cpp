#include "objectFactory.h"

std::list<Object*> makeOrbitalObjectSet(
    ObjectType typeDist[],
    GLuint population,
    GLuint samples,
    GLfloat objectMassMean,
    GLfloat objectMassVariance,
    GLfloat objectColor[3],
    GLfloat orbitalRadiusMean,
    GLfloat orbitalRadiusVariance,
    GLfloat orbitalVariance,
    Object *orbitalTarget) {

    std::list<Object*> objSet;
    std::mt19937 eng;
    eng.seed(time(0));

    std::normal_distribution<double> stdNormal(0, 1);
    std::uniform_real<double> uniform(0, 1);
    std::uniform_int<int> uniformInt(0, population - 1);

    for (GLuint i = 0; i < samples; ++i) {
        ObjectType type = typeDist[uniformInt(eng)];
        GLfloat objectMass = stdNormal(eng) * objectMassVariance + objectMassMean;
        GLfloat orbitalRadius = stdNormal(eng) * orbitalRadiusVariance + orbitalRadiusMean;
        GLfloat orbitalVarianceMag = stdNormal(eng) * orbitalVariance;
        GLfloat orbitalVarianceDir = uniform(eng) * 2 * PI;
        GLfloat orbitalVariance[] = {orbitalVarianceMag * cos(orbitalVarianceDir),
            orbitalVarianceMag * sin(orbitalVarianceDir)};
        objSet.push_back(makeOrbitalObject(type, objectMass, objectMass/25, objectColor, orbitalRadius, orbitalTarget, uniform(eng) * 2 * PI, orbitalVariance));
    }

    return objSet;
}

Object *makeOrbitalObject(
    ObjectType type,
    GLfloat objectMass,
    GLfloat objectRadius,
    GLfloat objectColor[3],
    GLfloat orbitalRadius,
    Object *orbitalTarget,
    GLfloat orbitalAngle,
    GLfloat orbitalVariance[2]) {

    if (orbitalAngle == NULL) {
        std::mt19937 eng;
        eng.seed(time(0));
        std::uniform_real<double> uniform(0, 1);
        orbitalAngle = uniform(eng) * 2 * PI;
    }

    const GLfloat *pos = orbitalTarget->getPosition();
    GLfloat objectPosition[] = {pos[0] + orbitalRadius * cos(orbitalAngle),
                                pos[1] + orbitalRadius * sin(orbitalAngle)};

    GLfloat orbitalVelocity = sqrt(GRAVITY_CONSTANT * orbitalTarget->getMass() / orbitalRadius);

    const GLfloat *vel = orbitalTarget->getVelocity();
    GLfloat objectVelocity[] = {vel[0] - orbitalVelocity * sin(orbitalAngle + orbitalVariance[0]),
                                vel[1] + orbitalVelocity * cos(orbitalAngle + orbitalVariance[1])};

    return makeObject(type, objectMass, objectRadius, objectColor, objectPosition, objectVelocity);
}

Object *makeObject(
    ObjectType type,
    GLfloat objectMass,
    GLfloat objectRadius,
    GLfloat objectColor[3],
    GLfloat objectPosition[2],
    GLfloat objectVelocity[2]) {

    switch (type) {
        case AsteroidType:    return new Asteroid(objectPosition, 0, objectMass, objectColor, objectVelocity, NULL, objectRadius);
        case GravityWellType: return new GravityWell(objectPosition, objectMass, objectVelocity, false, objectRadius);
    }
}

