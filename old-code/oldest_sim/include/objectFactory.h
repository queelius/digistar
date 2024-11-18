#ifndef OBJECT_FACTORY_H
#define OBJECT_FACTORY_H

#include <list>
#include "glut.h"
#include "object.h"
#include "asteroid.h"
#include "gravityWell.h"
#include "timer.h"
#include <random>

Object *makeObject(
    ObjectType type,
    GLfloat objectMass,
    GLfloat objectRadius,
    GLfloat objectColor[3],
    GLfloat objectPosition[2],
    GLfloat objectVelocity[2]);

Object *makeOrbitalObject(
    ObjectType type,
    GLfloat objectMass,
    GLfloat objectRadius,
    GLfloat objectColor[3],
    GLfloat orbitalRadius,
    Object *orbitalTarget,
    GLfloat orbitalAngle = NULL,
    GLfloat orbitalVariance[2] = NULL);

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
    Object *orbitalTarget);

#endif