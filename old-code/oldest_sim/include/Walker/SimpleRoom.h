////////////////////////////////////////////////
// SimpleRoom.h - SimpleRoom Class Definition //
////////////////////////////////////////////////

#ifndef SIMPLE_ROOM
#define SIMPLE_ROOM

#include "glut.h"
#include "Globals.h"

class SimpleRoom {
    SimpleRoom(GLfloat width, GLfloat depth, GLfloat height, GLuint wallTextureID);
    void draw();

    GLfloat coord[8][3];
    GLfloat normal[4][3];

    GLuint wallTextureID;
};

#endif