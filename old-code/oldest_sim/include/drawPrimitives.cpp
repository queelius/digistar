#include "drawPrimitives.h"

void drawCircle(const GLfloat pos[2], GLfloat radius, GLfloat slices, const glColor &color) {
    glPushMatrix();
        glColor3fv(color.toArray());
        glTranslatef(pos[0], pos[1], 0);
        glBegin(GL_TRIANGLE_FAN);
            glVertex3f(0, 0, 0);
            GLfloat theta = 2*PI/slices;
            for (unsigned i = 0; i <= slices; ++i)
                glVertex3f(radius*cos(i*theta), radius*sin(i*theta), 0);
        glEnd();
    glPopMatrix();
}

void drawTriangle(const GLfloat pos[2], GLfloat base, GLfloat height, GLfloat angle, const glColor &color) {
    glPushMatrix();
        glColor3fv(color.toArray());
        glTranslatef(pos[0], pos[1], 0);
        glRotatef((angle * 180 / PI) - 90, 0, 0, 1);
        glBegin(GL_POLYGON);
            glVertex2f(-base/2.0, -height/2.0);
            glVertex2f(base/2.0, -height/2.0);
            glVertex2f(0, height/2.0);
        glEnd();
    glPopMatrix();
}

void drawRectangle(const GLfloat pos[2], GLfloat width, GLfloat height, GLfloat angle, const glColor &color) {
    glPushMatrix();
        glColor3fv(color.toArray());
        glTranslatef(pos[0], pos[1], 0);
        glRotatef((angle * 180 / PI) - 90, 0, 0, 1);
        glBegin(GL_POLYGON);
            glVertex2f(-width/2.0, -height/2.0);
            glVertex2f(width/2.0, -height/2.0);
            glVertex2f(width/2.0, height/2.0);
            glVertex2f(-width/2.0, height/2.0);
        glEnd();
    glPopMatrix();
}