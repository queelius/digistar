#include "DrawShapes.h"

namespace PhysicsTest {

void DrawCircle(const Point &pt, GLfloat radius, const GLfloat color[2], GLuint slices) {
    if (slices == 0) slices = radius;

    glPushMatrix();
        glColor3fv(color);
        glTranslatef(pt.getX(), pt.getY(), 0.0);
        glBegin(GL_TRIANGLE_FAN);
            glVertex3f(0.0, 0.0, 0.0);
            GLfloat theta = 2 * PI / slices;
            for (GLuint i = 0; i <= slices; ++i)
                glVertex3f(radius * cos(i * theta), radius * sin(i * theta), 0.0);
        glEnd();
    glPopMatrix();
}

void DrawTriangle(const Point &pt, GLfloat base, GLfloat height, GLfloat angle, const GLfloat color[2]) {
    glPushMatrix();
        glColor3fv(color);
        glTranslatef(pt.getX(), pt.getY(), 0.0);
        glRotatef(angle * 180 / PI - 90.0, 0.0, 0.0, 1.0);
        glBegin(GL_POLYGON);
            glVertex2f(-base/2.0, -height/2.0);
            glVertex2f(base/2.0, -height/2.0);
            glVertex2f(0, height/2.0);
        glEnd();
    glPopMatrix();
}

void DrawRectangle(const Point &pt, GLfloat width, GLfloat height, GLfloat angle, const GLfloat color[2]) {
    glPushMatrix();
        glColor3fv(color);
        glTranslatef(pt.getX(), pt.getY(), 0.0);
        glRotatef(angle * 180 / PI - 90, 0.0, 0.0, 1.0);
        glBegin(GL_POLYGON);
            glVertex2f(-width / 2.0, -height / 2.0);
            glVertex2f(width / 2.0, -height / 2.0);
            glVertex2f(width / 2.0, height / 2.0);
            glVertex2f(-width / 2.0, height / 2.0);
        glEnd();
    glPopMatrix();
}

}