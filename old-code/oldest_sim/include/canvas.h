#ifndef CANVAS_H
#define CANVAS_H

#include "globals.h"
#include "object.h"
using namespace std;

extern vector<Object*> o;
extern Object *player;

//class OrthoCam {
class Canvas {
public:
    GLfloat pixelWidth, pixelHeight;
    GLfloat width, height;
    GLfloat center_x;
    GLfloat center_y;
    bool fullscreen;
    string title;

    void scale(GLfloat factor) {
        width *= factor;
        height *= factor;   
    };

    void setCenter(GLfloat x, GLfloat y) {
        center_x = x;
        center_y = y;
    };

    void setCenter(const GLfloat pos[]) {
        setCenter(pos[0], pos[1]);
    };

    void getPosition(GLfloat &x, GLfloat &y, GLint pixelX, GLint pixelY) {
        x = center_x + width * pixelX / pixelWidth - 0.5 * width;
        y = center_y + 0.5 * height - (height * pixelY / pixelHeight);
    };

    void getPixelPosition(GLfloat &pixelX, GLfloat &pixelY, GLfloat x, GLfloat y) {
        pixelX = pixelWidth * x / width - pixelWidth * center_x / width + 0.5 * pixelWidth;
        pixelY = -(pixelHeight / height) * (y - center_y - 0.5 * height);
    };

    void getPosition(GLfloat pos[2], const GLint pixelPos[2]) {
        getPosition(pos[0], pos[1], pixelPos[0], pixelPos[1]);
    };

    GLfloat getLeft() {
        return center_x - width / 2.0;
    };

    GLfloat getRight() {
        return center_x + width / 2.0;
    };

    GLfloat getTop() {
        return center_y + height / 2.0;
    };

    GLfloat getBottom() {
        return center_y - height / 2.0;
    };

    void initDisplay() {
        if (fullscreen) {
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
            glutGameModeString("1680x1050:32");
	        if (glutGameModeGet(GLUT_GAME_MODE_POSSIBLE)) 
		        glutEnterGameMode();
        }
        else {
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
            glutInitWindowSize(pixelWidth, pixelHeight);
            glutCreateWindow(title.c_str());
        }
    };

    void renderText(const GLfloat pos[2], const char *s, void *font = GLUT_BITMAP_8_BY_13) {
        glRasterPos2fv(pos);
        for (const char *c = s; *c != '\0'; c++)
            glutBitmapCharacter(font, *c);
    };

    void showInfo(Object *o, glColor color) {
        glPushMatrix();
            glColor3fv(color.toArray());
            const GLfloat *p = o->getPosition();

            GLfloat pos[] = {0.66, -0.9};
            stringstream text;
            string s;

            text.clear();
            text.precision(3);
            text << o->getSpeed() << " m/s";
            getline(text, s);
            pos[1] += 0.0333;
            renderText(pos, s.c_str());

            text.clear();
            text << o->getAngle() * 180 / PI << " deg";
            getline(text, s);
            pos[1] += 0.0333;
            renderText(pos, s.c_str());

            text.clear();
            text << o->getAngularVelocity() * 180 / PI << " deg/s";
            getline(text, s);
            pos[1] += 0.0333;
            renderText(pos, s.c_str());
        glPopMatrix();
    };
};


#endif