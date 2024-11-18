#ifndef GL_COLOR_H
#define GL_COLOR_H

#include <string>
#include "glut.h"

enum ColorName {White, Black, Red, Green, Blue, Yellow, Gold, GoldenRod};

const GLfloat COLOR_BLACK[]      = {0.0,         0.0,         0.0};
const GLfloat COLOR_WHITE[]      = {1.0,         1.0,         1.0};
const GLfloat COLOR_RED[]        = {1.0,         0.0,         0.0};
const GLfloat COLOR_GREEN[]      = {0.0,         1.0,         0.0};
const GLfloat COLOR_BLUE[]       = {0.0,         0.0,         1.0};
const GLfloat COLOR_YELLOW[]     = {1.0,         1.0,         0.0};
const GLfloat COLOR_GOLD[]       = {1.0,         0.843137255, 0.0};
const GLfloat COLOR_GOLDEN_ROD[] = {0.854901961, 0.647058824, 0.125490196};

class glColor {
    friend bool operator==(const glColor &lhs, const glColor &rhs);
    friend bool operator!=(const glColor &lhs, const glColor &rhs);
    friend glColor operator+(const glColor &lhs, const glColor &rhs);
    friend glColor operator+(ColorName lhs, ColorName rhs);
    friend glColor operator-(const glColor &lhs, const glColor &rhs);
public:
    glColor();
    glColor(ColorName name);
    glColor(const glColor &c);
    glColor(GLfloat r, GLfloat g, GLfloat b);
    glColor(const GLfloat rgb[3]);

    void setColor(ColorName name);
    void setColor(GLfloat red, GLfloat green, GLfloat blue);
    void setColor(const GLfloat rgb[3]);
    bool setBrightness(GLfloat level);

    void resetBrightness();

    bool incrRed(GLfloat incr = 0.01);
    bool incrBlue(GLfloat incr = 0.01);
    bool incrGreen(GLfloat incr = 0.01);
    bool incrBrightness(GLfloat incr = 0.01);

    const GLfloat *toArray() const;
    const GLfloat *baseColorToArray() const;

    glColor &operator=(const glColor &c);
    GLfloat &operator[](int index);
    glColor &operator++();
    glColor &operator++(int);
    
    GLfloat getBrightness();
    std::string getColor() const;

protected:
    GLfloat baseRGB[3];
    GLfloat adjustedRGB[3];
    GLfloat level;

    bool incHelper(GLfloat &v, GLfloat inc);
};

bool operator==(const glColor &lhs, const glColor &rhs);
bool operator!=(const glColor &lhs, const glColor &rhs);

#endif