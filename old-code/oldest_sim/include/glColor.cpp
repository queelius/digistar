#include "glColor.h"

glColor operator+(const glColor &lhs, const glColor &rhs) {
    const GLfloat *l = lhs.toArray();
    const GLfloat *r = rhs.toArray();
    GLfloat rgb[3] = { l[0] + r[0], l[1] + r[1], l[2] + r[2] };

    GLfloat max = rgb[0];
    for (unsigned i = 1; i < 3; ++i)
        if (rgb[i] > max)
            max = rgb[i];

    if (max > 1) {
        rgb[0] /= max;
        rgb[1] /= max;
        rgb[2] /= max;
    }

    return glColor(rgb);
}

glColor operator-(const glColor &lhs, const glColor &rhs) {
    const GLfloat *l = lhs.toArray();
    const GLfloat *r = rhs.toArray();

    GLfloat rgb[3] = { l[0] - r[0],  l[1] - r[1], l[2] - r[2] };

    if (rgb[0] < 0) rgb[0] = 0;
    if (rgb[1] < 0) rgb[1] = 0;
    if (rgb[2] < 0) rgb[2] = 0;

    return glColor(rgb);
}

glColor operator+(const ColorName lhs, const ColorName rhs) {
    return glColor(lhs) + glColor(rhs);
}

bool operator==(const glColor &lhs, const glColor &rhs) {
    return lhs.adjustedRGB[0] == rhs.adjustedRGB[0] &&
           lhs.adjustedRGB[1] == rhs.adjustedRGB[1] &&
           lhs.adjustedRGB[2] == rhs.adjustedRGB[2];
}

bool operator!=(const glColor &lhs, const glColor &rhs) {
    return !(lhs == rhs);
}

void glColor::resetBrightness() {
    setBrightness(0.5);
}

glColor::glColor() {
    // do nothing
}

glColor::glColor(const glColor &c) {
    setColor(c.baseRGB);
    setBrightness(c.level);
}

glColor::glColor(ColorName name) {
    setColor(name);
    setBrightness(0.5);
}

glColor::glColor(const GLfloat rgb[3]) {
    setColor(rgb);
    setBrightness(0.5);
}

glColor::glColor(GLfloat red, GLfloat green, GLfloat blue) {
    setColor(red, green, blue);
    setBrightness(0.5);
}

inline const GLfloat *glColor::toArray() const {
    return adjustedRGB;
}

bool glColor::incrRed(GLfloat incr) {
    return incHelper(baseRGB[0], incr);
}

bool glColor::incrGreen(GLfloat incr) {
    return incHelper(baseRGB[1], incr);
}

bool glColor::incrBlue(GLfloat incr) {
    return incHelper(baseRGB[2], incr);
}

void glColor::setColor(const GLfloat rgb[3]) {
    memcpy(this->baseRGB, rgb, 12);
    setBrightness(level);
}

void glColor::setColor(ColorName name) {
    switch (name) {
        case Black:  setColor(COLOR_BLACK); break;
        case White:  setColor(COLOR_WHITE); break;
        case Red:    setColor(COLOR_RED); break;
        case Green:  setColor(COLOR_GREEN); break;
        case Blue:   setColor(COLOR_BLUE); break;
        case Yellow: setColor(COLOR_YELLOW); break;
        case Gold:   setColor(COLOR_GOLD); break;
        case GoldenRod: setColor(COLOR_GOLDEN_ROD); break;
    }
}

void glColor::setColor(GLfloat red, GLfloat green, GLfloat blue) {
    baseRGB[0] = red;
    baseRGB[1] = green;
    baseRGB[2] = blue;

    setBrightness(level);
}

bool glColor::incHelper(GLfloat &v, GLfloat incr) {
    if (incr > 0 && v >= 1 || incr < 0 && v <= 0)
        return false;

    v += incr;
    if (v > 1) {
        for (unsigned i = 0; i < 3; ++i) {
            baseRGB[i] /= v;
        }
    }
    else if (v < 0) v = 0;

    setBrightness(level);

    return true;
}

bool glColor::incrBrightness(GLfloat incr) {
    return setBrightness(level + incr);
}

GLfloat glColor::getBrightness() {
    return (adjustedRGB[0] + adjustedRGB[1] + adjustedRGB[2]) / 3.0;
}

bool glColor::setBrightness(GLfloat level) {
    if (level > 1 || level < 0)
        return false;

    this->level = level;
    for (unsigned i = 0; i < 3; ++i) {
        adjustedRGB[i] = 2*(level - 0.5) + baseRGB[i];

        if (adjustedRGB[i] < 0)      adjustedRGB[i] = 0;
        else if (adjustedRGB[i] > 1) adjustedRGB[i] = 1;
    }

    return true;
}

glColor &glColor::operator=(const glColor &c) {
    memcpy(baseRGB, c.baseRGB, 12);
    memcpy(adjustedRGB, c.adjustedRGB, 12);
    level = c.level;

    return *this;
}