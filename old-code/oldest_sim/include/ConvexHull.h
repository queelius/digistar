#ifndef CONVEX_HULL_H
#define CONVEX_HULL_H

#include <stack>
#include <deque>
#include <functional>
#include "Entity.h"

class EntityStack: public std::stack<Entity*>
{
public:
    const Entity *underTop() const
    {
        return (c.size() > 2 ? *(c.end() - 2) : NULL);
    }
};

// polar sort function object; pass to sorting routines when sorting objects
// inheriting from superclass Entity to sort by polar coordinate (by asecnding ccw angle)
struct PolarSortFunctor: public std::binary_function<Entity*, Entity*, bool>
{
public:
    PolarSortFunctor();
    PolarSortFunctor(double polarX, double polarY);

    // set polar coordinate to be x, y
    void setPole(double x, double y);

    bool operator() (const Entity *ent1, const Entity *ent2) const;

protected:
    double polarX, polarY;
};

void traceHull(const SpringBody &body);
bool isLeftTurn(const Entity *a, const Entity *b, const Entity *c);

#endif

/*
#define DEBUG 0

#include <cmath>
#include <vector>
#include <stack>
#include <deque>
#include <functional>
#include "stdlib.h"
#include "glut.h"
#include "Random.h"
#include "SoftBody.h"

int findLowestBodyInComposite(const SpringBody &body);
void traceHull(SpringBody body);
bool isLeftTurn(const Node *a, const Node *b, const Node *c);

// change from type Node* to type Entity*?
class Stack: public std::stack<Node*> {
public:
    const Node *underTop() const
    {
        if (DEBUG)
        {
            cout << "UnderTop" << endl;
            cout << "Stack size: " << c.size() << endl;
        }

        return (c.size() > 2 ? *(c.end() - 2) : NULL);
    }
};

// polar sort function object; pass to sorting routines when sorting objects
// inheriting from superclass VisibleEntity to sort by polar coordinate (by asecnding ccw angle)
struct PolarSortFunctor: public std::binary_function<Node*, Node*, bool> {
public:
    PolarSortFunctor()
    {
        if (DEBUG)
            cout << "Empty constructor" << endl;
    };

    PolarSortFunctor(GLfloat polarX, GLfloat polarY)
    {
        if (DEBUG)
            cout << "Non-empty constructor" << endl;

        setPole(polarX, polarY);
    };

    // set polar coordinate to be x, y
    void setPole(GLfloat x, GLfloat y)
    {
        if (DEBUG)
            cout << "Set pole: " << x << ", " << y << endl;

        this->polarX = x;
        this->polarY = y;
    };

    bool operator() (const Node *arg1, const Node *arg2) const
    {
        const double *p1 = arg1->getPosition();
        const double *p2 = arg2->getPosition();

        double theta1 = atan2(p1[1] - polarY, p1[0] - polarX);
        double theta2 = atan2(p2[1] - polarY, p2[0] - polarX);

        if (DEBUG)
        {
            cout << "Node 1: " << arg1 << " (" << p1[0] << p1[1] << ")" << endl;
            cout << "Node 2: " << arg2 << " (" << p2[0] << p2[1] << ")" << endl;
            cout << "Theta 1: " << theta1 << endl;
            cout << "Theta 2: " << theta2 << endl;
            cout << "---" << endl;
        }

        return ((theta1 < theta2) || (theta1 == theta2 && p1[0] < p2[0]));
    };

private:
    GLfloat polarX, polarY;
};

int findLowestBodyInComposite(const SpringBody &body)
{
    if (body.inodes.size() == 0)
        return -1;
    else {
        unsigned int min = 0;
        const double *minYPos = body.inodes[min]->getPosition();
        double minY = minYPos[1];
        for (unsigned int i = 1; i < body.inodes.size(); ++i) {
            const double *pos = body.inodes[i]->getPosition();

            // also, select middle-most body with x if two or more bodies are at same height
            if (pos[1] < minY || body.inodes[min]->getPosition()[1] == pos[1] && pos[0] < body.inodes[min]->getPosition()[0])
            {
                min = i;
                minY = pos[1];
            }
        }
        return min;
    }
}

void traceHull(SpringBody body) {
    if (body.inodes.size() < 3) // actually, draw a line between them if size == 2?
        return;

    glColor3f(1, 1, 0);
    for (unsigned int i = 0; i < body.inodes.size(); ++i)
    {
        glBegin(GL_TRIANGLE_FAN);
            const double *p = body.inodes[i]->getPosition();
            glVertex2d(p[0], p[1]);
            for (unsigned int j = 0; j < body.inodes.size(); ++j)
            {
                if (j == i)
                    continue;

                const double *p2 = body.inodes[j]->getPosition();
                glVertex2d(p2[0], p2[1]);
            }
        glEnd();
    }

    return;


    int lowest = findLowestBodyInComposite(body);
    if (lowest != -1 && lowest != 0)
        swap(body.inodes[0], body.inodes[lowest]);

    const double *p = body.inodes[0]->getPosition();

    PolarSortFunctor polarSort;
    polarSort.setPole(p[0], p[1]);
    sort(body.inodes.begin() + 1, body.inodes.end(), polarSort);

    Stack S;
    for (size_t i = 0; i < 3; ++i)
    {
        if (DEBUG) cout << "Pushing in #" << i + 1 << ": " << body.inodes[i] << endl;
        S.push(body.inodes[i]);
    }

    for (size_t i = 3; i < body.inodes.size(); ++i)
    {
        while (true)
        {
            const Node *a = S.underTop();
            const Node *b = S.top();
            const Node *c = body.inodes[i];

            if (DEBUG) { cout << "Node a: " << a << "\nNode b: " << b << "\nNode c: " << c << endl; }

            if (!isLeftTurn(a, b, c))
            {
                if (DEBUG) cout << "Found a non-left turn!" << endl;
                break;
            }
            
            S.pop();
        }
        if (DEBUG) cout << "Pushing in #" << i + 1 << " of " << body.inodes.size() << endl;
        S.push(body.inodes[i]);
    }

    // color from body specified color
    // types of color specs: functor (e.g., stresses change colors, color changes by position,
    // etc.
    glBegin(GL_TRIANGLE_FAN);
        glColor3f(1, 1, 0);
        int n = 0;
        while (!S.empty()) {
            const double *p = S.top()->getPosition();

            if (DEBUG) cout << "Node " << n++ << " (" << p[0] << ", " << p[1] << ")" << endl;

            glVertex2d(p[0], p[1]);
            S.pop();
        }
    glEnd();
}

bool isLeftTurn(const Node *a, const Node *b, const Node *c)
{
    if (DEBUG) { cout << "isLeftTurn: Node a: " << a << "\nNode b: " << b << "\nNode c: " << c << endl; }

    if (a == NULL || b == NULL || c == NULL)
    {
        if (DEBUG)
        {
            if (a == NULL) cout << "isLeftTurn Node A is null" << endl;
            if (b == NULL) cout << "isLeftTurn Node B is null" << endl;
            if (c == NULL) cout << "isLeftTurn Node C is null" << endl;
        }
        return false;
    }

    const double *ap = a->getPosition();
    const double *bp = b->getPosition();
    const double *cp = c->getPosition();

    GLfloat val = (ap[0] - bp[0]) * (cp[1] - bp[1]) -
                  (cp[0] - bp[0]) * (ap[1] - bp[1]);
    return (val < 0);
}
*/