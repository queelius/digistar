

struct DelayedUpdate {
    Object *objectRef;
    Vector newVelocity;
};

extern std::list<Object*> world;
extern OrthoCam cam;


void checkCollisions();
void checkCollisions() {
    std::list<DelayedUpdate> u;
    for (auto i = world.begin(); i != world.end(); ++i) {
        const Point &pt = (*i)->getPosition();
        const GLfloat radius = (*i)->getRadius();

        for (auto j = world.begin(); j != world.end(); ++j) {
            if (getDistance(pt, (*j)->getPosition()) <= radius + (*j)->getRadius() && i != j) {
                GLfloat ma = (*i)->mass;
                GLfloat mb = (*j)->mass;

                GLfloat va_x = (*i)->velocity.pt[0];
                GLfloat va_y = (*i)->velocity.pt[1];
                    
                GLfloat vb_x = (*j)->velocity.pt[0];
                GLfloat vb_y = (*j)->velocity.pt[1];

                GLfloat v[] = {
                    (ma * va_x  + mb * vb_x) / (ma + mb),
                    (ma * va_y  + mb * vb_y) / (ma + mb)
                };
                DelayedUpdate tmp = {*i, Vector(v)};
                u.push_back(tmp);
            }
        }
    }

    for (auto i = u.begin(); i != u.end(); ++i)
        i->objectRef->velocity = i->newVelocity;
};
