#include "ParticleSource.h"
#include "Common.h"

SmokeParticle::SmokeParticle(GLfloat initPos[3], GLfloat theta, GLfloat phi, GLfloat initVel, bool jitter) {
    pos[0] = initPos[0];
    pos[1] = initPos[1];
    pos[2] = initPos[2];

    if (jitter) {
        pos[0] += getRand(-0.05, 0.05);
        pos[1] += getRand(-0.05, 0.05);
        pos[2] += getRand(-0.05, 0.05);

        phi   += getRand(-PI/8, PI/8);
        theta += getRand(-PI/8, PI/8);
        initVel *= getRand(DEFAULT_MIN_VARIANCE, DEFAULT_MAX_VARIANCE);
    }

    v[0] = initVel * cos(theta) * cos(phi);
    v[1] = initVel * sin(phi);
    v[2] = initVel * sin(theta) * cos(phi);

    age      = 0;
    lifeSpan = getRandInt(DEFAULT_SMOKE_MIN_LIFE_SPAN, DEFAULT_SMOKE_MAX_LIFE_SPAN);
    size     = DEFAULT_SMOKE_SIZE * getRand(DEFAULT_MIN_VARIANCE, DEFAULT_MAX_VARIANCE);
    gravity  = DEFAULT_GRAVITY * getRand(DEFAULT_MIN_VARIANCE, DEFAULT_MAX_VARIANCE);
};

ParticleType SmokeParticle::getType()   { return SMOKE; };
bool         SmokeParticle::isExpired() { return lifeSpan < age; };

void SmokeParticle::update() {
    ++age;

    a[0] = -v[0] * DEFAULT_DRAG_COEFF;
    a[1] = -v[1] * DEFAULT_DRAG_COEFF - gravity;
    a[2] = -v[2] * DEFAULT_DRAG_COEFF;

    v[0] += a[0];
    v[1] += a[1];
    v[2] += a[2];

    pos[0] += v[0];
    pos[1] += v[1];
    pos[2] += v[2];
};

// draw the smoke particle
void SmokeParticle::draw() {
    glPushMatrix();
        glTranslatef (pos[0], pos[1], pos[2]);
        glScalef(size, size, size);
        glBegin(GL_QUADS);
            glTexCoord2d (0, 0);
            glVertex3f (-1, -1, 0);
            glTexCoord2d (1, 0);
            glVertex3f (1, -1, 0);
            glTexCoord2d (1, 1);
            glVertex3f (1, 1, 0);
            glTexCoord2d (0, 1);
            glVertex3f (-1, 1, 0);
            //glTexCoord2f(0.0, 0.0); glVertex3f(pos[0]-size/2, pos[1]-size/2, pos[2]);
            //glTexCoord2f(1.0, 0.0); glVertex3f(pos[0]+size/2, pos[1]-size/2, pos[2]);
            //glTexCoord2f(1.0, 1.0); glVertex3f(pos[0]+size/2, pos[1]+size/2, pos[2]);
            //glTexCoord2f(0.0, 1.0); glVertex3f(pos[0]-size/2, pos[1]+size/2, pos[2]);
        glEnd();
    glPopMatrix();
}


void ParticleSource::init(const GLfloat pos[], const GLfloat theta, const GLfloat phi, const GLuint texID,
                          const GLfloat emitMag, const GLfloat emitRadius, const GLuint emitRate) {
    this->texID      = texID;    

    this->pos[0]     = pos[0];
    this->pos[1]     = pos[1];
    this->pos[2]     = pos[2];

    this->phi        = phi;
    this->theta      = theta;

    this->emitMag    = emitMag;
    this->emitRadius = emitRadius;
    this->emitRate   = emitRate;
};

void ParticleSource::incrRate() { ++emitRate; };
void ParticleSource::decrRate() { --emitRate; };
void ParticleSource::incrMag()  { ++emitMag; };
void ParticleSource::decrMag()  { --emitMag; };
void ParticleSource::update() {
    for (unsigned int i = 0; i < emitRate; ++i)
        particles.push_back(SmokeParticle(pos, theta, phi, emitMag));
}

void ParticleSource::emit() {
    glPushMatrix();
        glEnable(GL_TEXTURE_2D);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glBindTexture(GL_TEXTURE_2D, texID);
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE_MINUS_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR);

        std::list<SmokeParticle>::iterator i = particles.begin(), tmp;
        while (i != particles.end()) {
            i->update();
            if (i->isExpired()) { tmp = i; ++i; particles.erase(tmp); }
            else                { i->draw(); ++i; }
        }
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_TEXTURE_2D);
    glPopMatrix();
}

void ParticleSource::setPos(GLfloat pos[3]) {
}

void ParticleSource::setPos(GLfloat x, GLfloat y, GLfloat z) {
}
