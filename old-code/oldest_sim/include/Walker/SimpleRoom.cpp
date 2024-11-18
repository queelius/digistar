#include "SimpleRoom.h"

// Construtor: pre-compute the vertices for each corner of the SimpleRoom
SimpleRoom::SimpleRoom(GLfloat width, GLfloat depth, GLfloat height, GLuint wallTextureID) {
    this->wallTextureID = wallTextureID;

    coord[0][0] = -width/2; coord[0][1] = 0.0;    coord[0][2] = -depth/2;
    coord[1][0] = width/2;  coord[1][1] = 0.0;    coord[1][2] = -depth/2;
    coord[2][0] = width/2;  coord[2][1] = 0.0;    coord[2][2] = depth/2;
    coord[3][0] = -width/2; coord[3][1] = 0.0;    coord[3][2] = depth/2;
    coord[4][0] = -width/2; coord[4][1] = height; coord[4][2] = -depth/2;
    coord[5][0] = width/2;  coord[5][1] = height; coord[5][2] = -depth/2;
    coord[6][0] = width/2;  coord[6][1] = height; coord[6][2] = depth/2;
    coord[7][0] = -width/2; coord[7][1] = height; coord[7][2] = depth/2;

    normal[0][0] = 0.0;  normal[0][1] = 0.0; normal[0][2] = 1.0;
    normal[1][0] = 1.0;  normal[1][1] = 0.0; normal[1][2] = 1.0;
    normal[2][0] = 0.0;  normal[2][1] = 0.0; normal[2][2] = 1.0;
    normal[3][0] = -1.0; normal[3][1] = 0.0; normal[3][2] = 1.0;
}

// Draw the room. The room consists of four texture-mapped
// polygons (walls) and two untextured polygons (ceiling and floor)
void SimpleRoom::draw() {
    glPushMatrix();
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, wallTextureID);
        glBegin(GL_QUADS); // draw walls
	        for (int i = 0; i < 4; i++) {
                glNormal3f(normal[i][0], normal[i][1], normal[i][2]);
		        glTexCoord2f(0.0, 0.0);
		        glVertex3f(coord[i][0], coord[i][1], coord[i][2]);

                glTexCoord2f(4.0, 0.0);
		        glVertex3f(coord[(i+1)%4][0], coord[(i+1)%4][1], coord[(i+1)%4][2]);

                glTexCoord2f(4.0, 1.0);
		        glVertex3f(coord[(i+1)%4+4][0], coord[(i+1)%4+4][1], coord[(i+1)%4+4][2]);

                glTexCoord2f(0.0, 1.0);
		        glVertex3f(coord[i+4][0], coord[i+4][1], coord[i+4][2]);
	        }

            glColor3f(0.0, 0.6, 0.6); // floor and ceiling color

            glVertex3f(coord[0][0], coord[0][1], coord[0][2]); // draw floor
            glVertex3f(coord[3][0], coord[3][1], coord[3][2]);
            glVertex3f(coord[2][0], coord[2][1], coord[2][2]);
            glVertex3f(coord[1][0], coord[1][1], coord[1][2]);

            glVertex3f(coord[4][0], coord[4][1], coord[4][2]); // draw ceiling
            glVertex3f(coord[5][0], coord[5][1], coord[5][2]);
            glVertex3f(coord[6][0], coord[6][1], coord[6][2]);
            glVertex3f(coord[7][0], coord[7][1], coord[7][2]);
        glEnd();
    glPopMatrix();
}
