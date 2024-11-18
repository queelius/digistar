//////////////////////////////////////////////////////
// Walker.cpp - Walker Class Implementation //
//////////////////////////////////////////////////////

#include "Walker.h"

// Default Constructor
Walker::Walker() {
    reset();
}

// Reset walker to default values
void Walker::reset() {
    v[0] = 1; // x
    v[1] = 0; // y
    v[2] = 0; // z

    origin[0] = origin[1] = origin[2] = 0;

    pos[0] = 0;
    pos[1] = 0;
    pos[2] = 100;

    // Initalize Torso and Head "joint"
	TorsoAndHead.initialize(TORSO_AND_HEAD_SPHERE_RADIUS, TORSO_AND_HEAD_SPHERE_CENTER, TORSO_AND_HEAD_SPHERE_COLOR,
		                    TORSO_AND_HEAD_CYLINDER_RADIUS, TORSO_AND_HEAD_CYLINDER_LENGTH, TORSO_AND_HEAD_CYLINDER_COLOR,
                            TORSO_AND_HEAD_MIN_PITCH, TORSO_AND_HEAD_MIN_PITCH, TORSO_AND_HEAD_MAX_PITCH, TORSO_AND_HEAD_PITCH_INCREMENT);

    // Initialize Shoulders
    Shoulders.initialize(SHOULDERS_SPHERE_RADIUS, SHOULDERS_SPHERE_CENTER, SHOULDERS_SPHERE_COLOR,
		            SHOULDERS_CYLINDER_RADIUS, SHOULDERS_CYLINDER_LENGTH, SHOULDERS_CYLINDER_COLOR,
					SHOULDERS_ROLL, SHOULDERS_MIN_ROLL, SHOULDERS_MAX_ROLL, -SHOULDERS_ROLL_INCREMENT, 
					SHOULDERS_YAW, SHOULDERS_MIN_YAW, SHOULDERS_MAX_YAW, SHOULDERS_YAW_INCREMENT);

    // Initialize Humeruses
	LeftArm[0].initialize(HUMERUS_SPHERE_RADIUS, HUMERUS_SPHERE_CENTER, HUMERUS_SPHERE_COLOR,
		                  HUMERUS_CYLINDER_RADIUS, HUMERUS_CYLINDER_LENGTH, HUMERUS_CYLINDER_COLOR,
						  HUMERUS_MIN_PITCH, HUMERUS_MIN_PITCH, HUMERUS_MAX_PITCH, HUMERUS_PITCH_INCREMENT);
    RightArm[0].initialize(HUMERUS_SPHERE_RADIUS, HUMERUS_SPHERE_CENTER, HUMERUS_SPHERE_COLOR,
		                   HUMERUS_CYLINDER_RADIUS, HUMERUS_CYLINDER_LENGTH, HUMERUS_CYLINDER_COLOR,
						   HUMERUS_MAX_PITCH, HUMERUS_MIN_PITCH, HUMERUS_MAX_PITCH, -HUMERUS_PITCH_INCREMENT);

	// Initialize Forearms
	LeftArm[1].initialize(FOREARM_SPHERE_RADIUS, FOREARM_SPHERE_CENTER, FOREARM_SPHERE_COLOR,
		                  FOREARM_CYLINDER_RADIUS, FOREARM_CYLINDER_LENGTH, FOREARM_CYLINDER_COLOR,
						  FOREARM_MIN_PITCH, FOREARM_MIN_PITCH, FOREARM_MAX_PITCH, -FOREARM_PITCH_INCREMENT);
	RightArm[1].initialize(FOREARM_SPHERE_RADIUS, FOREARM_SPHERE_CENTER, FOREARM_SPHERE_COLOR,
		                   FOREARM_CYLINDER_RADIUS, FOREARM_CYLINDER_LENGTH, FOREARM_CYLINDER_COLOR,
						   FOREARM_MAX_PITCH, FOREARM_MIN_PITCH, FOREARM_MAX_PITCH, FOREARM_PITCH_INCREMENT);

	LeftArm[2].initialize(HAND_SPHERE_RADIUS, HAND_SPHERE_CENTER, HAND_SPHERE_COLOR,
		                  HAND_CYLINDER_RADIUS, HAND_CYLINDER_LENGTH, HAND_CYLINDER_COLOR,
						  HAND_MIN_PITCH, HAND_MIN_PITCH, HAND_MAX_PITCH, -HAND_PITCH_INCREMENT);
	RightArm[2].initialize(HAND_SPHERE_RADIUS, HAND_SPHERE_CENTER, HAND_SPHERE_COLOR,
		                   HAND_CYLINDER_RADIUS, HAND_CYLINDER_LENGTH, HAND_CYLINDER_COLOR,
						   HAND_MAX_PITCH, HAND_MIN_PITCH, HAND_MAX_PITCH, HAND_PITCH_INCREMENT);

	// Initialize Hips
	Hips.initialize(HIPS_SPHERE_RADIUS, HIPS_SPHERE_CENTER, HIPS_SPHERE_COLOR,
		            HIPS_CYLINDER_RADIUS, HIPS_CYLINDER_LENGTH, HIPS_CYLINDER_COLOR,
					HIPS_ROLL, HIPS_MIN_ROLL, HIPS_MAX_ROLL, -HIPS_ROLL_INCREMENT, 
					HIPS_YAW, HIPS_MIN_YAW, HIPS_MAX_YAW, HIPS_YAW_INCREMENT);

	// Initialize Thighs
	LeftLeg[0].initialize(THIGH_SPHERE_RADIUS, THIGH_SPHERE_CENTER, THIGH_SPHERE_COLOR,
		                  THIGH_CYLINDER_RADIUS, THIGH_CYLINDER_LENGTH, THIGH_CYLINDER_COLOR,
						  THIGH_MIN_PITCH, THIGH_MIN_PITCH, THIGH_MAX_PITCH, THIGH_PITCH_INCREMENT);
	RightLeg[0].initialize(THIGH_SPHERE_RADIUS, THIGH_SPHERE_CENTER, THIGH_SPHERE_COLOR,
		                   THIGH_CYLINDER_RADIUS, THIGH_CYLINDER_LENGTH, THIGH_CYLINDER_COLOR,
						   THIGH_MAX_PITCH, THIGH_MIN_PITCH, THIGH_MAX_PITCH, -THIGH_PITCH_INCREMENT);
	// Initialize Shins
	LeftLeg[1].initialize(SHIN_SPHERE_RADIUS, SHIN_SPHERE_CENTER, SHIN_SPHERE_COLOR,
		                  SHIN_CYLINDER_RADIUS, SHIN_CYLINDER_LENGTH, SHIN_CYLINDER_COLOR,
						  SHIN_MIN_PITCH, SHIN_MIN_PITCH, SHIN_MAX_PITCH, -SHIN_PITCH_INCREMENT);
	RightLeg[1].initialize(SHIN_SPHERE_RADIUS, SHIN_SPHERE_CENTER, SHIN_SPHERE_COLOR,
		                   SHIN_CYLINDER_RADIUS, SHIN_CYLINDER_LENGTH, SHIN_CYLINDER_COLOR,
						   SHIN_MAX_PITCH, SHIN_MIN_PITCH, SHIN_MAX_PITCH, SHIN_PITCH_INCREMENT);
	// Initialize Heels
	LeftLeg[2].initialize(HEEL_SPHERE_RADIUS, HEEL_SPHERE_CENTER, HEEL_SPHERE_COLOR,
		                  HEEL_CYLINDER_RADIUS, HEEL_CYLINDER_LENGTH, HEEL_CYLINDER_COLOR,
						  HEEL_MIN_PITCH, HEEL_MIN_PITCH, HEEL_MAX_PITCH, HEEL_PITCH_INCREMENT);
	RightLeg[2].initialize(HEEL_SPHERE_RADIUS, HEEL_SPHERE_CENTER, HEEL_SPHERE_COLOR,
		                   HEEL_CYLINDER_RADIUS, HEEL_CYLINDER_LENGTH, HEEL_CYLINDER_COLOR,
						   HEEL_MAX_PITCH, HEEL_MIN_PITCH, HEEL_MAX_PITCH, -HEEL_PITCH_INCREMENT);
	// Initialize Toes
	LeftLeg[3].initialize(TOE_SPHERE_RADIUS, TOE_SPHERE_CENTER, TOE_SPHERE_COLOR,
		                  TOE_CYLINDER_RADIUS, TOE_CYLINDER_LENGTH, TOE_CYLINDER_COLOR,
						  TOE_MIN_PITCH, TOE_MIN_PITCH, TOE_MAX_PITCH, TOE_PITCH_INCREMENT);
	RightLeg[3].initialize(TOE_SPHERE_RADIUS, TOE_SPHERE_CENTER, TOE_SPHERE_COLOR,
		                   TOE_CYLINDER_RADIUS, TOE_CYLINDER_LENGTH, TOE_CYLINDER_COLOR,
						   TOE_MAX_PITCH, TOE_MIN_PITCH, TOE_MAX_PITCH, -TOE_PITCH_INCREMENT);
}

// Set up the quadric object to display the
// graphical components of the Walker.
void Walker::draw() {
	float offset[3];
	GLUquadricObj *qObj;
	qObj = gluNewQuadric();
	gluQuadricDrawStyle(qObj, GLU_FILL);

	glPushMatrix();
        glTranslatef(pos[0], pos[1], pos[2]);
        glRotatef(-theta * 180 / PI, 0, 1, 0);

		Hips.getSphereCenter(offset);
		glTranslatef(offset[0],offset[1],offset[2]);
		glPushMatrix();
			glRotatef(Hips.getCurrentYaw(), 0.0, 1.0, 0.0);
			glRotatef(Hips.getCurrentRoll(), 0.0, 0.0, 1.0);
			Hips.draw(qObj);

            glPushMatrix();
                glTranslatef(0.0, TorsoAndHead.getCylinderLength(), 0.0);
                TorsoAndHead.draw(qObj);
	        glPopMatrix();

            glPushMatrix();
                Shoulders.getSphereCenter(offset);
                glTranslatef(offset[0],offset[1],offset[2]);
                glPushMatrix();
                    glRotatef(Shoulders.getCurrentYaw(), 0.0, 1.0, 0.0);
                    glRotatef(Shoulders.getCurrentRoll(), 0.0, 0.0, 1.0);

                    Shoulders.draw(qObj);

                    glRotatef(180.0, 0.0, 1.0, 0.0);
   			        glPushMatrix();
				        glTranslatef(Shoulders.getCylinderLength()/2, 0.0, 0.0 );
				        glRotatef(LeftArm[0].getCurrentPitch(), 1.0, 0.0, 0.0);
                        LeftArm[0].draw(qObj);
				        glTranslatef(0.0, -LeftArm[0].getCylinderLength(), 0.0);
				        glRotatef(LeftArm[1].getCurrentPitch(), 1.0, 0.0, 0.0);
				        LeftArm[1].draw(qObj);
                        glTranslatef(0.0, -LeftArm[1].getCylinderLength(), 0.0);
                        LeftArm[2].draw(qObj);
                    glPopMatrix();

   			        glPushMatrix();
				        glTranslatef(-Shoulders.getCylinderLength()/2, 0.0, 0.0 );
				        glRotatef(RightArm[0].getCurrentPitch(), 1.0, 0.0, 0.0);
                        RightArm[0].draw(qObj);
				        glTranslatef(0.0, -RightArm[0].getCylinderLength(), 0.0);
				        glRotatef(RightArm[1].getCurrentPitch(), 1.0, 0.0, 0.0);
				        RightArm[1].draw(qObj);
                        glTranslatef(0.0, -RightArm[1].getCylinderLength(), 0.0);
                        RightArm[2].draw(qObj);
                    glPopMatrix();
                glPopMatrix();
            glPopMatrix();        

			glPushMatrix();
				glTranslatef(Hips.getCylinderLength()/2, 0.0, 0.0 );
				glRotatef(LeftLeg[0].getCurrentPitch(), 1.0, 0.0, 0.0);
				LeftLeg[0].draw(qObj);
				glTranslatef(0.0, -LeftLeg[0].getCylinderLength(), 0.0);
				glRotatef(LeftLeg[1].getCurrentPitch(), 1.0, 0.0, 0.0);
				LeftLeg[1].draw(qObj);
				glTranslatef(0.0, -LeftLeg[1].getCylinderLength(), 0.0);
				glRotatef(LeftLeg[2].getCurrentPitch(), 1.0, 0.0, 0.0);
				LeftLeg[2].draw(qObj);
				glTranslatef(0.0, -LeftLeg[2].getCylinderLength(), 0.0);
				glRotatef(LeftLeg[3].getCurrentPitch(), 1.0, 0.0, 0.0);
				LeftLeg[3].draw(qObj);
			glPopMatrix();
			glPushMatrix();
				glTranslatef(-Hips.getCylinderLength()/2, 0.0, 0.0 );
				glRotatef(RightLeg[0].getCurrentPitch(), 1.0, 0.0, 0.0);
				RightLeg[0].draw(qObj);
				glTranslatef(0.0, -RightLeg[0].getCylinderLength(), 0.0);
				glRotatef(RightLeg[1].getCurrentPitch(), 1.0, 0.0, 0.0);
				RightLeg[1].draw(qObj);
				glTranslatef(0.0, -RightLeg[1].getCylinderLength(), 0.0);
				glRotatef(RightLeg[2].getCurrentPitch(), 1.0, 0.0, 0.0);
				RightLeg[2].draw(qObj);
				glTranslatef(0.0, -RightLeg[2].getCylinderLength(), 0.0);
				glRotatef(RightLeg[3].getCurrentPitch(), 1.0, 0.0, 0.0);
				RightLeg[3].draw(qObj);
			glPopMatrix();
		glPopMatrix();
	glPopMatrix();
	gluDeleteQuadric(qObj);
}

// Update the positions and orientations associated
// with the joints and cross-joints of the Walker.
void Walker::update(float animFactor) {
    GLfloat dx = origin[0] - pos[0];
    GLfloat dz = origin[2] - pos[2];

    theta = atan2(dz, dx);

    GLfloat mag = (v[0] * v[0] + v[2] * v[2]) / sqrt(dx*dx + dz*dz);

    a[0] = mag * cos(theta);
    a[2] = mag * sin(theta);

    v[0] += a[0];
    v[2] += a[2];

    pos[0] += v[0];
    pos[2] += v[2];

    //std::cout << pos[0] << ", " << pos[2] << std::endl;

	Hips.update(animFactor);
    Shoulders.update(animFactor);
    TorsoAndHead.update(animFactor);

	int i;
    for (i = 0; i < 3; i++) {
		LeftArm[i].update(animFactor);
		RightArm[i].update(animFactor);
    }
    for (i = 0; i < 4; i++) {
		LeftLeg[i].update(animFactor);
		RightLeg[i].update(animFactor);
    }
}