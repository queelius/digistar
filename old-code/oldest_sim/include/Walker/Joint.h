//////////////////////////////////////
// Joint.h - Joint Class Definition //
//////////////////////////////////////

#ifndef JOINT_H
#define JOINT_H

#include "glut.h"
#include <cmath>

class Joint {
	public:
		Joint();
		Joint(float sphRad, float sphCent[], float sphClr[],
			  float cylRad, float cylLen, float cylClr[],
			  float pit, float minPit, float maxPit, float pitInc);

		void initialize(const float sphRad, const float sphCent[], const float sphClr[],
			            const float cylRad, const float cylLen, const float cylClr[],
						const float pit, const float minPit, const float maxPit, const float pitInc);

		void setSphereRadius(float sphRad);
		float getSphereRadius();
		void setSphereCenter(float sphCent[]);
		void setSphereCenter(float sphCentX, float sphCentY, float sphCentZ);
		void getSphereCenter(float sphCent[]);
		void getSphereCenter(float &sphCentX, float &sphCentY, float &sphCentZ);
		void setSphereColor(float sphClr[]);
		void getSphereColor(float sphClr[]);
		void setCylinderRadius(float cylRad);
		float getCylinderRadius();
		void setCylinderLength(float cylLen);
		float getCylinderLength();
		void setCylinderColor(float cylClr[]);
		void getCylinderColor(float cylClr[]);
		void setCurrentPitch(float pit);
		float getCurrentPitch();
		void setMinPitch(float minPit);
		float getMinPitch();
		void setMaxPitch(float maxPit);
		float getMaxPitch();
		void setPitchIncrement(float pitInc);
		float getPitchIncrement();

		void update(float animFactor);
		void draw(GLUquadricObj *qObj);

	private:
		float SphereRadius;
		float SphereCenter[3];
		float SphereColor[3];
		float CylinderRadius;
		float CylinderLength;
		float CylinderColor[3];
		float CurrentPitch;
		float MinPitch;
		float MaxPitch;
		float PitchIncrement;
};

#endif
