////////////////////////////////////////////////
// CrossJoint.h - CrossJoint Class Definition //
////////////////////////////////////////////////

#ifndef CROSS_JOINT_H
#define CROSS_JOINT_H

#include "glut.h"
#include <cmath>

class CrossJoint {
	public:
		CrossJoint();
		CrossJoint(float sphRad, float sphCent[], float sphClr[],
			       float cylRad, float cylLen, float cylClr[], 
			       float rol, float minRol, float maxRol, float rolInc,  
				   float ya, float minYa, float maxYa, float yaInc);
		
		void CrossJoint::initialize(const float sphRad, const float sphCent[], const float sphClr[],
			                        const float cylRad, const float cylLen, const float cylClr[], 
									const float rol, const float minRol, const float maxRol, const float rolInc,  
									const float ya, const float minYa, const float maxYa, const float yaInc);

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
		void setCurrentRoll(float pit);
		float getCurrentRoll();
		void setMinRoll(float minRol);
		float getMinRoll();
		void setMaxRoll(float maxRol);
		float getMaxRoll();
		void setRollIncrement(float rolInc);
		float getRollIncrement();
		void setCurrentYaw(float ya);
		float getCurrentYaw();
		void setMinYaw(float minYa);
		float getMinYaw();
		void setMaxYaw(float maxYa);
		float getMaxYaw();
		void setYawIncrement(float yaInc);
		float getYawIncrement();

        void update(float animFactor);
		void draw(GLUquadricObj *qObj);

	private:
		float SphereRadius;
		float SphereCenter[3];
		float SphereColor[3];
		float CylinderRadius;
		float CylinderLength;
		float CylinderColor[3];
		float CurrentRoll;
		float MinRoll;
		float MaxRoll;
		float RollIncrement;
		float CurrentYaw;
		float MinYaw;
		float MaxYaw;
		float YawIncrement;
};

#endif
