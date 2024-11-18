//////////////////////////////////////////////////////
// CrossJoint.cpp - CrossJoint Class Implementation //
//////////////////////////////////////////////////////

#include "CrossJoint.h"

// Default Constructor
CrossJoint::CrossJoint()
{
}

// Initializing Constructor
CrossJoint::CrossJoint(float sphRad, float sphCent[], float sphClr[],
					   float cylRad, float cylLen, float cylClr[], 
					   float rol, float minRol, float maxRol, float rolInc, 
					   float ya, float minYa, float maxYa, float yaInc)
{
	SphereRadius     = sphRad;
	SphereCenter[0]  = sphCent[0];
	SphereCenter[1]  = sphCent[1];
	SphereCenter[2]  = sphCent[2];
	SphereColor[0]   = sphClr[0];
	SphereColor[1]   = sphClr[1];
	SphereColor[2]   = sphClr[2];
	CylinderRadius   = cylRad;
	CylinderLength   = cylLen;
	CylinderColor[0] = cylClr[0];
	CylinderColor[1] = cylClr[1];
	CylinderColor[2] = cylClr[2];
	CurrentRoll      = rol;
	MinRoll          = minRol;
	MaxRoll          = maxRol;
	RollIncrement    = rolInc;
	CurrentYaw       = ya;
	MinYaw           = minYa;
	MaxYaw           = maxYa;
	YawIncrement     = yaInc;
}

// Initialization, using parametreized values.
void CrossJoint::initialize(const float sphRad, const float sphCent[], const float sphClr[],
							const float cylRad, const float cylLen, const float cylClr[], 
							const float rol, const float minRol, const float maxRol, const float rolInc, 
							const float ya, const float minYa, const float maxYa, const float yaInc)
{
	SphereRadius     = sphRad;
	SphereCenter[0]  = sphCent[0];
	SphereCenter[1]  = sphCent[1];
	SphereCenter[2]  = sphCent[2];
	SphereColor[0]   = sphClr[0];
	SphereColor[1]   = sphClr[1];
	SphereColor[2]   = sphClr[2];
	CylinderRadius   = cylRad;
	CylinderLength   = cylLen;
	CylinderColor[0] = cylClr[0];
	CylinderColor[1] = cylClr[1];
	CylinderColor[2] = cylClr[2];
	CurrentRoll      = rol;
	MinRoll          = minRol;
	MaxRoll          = maxRol;
	RollIncrement    = rolInc;
	CurrentYaw       = ya;
	MinYaw           = minYa;
	MaxYaw           = maxYa;
	YawIncrement     = yaInc;
}

// Mutator, guaranteeing valid value.
void CrossJoint::setSphereRadius(float sphRad)
{
	if (sphRad >= 0.0)
		SphereRadius = sphRad;
}

// Accessor
float CrossJoint::getSphereRadius()
{
	return SphereRadius;
}

// Mutator, using array parameter.
void CrossJoint::setSphereCenter(float sphCent[])
{
	SphereCenter[0] = sphCent[0];
	SphereCenter[1] = sphCent[1];
	SphereCenter[2] = sphCent[2];
}

// Mutator, using separate values.
void CrossJoint::setSphereCenter(float sphCentX, float sphCentY, float sphCentZ)
{
	SphereCenter[0] = sphCentX;
	SphereCenter[1] = sphCentY;
	SphereCenter[2] = sphCentZ;
}

// Accessor, using array parameter.
void CrossJoint::getSphereCenter(float sphCent[])
{
	sphCent[0] = SphereCenter[0];
	sphCent[1] = SphereCenter[1];
	sphCent[2] = SphereCenter[2];
}

// Accessor, using separate values.
void CrossJoint::getSphereCenter(float &sphCentX, float &sphCentY, float &sphCentZ)
{
	sphCentX = SphereCenter[0];
	sphCentY = SphereCenter[1];
	sphCentZ = SphereCenter[2];
}

// Mutator
void CrossJoint::setSphereColor(float sphClr[])
{
	SphereColor[0] = sphClr[0];
	SphereColor[1] = sphClr[1];
	SphereColor[2] = sphClr[2];
}

// Accessor
void CrossJoint::getSphereColor(float sphClr[])
{
	sphClr[0] = SphereColor[0];
	sphClr[1] = SphereColor[1];
	sphClr[2] = SphereColor[2];
}

// Mutator, guaranteeing valid value.
void CrossJoint::setCylinderRadius(float cylRad)
{
	if (cylRad >= 0.0)
		CylinderRadius = cylRad;
}

// Accessor
float CrossJoint::getCylinderRadius()
{
	return CylinderRadius;
}

// Mutator, guaranteeing valid value.
void CrossJoint::setCylinderLength(float cylLen)
{
	if (cylLen >= 0.0)
		CylinderLength = cylLen;
}

// Accessor
float CrossJoint::getCylinderLength()
{
	return CylinderLength;
}

// Mutator
void CrossJoint::setCylinderColor(float cylClr[])
{
	CylinderColor[0] = cylClr[0];
	CylinderColor[1] = cylClr[1];
	CylinderColor[2] = cylClr[2];
}

// Accessor
void CrossJoint::getCylinderColor(float cylClr[])
{
	cylClr[0] = CylinderColor[0];
	cylClr[1] = CylinderColor[1];
	cylClr[2] = CylinderColor[2];
}

// Mutator
void CrossJoint::setCurrentRoll(float rol)
{
	CurrentRoll = rol;
}

// Accessor
float CrossJoint::getCurrentRoll()
{
	return CurrentRoll;
}

// Mutator
void CrossJoint::setMinRoll(float minRol)
{
	MinRoll = minRol;
}

// Accessor
float CrossJoint::getMinRoll()
{
	return MinRoll;
}

// Mutator
void CrossJoint::setMaxRoll(float maxRol)
{
	MaxRoll = maxRol;
}

// Accessor
float CrossJoint::getMaxRoll()
{
	return MaxRoll;
}

// Mutator
void CrossJoint::setRollIncrement(float rolInc)
{
	RollIncrement = rolInc;
}

// Accessor
float CrossJoint::getRollIncrement()
{
	return RollIncrement;
}

// Mutator
void CrossJoint::setCurrentYaw(float ya)
{
	CurrentYaw = ya;
}

// Accessor
float CrossJoint::getCurrentYaw()
{
	return CurrentYaw;
}

// Mutator
void CrossJoint::setMinYaw(float minYa)
{
	MinYaw = minYa;
}

// Accessor
float CrossJoint::getMinYaw()
{
	return MinYaw;
}

// Mutator
void CrossJoint::setMaxYaw(float maxYa)
{
	MaxYaw = maxYa;
}

// Accessor
float CrossJoint::getMaxYaw()
{
	return MaxYaw;
}

// Mutator
void CrossJoint::setYawIncrement(float yaInc)
{
	YawIncrement = yaInc;
}

// Accessor
float CrossJoint::getYawIncrement()
{
	return YawIncrement;
}

// Update CrossJoint's roll and yaw, altering
// increment values if necessary.
// animFactor specifies the rate at which it should change
// its pitch relative to its base pitch increment. animFactor
// equal to 1.0 would be no change, < 1.0 would be slower, > 1.0
// would be faster
void CrossJoint::update(float animFactor)
{
	CurrentRoll += animFactor * RollIncrement;
	if (CurrentRoll < MinRoll)
	{
		CurrentRoll = MinRoll;
		RollIncrement *= -1;
	}
	else if (CurrentRoll > MaxRoll)
	{
		CurrentRoll = MaxRoll;
		RollIncrement *= -1;
	}
	CurrentYaw += animFactor * YawIncrement;
	if (CurrentYaw < MinYaw)
	{
		CurrentYaw = MinYaw;
		YawIncrement *= -1;
	}
	else if (CurrentYaw > MaxYaw)
	{
		CurrentYaw = MaxYaw;
		YawIncrement *= -1;
	}
}

// Render the crossjoint as a cylinder on the x-axis,
// with a sphere emerging from its center (the origin).
void CrossJoint::draw(GLUquadricObj *qObj)
{
	gluQuadricNormals(qObj, GLU_SMOOTH);
	glPushMatrix();
			glColor3f( SphereColor[0], SphereColor[1], SphereColor[2] );
			gluSphere( qObj, SphereRadius, 16, 16 );
			glColor3f( CylinderColor[0], CylinderColor[1], CylinderColor[2] );
			glRotatef( 90.0, 0.0, 1.0, 0.0 );
			glTranslatef( 0.0, 0.0, -CylinderLength/2 );
			gluCylinder( qObj, CylinderRadius, CylinderRadius, CylinderLength, 12, 6 );
	glPopMatrix();
}