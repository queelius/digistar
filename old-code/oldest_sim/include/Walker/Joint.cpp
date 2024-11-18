////////////////////////////////////////////
// Joint.cpp - Joint Class Implementation //
////////////////////////////////////////////

#include "Joint.h"

// Default Constructor
Joint::Joint()
{
}

// Initializing Constructor
Joint::Joint(float sphRad, float sphCent[], float sphClr[],
			 float cylRad, float cylLen, float cylClr[],
			 float pit, float minPit, float maxPit, float pitInc)
{
	SphereRadius      = sphRad;
	SphereCenter[0]   = sphCent[0];
	SphereCenter[1]   = sphCent[1];
	SphereCenter[2]   = sphCent[2];
	SphereColor[0]    = sphClr[0];
	SphereColor[1]    = sphClr[1];
	SphereColor[2]    = sphClr[2];
	CylinderRadius    = cylRad;
	CylinderLength    = cylLen;
	CylinderColor[0]  = cylClr[0];
	CylinderColor[1]  = cylClr[1];
	CylinderColor[2]  = cylClr[2];
	CurrentPitch      = pit;
	MinPitch          = minPit;
	MaxPitch          = maxPit;
	PitchIncrement    = pitInc;
}

// Initialization, using parametreized values.
void Joint::initialize(const float sphRad, const float sphCent[], const float sphClr[],
					   const float cylRad, const float cylLen, const float cylClr[],
					   const float pit, const float minPit, const float maxPit, const float pitInc)
{
	SphereRadius      = sphRad;
	SphereCenter[0]   = sphCent[0];
	SphereCenter[1]   = sphCent[1];
	SphereCenter[2]   = sphCent[2];
	SphereColor[0]    = sphClr[0];
	SphereColor[1]    = sphClr[1];
	SphereColor[2]    = sphClr[2];
	CylinderRadius    = cylRad;
	CylinderLength    = cylLen;
	CylinderColor[0]  = cylClr[0];
	CylinderColor[1]  = cylClr[1];
	CylinderColor[2]  = cylClr[2];
	CurrentPitch      = pit;
	MinPitch          = minPit;
	MaxPitch          = maxPit;
	PitchIncrement    = pitInc;
}

// Mutator, guaranteeing valid value.
void Joint::setSphereRadius(float sphRad)
{
	if (sphRad >= 0.0)
		SphereRadius = sphRad;
}

// Accessor
float Joint::getSphereRadius()
{
	return SphereRadius;
}

// Mutator, using array parameter.
void Joint::setSphereCenter(float sphCent[])
{
	SphereCenter[0] = sphCent[0];
	SphereCenter[1] = sphCent[1];
	SphereCenter[2] = sphCent[2];
}

// Mutator, using separate values.
void Joint::setSphereCenter(float sphCentX, float sphCentY, float sphCentZ)
{
	SphereCenter[0] = sphCentX;
	SphereCenter[1] = sphCentY;
	SphereCenter[2] = sphCentZ;
}

// Accessor, using array parameter.
void Joint::getSphereCenter(float sphCent[])
{
	sphCent[0] = SphereCenter[0];
	sphCent[1] = SphereCenter[1];
	sphCent[2] = SphereCenter[2];
}

// Accessor, using separate values.
void Joint::getSphereCenter(float &sphCentX, float &sphCentY, float &sphCentZ)
{
	sphCentX = SphereCenter[0];
	sphCentY = SphereCenter[1];
	sphCentZ = SphereCenter[2];
}

// Mutator
void Joint::setSphereColor(float sphClr[])
{
	SphereColor[0] = sphClr[0];
	SphereColor[1] = sphClr[1];
	SphereColor[2] = sphClr[2];
}

// Accessor
void Joint::getSphereColor(float sphClr[])
{
	sphClr[0] = SphereColor[0];
	sphClr[1] = SphereColor[1];
	sphClr[2] = SphereColor[2];
}

// Mutator, guaranteeing valid value.
void Joint::setCylinderRadius(float cylRad)
{
	if (cylRad >= 0.0)
		CylinderRadius = cylRad;
}

// Accessor
float Joint::getCylinderRadius()
{
	return CylinderRadius;
}

// Mutator, guaranteeing valid value.
void Joint::setCylinderLength(float cylLen)
{
	if (cylLen >= 0.0)
		CylinderLength = cylLen;
}

// Accessor
float Joint::getCylinderLength()
{
	return CylinderLength;
}

// Mutator
void Joint::setCylinderColor(float cylClr[])
{
	CylinderColor[0] = cylClr[0];
	CylinderColor[1] = cylClr[1];
	CylinderColor[2] = cylClr[2];
}

// Accessor
void Joint::getCylinderColor(float cylClr[])
{
	cylClr[0] = CylinderColor[0];
	cylClr[1] = CylinderColor[1];
	cylClr[2] = CylinderColor[2];
}

// Mutator
void Joint::setCurrentPitch(float pit)
{
	CurrentPitch = pit;
}

// Accessor
float Joint::getCurrentPitch()
{
	return CurrentPitch;
}

// Mutator
void Joint::setMinPitch(float minPit)
{
	MinPitch = minPit;
}

// Accessor
float Joint::getMinPitch()
{
	return MinPitch;
}

// Mutator
void Joint::setMaxPitch(float maxPit)
{
	MaxPitch = maxPit;
}

// Accessor
float Joint::getMaxPitch()
{
	return MaxPitch;
}

// Mutator
void Joint::setPitchIncrement(float pitInc)
{
	PitchIncrement = pitInc;
}

// Accessor
float Joint::getPitchIncrement()
{
	return PitchIncrement;
}

// Update Joint's pitch, altering
// increment values if necessary.
//
// animFactor specifies the rate at which it should change
// its pitch relative to its base pitch increment. animFactor
// equal to 1.0 would be no change, < 1.0 would be slower, > 1.0
// would be faster
void Joint::update(float animFactor)
{
    CurrentPitch += animFactor * PitchIncrement;
    
	if (CurrentPitch < MinPitch)
	{
		CurrentPitch = MinPitch;
		PitchIncrement *= -1;
	}
	else if (CurrentPitch > MaxPitch)
	{
		CurrentPitch = MaxPitch;
		PitchIncrement *= -1;
	}
}

// Render the joint as a cylinder on the z-axis, with
// a sphere centered at its upper end at the origin.
void Joint::draw(GLUquadricObj *qObj)
{
	gluQuadricNormals(qObj, GLU_SMOOTH);
	glPushMatrix();
		glColor3f( SphereColor[0], SphereColor[1], SphereColor[2] );
		gluSphere( qObj, SphereRadius, 16, 16 );
		glColor3f( CylinderColor[0], CylinderColor[1], CylinderColor[2] );
		glRotatef(90.0, 1.0, 0.0, 0.0);
		gluCylinder( qObj, CylinderRadius, CylinderRadius, CylinderLength, 12, 6 );
	glPopMatrix();
}