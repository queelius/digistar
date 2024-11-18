#include <string>
#include <iostream>
#include <cassert>
#include <cmath>
#include <list>
#include <vector>
using namespace std;

class IServoPreference
{
	virtual long id() const			= 0;
};

class IServoConstraint
{
public:
	virtual long id() const			= 0;

	virtual bool isValid() const	= 0;
};

class IServo
{
public:
	virtual long id() const						= 0;

	virtual bool set(double theta)				= 0; // absolute angular displacement (radians)
	virtual bool set(double theta, double t)	= 0; // absolute angular displacement (radians); t seconds to reach theta
	virtual bool rotate(double w, double t)		= 0; // rotate at angular velocity w (radians / sec) for time t (sec)

	virtual list<IServoConstraint*> getConstraints() = 0;
	virtual IServoConstraint* getConstraint(long constraintId) = 0;
};

// SSC32 has fairly powerful group commands that
// this interface does not yet expose
//
// try to find elegant way to expose it
class IServoBatchOp
{
	virtual long	id() const				= 0;

	virtual bool	add(IServo* servo)		= 0;
	virtual bool	remove(IServo* servo)	= 0;
	virtual bool	remove(long servoId)	= 0;
	virtual IServo*	get(long servoId)		= 0;

	virtual bool	setServo(long servoId, double theta, double t);
	virtual bool	setServo(IServo* servo, double theta, double t);
	virtual bool	rotateServo(long servoId, double w, double t);
};

class IServoGroup
{
};

class SSC32: public IServoGroup // ssc32 interface
{
public:
	static SSC32	getSSC32();

	bool			add(IServo* servo);
	bool			remove(IServo* servo);
	bool			remove(long servoId);
	long			id();
	IServo*			get(long servoId);
	
private:
	SSC32(); // private constructor
}

int main()
{
}