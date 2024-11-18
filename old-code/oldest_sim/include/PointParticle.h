#ifndef POINT_PARTICLE_H
#define POINT_PARTICLE_H

#include "vec2.h"
#include "Random.h"
#include "constants.h"
#include "Particle.h"
#include "BehaviorModifier.h"
#include <map>
#include <list>

class PointParticle: public Particle
{
public:
    PointParticle(double time): Particle(time)
    {
    };

    void addHeat(double amount)
    {
        double v = sqrt(2 * amount / _mass);
        _velocity += vec2(v * cos(TWO_PI * _rnd.get0_1()), v * sin(TWO_PI * _rnd.get0_1()));
    };

    double getKineticEnergy() const     { return 0.5 * _mass * _velocity.mag2(); };
    vec2 getAcceleration() const        { return _lastUpdateTime ? (1 / _lastUpdateTime) * _deltaVelocity : vec2::zero; };
    vec2 getMomentum() const            { return _mass * _velocity; };
    double getSpeed() const             { return _velocity.mag();   };
    void setFixed(bool value = true)    { _fixed = value; };
    double getTotalDistance() const     { return sqrt(_totalDistanceSquared); };
    double getMass() const              { return _mass;         };
    double getCharge() const            { return _charge;       };
    vec2 getPosition() const            { return _position;     };
    vec2 getVelocity() const            { return _velocity;     };
    void setPosition(const vec2& position)  { _position = position; };
    void setVelocity(const vec2& velocity)  { _velocity = velocity; };
    //PointParticle(
    //    double time, unsigned types, const vec2& position = vec2::zero,  const vec2& velocity = vec2::zero,
    //    double mass = 1.0, double charge = 0.0): Particle(time, POINT_PARTICLE | types)
    //{
    //    _position = position;
    //    _velocity = velocity;
    //    _deltaVelocity = vec2::zero;
    //    _mass = mass;
    //    _charge = charge;
    //    _totalDistanceSquared = 0;
    //    _fixed = false;
    //};
    void addBehaviorModifier(BehaviorModifier* f)
    {
        f->attachTo(this);
        _behaviorFns.push_back(f);
    };

    void updateImp(double time)
    {
        _totalImpulse = vec2::zero;
        auto imp = _impulse.begin();
        while (imp->first < time && imp != _impulse.end())
        {
            double duration = min(imp->second.tf - _lastUpdateTime, time - _lastUpdateTime);
            if (duration > 0)
            {
                _totalImpulse += duration * imp->second.f;
                imp++;
            }
            else
            {
                auto tmp = imp;
                imp++;
                _impulse.erase(tmp);
            }
        }

        auto f = _behaviorFns.begin();
        while (f != _behaviorFns.end())
        {
            if ((*f)->expire())
            {
                auto tmp = f;
                f++;
                delete *tmp;
                _behaviorFns.erase(tmp);
            }
            else
            {
                (*f)->update();
                f++;
            }
        }

//        for (auto i = os.begin(); i != os.end(); i++)
//        {
//            object& o = **i;
//
//            o.v = o.v + (1/o.mass) * o.fnet;
//            o.s = o.s + TIME_STEP * o.v + 0.5 / (1/o.mass) * TIME_STEP * TIME_STEP * o.fnet;
//            o.fnet = vec2(0, 0);
//        }

        _deltaVelocity = (1 / _mass) * _totalImpulse;
        _deltaPosition = timeStep * _velocity + 0.5 * timeStep * _deltaVelocity;
        _totalDistanceSquared += _deltaPosition.mag2();        
        _position += _deltaPosition;
        _velocity += _deltaVelocity;     
    };

    void addImpulse(const vec2& force, double startTime, double endTime)
    {
        if (startTime < _localTime || endTime < startTime)
            return;

        impulse imp = { startTime, endTime, force };        
        _impulse.insert(pair<double, impulse>(startTime, imp));
    };

protected:
    struct impulse
    {
        double ti;
        double tf;
        vec2 f;
    };

    double _mass;
    double _charge;
    bool _fixed;
    double _totalDistanceSquared;

    vec2 _velocity;
    vec2 _position;
    vec2 _totalImpulse;

    vec2 _deltaVelocity;
    vec2 _deltaPosition;

    Random _rnd;

    multimap<double, impulse> _impulse;
    list<BehaviorModifier*> _behaviorFns;
};

#endif