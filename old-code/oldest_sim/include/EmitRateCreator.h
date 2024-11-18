#ifndef EMIT_RATE_PROPERTY_H
#define EMIT_RATE_PROPERTY_H

const double DEFAULT_INITIAL_ACCUMULATOR        = 0;
const double DEFAULT_INITIAL_ACCUMULATOR_RATE   = 0;
const double DEFAULT_ACCUMULATOR_RATE_OF_RATE   = 0;
const double DEFAULT_MAX_ACCUMULATOR            = 1000000;
const double DEFAULT_MIN_ACCUMULATOR            = 0;
const double DEFAULT_MAX_ACCUMULATOR_RATE       = 1000000;
const double DEFAULT_MIN_ACCUMULATOR_RATE       = -1000000;

class EmitRateCreator
{
public:
    virtual void            tick()         = 0;
    virtual unsigned int    getEmissions() = 0;
};

// implement an emit creator that implements a poisson distribution

class DefaultEmitRateCreator: public EmitRateCreator
{
public:
    DefaultEmitRateCreator(
        double initialAccumulator       = DEFAULT_INITIAL_ACCUMULATOR,
        double initialAccumulatorRate   = DEFAULT_INITIAL_ACCUMULATOR_RATE,
        double accumulatorRateOfRate    = DEFAULT_ACCUMULATOR_RATE_OF_RATE,
        double maxAccumulator           = DEFAULT_MAX_ACCUMULATOR,
        double minAccumulator           = DEFAULT_MIN_ACCUMULATOR,
        double maxAccumulatorRate       = DEFAULT_MAX_ACCUMULATOR_RATE,
        double minAccumulatorRate       = DEFAULT_MIN_ACCUMULATOR_RATE
        )
    {
        this->accumulator           = initialAccumulator;
        this->accumulatorRate       = initialAccumulatorRate;
        this->accumulatorRateOfRate = accumulatorRateOfRate;
        this->minAccumulator        = minAccumulator;
        this->maxAccumulator        = maxAccumulator;
        this->minAccumulatorRate    = minAccumulatorRate;
        this->maxAccumulatorRate    = minAccumulatorRate;
    };

    void tick()
    {
        accumulatorRate += accumulatorRateOfRate;
        if (accumulatorRate > maxAccumulatorRate)
            accumulatorRate = maxAccumulatorRate;
        else if (accumulatorRate < minAccumulatorRate)
            accumulatorRate = minAccumulatorRate;

        accumulator += accumulatorRate;
        if (accumulator > maxAccumulator)
            accumulator = maxAccumulator;
        if (accumulator < minAccumulator)
            accumulator = minAccumulator;
    };

    unsigned int getEmissions()
    {
        unsigned int emissions = (unsigned int)accumulator;
        accumulator -= emissions;

        return emissions;
    };

protected:
    double accumulatorRateOfRate;
    double accumulatorRate;
    double accumulator;

    double minAccumulator;
    double maxAccumulator;
    double minAccumulatorRate;
    double maxAccumulatorRate;
};

#endif