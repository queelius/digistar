#include "Emitter.h"

Emitter::Emitter(EmitRateCreator *emitRate)
{
    if (emitRate == NULL)
        this->emitRate = new DefaultEmitRateCreator();
    else
        this->emitRate = emitRate;
}

void Emitter::tick()
{
    emitRate->tick();
    unsigned int emitCount = emitRate->getEmissions();
    if (emitCount > 0)
    {
        for (unsigned int i = 0; i < emitCount; ++i)
        {
            // make new particle and release into world
        }
    }
}
