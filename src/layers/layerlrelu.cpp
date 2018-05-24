#include "layerlrelu.h"
#include "../rand.h"
#include "../data_structures/vec.h"

LayerLReLU::LayerLReLU(size_t _numUnits)
: Layer(_numUnits, _numUnits)
{
    weighted = false;
}

LayerLReLU::~LayerLReLU()
{

}

void LayerLReLU::init(Rand* rng)
{

}

void LayerLReLU::activate(const Vec& x)
{
    for(size_t i = 0; i < numInputs; i++)
    {
        if(x[i] >= 0)
            (*activation)[i] = x[i];
        else
            (*activation)[i] = 0.01 * x[i];
    }
}

void LayerLReLU::backprop(Vec& prevBlame)
{
    for(size_t i = 0; i < numInputs; i++)
    {
        prevBlame[i] = (*blame)[i];
        if((*activation)[i] < 0)
            prevBlame[i] *= 0.01;
    }
}

void LayerLReLU::update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient)
{
        //This is an activation layer, so there's no gradient
}
