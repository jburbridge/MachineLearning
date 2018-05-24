#include <cmath>
#include "layertanh.h"
#include "../rand.h"
#include "../data_structures/vec.h"

LayerTanh::LayerTanh(size_t _numUnits) :
Layer(_numUnits, _numUnits)
{
    weighted = false;
}

LayerTanh::~LayerTanh()
{

}

void LayerTanh::init(Rand* rng)
{

}

void LayerTanh::activate(const Vec& x)
{
    for(size_t i = 0; i < numInputs; i++)
        (*activation)[i] = tanh(x[i]);
}

void LayerTanh::backprop(Vec& prevBlame)
{
    for(size_t i = 0; i < numInputs; i++)
        prevBlame[i] = (*blame)[i] * (1.0 - (*activation)[i] * (*activation)[i]);
}


void LayerTanh::update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient)
{
        //This is an activation layer, so there's no gradient
}
