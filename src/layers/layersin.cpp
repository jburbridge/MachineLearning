#include <cmath>
#include "layer.h"
#include "layersin.h"
#include "../data_structures/matrix.h"
#include "../rand.h"
#include "../data_structures/vec.h"

LayerSin::LayerSin(size_t _numUnits)
: Layer(_numUnits, _numUnits)
{
    weighted = false;
}

LayerSin::~LayerSin()
{

}

void LayerSin::init(Rand* rng)
{
    //Do nothing
}

void LayerSin::activate(const Vec& x)
{
    for(size_t i = 0; i < numInputs - 1; i++)
        (*activation)[i] = sin(x[i]);
    (*activation)[numInputs-1] = x[numInputs-1];
}

void LayerSin::backprop(Vec& prevBlame)
{
    for(size_t i = 0; i < numInputs - 1; i++)
        prevBlame[i] = (*blame)[i] * cos((*activation)[i]);
    prevBlame[numInputs-1] = (*blame)[numInputs-1];
}

void LayerSin::update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient)
{
    //Do nothing
}
