#include "layer.h"
#include "../data_structures/matrix.h"
#include "../data_structures/vec.h"

Layer::Layer(size_t _numInputs, size_t _numOutputs)
{
    if(_numInputs < 1 || _numOutputs < 1)
        throw Ex("Each layer must have at least 1 input and 1 output");

    numInputs = _numInputs;
    numOutputs = _numOutputs;
	activation = new Vec(numOutputs);
	activation->fill(0.0);
    blame = new Vec(numOutputs);
    blame->fill(0.0);
}

Layer::~Layer()
{
	delete activation;
    delete blame;
}

void Layer::reset()
{
    activation->fill(0.0);
    blame->fill(0.0);
}

void Layer::print()
{
    
}

size_t Layer::getNumInputs()
{
    return numInputs;
}

size_t Layer::getNumOutputs()
{
    return numOutputs;
}
