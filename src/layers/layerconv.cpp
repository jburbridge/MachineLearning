
#include "layerconv.h"
#include "../rand.h"
#include "../data_structures/matrix.h"
#include "../data_structures/tensor3d.h"
#include "../data_structures/vec.h"

LayerConv::LayerConv(const std::vector<size_t> _inputDims,
                     const size_t _num_filters,
                     const std::vector<size_t> _filterDims,
                     const std::vector<size_t> _outputDims)
: Layer(Tensor3D::countTensorSize(_inputDims), Tensor3D::countTensorSize(_outputDims))
{
    if (_inputDims.size() != 3 || _filterDims.size() != 3 || _outputDims.size() != 3)
        throw Ex("Input tensor, filter, and output tensor must all have 3 dimensions");

    if (_inputDims[2] != _filterDims[2])
        throw Ex("The depth of each filter must match the depth of the input tensor. The convolution only happens in the x and y dimensions.");

    if (_num_filters != _outputDims[2])
        throw Ex("The number of filters must equal the depth of the output tensor");


    weighted = false;
    num_filters = _num_filters;

    inputDims = _inputDims;
    filterDims = _filterDims;
    outputDims = _outputDims;

    biases = new Vec(num_filters);
}

LayerConv::~LayerConv()
{
    for(size_t i = 0; i < filters.size(); i++)
        delete filters[i];
    delete biases;
}

void LayerConv::init(Rand* rng)
{
    size_t filter_size = filterDims[0] * filterDims[1] * filterDims[2];
    for(size_t f = 0; f < num_filters; f++)
    {
        Vec v(filter_size);
        for(size_t i = 0; i < filter_size; i++)
            v[i] = rng->normal() / (filter_size);

        filters.push_back(new Tensor3D(v, filterDims));
    }
}

void LayerConv::activate(const Vec& x)
{
    Tensor3D* input = new Tensor3D(x, inputDims);
    Vec _activation(Tensor3D::countTensorSize(outputDims));

    for(size_t i = 0; i < num_filters; i++)
    {
        Tensor3D* output = new Tensor3D({outputDims[0], outputDims[1], 1});

        Tensor3D::convolve(*input, *filters[i], *output, false, 1);

        Vec* o = output->get_data();
        (*o) += (*biases)[i];
        _activation.put(i * o->size(), *o);

        delete o;
        delete output;
    }

    activation->copy(_activation);
    delete input;
}

void LayerConv::backprop(Vec& prevBlame)
{
    //Convolve blame with flipped filters
    Tensor3D tblame(*blame, outputDims);
    Tensor3D tprevBlame(inputDims);

    Vec _prevblame(Tensor3D::countTensorSize(inputDims));

    for(size_t i = 0; i < num_filters; i++)
    {
        Tensor3D* output = new Tensor3D({inputDims[0], inputDims[1], 1});

        Tensor3D::convolve(tblame, *filters[i], *output, true, 1);

        Vec* o = output->get_data();
        _prevblame.put(i * o->size(), *o);

        delete o;
        delete output;
    }

    prevBlame.copy(_prevblame);
}

void LayerConv::update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient)
{
    //Convolve blame with x
    Tensor3D tx(x, inputDims);
    Tensor3D tblame(*blame, outputDims);

    for(size_t i = 0; i < num_filters; i++)
    {
        //The sum of vblameslice is also the gradient for the bias for this filter
        Vec* vblameSlice = tblame.copy(outputDims[0] * outputDims[1] * i, outputDims[0] * outputDims[1]);
        Tensor3D tblameSlice(*vblameSlice, {outputDims[0], outputDims[1], 1});
        Tensor3D gSlice({filterDims[0], filterDims[1], 1});
        Tensor3D::convolve(tx, tblameSlice, gSlice, false, 1);

        //Don't hard code the learning rate!
        gSlice *= 0.01;
        (*filters[i]) += gSlice;
        bias_gradient[i] += vblameSlice->sum() * 0.01;

        delete vblameSlice;
    }
}

void LayerConv::print()
{
    //std::cout << "Input dims: " << inputDims[0] << ", " << inputDims[1] << ", " << inputDims[2] << "\n";
    //std::cout << "Filter dims: " << filterDims[0] << ", " << filterDims[1] << ", " << filterDims[2] << "\n";
    //std::cout << "Output dims: " << outputDims[0] << ", " << outputDims[1] << ", " << outputDims[2] << "\n";
    //std::cout << "Num_filters: " << num_filters << "\n";

    std::cout << "Initial weights:\n";

    for(size_t i = 0; i < num_filters; i++)
    {
        std::cout << "Filter " << i << ":\n";
        filters[i]->print();
        std::cout << "Bias " << i << ": " << (*biases)[i] << "\n\n";
    }

    //std::cout << "Most recent activation:\n";
    //activation->print();
    //std::cout << "\n";
}
