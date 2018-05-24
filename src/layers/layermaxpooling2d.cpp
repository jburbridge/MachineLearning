#include <vector>
#include "layermaxpooling2d.h"
#include "../data_structures/tensor3d.h"

LayerMaxPooling2D::LayerMaxPooling2D(std::vector<size_t> _inputDims, std::vector<size_t> _outputDims)
: Layer(Tensor3D::countTensorSize(_inputDims), Tensor3D::countTensorSize(_outputDims))
{
    if (_inputDims.size() != 3 || _outputDims.size() != 3)
        throw Ex("Input and output tensors in max pooling layer must all have 3 dimensions");

    if(_inputDims[2] != _outputDims[2])
        throw Ex("This is a 2D pooling layer, so input and output tensors should have the same depth");

    weighted = false;

    //Assumes images are square
    pool_size = _inputDims[0] / _outputDims[0];

    inputDims = _inputDims;
    outputDims = _outputDims;
    maxActivation = new Tensor3D(inputDims);
}

LayerMaxPooling2D::~LayerMaxPooling2D()
{

}

void LayerMaxPooling2D::init(Rand* rng)
{

}

void LayerMaxPooling2D::activate(const Vec& x)
{
    Tensor3D* in = new Tensor3D(x, inputDims);
    Tensor3D* out = new Tensor3D(outputDims);

    //Do max pooling
    for(size_t depth = 0; depth < inputDims[2]; depth++)
    {
        for(size_t outRow = 0; outRow < outputDims[0]; outRow++)
        {
            for(size_t outCol = 0; outCol < outputDims[1]; outCol++)
            {
                double max = in->at(outRow * pool_size, outCol * pool_size, depth);
                size_t maxRow = outRow * pool_size, maxCol = outCol * pool_size;

                for(size_t windowRow = 0; windowRow < pool_size; windowRow++)
                    for(size_t windowCol = 0; windowCol < pool_size; windowCol++)
                        if(in->at(outRow * pool_size + windowRow, outCol * pool_size + windowCol, depth) > max)
                        {
                            maxRow = outRow * pool_size + windowRow;
                            maxCol = outCol * pool_size + windowCol;
                            max = in->at(maxRow, maxCol, depth);
                        }

                /*std::cout << "Depth = " << depth << "\n";
                std::cout << "outRow = " << outRow << "\n";
                std::cout << "outCol = " << outCol << "\n";
                std::cout << "maxRow = " << maxRow << "\n";
                std::cout << "maxCol = " << maxCol << "\n";
                std::cout << "max = " << max << "\n";*/
                maxActivation->at(maxRow, maxCol, depth) = 1;
                out->at(outRow, outCol, depth) = max;
            }
        }
    }
    Vec* o = out->get_data();
    activation->copy(*o);

    delete o;
    delete in;
    delete out;
}

void LayerMaxPooling2D::backprop(Vec& prevBlame)
{
    Tensor3D tblame(*blame, outputDims);
    Tensor3D tprevBlame(inputDims);

    /*std::cout << "Blame:\n";
    tblame.print();
    std::cout << "\n";

    std::cout << "Prev blame:\n";
    tprevBlame.print();
    std::cout << "\n";

    std::cout << "Max activation:\n";
    maxActivation->print();
    std::cout << "\n";*/

    for(size_t depth = 0; depth < inputDims[2]; depth++)
    {
        for(size_t outRow = 0; outRow < outputDims[0]; outRow++)
        {
            for(size_t outCol = 0; outCol < outputDims[1]; outCol++)
            {
                for(size_t windowRow = 0; windowRow < pool_size; windowRow++)
                    for(size_t windowCol = 0; windowCol < pool_size; windowCol++)
                        if (maxActivation->at(outRow * pool_size + windowRow, outCol * pool_size + windowCol, depth) == 1)
                            tprevBlame.at(outRow * pool_size + windowRow, outCol * pool_size + windowCol, depth) = tblame.at(outRow, outCol, depth);
            }
        }
    }

    /*std::cout << "Blame:\n";
    tblame.print();
    std::cout << "\n";

    std::cout << "Prev blame:\n";
    tprevBlame.print();
    std::cout << "\n";*/

    Vec* o = tprevBlame.get_data();
    prevBlame.copy(*o);
    delete o;
}

void LayerMaxPooling2D::update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient)
{
    //Layer is not weighted, so there's no gradient to update
}
