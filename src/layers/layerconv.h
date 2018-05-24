#ifndef LAYERCONV_H
#define LAYERCONV_H

#include <vector>
#include "layer.h"
#include "../data_structures/matrix.h"
#include "../rand.h"
#include "../data_structures/tensor3d.h"
#include "../data_structures/vec.h"

class LayerConv : public Layer
{
    public:
        LayerConv(const std::vector<size_t> _inputDims,
                  const size_t _num_filters,
                  const std::vector<size_t> _filterDims,
                  const std::vector<size_t> _outputDims);
        ~LayerConv();

        void init(Rand* rng);
        void activate(const Vec& x);
        void backprop(Vec& prevBlame);
        void update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient);

        //Print information about this layer
        void print();

    private:
        std::vector<size_t> inputDims;
        std::vector<size_t> filterDims;
        std::vector<size_t> outputDims;
        size_t num_filters;
        std::vector<Tensor3D*> filters;
        Vec* biases;
};

#endif
