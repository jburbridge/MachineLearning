#ifndef LAYERMAXPOOLING2D_H
#define LAYERMAXPOOLING2D_H

#include <vector>
#include "layer.h"
#include "../data_structures/matrix.h"
#include "../rand.h"
#include "../data_structures/tensor3d.h"
#include "../data_structures/vec.h"

class LayerMaxPooling2D : public Layer
{
    public:
        LayerMaxPooling2D(std::vector<size_t> _inputDims, std::vector<size_t> _outputDims);
        ~LayerMaxPooling2D();

        void init(Rand* rng);
        void activate(const Vec& x);
        void backprop(Vec& prevBlame);
        void update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient);

    private:
        std::vector<size_t> inputDims;
        std::vector<size_t> outputDims;
        size_t pool_size;
        Tensor3D* maxActivation;
};

#endif
