#ifndef LAYERLRELU_H
#define LAYERLRELU_H

#include "layer.h"
#include "../data_structures/matrix.h"
#include "../rand.h"
#include "../data_structures/vec.h"

class LayerLReLU : public Layer
{
    public:
        LayerLReLU(size_t _numUnits);
        ~LayerLReLU();

        void init(Rand* rng);
        void activate(const Vec& x);
        void backprop(Vec& prevBlame);
        void update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient);

    private:
};

#endif
