#ifndef LAYERSIN_H
#define LAYERSIN_H

#include "layer.h"
#include "../data_structures/matrix.h"
#include "../rand.h"
#include "../data_structures/vec.h"

class LayerSin : public Layer
{
    public:
        LayerSin(size_t _numUnits);
        ~LayerSin();

        void init(Rand* rng);
        void activate(const Vec& x);
        void backprop(Vec& prevBlame);
        void update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient);

    private:

};

#endif
