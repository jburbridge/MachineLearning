#ifndef LAYERTANH_H
#define LAYERTANH_H

#include "layer.h"
#include "../data_structures/matrix.h"
#include "../rand.h"
#include "../data_structures/vec.h"

class LayerTanh : public Layer
{
    public:
        LayerTanh(size_t _numUnits);
        ~LayerTanh();

        void init(Rand* rng);
        void activate(const Vec& x);
        void backprop(Vec& prevBlame);
        void update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient);

    private:
};

#endif
