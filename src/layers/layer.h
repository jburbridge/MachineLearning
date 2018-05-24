#ifndef LAYER_H
#define LAYER_H

#include "../data_structures/matrix.h"
#include "../data_structures/vec.h"

class Layer
{
    friend class NeuralNet;

    protected:
        size_t numInputs;
        size_t numOutputs;
        Vec* activation;
        Vec* blame;
        Matrix* weights;
        Vec* biases;

        bool weighted;

    public:
        Layer(size_t _numInputs, size_t _numOutputs);
        virtual ~Layer();

        virtual void init(Rand* rng) = 0;
        virtual void activate(const Vec& x) = 0;
        virtual void backprop(Vec& prevBlame) = 0;
        virtual void update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient) = 0;
        virtual void reset();
        virtual void print();

        size_t getNumInputs();
        size_t getNumOutputs();
};

#endif
