#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include "../layers/layerlinear.h"
#include "../rand.h"
#include "supervised.h"
#include "../data_structures/vec.h"

class NeuralNet : public SupervisedLearner
{
    public:
        NeuralNet();
        ~NeuralNet();
        const char* name(); 

        //Adds a layer to the end of the vector. _inputs must equal _outputs from previous layer.
        Layer& addLayer(Layer* l);

        //Initializes the weights for each layer
        void initWeights();

        void refineWeights(const Matrix& features, const Matrix& labels);

        void setInitialLearningRate(double _alpha);
        void setLearningRate(double _learning_rate);
        void setDecay(double _decay);
        void setMomentum(double _mu);
        void setLambda(double _lambda);
        void setBatchSize(size_t _batch_size);

        const Vec& predict(const Vec& in);
        void backprop(const Vec& targets);
        void update_gradient(const Vec& x);
        void train(const Matrix& features, const Matrix& labels);
        void forget();

        Matrix* train_with_images(const Matrix& x);

    private:
        std::vector<Layer*> layers;
        std::vector<Matrix*> gradients;
        std::vector<Vec*> bias_gradients;

        size_t numLayers;                   //Linear layers are separate from non-linear activations
        double alpha;                       //Initial learning rate, before decay
        double learning_rate;               //Decay this, not alpha
        double decay;                       //Decay factor
        double mu;                          //Momentum
        double lambda;                      //Regularization factor
        size_t batch_size;                  //Batch size
        double GRAD_MAX;              //Gradient clipping max magnitude
};

#endif
