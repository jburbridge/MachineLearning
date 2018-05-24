#include <algorithm>
#include <cmath>
#include <ctime>
#include "../layers/layerlinear.h"
#include "../data_structures/matrix.h"
#include "neuralnet.h"
#include "../rand.h"
#include "../data_structures/vec.h"

NeuralNet::NeuralNet()
{
    numLayers = 0;
    alpha = 1.0;
    learning_rate = alpha;
    mu = 0.0;
    lambda = 0.0;
    batch_size = 1;
    GRAD_MAX = 1e2;
}

NeuralNet::~NeuralNet()
{
    for(size_t i = 0; i < numLayers; i++)
    {
        delete layers[i];
        delete gradients[i];
        delete bias_gradients[i];
    }
}

const char* NeuralNet::name()
{
    return "NeuralNet";
}

Layer& NeuralNet::addLayer(Layer* l)
{
    if(numLayers != 0 && layers[numLayers-1]->getNumOutputs() != l->getNumInputs())
        throw Ex("New layer's number of inputs should equal prev layer's number of outputs");

    layers.push_back(l);

    //If the layer is unweighted, push a dummy matrix on the gradients vector
    //This is terrible design, so remember to fix this later
    if(l->weighted)
    {
        gradients.push_back(new Matrix(l->getNumOutputs(), l->getNumInputs()));
        bias_gradients.push_back(new Vec(l->getNumOutputs()));
    }
    else
    {
        gradients.push_back(new Matrix(1, 1));
        bias_gradients.push_back(new Vec(1));
    }

    gradients.back()->fill(0.0);
    bias_gradients.back()->fill(0.0);
    numLayers++;
    return *layers[numLayers-1];
}

void NeuralNet::initWeights()
{
    for(size_t l = 0; l < numLayers; l++)
    {
            layers[l]->init(rng);
            /*if(layers[l]->weighted)
            {
                for(size_t i = 0; i < layers[l]->weights->rows(); i++)
                {
                    for(size_t j = 0; j < layers[l]->weights->cols(); j++)
                        (*layers[l]->weights)[i][j] = double(0.007) * int(i) + double(0.003) * int(j);
                    (*layers[l]->biases)[i] = double(0.001) * int(i);
                }
            }*/
    }
}

void NeuralNet::refineWeights(const Matrix& features, const Matrix& labels)
{
    //Scale the gradients for this new batch
    for(size_t i = 0; i < numLayers; i++)
    {
        *gradients[i] *= mu;
        *bias_gradients[i] *= mu;
    }

    //Run each example through the net and add to the gradient
    for(size_t i = 0; i < features.rows(); i++)
    {
        predict(features[i]);
        backprop(labels[i]);
        update_gradient(features[i]);
    }

    //Add the gradients into the weights
    for(size_t l = 0; l < numLayers; l++)
        if(layers[l]->weighted)
        {
            //layers[l]->weights->subtractScalar(lambda); //L1 Regularization
            *layers[l]->weights *= 1 - lambda; //L2 Regularization
            *layers[l]->weights += *gradients[l] * learning_rate;

            //*layers[l]->biases -= lambda; //L1
            *layers[l]->biases *= 1 - lambda; //L2
            *layers[l]->biases += *bias_gradients[l] * learning_rate;
        }
}

void NeuralNet::setInitialLearningRate(double _alpha)
{
    alpha = _alpha;
    learning_rate = alpha;
}

void NeuralNet::setLearningRate(double _learning_rate)
{
    learning_rate = _learning_rate;
}

void NeuralNet::setDecay(double _decay)
{
    decay = _decay;
}

void NeuralNet::setMomentum(double _mu)
{
    mu = _mu;
}

void NeuralNet::setLambda(double _lambda)
{
    lambda = _lambda;
}

void NeuralNet::setBatchSize(size_t _batch_size)
{
    batch_size = _batch_size;
}

const Vec& NeuralNet::predict(const Vec& in)
{
    layers[0]->activate(in);
    /*std::cout << "Layer 0 activation:\n";
    layers[0]->activation->print();
    std::cout << "\n";*/

	for(size_t i = 1; i < numLayers; i++)
    {
		layers[i]->activate(*layers[i-1]->activation);

        /*std::cout << "Layer " << i << " activation:\n";
        layers[i]->activation->print();
        std::cout << "\n";*/
    }
    /*std::cout << "Layer 0 weights:\n";
    layers[0]->weights->print(std::cout);
    std::cout << "\nLayer 0 biases:\n";
    layers[0]->biases->print(std::cout);
    std::cout << "\nLayer 2 weights:\n";
    layers[2]->weights->print(std::cout);
    std::cout << "\nLayer 2 biases:\n";
    layers[2]->biases->print(std::cout);
    std::cout << "\n";*/

    return *layers[numLayers-1]->activation;
}

void NeuralNet::backprop(const Vec& targets)
{
    layers[numLayers-1]->blame->copy(targets - *layers[numLayers-1]->activation);

    for(int i = numLayers-2; i >= 0; i--)
        layers[i+1]->backprop(*layers[i]->blame);
}

void NeuralNet::update_gradient(const Vec& x)
{
    Vec input;
    input.copy(x);

    for(size_t i = 0; i < numLayers; i++)
    {
        layers[i]->update_gradient(input, *gradients[i], *bias_gradients[i]);
        input.copy(*layers[i]->activation);
    }

    //Gradient clipping
    for(size_t l = 0; l < numLayers; l++)
    {
        for(size_t i = 0; i < gradients[l]->rows(); i++)
            for(size_t j = 0; j < gradients[l]->cols(); j++)
                if(abs((*gradients[l])[i][j]) > GRAD_MAX)
                    (*gradients[l])[i][j] = GRAD_MAX;

        for(size_t i = 0; i < bias_gradients[l]->size(); i++)
            if(abs((*bias_gradients[l])[i]) > GRAD_MAX)
                (*bias_gradients[l])[i] = GRAD_MAX;
    }
}

void NeuralNet::train(const Matrix& features, const Matrix& labels)
{
    Matrix f_batch(batch_size, features.cols());
    Matrix l_batch(batch_size, labels.cols());

    for(size_t i = 0; i < ceil(features.rows() / batch_size); i++)
    {
        size_t batchStart = i * batch_size;
        size_t batchEnd = i * batch_size + batch_size;
        if(batchEnd > features.rows())
            batchEnd = features.rows();

        f_batch.copyBlock(0, 0, features, batchStart, 0, batchEnd - batchStart, features.cols());
        l_batch.copyBlock(0, 0, labels, batchStart, 0, batchEnd - batchStart, labels.cols());
        refineWeights(f_batch, l_batch);
    }

    setLearningRate(learning_rate * decay);
}

void NeuralNet::forget()
{
    //Erase all memory
    for(size_t i = 0; i < numLayers; i++)
    {
        layers[i]->reset();
        gradients[i]->fill(0.0);
    }

    //Reset the whole network
    learning_rate = alpha;
    initWeights();
}

Matrix* NeuralNet::train_with_images(const Matrix& x)
{
    size_t height = 48;
    size_t width = 64;
    //images are 48x68, so 3 channels
    size_t channels = x.cols() / (height * width);

    size_t n = x.rows();
    size_t k = 2; //2 degrees of freedom

    Matrix* v = new Matrix(n, k);
    v->fill(0.0);

    int e = 10;
    for(size_t j = 0; j < e; j++)
    {
        std::cout << "j = " << j + 1 << "/" << e << "\n";
        for(size_t i = 0; i < 10000000; i++)
        {
            int t = rng->next(x.rows());
            int p = rng->next(width);
            int q = rng->next(height);

            //Debug spew
            /*std::cout << "j=" << j << ", i=" << i << "\n";
            for(size_t l = 0; l < numLayers; l++)
            {
                if(layers[l]->weighted)
                {
                    std::cout << "Layer " << l << " biases:\n";
                    layers[l]->biases->print(std::cout);
                    std::cout << "\nLayer " << l << " weights:\n";
                    layers[l]->weights->print(std::cout);
                    std::cout << "\n";
                }
            }

            int t = i % 1000;
            int p = (i * 31) % 64;
            int q = (i * 19) % 48;*/

            Vec features(2 + k);
            features[0] = (float)p/width;
            features[1] = (float)q/height;
            features.put(2, (*v)[t]);

            int s = channels * (width * q + p);

            Vec label(channels);
            label.put(0, x[t], s, 3);

            //Scale the gradients for this new batch
            for(size_t i = 0; i < numLayers; i++)
            {
                *gradients[i] *= mu;
                *bias_gradients[i] *= mu;
            }

            Vec pred;
            pred.copy(predict(features));
            backprop(label);

            //compute gradient on v[t]
            Matrix* blame_column = layers[0]->blame->toColumnVector();
            Matrix* product = Matrix::multiply(*layers[0]->weights, *blame_column, true, false);
            Vec* grad = product->toVec();
            *grad *= -1;

            //std::cout << "Grad =\n";
            //grad->print(std::cout);
            //std::cout << "\n";

            //update v[t]
            for(size_t g = 2; g < 2 + k; g++)
                (*v)[t][g-2] = (*v)[t][g-2] - learning_rate * (*grad)[g];

            update_gradient(features);

            //Add the gradients into the weights
            for(size_t l = 0; l < numLayers; l++)
                if(layers[l]->weighted)
                {
                    //layers[l]->weights->subtractScalar(lambda); //L1 Regularization
                    *layers[l]->weights *= 1 - lambda; //L2 Regularization
                    *layers[l]->weights += *gradients[l] * learning_rate;

                    //*layers[l]->biases -= lambda; //L1
                    *layers[l]->biases *= 1 - lambda; //L2
                    *layers[l]->biases += *bias_gradients[l] * learning_rate;
                }

            //Debug spew
            /*std::cout << "t=" << t << ", p=" << p << ", q=" << q << "\n";
            std::cout << "in = ";
            features.print(std::cout);
            std::cout << "\ntarget = ";
            label.print(std::cout);
            std::cout << "\nprediction = ";
            pred.print(std::cout);
            std::cout << "\n";

            for(size_t l = 0; l < numLayers; l++)
            {
                std::cout << "Layer " << l << " blame:\n";
                layers[l]->blame->print(std::cout);
                std::cout << "\n";
            }

            std::cout << "v_gradient = ";
            grad->print(std::cout);
            std::cout << "\n";
            std::cout << "\nUpdated v[t] = ";
            (*v)[t].print(std::cout);
            std::cout << "\n";*/

            delete blame_column;
            delete product;
            delete grad;
            //delete pred;
        }
        setLearningRate(learning_rate * decay);
    }

    v->print(std::cout);
    return v;
}
