#include "layerlinear.h"
#include "../data_structures/matrix.h"
#include "../rand.h"
#include "../data_structures/vec.h"

LayerLinear::LayerLinear(size_t _numInputs, size_t _numOutputs)
: Layer(_numInputs, _numOutputs)
{
    weighted = true;
    weights = new Matrix(numOutputs, numInputs);
    weights->fill(0.0);
    biases = new Vec(numOutputs);
    biases->fill(0.0);
}

LayerLinear::~LayerLinear()
{
    delete weights;
	delete biases;
}

void LayerLinear::init(Rand* rng)
{
    for(size_t i = 0; i < weights->rows(); i++)
        for(size_t j = 0; j < weights->cols(); j++)
            (*weights)[i][j] = std::max(0.03, 1.0/(weights->cols())) * rng->normal();

    for(size_t b = 0; b < biases->size(); b++)
        (*biases)[b] = std::max(0.03, 1.0/(weights->cols())) * rng->normal();
}

void LayerLinear::activate(const Vec& x)
{
    for(size_t i = 0; i < numOutputs; i++)
        (*activation)[i] = (*weights)[i].dotProduct(x) + (*biases)[i];
}

void LayerLinear::backprop(Vec& prevBlame)
{
    Matrix* column = blame->toColumnVector();
    Matrix* product = Matrix::multiply(*weights, *column, true, false);
    Vec* flattened = product->toVec();

    prevBlame.copy(*flattened);

    delete column;
    delete product;
    delete flattened;
}

void LayerLinear::update_gradient(const Vec& x, Matrix& gradient, Vec& bias_gradient)
{
    bias_gradient += *blame;
    gradient += Vec::outerProduct(*blame, x);
}

//virtual
void LayerLinear::reset()
{
    weights->fill(0.0);
    biases->fill(0.0);
    activation->fill(0.0);
    blame->fill(0.0);
}

void LayerLinear::ordinary_least_squares(const Matrix& x, const Matrix& y)
{
    //Gashler's equation
    Vec x_centroid(x.cols());
    x_centroid.fill(0.0);
    Vec y_centroid(y.cols());
    y_centroid.fill(0.0);

    //Compute X centroid
    for(size_t i = 0; i < x.rows(); i++)
        x_centroid += x[i];
    x_centroid /= x.rows();

    //Compute Y centroid
    for(size_t i = 0; i < y.rows(); i++)
        y_centroid += y[i];
    y_centroid /= y.rows();

    //Create matrices to hold everything
    Matrix left(y.cols(), x.cols());
    left.fill(0.0);
    Matrix right(x.cols(), x.cols());
    right.fill(0.0);
    Matrix theta(x.cols(), y.cols());
    theta.fill(0.0);

    //Compute M
    for(size_t i = 0; i < x.rows(); i++)
    {
        left += Vec::outerProduct(y[i] - y_centroid, x[i] - x_centroid);
        right += Vec::outerProduct(x[i] - x_centroid, x[i] - x_centroid);
    }
    theta = *Matrix::multiply(left, *right.pseudoInverse(), false, false);

    //Compute B
    Matrix b(x.rows(), 1);
    b = *y_centroid.toColumnVector() - *Matrix::multiply(theta, *x_centroid.toColumnVector(), false, false);

    //Fill weights and biases
    for(size_t i = 0; i < numOutputs; i++)
        for(size_t j = 0; j < numInputs; j++)
            (*weights)[i][j] = theta[i][j];

    for(size_t i = 0; i < numOutputs; i++)
        (*biases)[i] = b[i][0];

    //Normal equation
    //Matrix theta = Matrix::multiply(Matrix::multiply(*Matrix::multiply(x, x, true, false)->pseudoInverse(), x, false, true), y, false, false);
    //theta.print(std::cout);
}
