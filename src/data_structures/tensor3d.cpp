#include <cmath>
#include <vector>
#include "matrix.h"
#include "tensor3d.h"
#include "vec.h"

Tensor3D::Tensor3D()
{

}

Tensor3D::Tensor3D(std::vector<size_t> _dims)
{
    if(_dims.size() != 3)
        throw Ex("Tensor3D should have 3 dimensions.");

    dims = _dims;
    data = new Vec(size());
}

Tensor3D::Tensor3D(const Vec& _data, std::vector<size_t> _dims)
{
    if(_dims.size() != 3)
        throw Ex("Tensor3D should have 3 dimensions.");

    if(_dims[0] * _dims[1] * _dims[2] != _data.size())
        throw Ex("Cannot create tensor because this vec does not fit perfectly with the given dimensions.");

    dims = _dims;
    data = new Vec(0);
    data->copy(_data);
}

Tensor3D::~Tensor3D()
{
    delete data;
}

double& Tensor3D::at(size_t row, size_t col, size_t depth)
{
    return (*data)[depth * dims[0] * dims[1] + row * dims[1]  + col];
}

//static
void Tensor3D::convolve(const Tensor3D& in, const Tensor3D& filter, Tensor3D& out, bool flipFilter, size_t stride)
{
	// Precompute some values
	size_t dc = in.dims.size();
	if(dc != filter.dims.size())
		throw Ex("Expected tensors with the same number of dimensions");
	if(dc != out.dims.size())
		throw Ex("Expected tensors with the same number of dimensions");

	size_t* kinner = (size_t*)malloc(sizeof(size_t) * 5 * dc);
	size_t* kouter = kinner + dc;
	size_t* stepInner = kouter + dc;
	size_t* stepFilter = stepInner + dc;
	size_t* stepOuter = stepFilter + dc;

	// Compute step sizes
	stepInner[0] = 1;
	stepFilter[0] = 1;
	stepOuter[0] = 1;
	for(size_t i = 1; i < dc; i++)
	{
		stepInner[i] = stepInner[i - 1] * in.dims[i - 1];
		stepFilter[i] = stepFilter[i - 1] * filter.dims[i - 1];
		stepOuter[i] = stepOuter[i - 1] * out.dims[i - 1];
	}
	size_t filterTail = stepFilter[dc - 1] * filter.dims[dc - 1] - 1;

	// Do convolution
	size_t op = 0;
	size_t ip = 0;
	size_t fp = 0;
	for(size_t i = 0; i < dc; i++)
	{
		kouter[i] = 0;
		kinner[i] = 0;
		int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
		int adj = (padding - std::min(padding, (int)kouter[i])) - kinner[i];
		kinner[i] += adj;
		fp += adj * stepFilter[i];
	}
	while(true) // kouter
	{
		double val = 0.0;

		// Fix up the initial kinner positions
		for(size_t i = 0; i < dc; i++)
		{
			int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
			int adj = (padding - std::min(padding, (int)kouter[i])) - kinner[i];
			kinner[i] += adj;
			fp += adj * stepFilter[i];
			ip += adj * stepInner[i];
		}
		while(true) // kinner
		{
            //std::cout << "ip = " << ip << ", fp = " << fp << "\n";
			val += ((*in.data)[ip] * (*filter.data)[flipFilter ? filterTail - fp : fp]);

			// increment the kinner position
			size_t i;
			for(i = 0; i < dc; i++)
			{
				kinner[i]++;
				ip += stepInner[i];
				fp += stepFilter[i];
				int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
				if(kinner[i] < filter.dims[i] && kouter[i] + kinner[i] - padding < in.dims[i])
					break;
				int adj = (padding - std::min(padding, (int)kouter[i])) - kinner[i];
				kinner[i] += adj;
				fp += adj * stepFilter[i];
				ip += adj * stepInner[i];
			}
			if(i >= dc)
				break;

		}
		(*out.data)[op] += val;

		// increment the kouter position
		size_t i;
		for(i = 0; i < dc; i++)
		{
			kouter[i]++;
			op += stepOuter[i];
			ip += stride * stepInner[i];
			if(kouter[i] < out.dims[i])
				break;
			op -= kouter[i] * stepOuter[i];
			ip -= kouter[i] * stride * stepInner[i];
			kouter[i] = 0;
		}
		if(i >= dc)
			break;
	}

	free(kinner);
}

Vec* Tensor3D::get_data()
{
    Vec* out(data);
    return out;
}

size_t Tensor3D::size()
{
    return dims[0] * dims[1] * dims[2];
}

void Tensor3D::print()
{
    for(size_t d = 0; d < dims[2]; d++)
    {
        for(size_t r = 0; r < dims[0]; r++)
        {
            for(size_t c = 0; c < dims[1]; c++)
            {
                    std::cout << at(r, c, d) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

double Tensor3D::max()
{
    if(data->size() < 1)
        throw Ex("Cannot get max element of an empty vector");

    double max = (*data)[0];
    for(size_t i = 1; i < data->size(); i++)
        if((*data)[i] > max)
            max = (*data)[i];
    return max;
}

Vec* Tensor3D::copy(size_t start, size_t len)
{
    if(len == (size_t) - 1)
        len = data->size();

    Vec* c = new Vec(len);
    for(size_t i = 0; i < len; i++)
        (*c)[i] = (*data)[i + start];

    return c;
}

void Tensor3D::operator+=(const Tensor3D& that)
{
    if(Tensor3D::countTensorSize(dims) != Tensor3D::countTensorSize(that.dims))
        throw Ex("Cannot add Tensor3Ds of unequal size");

    (*data) += (*that.data);
}

void Tensor3D::operator*=(const double scalar)
{
    (*data) *= scalar;
}

//static
size_t Tensor3D::countTensorSize(std::vector<size_t> _dims)
{
    size_t total = 1;
    for(size_t i = 0; i < _dims.size(); i++)
        total *= _dims[i];
    return total;
}

// static
void Tensor3D::test()
{
	{
		// 1D test
		Vec in({2,3,1,0,1});
		Tensor3D tin(in, {5, 1, 1});

		Vec k({1, 0, 2});
		Tensor3D tk(k, {3, 1, 1});

		Vec out(7);
		Tensor3D tout(out, {7, 1, 1});

		Tensor3D::convolve(tin, tk, tout, true, 1);

		//     2 3 1 0 1
		// 2 0 1 --->
		Vec expected({2, 3, 5, 6, 3, 0, 2});
		if(sqrt(tout.get_data()->squaredDistance(expected)) > 1e-10)
			std::cout << "1D test failed\n";
        else
            std::cout << "1D test passed!\n";
	}

	{
		// 2D test
		Vec in(
			{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9
			}
		);
		Tensor3D tin(in, {3, 3, 1});

		Vec k(
			{
				 1,  2,  1,
				 0,  0,  0,
				-1, -2, -1
			}
		);
		Tensor3D tk(k, {3, 3, 1});

		Vec out(9);
		Tensor3D tout(out, {3, 3, 1});

		Tensor3D::convolve(tin, tk, tout, false, 1);

		Vec expected(
			{
				-13, -20, -17,
				-18, -24, -18,
				 13,  20,  17
			}
		);
		if(sqrt(tout.get_data()->squaredDistance(expected)) > 1e-10)
			std::cout << "2D test failed\n";
        else
            std::cout << "2D test passed!\n";
	}

    {
		// 3D test
		Vec in(
			{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,

                10, 11, 12,
                13, 14, 15,
                16, 17, 18
			}
		);
		Tensor3D tin(in, {3, 3, 2});

		Vec k(
			{
				 1,  1,
				 0,  0,

                1, 0,
                0, 0
			}
		);
		Tensor3D tk(k, {2, 2, 2});

		Vec out(4);
		Tensor3D tout(out, {2, 2, 1});

		Tensor3D::convolve(tin, tk, tout, false, 1);

		Vec expected(
			{
				13, 16,
				22, 25,
			}
		);
		if(sqrt(tout.get_data()->squaredDistance(expected)) > 1e-10)
			std::cout << "3D test failed\n";
        else
            std::cout << "3D test passed!\n";

        tout.get_data()->print(std::cout);
        std::cout << "\n";
	}
}
