#include <cmath>
#include <vector>
#include "matrix.h"
#include "tensor.h"
#include "vec.h"
#include "vecwrapper.h"

//Empty constructor...not sure what to do with this, but the compiler was getting mad that it didn't exist
Tensor::Tensor()
{

}

Tensor::Tensor(const Vec& vals, const std::vector<size_t> _dims)
: VecWrapper(vals)
{
	size_t tot = 1;
	for(size_t i = 0; i < _dims.size(); i++)
	{
		dims.push_back(_dims[i]);
		tot *= _dims[i];
	}

	if(tot != vals.size())
		throw Ex("Mismatching sizes. Vec has ", to_str(m_size), ", Tensor has ", to_str(tot));
}

Tensor::Tensor(const Tensor& copyMe)
: VecWrapper(copyMe.m_data, copyMe.m_size)
{
	for(size_t i = 0; i < copyMe.dims.size(); i++)
		dims.push_back(copyMe.dims[i]);
}

Tensor::Tensor(double* buf, std::vector<size_t> _dims)
: VecWrapper(buf, buf ? countTensorSize(_dims) : 0)
{
	size_t tot = 1;
	for(size_t i = 0; i < _dims.size(); i++)
	{
		dims.push_back(_dims[i]);
		tot *= _dims[i];
	}

	if(tot != m_size)
		throw Ex("Mismatching sizes. Vec has ", to_str(m_size), ", Tensor has ", to_str(tot));
}

Tensor::~Tensor()
{

}

void Tensor::print()
{
	for(size_t i = 0; i < dims[0]; i++)
	{
		for(size_t j = 0; j < dims[1]; j++)
			std::cout << m_data[i*dims[1]+j] << ",";
		std::cout << "\n";
	}
}

Vec* Tensor::toVec()
{
	Vec* v = new Vec(countTensorSize(dims));
	for(size_t i = 0; i < v->size(); i++)
		(*v).m_data[i] = m_data[i];
	return v;
}

void Tensor::convolve(const Tensor& in, const Tensor& filter, Tensor& out, bool flipFilter, size_t stride)
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
			val += (in[ip] * filter[flipFilter ? filterTail - fp : fp]);

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
		out[op] += val;

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

void Tensor::convolve2(const Tensor& in, const Tensor& filter, Tensor& out, size_t padding, size_t stride)
{
	//New plan: in this function, break the tensor into a vector of matrices, add the padding, do convolution
	Matrix* m_in = in.toMatrix(in.dims[0]);
	Matrix* m_f = filter.toMatrix(filter.dims[0]);
	Matrix* m_out = out.toMatrix(out.dims[0]);

	std::vector<Matrix*> image;
	std::vector<Matrix*> kernal;
	std::vector<Matrix*> output;

	//For each layer (color channel)
	for(size_t i = 0; i < in.dims[2]; i++)
	{
		//Create each layer
		Matrix* input_channel = new Matrix(in.dims[0] + 2 * padding, in.dims[1] + 2 * padding);
		Matrix* filter_layer = new Matrix(filter.dims[0], filter.dims[1]);

		//Fill with 0.0
		input_channel->fill(0.0);
		filter_layer->fill(0.0);

		//Copy vals from tensor to layer (adding padding for image)
		input_channel->copyBlock(padding, padding, *m_in, i * in.dims[0], 0, in.dims[0], in.dims[1]);
		filter_layer->copyBlock(0, 0, *m_f, i * filter.dims[0], 0, filter.dims[0], filter.dims[1]);

		//Add layer to the vector
		image.push_back(input_channel);
		kernal.push_back(filter_layer);
	}

	//Now you can use triple indices [][][] and everything is the correct size
	//You're welcome.
	//Goodnight.

	if(in.dims.size() != 3 || filter.dims.size() != 3 || out.dims.size() != 2)
		throw Ex("Input and filter should be 3D tensors, which makes the output 2D");

	for(size_t i = 0; i < out.dims[0]; i++)
	{
		for(size_t j = 0; j < out.dims[1]; j++)
		{
			double out_pixel = 0.0;

			for(size_t z = 0; z < filter.dims[2]; z++)
			{
				for(size_t x = 0; x < filter.dims[0]; x++)
				{
					for(size_t y = 0; y < filter.dims[1]; y++)
					{
						//std::cout << "i: " << i << ", j: " << j << ", z: " << z << ", x: " << x << ", y: " << y << "\n";
						//std::cout << "Index: " << i * stride * in.dims[0] + j * stride + z*filter.dims[2] + x*filter.dims[0] + y << "\n";
						size_t in_index = i * stride * in.dims[0] + j * stride + z*filter.dims[2] + x*filter.dims[0] + y;
						size_t filter_index = z*filter.dims[2] + x*filter.dims[0] + y;

						out_pixel += in[in_index] * filter[filter_index];
					}
				}
			}
			out[i*out.dims[0] + j] = out_pixel;
		}
	}


}

//static
size_t Tensor::countTensorSize(std::vector<size_t> _dims)
{
	size_t n = 1;
	for(size_t i = 0; i < _dims.size(); i++)
		n *= _dims[i];
	return n;
}

// static
void Tensor::test()
{
	{
		// 1D test
		Vec in({2,3,1,0,1});
		Tensor tin(in, {5});

		Vec k({1, 0, 2});
		Tensor tk(k, {3});

		Vec out(7);
		Tensor tout(out, {7});

		Tensor::convolve(tin, tk, tout, true, 1);

		//     2 3 1 0 1
		// 2 0 1 --->
		Vec expected({2, 3, 5, 6, 3, 0, 2});
		if(sqrt(out.squaredDistance(expected)) > 1e-10)
			throw Ex("wrong");
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
		Tensor tin(in, {3, 3});

		Vec k(
			{
				 1,  2,  1,
				 0,  0,  0,
				-1, -2, -1
			}
		);
		Tensor tk(k, {3, 3});

		Vec out(9);
		Tensor tout(out, {3, 3});

		Tensor::convolve(tin, tk, tout, false, 1);

		Vec expected(
			{
				-13, -20, -17,
				-18, -24, -18,
				 13,  20,  17
			}
		);
		if(sqrt(out.squaredDistance(expected)) > 1e-10)
			throw Ex("wrong");
	}
}
