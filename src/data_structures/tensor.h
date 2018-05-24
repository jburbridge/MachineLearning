#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include "vec.h"
#include "vecwrapper.h"

/// A tensor class.
class Tensor : public VecWrapper
{
public:
	std::vector<size_t> dims;

	Tensor();

	/// General-purpose constructor. Example:
	/// Tensor t(v, {5, 7, 3});
	Tensor(const Vec& vals, const std::vector<size_t> _dims);

	/// Copy constructor. Copies the dimensions. Wraps the same vector.
	Tensor(const Tensor& copyMe);

	/// Constructor for dynamically generated sizes
	Tensor(double* buf, std::vector<size_t> _dims);

	/// Destructor
	virtual ~Tensor();

	void print();
	Vec* toVec();

	/// The result is added to the existing contents of out. It does not replace the existing contents of out.
	/// Padding is computed as necessary to fill the the out tensor.
	/// filter is the filter to convolve with in.
	/// If flipFilter is true, then the filter is flipped in all dimensions.
	static void convolve(const Tensor& in, const Tensor& filter, Tensor& out, bool flipFilter = false, size_t stride = 1);
	static void convolve2(const Tensor& in, const Tensor& filter, Tensor& out, size_t padding = 0, size_t stride = 1);

	static size_t countTensorSize(std::vector<size_t> _dims);

	/// Throws an exception if something is wrong.
	static void test();
};

#endif
