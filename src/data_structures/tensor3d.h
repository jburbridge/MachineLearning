#ifndef TENSOR3D_H
#define TENSOR3D_H

#include <vector>
#include "vec.h"

class Tensor3D
{
    public:
        Tensor3D();
        Tensor3D(std::vector<size_t> _dims);
        Tensor3D(const Vec& _data, std::vector<size_t> _dims);
        ~Tensor3D();

        double& at(size_t row, size_t col, size_t depth);
        static void convolve(const Tensor3D& in, const Tensor3D& filter, Tensor3D& out, bool flipFilter = false, size_t stride = 1);

        Vec* get_data();
        size_t size();
        void print();
        double max();
        Vec* copy(size_t start = 0, size_t len = (size_t) - 1);
        void operator+=(const Tensor3D& that);
        void operator*=(const double scalar);

        static size_t countTensorSize(std::vector<size_t> _dims);
        static void test();

    private:
        Vec* data;
        std::vector<size_t> dims;
};

#endif
