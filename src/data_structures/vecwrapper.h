#ifndef VECWRAPPER_H
#define VECWRAPPER_H

#include "vec.h"

class VecWrapper : public Vec
{
    public:
        VecWrapper(const Vec& vec, size_t start = 0, size_t len = (size_t) - 1);

        VecWrapper(double* buf = nullptr, size_t size = 0);

        virtual ~VecWrapper();

        void setData(const Vec& vec, size_t start = 0, size_t len = (size_t) - 1);

        void setData(double* buf, size_t size);

        void setSize(size_t size);
};
#endif
