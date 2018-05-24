#include "vec.h"
#include "vecwrapper.h"

VecWrapper::VecWrapper(const Vec& vec, size_t start, size_t len)
: Vec(0)
{
    setData(vec, start, len);
}

VecWrapper::VecWrapper(double* buf, size_t size)
: Vec(0)
{
    setData(buf, size);
}

VecWrapper::~VecWrapper()
{
    m_data = NULL;
    m_size = 0;
}

void VecWrapper::setData(const Vec& vec, size_t start, size_t len)
{
    m_data = vec.m_data + start;
    m_size = std::min(len, vec.size() - start);
}

void VecWrapper::setData(double* buf, size_t size)
{
    m_data = buf;
    m_size = size;
}

void VecWrapper::setSize(size_t size)
{
    m_size = size;
}
