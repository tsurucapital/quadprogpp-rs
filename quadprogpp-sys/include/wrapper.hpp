#pragma once

#include <memory>
#include "../upstream/src/Array.hh"
#include "../upstream/src/QuadProg++.hh"

namespace quadprogpp {
typedef quadprogpp::Vector<double> VectorF64;

std::unique_ptr<VectorF64>
new_vector(const unsigned int n)
{
    return std::make_unique<VectorF64>(n);
}

std::unique_ptr<VectorF64>
new_vector_from_ptr(const double* a, const unsigned int n)
{
    return std::make_unique<VectorF64>(a, n);
}

double
vector_index(const VectorF64& v, const unsigned int i)
{
    return v[i];
}

typedef quadprogpp::Matrix<double> MatrixF64;

std::unique_ptr<MatrixF64>
new_matrix_from_ptr(const double* a, const unsigned int n, const unsigned int m)
{
    return std::make_unique<MatrixF64>(a, n, m);
}
} // namespace quadprogpp
