#pragma once

#include <cstdint>

namespace rd {

template <typename T>
void cuda_reduce(int size, int threads, int blocks, int whichKernel,
    T *d_idata, T *d_odata);

template <typename T>
void cuda_reduce_arbirary_threads(int size, int threads, int blocks,
    int whichKernel, T *d_idata, T *d_odata);

} // namespace rd
