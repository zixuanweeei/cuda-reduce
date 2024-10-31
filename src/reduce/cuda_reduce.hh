#pragma once

#include <cstdint>

namespace rd {

template <typename T>
void cuda_reduce(int size, int threads, int blocks, int whichKernel,
    T *d_idata, T *d_odata, bool clean_final_block = false);

template <typename T>
void cuda_reduce_arbirary_threads(int size, int threads, int blocks,
    int whichKernel, T *d_idata, T *d_odata, bool clean_final_block = false);

} // namespace rd
