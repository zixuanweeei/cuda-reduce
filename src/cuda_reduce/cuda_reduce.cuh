#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace rd {

template <typename T>
__global__ void reduce0(T *in, T *out, uint32_t numel);

template <typename T>
__global__ void reduce1(T *in, T *out, uint32_t numel);

template <typename T>
__global__ void reduce2(T *in, T *out, uint32_t numel);

template <typename T>
__global__ void reduce3(T *in, T *out, uint32_t numel);

template <typename T, uint32_t threads>
__global__ void reduce4(T *in, T *out, uint32_t numel);

template <typename T, uint32_t threads, int kern_no>
struct reduce {
  void doit(dim3 dim_grid, dim3 dim_block, uint32_t smem_size, T *in, T *out,
      uint32_t numel);
};

} // namespace rd
