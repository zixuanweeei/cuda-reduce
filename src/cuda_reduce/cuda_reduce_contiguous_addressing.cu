#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "cuda_reduce/cuda_reduce.cuh"
#include "shared_memory.cuh"

namespace cg = cooperative_groups;

namespace rd {

template <typename T>
__global__ void reduce2(T *in, T *out, uint32_t numel) {
  cg::thread_block cta = cg::this_thread_block();
  T *smem = shared_memory_t<T> {};

  uint32_t tid = threadIdx.x;
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;

  smem[tid] = (i < numel) ? in[i] : 0;

  cg::sync(cta);

  for (uint32_t strides = blockDim.x / 2; strides > 0; strides >>= 1) {
    // Use sequential/contiguous addressing
    //
    // step 0: 0 1 2 3 + (4 5 6 7)
    // step 1: 0 1 + (2 3)
    // step 2: 0 + (1)
    if (tid < strides) { smem[tid] += smem[tid + strides]; }

    cg::sync(cta);
  }

  if (tid == 0) out[blockIdx.x] = smem[0];
}

template __global__ void reduce2<float>(float *in, float *out, uint32_t numel);
template __global__ void reduce2<int32_t>(
    int32_t *in, int32_t *out, uint32_t numel);
template __global__ void reduce2<double>(
    double *in, double *out, uint32_t numel);

/// Use half threads to do reduction.
template <typename T>
__global__ void reduce3(T *in, T *out, uint32_t numel) {
  cg::thread_block cta = cg::this_thread_block();
  T *smem = shared_memory_t<T> {};

  uint32_t tid = threadIdx.x;
  uint32_t i = blockDim.x * 2 * blockIdx.x + threadIdx.x;

  T sum = (i < numel) ? in[i] : 0;

  // Perform first level reduction when reading from global memory
  if (i + blockDim.x < numel) sum += in[i + blockDim.x];

  smem[tid] = sum;
  cg::sync(cta);

  for (uint32_t strides = blockDim.x / 2; strides > 0; strides >>= 1) {
    // Use sequential/contiguous addressing
    //
    // step 0: 0 1 2 3 + (4 5 6 7)
    // step 1: 0 1 + (2 3)
    // step 2: 0 + (1)
    if (tid < strides) { smem[tid] += smem[tid + strides]; }

    cg::sync(cta);
  }

  if (tid == 0) out[blockIdx.x] = smem[0];
}

template __global__ void reduce3<float>(float *in, float *out, uint32_t numel);
template __global__ void reduce3<int32_t>(
    int32_t *in, int32_t *out, uint32_t numel);
template __global__ void reduce3<double>(
    double *in, double *out, uint32_t numel);

} // namespace rd
