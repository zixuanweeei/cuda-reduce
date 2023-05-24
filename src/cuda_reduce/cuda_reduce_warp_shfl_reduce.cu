#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdint>
#include <cstdio>

#include "cuda_reduce/cuda_reduce.cuh"
#include "shared_memory.cuh"

namespace cg = cooperative_groups;

namespace rd {

/// Use half threads to do reduction.
template <typename T, uint32_t block_size>
__global__ void reduce4(T *in, T *out, uint32_t numel) {
  cg::thread_block cta = cg::this_thread_block();
  T *smem = shared_memory_t<T> {};

  uint32_t tid = threadIdx.x;
  uint32_t i = blockDim.x * 2 * blockIdx.x + threadIdx.x;

  T sum = (i < numel) ? in[i] : 0;

  // Perform first level reduction when reading from global memory
  if (i + block_size < numel) sum += in[i + block_size];

  smem[tid] = sum;
  cg::sync(cta);

  for (uint32_t strides = blockDim.x / 2; strides > 32; strides >>= 1) {
    // Use sequential/contiguous addressing
    //
    // step 0: 0 1 2 3 + (4 5 6 7)
    // step 1: 0 1 + (2 3)
    // step 2: 0 + (1)
    if (tid < strides) {
      sum += smem[tid + strides];
      smem[tid] = sum;
    }

    cg::sync(cta);
  }

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    if (block_size >= 64) sum += smem[tid + 32];

    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      sum += tile32.shfl_down(sum, offset);
    }
  }

  if (cta.thread_rank() == 0) { out[blockIdx.x] = sum; }
}

template <typename T, uint32_t block_size>
struct reduce<T, block_size, 4> {
  void doit(dim3 dim_grid, dim3 dim_block, uint32_t smem_size, T *in, T *out,
      uint32_t numel) {
    reduce4<T, block_size><<<dim_grid, dim_block, smem_size>>>(in, out, numel);
  }
};

template struct reduce<int, 512, 4>;
template struct reduce<int, 256, 4>;
template struct reduce<int, 128, 4>;
template struct reduce<int, 64, 4>;
template struct reduce<int, 32, 4>;
template struct reduce<int, 16, 4>;
template struct reduce<int, 8, 4>;
template struct reduce<int, 4, 4>;
template struct reduce<int, 2, 4>;
template struct reduce<int, 1, 4>;

template struct reduce<float, 512, 4>;
template struct reduce<float, 256, 4>;
template struct reduce<float, 128, 4>;
template struct reduce<float, 64, 4>;
template struct reduce<float, 32, 4>;
template struct reduce<float, 16, 4>;
template struct reduce<float, 8, 4>;
template struct reduce<float, 4, 4>;
template struct reduce<float, 2, 4>;
template struct reduce<float, 1, 4>;

template struct reduce<double, 512, 4>;
template struct reduce<double, 256, 4>;
template struct reduce<double, 128, 4>;
template struct reduce<double, 64, 4>;
template struct reduce<double, 32, 4>;
template struct reduce<double, 16, 4>;
template struct reduce<double, 8, 4>;
template struct reduce<double, 4, 4>;
template struct reduce<double, 2, 4>;
template struct reduce<double, 1, 4>;

} // namespace rd
