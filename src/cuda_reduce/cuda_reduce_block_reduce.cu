#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdint>

#include "cuda_reduce/cuda_reduce.cuh"
#include "shared_memory.cuh"
#include "utils.hh"
#include "warp_reduce_sum.cuh"

namespace cg = cooperative_groups;

namespace rd {

/// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
template <typename T, uint32_t block_size, bool is_pow2>
__global__ void reduce7(T *in, T *out, uint32_t numel) {
  cg::thread_block cta = cg::this_thread_block();
  T *smem = shared_memory_t<T> {};

  uint32_t tid = threadIdx.x;
  uint32_t grid_size = blockDim.x * gridDim.x;
  uint32_t mask_len = (block_size & 31);
  mask_len = (mask_len > 0) ? (32 - mask_len) : mask_len;
  const uint32_t mask = (0xff'ff'ff'ff) >> mask_len;

  T sum = 0;

  if (is_pow2) {
    uint32_t i = blockDim.x * 2 * blockIdx.x + threadIdx.x;
    grid_size <<= 1;

    while (i < numel) {
      sum += in[i];

      if (i + block_size < numel) { sum += in[i + block_size]; }

      i += grid_size;
    }
  } else {
    uint32_t i = blockIdx.x * block_size + threadIdx.x;

    while (i < numel) {
      sum += in[i];
      i += grid_size;
    }
  }

  sum = warp_reduce_sum<T>(mask, sum);

  // Each warp puts its local sum into shared memory in the first thread
  if ((tid % warpSize) == 0) { smem[tid / warpSize] = sum; }

  cg::sync(cta);

  const uint32_t shmem_extent
      = (block_size / warpSize) > 0 ? (block_size / warpSize) : 1;
  const uint32_t ballot_result = __ballot_sync(mask, tid < shmem_extent);
  if (tid < shmem_extent) {
    sum = smem[tid];
    sum = warp_reduce_sum<T>(ballot_result, sum);
  }

  if (cta.thread_rank() == 0) { out[blockIdx.x] = sum; }
}

template <typename T, uint32_t block_size>
struct reduce<T, block_size, 7> {
  void doit(dim3 dim_grid, dim3 dim_block, uint32_t smem_size, T *in, T *out,
      uint32_t numel) {
    smem_size = ((dim_block.x / 32) + 1) * sizeof(T);
    if (is_pow2(numel)) {
      reduce7<T, block_size, true>
          <<<dim_grid, dim_block, smem_size>>>(in, out, numel);
    } else {
      reduce7<T, block_size, false>
          <<<dim_grid, dim_block, smem_size>>>(in, out, numel);
    }
  }
};

template struct reduce<int, 1024, 7>;
template struct reduce<int, 512, 7>;
template struct reduce<int, 256, 7>;
template struct reduce<int, 128, 7>;
template struct reduce<int, 64, 7>;
template struct reduce<int, 32, 7>;
template struct reduce<int, 16, 7>;
template struct reduce<int, 8, 7>;
template struct reduce<int, 4, 7>;
template struct reduce<int, 2, 7>;
template struct reduce<int, 1, 7>;

template struct reduce<float, 1024, 7>;
template struct reduce<float, 512, 7>;
template struct reduce<float, 256, 7>;
template struct reduce<float, 128, 7>;
template struct reduce<float, 64, 7>;
template struct reduce<float, 32, 7>;
template struct reduce<float, 16, 7>;
template struct reduce<float, 8, 7>;
template struct reduce<float, 4, 7>;
template struct reduce<float, 2, 7>;
template struct reduce<float, 1, 7>;

template struct reduce<double, 1024, 7>;
template struct reduce<double, 512, 7>;
template struct reduce<double, 256, 7>;
template struct reduce<double, 128, 7>;
template struct reduce<double, 64, 7>;
template struct reduce<double, 32, 7>;
template struct reduce<double, 16, 7>;
template struct reduce<double, 8, 7>;
template struct reduce<double, 4, 7>;
template struct reduce<double, 2, 7>;
template struct reduce<double, 1, 7>;

} // namespace rd