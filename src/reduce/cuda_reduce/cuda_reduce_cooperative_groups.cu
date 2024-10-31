#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdint>
#include <cstdio>

#include "cuda_reduce/cuda_reduce.cuh"
#include "cuda_reduce/cuda_reduce_last_block_clean.cuh"
#include "shared_memory.cuh"
#include "utils.hh"
#include "warp_reduce_sum.cuh"

namespace cg = cooperative_groups;

namespace rd {

// Performs a reduction step and updates numTotal with how many are remaining
template <typename T, typename Group>
__device__ T cg_reduce_n(T in, Group &threads) {
  return cg::reduce(threads, in, cg::plus<T>());
}

template <typename T>
__global__ void cg_reduce(
    T *in, T *out, uint32_t numel, bool clean_final_block) {
  T *smem = shared_memory_t<T> {};
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Handle to tile in thread block
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

  uint32_t cta_size = cta.size();
  uint32_t num_ctas = gridDim.x;
  uint32_t thread_rank = cta.thread_rank();
  uint32_t thread_idx = (blockIdx.x * cta_size) + thread_rank;

  T local_sum = 0;
  {
    uint32_t i = thread_idx;
    uint32_t i_stride = num_ctas * cta_size;
    while (i < numel) {
      local_sum += in[i];
      i += i_stride;
    }
    smem[thread_rank] = local_sum;
  }

  // Wait all tiles to finish and reduce within CTA
  {
    uint32_t cta_strides = cta_size >> 1;
    while (cta_strides >= 32) {
      cta.sync();
      if (thread_rank < cta_strides) {
        local_sum += smem[thread_rank + cta_strides];
        smem[thread_rank] = local_sum;
      }

      cta_strides >>= 1;
    }
  }

  // shuffle redux instead of smem redux
  {
    cta.sync();
    if (tile.meta_group_rank() == 0) {
      local_sum = cg_reduce_n(local_sum, tile);
    }
  }

  if (thread_rank == 0) out[blockIdx.x] = local_sum;
  if (clean_final_block) reduce_last_block_clean(smem, out);
}

template __global__ void cg_reduce<float>(
    float *in, float *out, uint32_t numel, bool);
template __global__ void cg_reduce<int32_t>(
    int32_t *in, int32_t *out, uint32_t numel, bool);
template __global__ void cg_reduce<double>(
    double *in, double *out, uint32_t numel, bool);

template <class T, size_t BlockSize, size_t MultiWarpGroupSize>
__global__ void multi_warp_cg_reduce(T *in, T *out, uint32_t n) {
  // Shared memory for intermediate steps
  T *sdata = shared_memory_t<T>();
  __shared__ cg::block_tile_memory<BlockSize> scratch;

  // Handle to thread block group
  auto cta = cg::this_thread_block(scratch);
  // Handle to multi_warp_tile in thread block
  auto multi_warp_tile = cg::tiled_partition<MultiWarpGroupSize>(cta);

  uint32_t grid_size = BlockSize * gridDim.x;
  T local_sum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger grid_size and therefore fewer elements per thread
  int nIsPow2 = !(n & n - 1);
  if (nIsPow2) {
    uint32_t i = blockIdx.x * BlockSize * 2 + threadIdx.x;
    grid_size = grid_size << 1;

    while (i < n) {
      local_sum += in[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + BlockSize) < n) { local_sum += in[i + blockDim.x]; }
      i += grid_size;
    }
  } else {
    uint32_t i = blockIdx.x * BlockSize + threadIdx.x;
    while (i < n) {
      local_sum += in[i];
      i += grid_size;
    }
  }

  local_sum = cg_reduce_n(local_sum, multi_warp_tile);

  if (multi_warp_tile.thread_rank() == 0) {
    sdata[multi_warp_tile.meta_group_rank()] = local_sum;
  }
  cg::sync(cta);

  if (threadIdx.x == 0) {
    local_sum = 0;
    for (int i = 0; i < multi_warp_tile.meta_group_size(); i++) {
      local_sum += sdata[i];
    }
    out[blockIdx.x] = local_sum;
  }
}

} // namespace rd
