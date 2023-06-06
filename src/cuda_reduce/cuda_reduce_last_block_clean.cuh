#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>

#include "warp_reduce_sum.cuh"

namespace rd {

__device__ static uint32_t semaphore = 0;
__shared__ bool is_last_block_done;

template <typename T>
__device__ void reduce_last_block_clean(T *smem, T *in) {
  uint32_t segment_size = gridDim.x;
  uint32_t this_block_size = blockDim.x;
  uint32_t tid = threadIdx.x;

  if (threadIdx.x == 0) {
    // Thread 0 makes sure that the incrementation of the "semaphore" variable
    // is only performed after the partial sum has been written to global
    // memory.
    __threadfence();
    // Thread 0 signals that it is done.
    uint32_t value = atomicInc(&semaphore, gridDim.x);
    // Thread 0 determines if its block is the last block to be done.
    is_last_block_done = (value == (gridDim.x - 1));
  }

  __syncthreads();

  if (!is_last_block_done) return;

  uint32_t mask_len = (this_block_size & 31);
  mask_len = (mask_len > 0) ? (32 - mask_len) : mask_len;
  const uint32_t mask = (0xff'ff'ff'ff) >> mask_len;

  T local_sum = 0;
  uint32_t off = tid;
  while (off < segment_size) {
    local_sum += in[off];
    off += this_block_size;
  }
  local_sum = warp_reduce_sum<T>(mask, local_sum);

  // Each warp puts its local sum into shared memory in the first thread
  if ((tid % warpSize) == 0) { smem[tid / warpSize] = local_sum; }

  __syncthreads();

  const uint32_t shmem_extent
      = (this_block_size / warpSize) > 0 ? (this_block_size / warpSize) : 1;
  const uint32_t ballot_result = __ballot_sync(mask, tid < shmem_extent);
  if (tid < shmem_extent) {
    local_sum = smem[tid];
    local_sum = warp_reduce_sum<T>(ballot_result, local_sum);
  }

  if (tid == 0) {
    *in = local_sum;
    semaphore = 0;
  }
}

} // namespace rd
