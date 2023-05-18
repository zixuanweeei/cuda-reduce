#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "common/helper_cuda.h"

namespace rd {

uint32_t next_pow2(uint32_t x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;

  return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction
// kernel For the kernels >= 3, we set threads / block to the minimum of
// max_threads and n/2. For kernels < 3, we set to the minimum of max_threads and
// n.  For kernel 6, we observe the maximum specified number of blocks, because
// each thread in that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void get_num_blocks_and_threads(int32_t which_kernel, int32_t n,
    int32_t max_blocks, int32_t max_threads, int32_t &blocks,
    int32_t &threads) {
  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int32_t device;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));

  if (which_kernel < 3) {
    threads = (n < max_threads) ? next_pow2(n) : max_threads;
    blocks = (n + threads - 1) / threads;
  } else {
    threads = (n < max_threads * 2) ? next_pow2((n + 1) / 2) : max_threads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
  }

  if ((float)threads * blocks
      > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
    printf("n is too large, please choose a smaller number!\n");
  }

  if (blocks > prop.maxGridSize[0]) {
    printf(
        "Grid size <%d> exceeds the device capability <%d>, set block size as "
        "%d (original %d)\n",
        blocks, prop.maxGridSize[0], threads * 2, threads);

    blocks /= 2;
    threads *= 2;
  }

  if (which_kernel >= 6) { blocks = std::min(max_blocks, blocks); }
  printf(
      "The selected block and thread numbers are (%d, %d)\n", blocks, threads);
}

} // namespace rd
