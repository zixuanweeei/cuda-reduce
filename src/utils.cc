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

int get_env(const char *name, char *buffer, size_t buffer_size) {
  if (!name || buffer_size == 0 || !buffer) return -1;

  int writen = 0;
  int terminated_idx = 0;
  size_t value_len = 0;

  const char *value = ::getenv(name);
  value_len = value == nullptr ? 0 : strlen(value);

  if (value_len > INT_MAX) {
    // integer overflow
    writen = -1;
  } else {
    int int_value_len = static_cast<int>(value_len);
    if (value_len >= buffer_size) {
      // buffer overflow
      writen = -int_value_len;
    } else {
      terminated_idx = int_value_len;
      writen = int_value_len;

      if (value) strncpy(buffer, value, buffer_size - 1);
    }
  }

  if (buffer) buffer[terminated_idx] = '\0';
  return writen;
}

int get_env_int(const char *name, int default_value) {
  int value = default_value;
  // len(str(INT_MIN)) + terminated == 12
  constexpr int len = 12;
  char str[len] = {'\0'};

  if (get_env(name, str, len) > 0) value = std::atoi(str);
  return value;
}

} // namespace rd
