#pragma once

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace rd {

uint32_t next_pow2(uint32_t x);

void get_num_blocks_and_threads(int32_t which_kern, int32_t n,
    int32_t max_blocks, int32_t max_threads, int32_t &blocks,
    int32_t &threads);

int get_env(const char *name, char *buffer, size_t buffer_size);

int get_env_int(const char *name, int default_value);

bool is_pow2(int32_t n);

#define CP_TIMEIT(stream, ...) \
  do { \
    if (get_env_int("CP_PROFILE", 0)) { \
      size_t warmup = 20; \
      size_t dryrun = 100; \
\
      cudaEvent_t start, stop; \
      cudaEventCreate(&start, 0); \
      cudaEventCreate(&stop, 0); \
\
      for (size_t i = 0; i < warmup; ++i) { \
        { __VA_ARGS__; } \
      } \
\
      cudaEventRecord(start, stream); \
      for (size_t i = 0; i < dryrun; ++i) { \
        { __VA_ARGS__; } \
      } \
      cudaEventRecord(stop, stream); \
      cudaEventSynchronize(stop); \
\
      float total_msec = 0.f; \
      cudaEventElapsedTime(&total_msec, start, stop); \
      printf("Elapsed time: %.4f\n", total_msec / dryrun); \
\
      cudaEventDestroy(stop); \
      cudaEventDestroy(start); \
    } else { \
      { __VA_ARGS__; } \
    } \
  } while (0)

} // namespace rd
