#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace rd {

template <typename T>
struct shared_memory_t {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

} // namespace rd
