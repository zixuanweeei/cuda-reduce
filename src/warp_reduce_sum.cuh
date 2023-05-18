#pragma once

#include <cstddef>
#include <crt/device_functions.h>

namespace rd {

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(uint32_t mask, T acc) {
  for (int off = warpSize / 2; off > 0; off /= 2) {
    acc += __shfl_down_sync(mask, acc, offset);
  }
  return acc;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-reduce-functions
// Specialize warp_reduce_sum for int inputs to use __reduce_add_sync intrinsic
// on SM 8.0 or higher.
//
// The `__reduce_sync(unsigned mask, T value)` intrinsics perform a reduction
// operation on the data provided in `value` after synchronizing threads named
// in mask. `T` can be unsigned or signed for {add, min, max} and unsigned only
// for {and, or, xor} operations.
template <>
__device__ __forceinline__ int warp_reduce_sum<int>(uint32_t mask, int acc) {
  acc = __reduce_add_sync(mask, acc);
  return acc;
}
#endif

} // namespace rd
