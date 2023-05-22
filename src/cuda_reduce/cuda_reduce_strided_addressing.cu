#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "cuda_reduce/cuda_reduce.cuh"
#include "shared_memory.cuh"

namespace cg = cooperative_groups;

namespace rd {

template <typename T>
__global__ void reduce1(T *in, T *out, uint32_t numel) {
  cg::thread_block cta = cg::this_thread_block();
  T *smem = shared_memory_t<T> {};

  uint32_t tid = threadIdx.x;
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  smem[tid] = (i < numel) ? in[i] : 0;

  cg::sync(cta);

  for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
    // Use contiguous threads to do reduction. The addressing is strided.
    //
    // 0 1 2 3 4 5 6 7 ...
    // |/  |/  |/  |/
    // 0   2   4   6 ...
    // |  /    |  /
    // | /     | /
    // 0       4 ...
    // |....../
    // 0
    int32_t idx = 2 * stride * tid;

    // The strided addressing results in many shared memory back conflicts.
    if (idx < blockDim.x) { smem[idx] = smem[idx + stride]; }

    cg::sync(cta);
  }

  if (tid == 0) out[blockIdx.x] = smem[0];
}

template __global__ void reduce1<float>(float *in, float *out, uint32_t numel);
template __global__ void reduce1<int32_t>(
    int32_t *in, int32_t *out, uint32_t numel);
template __global__ void reduce1<double>(
    double *in, double *out, uint32_t numel);

} // namespace rd
