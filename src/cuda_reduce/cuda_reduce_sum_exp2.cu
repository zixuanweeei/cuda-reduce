#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "cuda_reduce/cuda_reduce.cuh"
#include "shared_memory.cuh"

namespace cg = cooperative_groups;

namespace rd {

template <typename T>
__global__ void reduce0(T *in, T *out, uint32_t numel) {
  uint32_t tid = threadIdx.x;
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  T *smem = shared_memory_t<T> {};
  smem[tid] = (idx < numel) ? in[idx] : 0;

  cg::thread_block cta = cg::this_thread_block();
  cg::sync(cta);

  for (uint32_t sidx = 1; sidx < blockDim.x; sidx *= 2) {
    if ((tid % (2 * sidx)) == 0) smem[tid] += smem[tid + sidx];

    cg::sync(cta);
  }

  if (tid == 0) out[blockIdx.x] = smem[0];
}

template __global__ void reduce0<float>(float *in, float *out, uint32_t numel);
template __global__ void reduce0<int32_t>(
    int32_t *in, int32_t *out, uint32_t numel);
template __global__ void reduce0<double>(
    double *in, double *out, uint32_t numel);

} // namespace rd
