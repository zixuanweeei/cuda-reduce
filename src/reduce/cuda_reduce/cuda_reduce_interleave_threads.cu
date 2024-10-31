#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "cuda_reduce/cuda_reduce.cuh"
#include "cuda_reduce/cuda_reduce_last_block_clean.cuh"
#include "shared_memory.cuh"

namespace cg = cooperative_groups;

namespace rd {

template <typename T>
__global__ void reduce0(
    T *in, T *out, uint32_t numel, bool clean_final_block) {
  uint32_t tid = threadIdx.x;
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  T *smem = shared_memory_t<T> {};
  smem[tid] = (idx < numel) ? in[idx] : 0;

  cg::thread_block cta = cg::this_thread_block();
  cg::sync(cta);

  for (uint32_t sidx = 1; sidx < blockDim.x; sidx *= 2) {
    // Interleaving active threads using the modulo.
    //
    // 0 1 2 3 4 5 6 7 ...
    // |/  |/  |/  |/
    // 0   2   4   6 ...
    // |  /    |  /
    // | /     | /
    // 0       4 ...
    // |....../
    // 0
    //
    // In this case, no whole warps is active which causes high warp divergency.
    // Besides, the `%` is very slow.
    if ((tid % (2 * sidx)) == 0) smem[tid] += smem[tid + sidx];

    cg::sync(cta);
  }

  if (tid == 0) out[blockIdx.x] = smem[0];
  if (clean_final_block) reduce_last_block_clean(smem, out);
}

template __global__ void reduce0<float>(
    float *in, float *out, uint32_t numel, bool);
template __global__ void reduce0<int32_t>(
    int32_t *in, int32_t *out, uint32_t numel, bool);
template __global__ void reduce0<double>(
    double *in, double *out, uint32_t numel, bool);

} // namespace rd
