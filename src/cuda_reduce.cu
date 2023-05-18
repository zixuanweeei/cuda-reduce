#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

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

template <typename T>
void cuda_reduce(int32_t numel, int32_t num_threads, int32_t num_blocks,
    int32_t which_kern, T *in, T *out) {
  // launch config
  dim3 dim_block(num_threads, 1, 1);
  dim3 dim_grid(num_blocks, 1, 1);

  int32_t smem_size = (num_threads <= 32) ? (2 * num_threads * sizeof(T))
                                          : (num_threads * sizeof(T));

  reduce0<T><<<dim_grid, dim_block, smem_size>>>(in, out, numel);
}

template void cuda_reduce<int32_t>(int32_t numel, int32_t num_threads,
    int32_t num_blocks, int32_t which_kern, int32_t *in, int32_t *out);
template void cuda_reduce<float>(int32_t numel, int32_t num_threads,
    int32_t num_blocks, int32_t which_kern, float *in, float *out);
template void cuda_reduce<double>(int32_t numel, int32_t num_threads,
    int32_t num_blocks, int32_t which_kern, double *in, double *out);

} // namespace rd
