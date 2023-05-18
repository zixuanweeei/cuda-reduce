#include "cuda_reduce.hh"
#include "cuda_reduce/cuda_reduce.cuh"

namespace rd {

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
