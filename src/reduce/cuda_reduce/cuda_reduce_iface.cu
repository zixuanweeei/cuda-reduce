#include <cassert>
#include <cstdint>
#include <cstdio>

#include "common/helper_cuda.h"
#include "cuda_reduce.hh"
#include "cuda_reduce/cuda_reduce.cuh"
#include "utils.hh"

namespace rd {

template <typename T, int kern_no, uint32_t threads>
struct reduce_kern_exec {
  void operator()(dim3 grid_dim, dim3 block_dim, uint32_t smem_size, T *in,
      T *out, int32_t numel, bool clean_final_block) {
    reduce<T, threads, kern_no> {}.doit(
        grid_dim, block_dim, smem_size, in, out, numel, clean_final_block);
  }
};

template <typename T, int kern_no = 4, int max_blocks = 512>
void cuda_reduce_arbirary_blocks(int numel, int num_threads, int num_blocks,
    T *in, T *out, bool clean_final_block) {
  // launch config
  dim3 dim_block(num_threads, 1, 1);
  dim3 dim_grid(num_blocks, 1, 1);

  int32_t smem_size = (num_threads <= 32) ? (2 * num_threads * sizeof(T))
                                          : (num_threads * sizeof(T));

  if (max_blocks >= 1024 && num_threads == 1024) {
    reduce_kern_exec<T, kern_no, 1024> {}(
        dim_grid, dim_block, smem_size, in, out, numel, clean_final_block);
    return;
  }

  switch (num_threads) {
    case 512:
      reduce_kern_exec<T, kern_no, 512> {}(
          dim_grid, dim_block, smem_size, in, out, numel, clean_final_block);
      break;
    case 256:
      reduce_kern_exec<T, kern_no, 256> {}(
          dim_grid, dim_block, smem_size, in, out, numel, clean_final_block);
      break;
    case 128:
      reduce_kern_exec<T, kern_no, 128> {}(
          dim_grid, dim_block, smem_size, in, out, numel, clean_final_block);
      break;
    case 64:
      reduce_kern_exec<T, kern_no, 64> {}(
          dim_grid, dim_block, smem_size, in, out, numel, clean_final_block);
      break;
    case 32:
      reduce_kern_exec<T, kern_no, 32> {}(
          dim_grid, dim_block, smem_size, in, out, numel, clean_final_block);
      break;
    case 16:
      reduce_kern_exec<T, kern_no, 16> {}(
          dim_grid, dim_block, smem_size, in, out, numel, clean_final_block);
      break;
    case 8:
      reduce_kern_exec<T, kern_no, 8> {}(
          dim_grid, dim_block, smem_size, in, out, numel, clean_final_block);
      break;
    case 4:
      reduce_kern_exec<T, kern_no, 4> {}(
          dim_grid, dim_block, smem_size, in, out, numel, clean_final_block);
      break;
    case 2:
      reduce_kern_exec<T, kern_no, 2> {}(
          dim_grid, dim_block, smem_size, in, out, numel, clean_final_block);
      break;
    case 1:
      reduce_kern_exec<T, kern_no, 1> {}(
          dim_grid, dim_block, smem_size, in, out, numel, clean_final_block);
      break;

    default: assert(!"not implemented");
  }
}

template <typename T>
void cuda_reduce(int32_t numel, int32_t num_threads, int32_t num_blocks,
    int32_t which_kern, T *in, T *out, bool clean_final_block) {
  // launch config
  dim3 dim_block(num_threads, 1, 1);
  dim3 dim_grid(num_blocks, 1, 1);

  int32_t smem_size = (num_threads <= 32) ? (2 * num_threads * sizeof(T))
                                          : (num_threads * sizeof(T));

  CP_TIMEIT(0, {
    switch (which_kern) {
      case 0:
        reduce0<T><<<dim_grid, dim_block, smem_size>>>(
            in, out, numel, clean_final_block);
        break;
      case 1:
        reduce1<T><<<dim_grid, dim_block, smem_size>>>(
            in, out, numel, clean_final_block);
        break;
      case 2:
        reduce2<T><<<dim_grid, dim_block, smem_size>>>(
            in, out, numel, clean_final_block);
        break;
      case 3:
        reduce3<T><<<dim_grid, dim_block, smem_size>>>(
            in, out, numel, clean_final_block);
        break;
      case 4:
        cuda_reduce_arbirary_blocks<T, 4, 1024>(
            numel, num_threads, num_blocks, in, out, clean_final_block);
        break;
      case 5:
        cuda_reduce_arbirary_blocks<T, 5, 1024>(
            numel, num_threads, num_blocks, in, out, clean_final_block);
        break;
      case 6:
        cuda_reduce_arbirary_blocks<T, 6, 1024>(
            numel, num_threads, num_blocks, in, out, clean_final_block);
        break;
      case 7:
        cuda_reduce_arbirary_blocks<T, 7, 1024>(
            numel, num_threads, num_blocks, in, out, clean_final_block);
        break;
      case 8:
        cg_reduce<T><<<dim_grid, dim_block, smem_size>>>(
            in, out, numel, clean_final_block);
        break;

      default: assert(!"not implemented");
    }
  });

  getLastCudaError("kernel execution failed");
}

template void cuda_reduce<int32_t>(int32_t numel, int32_t num_threads,
    int32_t num_blocks, int32_t which_kern, int32_t *in, int32_t *out, bool);
template void cuda_reduce<float>(int32_t numel, int32_t num_threads,
    int32_t num_blocks, int32_t which_kern, float *in, float *out, bool);
template void cuda_reduce<double>(int32_t numel, int32_t num_threads,
    int32_t num_blocks, int32_t which_kern, double *in, double *out, bool);

} // namespace rd
