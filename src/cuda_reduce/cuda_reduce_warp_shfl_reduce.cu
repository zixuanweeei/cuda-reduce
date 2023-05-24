#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdint>

#include "cuda_reduce/cuda_reduce.cuh"
#include "shared_memory.cuh"
#include "utils.hh"

namespace cg = cooperative_groups;

namespace rd {

/// Use half threads to do reduction.
template <typename T, uint32_t block_size>
__global__ void reduce4(T *in, T *out, uint32_t numel) {
  cg::thread_block cta = cg::this_thread_block();
  T *smem = shared_memory_t<T> {};

  uint32_t tid = threadIdx.x;
  uint32_t i = blockDim.x * 2 * blockIdx.x + threadIdx.x;

  T sum = (i < numel) ? in[i] : 0;

  // Perform first level reduction when reading from global memory
  if (i + block_size < numel) sum += in[i + block_size];

  smem[tid] = sum;
  cg::sync(cta);

  for (uint32_t strides = blockDim.x / 2; strides > 32; strides >>= 1) {
    // Use sequential/contiguous addressing
    //
    // step 0: 0 1 2 3 + (4 5 6 7)
    // step 1: 0 1 + (2 3)
    // step 2: 0 + (1)
    if (tid < strides) {
      sum += smem[tid + strides];
      smem[tid] = sum;
    }

    // This synchronization happens among the reductions in multiple wraps. At
    // the last level reduction, say strides <= 32, there is only a active warp
    // deriving the very final result where the instructions are executed
    // sychronously as if SIMD instructions. Therefore, `cg::sync()` is not
    // needed.
    cg::sync(cta);
  }

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    if (block_size >= 64) sum += smem[tid + 32];

    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      sum += tile32.shfl_down(sum, offset);
    }
  }

  if (cta.thread_rank() == 0) { out[blockIdx.x] = sum; }
}

template <typename T, uint32_t block_size>
struct reduce<T, block_size, 4> {
  void doit(dim3 dim_grid, dim3 dim_block, uint32_t smem_size, T *in, T *out,
      uint32_t numel) {
    reduce4<T, block_size><<<dim_grid, dim_block, smem_size>>>(in, out, numel);
  }
};

template struct reduce<int, 512, 4>;
template struct reduce<int, 256, 4>;
template struct reduce<int, 128, 4>;
template struct reduce<int, 64, 4>;
template struct reduce<int, 32, 4>;
template struct reduce<int, 16, 4>;
template struct reduce<int, 8, 4>;
template struct reduce<int, 4, 4>;
template struct reduce<int, 2, 4>;
template struct reduce<int, 1, 4>;

template struct reduce<float, 512, 4>;
template struct reduce<float, 256, 4>;
template struct reduce<float, 128, 4>;
template struct reduce<float, 64, 4>;
template struct reduce<float, 32, 4>;
template struct reduce<float, 16, 4>;
template struct reduce<float, 8, 4>;
template struct reduce<float, 4, 4>;
template struct reduce<float, 2, 4>;
template struct reduce<float, 1, 4>;

template struct reduce<double, 512, 4>;
template struct reduce<double, 256, 4>;
template struct reduce<double, 128, 4>;
template struct reduce<double, 64, 4>;
template struct reduce<double, 32, 4>;
template struct reduce<double, 16, 4>;
template struct reduce<double, 8, 4>;
template struct reduce<double, 4, 4>;
template struct reduce<double, 2, 4>;
template struct reduce<double, 1, 4>;

/// Use half threads to do reduction. Unrolled version.
template <typename T, uint32_t block_size>
__global__ void reduce5(T *in, T *out, uint32_t numel) {
  cg::thread_block cta = cg::this_thread_block();
  T *smem = shared_memory_t<T> {};

  uint32_t tid = threadIdx.x;
  uint32_t i = blockDim.x * 2 * blockIdx.x + threadIdx.x;

  T sum = (i < numel) ? in[i] : 0;

  // Perform first level reduction when reading from global memory
  if (i + block_size < numel) sum += in[i + block_size];

  smem[tid] = sum;
  cg::sync(cta);

  if (block_size >= 512 && tid < 256) {
    sum += smem[tid + 256];
    smem[tid] = sum;
  }
  cg::sync(cta);

  if (block_size >= 256 && tid < 128) {
    sum += smem[tid + 128];
    smem[tid] = sum;
  }
  cg::sync(cta);

  if (block_size >= 128 && tid < 64) {
    sum += smem[tid + 64];
    smem[tid] = sum;
  }
  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    if (block_size >= 64) sum += smem[tid + 32];

    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      sum += tile32.shfl_down(sum, offset);
    }
  }

  if (cta.thread_rank() == 0) { out[blockIdx.x] = sum; }
}

template <typename T, uint32_t block_size>
struct reduce<T, block_size, 5> {
  void doit(dim3 dim_grid, dim3 dim_block, uint32_t smem_size, T *in, T *out,
      uint32_t numel) {
    reduce5<T, block_size><<<dim_grid, dim_block, smem_size>>>(in, out, numel);
  }
};

template struct reduce<int, 512, 5>;
template struct reduce<int, 256, 5>;
template struct reduce<int, 128, 5>;
template struct reduce<int, 64, 5>;
template struct reduce<int, 32, 5>;
template struct reduce<int, 16, 5>;
template struct reduce<int, 8, 5>;
template struct reduce<int, 4, 5>;
template struct reduce<int, 2, 5>;
template struct reduce<int, 1, 5>;

template struct reduce<float, 512, 5>;
template struct reduce<float, 256, 5>;
template struct reduce<float, 128, 5>;
template struct reduce<float, 64, 5>;
template struct reduce<float, 32, 5>;
template struct reduce<float, 16, 5>;
template struct reduce<float, 8, 5>;
template struct reduce<float, 4, 5>;
template struct reduce<float, 2, 5>;
template struct reduce<float, 1, 5>;

template struct reduce<double, 512, 5>;
template struct reduce<double, 256, 5>;
template struct reduce<double, 128, 5>;
template struct reduce<double, 64, 5>;
template struct reduce<double, 32, 5>;
template struct reduce<double, 16, 5>;
template struct reduce<double, 8, 5>;
template struct reduce<double, 4, 5>;
template struct reduce<double, 2, 5>;
template struct reduce<double, 1, 5>;

/// Use half threads to do reduction. Unrolled version + grid strides.
template <typename T, uint32_t block_size, bool is_pow2>
__global__ void reduce6(T *in, T *out, uint32_t numel) {
  cg::thread_block cta = cg::this_thread_block();
  T *smem = shared_memory_t<T> {};

  uint32_t tid = threadIdx.x;
  uint32_t grid_size = blockDim.x * gridDim.x;

  T sum = 0;

  if (is_pow2) {
    uint32_t i = blockDim.x * 2 * blockIdx.x + threadIdx.x;
    grid_size <<= 1;

    while (i < numel) {
      sum += in[i];

      if (i + block_size < numel) { sum += in[i + block_size]; }

      i += grid_size;
    }
  } else {
    uint32_t i = blockIdx.x * block_size + threadIdx.x;

    while (i < numel) {
      sum += in[i];
      i += grid_size;
    }
  }

  smem[tid] = sum;
  cg::sync(cta);

  if (block_size >= 512 && tid < 256) {
    sum += smem[tid + 256];
    smem[tid] = sum;
  }
  cg::sync(cta);

  if (block_size >= 256 && tid < 128) {
    sum += smem[tid + 128];
    smem[tid] = sum;
  }
  cg::sync(cta);

  if (block_size >= 128 && tid < 64) {
    sum += smem[tid + 64];
    smem[tid] = sum;
  }
  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    if (block_size >= 64) sum += smem[tid + 32];

    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      sum += tile32.shfl_down(sum, offset);
    }
  }

  if (cta.thread_rank() == 0) { out[blockIdx.x] = sum; }
}

template <typename T, uint32_t block_size>
struct reduce<T, block_size, 6> {
  void doit(dim3 dim_grid, dim3 dim_block, uint32_t smem_size, T *in, T *out,
      uint32_t numel) {
    if (is_pow2(numel)) {
      reduce6<T, block_size, true>
          <<<dim_grid, dim_block, smem_size>>>(in, out, numel);
    } else {
      reduce6<T, block_size, false>
          <<<dim_grid, dim_block, smem_size>>>(in, out, numel);
    }
  }
};

template struct reduce<int, 512, 6>;
template struct reduce<int, 256, 6>;
template struct reduce<int, 128, 6>;
template struct reduce<int, 64, 6>;
template struct reduce<int, 32, 6>;
template struct reduce<int, 16, 6>;
template struct reduce<int, 8, 6>;
template struct reduce<int, 4, 6>;
template struct reduce<int, 2, 6>;
template struct reduce<int, 1, 6>;

template struct reduce<float, 512, 6>;
template struct reduce<float, 256, 6>;
template struct reduce<float, 128, 6>;
template struct reduce<float, 64, 6>;
template struct reduce<float, 32, 6>;
template struct reduce<float, 16, 6>;
template struct reduce<float, 8, 6>;
template struct reduce<float, 4, 6>;
template struct reduce<float, 2, 6>;
template struct reduce<float, 1, 6>;

template struct reduce<double, 512, 6>;
template struct reduce<double, 256, 6>;
template struct reduce<double, 128, 6>;
template struct reduce<double, 64, 6>;
template struct reduce<double, 32, 6>;
template struct reduce<double, 16, 6>;
template struct reduce<double, 8, 6>;
template struct reduce<double, 4, 6>;
template struct reduce<double, 2, 6>;
template struct reduce<double, 1, 6>;

} // namespace rd
