#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#include "common/helper_cuda.h"
#include "common/helper_string.h"
#include "cpu_reduce.hh"
#include "cuda_reduce.hh"
#include "device_storage.hh"
#include "utils.hh"

namespace rd {

template <typename T>
bool do_reduce(int argc, char **argv) {
  int size = 1 << 24; // number of elements to reduce
  int max_threads = 256; // number of threads per block
  int which_kernel = 0; // kernel to be used
  int max_blocks = 64;
  bool cpu_final_reduction = false;
  int cpu_final_threshold = 1;

  if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
    which_kernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
    max_threads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "max_blocks")) {
    max_blocks
        = getCmdLineArgumentInt(argc, (const char **)argv, "max_blocks");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
    size = getCmdLineArgumentInt(argc, (const char **)argv, "n");
  }

  printf("which kernel: %d\n", which_kernel);
  fflush(stdout);

  // initialize random data on host
  std::vector<T> host_input(size, (T)0.f);
  for (size_t i = 0; i < host_input.size(); ++i) {
    host_input[i] = std::sin((T)i);
  }

  int num_blocks = 0;
  int num_threads = 0;
  get_num_blocks_and_threads(
      which_kernel, size, max_blocks, max_threads, num_blocks, num_threads);

  if (num_blocks == 1) { cpu_final_threshold = 1; }

  // allocate memory for the result on host side
  std::vector<T> host_output(num_blocks, (T)0.f);

  // allocate device memory
  device_storage ds_input;
  checkCudaErrors(make_device_storage<T>(ds_input, size));

  device_storage ds_output;
  checkCudaErrors(make_device_storage<T>(ds_output, num_blocks));

  // copy data to device memory directly
  checkCudaErrors(cudaMemcpy(ds_input.data(), host_input.data(),
      size * sizeof(T), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ds_output.data(), host_input.data(),
      num_blocks * sizeof(T), cudaMemcpyHostToDevice));

  cuda_reduce<T>(size, num_threads, num_blocks, which_kernel,
      ds_input.data<T>(), ds_output.data<T>());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(host_output.data(), ds_output.data(),
      num_blocks * sizeof(T), cudaMemcpyDeviceToHost));

  T gpu_result
      = std::accumulate(host_output.begin(), host_output.end(), T(0.f));

  T cpu_result = cpu_reduce(host_input.data(), size);

  printf("\nGPU result = %f\n", (double)gpu_result);
  printf("CPU result = %f\n", (double)cpu_result);

  return std::abs(gpu_result - cpu_result) < 5e-4;
}

} // namespace rd

int main(int argc, char **argv) {
  bool reduce_result = false;
  reduce_result = rd::do_reduce<float>(argc, argv);
  printf(reduce_result ? "Test passed\n" : "Test failed!\n");

  return reduce_result ? 0 : 233;
}
