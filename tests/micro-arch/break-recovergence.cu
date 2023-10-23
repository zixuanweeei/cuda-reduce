#include <cuda.h>

#include <cstdint>
#include <cstdio>

__global__ void break_recoverage_test() {
  __shared__ volatile int32_t shared_var;

  const int32_t tid = threadIdx.x;

  if (tid == 0) { shared_var = 0; }

  while (shared_var != tid)
    ;
  shared_var++;

  return;
}

int main() {
  printf("start to run test...\n");

  break_recoverage_test<<<1, 32>>>();
  cudaDeviceSynchronize();

  printf("end...\n");

  return 0;
}
