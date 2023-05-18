#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace rd {

template <typename T>
__global__ void reduce0(T *in, T *out, uint32_t numel);

}
