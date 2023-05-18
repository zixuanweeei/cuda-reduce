#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rd {

struct device_storage {
  device_storage() = default;
  device_storage(size_t size) {
    union {
      void *void_p;
      uint8_t *u8_p;
    } data_p;

    cudaMalloc(&data_p.void_p, size);
    _data.reset(data_p.u8_p, device_storage::_device_storage_free);
  }

  void *data() const { return _data.get(); }

  template <typename T>
  T *data() const {
    return reinterpret_cast<T *>(_data.get());
  }

  operator bool() { return nullptr == _data.get(); }

 private:
  std::shared_ptr<uint8_t> _data {nullptr};

  static void _device_storage_free(void *p) {
    if (p) { cudaFree(p); }
  }
};

template <typename T>
inline cudaError_t make_device_storage(device_storage &storage, size_t numel) {
  storage = device_storage(numel * sizeof(T));
  return cudaGetLastError();
}

} // namespace rd
