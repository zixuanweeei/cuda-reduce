#pragma once

#include <cstddef>
#include <cstdint>

namespace rd {

template <typename T>
T cpu_reduce(T *data, int32_t numel) {
  T acc = data[0];
  T c = (T)0.0;

  for (int i = 1; i < numel; ++i) {
    T y = data[i] - c;
    T t = acc + y;
    c = (t - acc) - y;
    acc = t;
  }

  return acc;
}

} // namespace rd
