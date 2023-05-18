#pragma once

#include <cstdint>

namespace rd {

uint32_t next_pow2(uint32_t x);

void get_num_blocks_and_threads(int32_t which_kern, int32_t n,
    int32_t max_blocks, int32_t max_threads, int32_t &blocks,
    int32_t &threads);

} // namespace rd
