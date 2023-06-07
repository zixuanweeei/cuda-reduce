CUDA Reduce
-----------

# Introduction

This project is for personal practice on Reduce operation in CUDA language. The
algorithms implemented are mainly referenced to [cuda-samples] of the reduction
example without multiple-block cooperative groups feature. The implementations
here **finish the last mile** including:

1. synchronize across all blocks after partial results are derived in each block
1. reduce up the partial results to derive the final result in one block instead
   of doing it on CPU as the cuda-samples does

With the last mile delivery it might impact the profiling results. Thus, an
option `final_reduce` is provided to control the operation on partial results.
Please refer to the below description for the command line usages.

# Prerequisites

- CUDA Toolkit >= 11
- Environment variable:
  - `CUDACXX`: nvcc binary path to identify the CUDA Toolkit environment. The
    default value configured in project is `/usr/local/cuda/bin/nvcc`
- gcc >= 10

# Building

The project uses CMake for the build automation. With the following commands,
anyone could have the project built and get the binary `src/do_reduce` in
building directory.

```bash
mkdir -p build && cd build
cmake .. -GNinja
ninja
```

# CLI

```bash
do_reduce --kernel=<int>    # Which kernel to be used. Supports 0-8
    --threads=<int>         # Block dimension, i.e. number of threads in a block
    --max_blocks=<int>      # Maximum number of blocks to be used
    --n=<int>               # The number of elements to be reduced up
    --final_reduce=<int>    # Whether to deliver the last mile. **0**, 1.
```

Due to the limitations in CUDA architecture, we cannot specify any value for
some options. For example, the maximum number of x-axis threads in a block is
1024. Thus, `--threads=1025` is illegal.


[cuda-samples]: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction
