cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
if(NOT DEFINED ENV{CUDACXX})
  message(
    STATUS
      "CUDACXX is not defined. Using default nvcc: /usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

set(CMAKE_CUDA_HOST_COMPILER g++ CACHE INTERNAL "" FORCE)
set(CMAKE_CXX_STANDARD 17)

project(CUDA_PROF LANGUAGES CXX CUDA)

if(POLICY CMP0104)
  cmake_policy(SET CMP0104 OLD)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

set(CMAKE_EXPORT_COMPILE_COMMANDS True)

find_package(OpenMP)
if(NOT OPENMP_FOUND)
  message(STATUS "OpenMP not found.")
else()
  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-Xcompiler=${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS})
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(src)
