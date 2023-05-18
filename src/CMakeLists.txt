include_directories(${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/common)

file(GLOB_RECURSE sources *.cc *.cu)

add_executable(do_reduce ${sources})

target_link_libraries(do_reduce PRIVATE
  ${CUDA_RUNTIME_LIBRARY})

set_property(TARGET do_reduce PROPERTY
  CUDA_ARCHITECTURES 50 53 60 61 62 70 72 75 80 86 87)
