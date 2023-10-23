file(GLOB_RECURSE sources *.cc *.cu)

foreach(src ${sources})
  file(RELATIVE_PATH src_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${src})
  string(REGEX REPLACE "[/_\\.]" "-" example_name ${src_rel_path})

  add_executable("test-${example_name}" ${src})
  target_link_libraries("test-${example_name}" PRIVATE
      ${CUDA_RUNTIME_LIBRARY})
  set_property(TARGET "test-${example_name}" PROPERTY
      CUDA_ARCHITECTURES 86 87)
endforeach()
