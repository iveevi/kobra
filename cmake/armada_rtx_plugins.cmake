project(NaivePathTracer CUDA)
project(ExperimentalPathTracer CUDA)

set(ARMADA_RTX_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source/amadeus)

# Set prefix and suffix for all shared libraries
add_library(NaivePathTracer SHARED ${ARMADA_RTX_PLUGIN_SOURCE_DIR}/naive_path_tracer.cu)
set_target_properties(NaivePathTracer PROPERTIES PREFIX "")
set_target_properties(NaivePathTracer PROPERTIES SUFFIX ".rtxa")

add_library(ExperimentalPathTracer SHARED ${ARMADA_RTX_PLUGIN_SOURCE_DIR}/experimental_path_tracer.cu)
set_target_properties(ExperimentalPathTracer PROPERTIES PREFIX "")
set_target_properties(ExperimentalPathTracer PROPERTIES SUFFIX ".rtxa")
