project(ExperimentalPathTracer CUDA)

set(ARMADA_RTX_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source/amadeus)

# Set library sources -- Optimized path tracer
add_library(ExperimentalPathTracer SHARED ${ARMADA_RTX_PLUGIN_SOURCE_DIR}/experimental_path_tracer.cu)
set_target_properties(ExperimentalPathTracer PROPERTIES PREFIX "")
set_target_properties(ExperimentalPathTracer PROPERTIES SUFFIX ".rtxa")
