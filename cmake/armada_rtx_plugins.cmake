project(optimized_path_tracer CUDA)

set(ARMADA_RTX_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source/amadeus)
set(PLUGIN_BUILDER ${CMAKE_CURRENT_SOURCE_DIR}/cmake/make_armada_rtx_plugin.sh)

# Set library sources -- Optimized path tracer
# TODO: also compile the optix shader...
add_library(optimized_path_tracer SHARED ${ARMADA_RTX_PLUGIN_SOURCE_DIR}/optimized_path_tracer.cu)

# After build, run script to create the plugin medatada file
add_custom_command(TARGET optimized_path_tracer POST_BUILD
	COMMAND bash ${PLUGIN_BUILDER} $<TARGET_FILE:optimized_path_tracer>
)
