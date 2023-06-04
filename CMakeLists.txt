# cmake_minimum_required(VERSION 3.12)
project(kobra CXX CUDA)

# CXX options
set(CMAKE_CXX_STANDARD 17)

# Compiler (clang default)
if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
endif()

# Color flags
set(COLOR_FLAGS -fdiagnostics-color=always)

# Add flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Disabling warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

# Vulkan
find_package(Vulkan REQUIRED)

if (Vulkan_FOUND)
    message(STATUS "Vulkan found")
    message(STATUS "Vulkan include dir: ${Vulkan_INCLUDE_DIR}")
    message(STATUS "Vulkan libraries: ${Vulkan_LIBRARIES}")
else()
    message(FATAL_ERROR "Vulkan not found")
endif()

# GLFW
find_package(glfw3 REQUIRED)

if (glfw3_FOUND)
    message(STATUS "GLFW found")
    message(STATUS "GLFW include dir: ${glfw3_INCLUDE_DIR}")
    message(STATUS "GLFW libraries: ${glfw3_LIBRARIES}")
else()
    message(FATAL_ERROR "GLFW not found")
endif()

# OpenCV
find_package(OpenCV COMPONENTS core videoio REQUIRED)

if (OpenCV_FOUND)
    message(STATUS "OpenCV found")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# Add NativeFileDialog extended
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/nativefiledialog-extended)

# Macros
add_compile_definitions(KOBRA_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Macros for ImageMagick
add_compile_definitions(MAGICKCORE_QUANTUM_DEPTH=16)
add_compile_definitions(MAGICKCORE_HDRI_ENABLE=0)

# Include directories
include_directories(
	${CMAKE_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/freetype/include
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glm
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/examples
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/json/single_include
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/nativefiledialog-extended/src/include
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/optix
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/taskflow
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/termcolor/include
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tinyexr/deps/miniz
	${OpenCV_INCLUDE_DIRS}
	${ImageMagick_INCLUDE_DIRS}
)

# Common sources
set(ImGui_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_draw.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_widgets.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_demo.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_tables.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/backends/imgui_impl_glfw.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/backends/imgui_impl_vulkan.cpp
)

set(ImExtra_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/implot/implot.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/implot/implot_items.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/implot/implot_demo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ImGuizmo/ImGuizmo.cpp
)

set(Glslang_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glslang/StandAlone/ResourceLimits.cpp)

file(GLOB Kobra_GENERIC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
file(GLOB Kobra_LAYERS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/layers/*.cpp)
file(GLOB Kobra_DAEMONS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/daemons/*.cpp)
file(GLOB Kobra_IO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/io/*.cpp)
file(GLOB Kobra_ENGINE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/engine/*.cpp)

# Exclude raster.cpp
list(REMOVE_ITEM Kobra_LAYERS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/layers/raster.cpp)

set(Kobra_SOURCES
	${Kobra_GENERIC_SOURCES}
	${Kobra_LAYERS_SOURCES}
        ${Kobra_DAEMONS_SOURCES}
	${Kobra_IO_SOURCES}
	${Kobra_ENGINE_SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/core.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/layers/denoiser.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/amadeus/armada.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/daemons/mesh.cu
)

# Create object library for the shared source
add_library(Kobra_COMMON OBJECT
	${Kobra_SOURCES}
	${ImGui_SOURCES}
        ${ImExtra_SOURCES}
	${Glslang_SOURCES}
)

# Set executable sources -- Editor
add_executable(editor
        editor.cu
        editor/editor_viewport.cu
        editor/inspector.cpp
        editor/mamba.cu
        editor/material_preview.cpp
        editor/scene_graph.cpp
        editor/shaders.cpp
        editor/sparse_gi.cu
        editor/startup.cpp
        editor/ui/material_editor.cpp
        editor/viewport_configure.cu
        editor/viewport_render.cu
        $<TARGET_OBJECTS:Kobra_COMMON>
)

target_link_libraries(editor
	${Vulkan_LIBRARIES}
	glfw
	glslang
	SPIRV
	assimp
	nvidia-ml
	nvrtc
	${OpenCV_LIBS}
	${ImageMagick_LIBRARIES}
	nfd
)

# Set executable sources -- experimental
add_executable(api
        experimental/api/main.cpp
        $<TARGET_OBJECTS:Kobra_COMMON>
)

target_link_libraries(api
	${Vulkan_LIBRARIES}
	glfw
	glslang
	SPIRV
	assimp
	nvidia-ml
	nvrtc
	${OpenCV_LIBS}
	${ImageMagick_LIBRARIES}
	nfd
)

# Set executable sources -- experimental
add_executable(snerf
        experimental/snerf/snerf.cu
        $<TARGET_OBJECTS:Kobra_COMMON>
)

target_link_libraries(snerf
	${Vulkan_LIBRARIES}
	glfw
	glslang
	SPIRV
	assimp
	nvidia-ml
	nvrtc
	${OpenCV_LIBS}
	${ImageMagick_LIBRARIES}
	nfd
)

# Extra outputs
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiling OptiX shaders
# TODO: compile at run time using NVRTC
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ptx)

file(GLOB_RECURSE OPTIX_SHADERS
	${CMAKE_CURRENT_SOURCE_DIR}/editor/gbuffer_rtx_shader.cu
	${CMAKE_CURRENT_SOURCE_DIR}/editor/optix/mamba_shader.cu
	${CMAKE_CURRENT_SOURCE_DIR}/editor/optix/sparse_gi_shader.cu
	${CMAKE_CURRENT_SOURCE_DIR}/editor/path_tracer.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/amadeus_path_tracer.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/amadeus_repg.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/amadeus_restir.cu
)

add_library(optix_shaders
	OBJECT ${OPTIX_SHADERS}
)

set_target_properties(optix_shaders PROPERTIES
        # CUDA_OPTIX_IR_COMPILATION ON
	COMPILE_FLAGS "-optix-ir -lineinfo"
)

# Fake target to copy shader outputs to bin folder
# TODO: Any cleaner way of doing this? (run a script for example)
add_custom_target(
	optix_shaders_copy
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/optix_shaders.dir/source/optix
		${CMAKE_BINARY_DIR}/bin/ptx
	COMMAND ${CMAKE_COMMAND} -E copy
		${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/optix_shaders.dir/editor/*.ptx
		${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/optix_shaders.dir/editor/optix/*.ptx
		${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/optix_shaders.dir/editor/*.o
		${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/optix_shaders.dir/editor/optix/*.o
		${CMAKE_BINARY_DIR}/bin/ptx
	DEPENDS optix_shaders
)

add_dependencies(editor optix_shaders_copy)

# Compile plugins as well
include(cmake/armada_rtx_plugins.cmake)
