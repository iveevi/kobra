project(kobra CXX CUDA)
project(stress CXX)

# CXX options
set(CMAKE_CXX_STANDARD 17)

# Compiler (clang default)
if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
endif()

# Color flags
set(COLOR_FLAGS -fdiagnostics-color=always)

# Add flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COLOR_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${COLOR_FLAGS}\"")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=20012 --diag_suppress=20013 --diag_suppress=20014\"")

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

# ImageMagick
find_package(ImageMagick COMPONENTS Magick++ REQUIRED)

if (ImageMagick_FOUND)
    message(STATUS "ImageMagick found")
    message(STATUS "ImageMagick version: ${ImageMagick_VERSION}")
    message(STATUS "ImageMagick include path: ${ImageMagick_INCLUDE_DIRS}")
    message(STATUS "ImageMagick libraries: ${ImageMagick_LIBRARIES}")
else()
    message(FATAL_ERROR "ImageMagick not found")
endif()

# Macros
add_compile_definitions(KOBRA_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Include directories
include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glm
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/optix
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/termcolor/include
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/freetype/include
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/examples
	${OpenCV_INCLUDE_DIRS}
	${ImageMagick_INCLUDE_DIRS}
)

# Sources
set(ImGui_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_draw.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_widgets.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_demo.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_tables.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/backends/imgui_impl_glfw.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/backends/imgui_impl_vulkan.cpp
)

set(ImPlot_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/implot/implot.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/implot/implot_items.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/implot/implot_demo.cpp
)

set(Glslang_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glslang/SPIRV/GlslangToSpv.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glslang/SPIRV/SpvBuilder.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glslang/StandAlone/ResourceLimits.cpp
)

file(GLOB Kobra_GENERIC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
file(GLOB Kobra_LAYERS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/layers/*.cpp)
file(GLOB Kobra_IO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/io/*.cpp)

set(Kobra_SOURCES
	${Kobra_GENERIC_SOURCES}
	${Kobra_LAYERS_SOURCES}
	${Kobra_IO_SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/core.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/layers/basilisk.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/layers/denoiser.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/layers/mesh_memory_cuda.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/layers/wssr_grid.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/amadeus/armada.cu
)

# Set executable sources -- Kobra
add_executable(kobra
	main.cu
	${ImGui_SOURCES}
	${ImPlot_SOURCES}
	${Glslang_SOURCES}
	${Kobra_SOURCES}
)

target_link_libraries(kobra
	${Vulkan_LIBRARIES}
	glfw
	glslang
	SPIRV
	assimp
	nvidia-ml
	${OpenCV_LIBS}
	${ImageMagick_LIBRARIES}
)

# TODO: create a shared library of library facilities

# Set executable sources -- Stress
add_executable(stress
	experimental/stress.cu
	${ImGui_SOURCES}
	${ImPlot_SOURCES}
	${Glslang_SOURCES}
	${Kobra_SOURCES}
)

target_link_libraries(stress
	${Vulkan_LIBRARIES}
	glfw
	glslang
	SPIRV
	assimp
	nvidia-ml
	${OpenCV_LIBS}
	${ImageMagick_LIBRARIES}
)

# Extra outputs
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiling GLSL shaders
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/spv)

file(GLOB_RECURSE GLSL_VERTEX_SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert)
file(GLOB_RECURSE GLSL_FRAGMENT_SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag)

list(APPEND GLSL_SHADERS ${GLSL_VERTEX_SHADERS} ${GLSL_FRAGMENT_SHADERS})

foreach(SHADER ${GLSL_SHADERS})
	get_filename_component(FILE_NAME ${SHADER} NAME)

	# Format as shader_{type}.spv
	string(REGEX REPLACE "(.*)\\.(vert|frag)" "\\1_\\2.spv" OUTPUT_FILE_NAME ${FILE_NAME})

	add_custom_command(
		OUTPUT ${CMAKE_BINARY_DIR}/bin/spv/${OUTPUT_FILE_NAME}
		COMMAND glslc ${SHADER} -o ${CMAKE_BINARY_DIR}/bin/spv/${OUTPUT_FILE_NAME}
		DEPENDS ${SHADER}
	)

	list(APPEND SPIRV_BINARY_FILES ${CMAKE_BINARY_DIR}/bin/spv/${OUTPUT_FILE_NAME})
endforeach(SHADER)

# Add as a dependency of the executable
add_custom_target(
    glsl_shaders
    DEPENDS ${SPIRV_BINARY_FILES}
)

add_dependencies(kobra glsl_shaders)

# Compiling OptiX shaders
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ptx)

file(GLOB_RECURSE OPTIX_SHADERS
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/basilisk_rt.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/basilisk_restir.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/basilisk_voxel.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/wssr_grid.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/amadeus_path_tracer.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/amadeus_restir.cu
	${CMAKE_CURRENT_SOURCE_DIR}/source/optix/amadeus_repg.cu
)

add_library(optix_shaders
	OBJECT ${OPTIX_SHADERS}
)

set_target_properties(optix_shaders PROPERTIES
	CUDA_PTX_COMPILATION ON
	COMPILE_FLAGS "--expt-relaxed-constexpr"
)

# Fake target to copy shader outputs to bin folder
add_custom_target(
	optix_shaders_copy
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/optix_shaders.dir/source/optix
		${CMAKE_BINARY_DIR}/bin/ptx
	DEPENDS optix_shaders
)

# Add binaries copy as a dependency of the executable
add_custom_target(
    copy_resources
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/bin ${CMAKE_CURRENT_SOURCE_DIR}/bin
    DEPENDS glsl_shaders optix_shaders_copy
)

add_dependencies(kobra copy_resources)
