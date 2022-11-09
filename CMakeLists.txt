cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_STANDARD 17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

include(CPack)

project(kobra VERSION 0.1.0)

include(CTest)

enable_testing()
enable_language(CUDA)

add_compile_options(-fdiagnostics-color=always)

# Build mode
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG)
endif()

set(KOBRA_SOURCE
        source/app.cpp
        source/backend.cpp
        source/bvh.cpp
        source/capture.cpp
        source/ecs.cpp
        source/extensions.cpp
        source/formats.cpp
        source/io/event.cpp
        source/layers/denoiser.cu
        source/layers/forward_renderer.cpp
        source/layers/framer.cpp
        source/layers/gizmo.cpp
        source/layers/hybrid_tracer.cu
        source/layers/objectifier.cpp
        source/layers/optix_tracer.cu
        source/layers/raster.cpp
        source/layers/raytracer.cpp
        source/layers/basilisk.cu
        source/logger.cpp
        source/material.cpp
        source/mesh.cpp
        source/renderer.cpp
        source/scene.cpp
        source/texture_manager.cpp
        source/timer.cpp
        source/vertex.cpp
)

add_executable(kobra
        main.cu
        ${KOBRA_SOURCE}
)

# Find ImageMagick
find_package(ImageMagick COMPONENTS Magick++ REQUIRED)

# Find opencv
find_package(OpenCV COMPONENTS core videoio imgcodecs imgproc REQUIRED)

include_directories(
        thirdparty
        thirdparty/glm
        thirdparty/termcolor/include
        thirdparty/freetype/include
        thirdparty/assimp/assimp
        thirdparty/tinyfiledialogs
        thirdparty/optix
        thirdparty/tinyexr
        thirdparty/stb
        thirdparty/termcolor/include
        /usr/include/eigen3
        ${ImageMagick_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(kobra
        glfw
        vulkan
        assimp
        pthread
        freetype
        ${ImageMagick_LIBRARIES}
        ${OpenCV_LIBS}
)

add_compile_definitions(MAGICKCORE_QUANTUM_DEPTH=16)
add_compile_definitions(MAGICKCORE_HDRI_ENABLE=0)
