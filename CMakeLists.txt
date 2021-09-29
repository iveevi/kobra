# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Settings
set(CMAKE_CXX_COMPILER "/usr/bin/g++-8")
set(CMAKE_CXX_STANDARD 14)

if (${CMAKE_BUILD_TYPE} MATCHES Debug)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
endif()

# Macros
add_compile_definitions(MERCURY_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Projects
project(mercury)

add_executable(mercury
	source/camera.cpp
	source/init.cpp
	source/main.cpp
	source/model.cpp
	source/mouse_bus.cpp
	source/shader.cpp
	source/ui/button.cpp
	source/ui/pure_rect.cpp
	source/ui/rect.cpp
	source/ui/text.cpp
	source/ui/ui_layer.cpp

	glad/glad.c
)

# Include directories
include_directories(mercury .)
include_directories(mercury glad)
include_directories(mercury thirdparty/glm)
include_directories(mercury thirdparty/assimp/include)
include_directories(mercury thirdparty/freetype2/include)

# Import CMake modules
find_package(OpenGL REQUIRED)
#find_package(ASSIMP REQUIRED)

# Import libraries
find_library(ASSIMP_LIBRARY
	NAMES assimp
	HINTS ${PROJECT_SOURCE_DIR}/thirdparty/assimp/bin
	REQUIRED
)

# Link libraries
target_link_libraries(mercury
	glfw
	freetype
	OpenGL::GL
	${CMAKE_DL_LIBS}
	${ASSIMP_LIBRARY}
)
