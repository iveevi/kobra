# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Settings
set(CMAKE_CXX_COMPILER "/usr/bin/g++-8")
set(CMAKE_CXX_STANDARD 14)

if (${CMAKE_BUILD_TYPE} MATCHES Debug)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -DMERCURY_DEBUG=0")
endif()

# Macros
add_compile_definitions(MERCURY_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Projects
project(mercury)
project(ui_designer)

# Source variables
set(MERCURY_SOURCE
	source/camera.cpp
	source/init.cpp
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

set(MERCURY_LIBS
	glfw
	freetype
	OpenGL::GL
	${CMAKE_DL_LIBS}
	${ASSIMP_LIBRARY}
)

# Adding sources
add_executable(mercury
	source/main.cpp
	${MERCURY_SOURCE}
)

add_executable(ui_designer
	source/bootstrap/ui_designer.cpp
	${MERCURY_SOURCE}
)

# Include directories
include_directories(.
	glad
	thirdparty/glm
	thirdparty/assimp/include
	thirdparty/freetype2/include
)

# Import CMake modules
find_package(OpenGL REQUIRED)

# Import libraries
find_library(ASSIMP_LIBRARY
	NAMES assimp
	HINTS ${PROJECT_SOURCE_DIR}/thirdparty/assimp/bin
	REQUIRED
)

# Link libraries
target_link_libraries(mercury
	${MERCURY_LIBS}
)

target_link_libraries(ui_designer
	${MERCURY_LIBS}
)
