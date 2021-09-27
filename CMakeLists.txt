# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Settings
if (${CMAKE_BUILD_TYPE} MATCHES Debug)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
endif()

# Projects
project(mercury)

add_executable(mercury
	source/camera.cpp
	source/init.cpp
	source/main.cpp
	source/model.cpp
	source/shader.cpp
	source/text.cpp

	glad/glad.c
)

include_directories(mercury .)
include_directories(mercury thirdparty/freetype2/include)

# Import CMake modules
find_package(OpenGL REQUIRED)
find_package(ASSIMP REQUIRED)

# Link libraries
target_link_libraries(mercury
	glfw
	freetype
	OpenGL::GL
	${CMAKE_DL_LIBS}
	${ASSIMP_LIBRARIES}
)
