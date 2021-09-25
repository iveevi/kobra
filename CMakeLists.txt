# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Projects
project(mercury)

add_executable(mercury
	source/main.cpp
	source/shader.cpp
	source/model.cpp

	glad/glad.c
)

include_directories(mercury .)

# Import CMake modules
find_package(OpenGL REQUIRED)
find_package(ASSIMP REQUIRED)

# Link libraries
target_link_libraries(mercury
	glfw
	OpenGL::GL
	${CMAKE_DL_LIBS}
	${ASSIMP_LIBRARIES}
)
