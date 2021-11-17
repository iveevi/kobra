# Set CMake version
cmake_minimum_required(VERSION 3.14)

# TODO: disable rtti (-fno-rtti)

# Settings
if (EXISTS "/usr/bin/clang++-10")
	message("${warning}Using Clang compiler${reset}")
	set(CMAKE_CXX_COMPILER "/usr/bin/clang++-10")
elseif (EXISTS "/usr/bin/g++-8")
	message("${warning}Using GCC compiler${reset}")
	set(CMAKE_CXX_COMPILER "/usr/bin/g++-8")
elseif (EXISTS "/usr/bin/clang++")
	message("${warning}Using default Clang compiler${reset}")
	set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
else()
	message("Requiring GCC compiler at /usr/bin/local/")
	set(CMAKE_CXX_COMPILER "${warning}/usr/local/bin/g++-8${reset}")
endif()

set(CMAKE_CXX_STANDARD 14)

# Set default compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable")

# Set debug compiler flags
if (${CMAKE_BUILD_TYPE} MATCHES Debug)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -DMERCURY_DEBUG=0")
endif()

# Macros
add_compile_definitions(MERCURY_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Projects
project(mercury)
project(ui_designer)
project(file_browser)
project(curses)

# Source variables
set(MERCURY_SOURCE
	source/common.cpp
	source/init.cpp
	source/lighting.cpp
	source/logger.cpp
	source/model.cpp
	source/mouse_bus.cpp
	source/rendering.cpp
	source/shader.cpp

	source/engine/camera.cpp
	source/engine/monitors.cpp
	source/engine/skybox.cpp

	source/mesh/basic.cpp
	source/mesh/sphere.cpp
	source/mesh/cuboid.cpp

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
	ncurses
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

add_executable(file_browser
	source/bootstrap/file_browser.cpp
	${MERCURY_SOURCE}
)

add_executable(curses
	source/bootstrap/curses.cpp
	${MERCURY_SOURCE}
)

# Include directories
include_directories(.
	glad
	thirdparty/assimp/include
	thirdparty/freetype2/include
	thirdparty/glm
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

target_link_libraries(file_browser
	${MERCURY_LIBS}
)

target_link_libraries(curses
	${MERCURY_LIBS}
)