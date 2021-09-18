# Set CMake version
cmake_minimum_required(VERSION 3.14)

project(mercury)

add_executable(mercury main.cpp glad.c)

include_directories(mercury .)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(mercury glfw OpenGL::GL ${CMAKE_DL_LIBS})
