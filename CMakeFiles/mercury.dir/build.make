# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/venki/mercury

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/venki/mercury

# Include any dependencies generated for this target.
include CMakeFiles/mercury.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mercury.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mercury.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mercury.dir/flags.make

CMakeFiles/mercury.dir/main.cpp.o: CMakeFiles/mercury.dir/flags.make
CMakeFiles/mercury.dir/main.cpp.o: main.cpp
CMakeFiles/mercury.dir/main.cpp.o: CMakeFiles/mercury.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/venki/mercury/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mercury.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mercury.dir/main.cpp.o -MF CMakeFiles/mercury.dir/main.cpp.o.d -o CMakeFiles/mercury.dir/main.cpp.o -c /home/venki/mercury/main.cpp

CMakeFiles/mercury.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mercury.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/venki/mercury/main.cpp > CMakeFiles/mercury.dir/main.cpp.i

CMakeFiles/mercury.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mercury.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/venki/mercury/main.cpp -o CMakeFiles/mercury.dir/main.cpp.s

CMakeFiles/mercury.dir/glad.c.o: CMakeFiles/mercury.dir/flags.make
CMakeFiles/mercury.dir/glad.c.o: glad.c
CMakeFiles/mercury.dir/glad.c.o: CMakeFiles/mercury.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/venki/mercury/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/mercury.dir/glad.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/mercury.dir/glad.c.o -MF CMakeFiles/mercury.dir/glad.c.o.d -o CMakeFiles/mercury.dir/glad.c.o -c /home/venki/mercury/glad.c

CMakeFiles/mercury.dir/glad.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mercury.dir/glad.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/venki/mercury/glad.c > CMakeFiles/mercury.dir/glad.c.i

CMakeFiles/mercury.dir/glad.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mercury.dir/glad.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/venki/mercury/glad.c -o CMakeFiles/mercury.dir/glad.c.s

# Object files for target mercury
mercury_OBJECTS = \
"CMakeFiles/mercury.dir/main.cpp.o" \
"CMakeFiles/mercury.dir/glad.c.o"

# External object files for target mercury
mercury_EXTERNAL_OBJECTS =

mercury: CMakeFiles/mercury.dir/main.cpp.o
mercury: CMakeFiles/mercury.dir/glad.c.o
mercury: CMakeFiles/mercury.dir/build.make
mercury: /usr/lib/x86_64-linux-gnu/libglfw.so.3.3
mercury: /usr/lib/x86_64-linux-gnu/libGLX.so
mercury: /usr/lib/x86_64-linux-gnu/libOpenGL.so
mercury: CMakeFiles/mercury.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/venki/mercury/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable mercury"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mercury.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mercury.dir/build: mercury
.PHONY : CMakeFiles/mercury.dir/build

CMakeFiles/mercury.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mercury.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mercury.dir/clean

CMakeFiles/mercury.dir/depend:
	cd /home/venki/mercury && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/venki/mercury /home/venki/mercury /home/venki/mercury /home/venki/mercury /home/venki/mercury/CMakeFiles/mercury.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mercury.dir/depend

