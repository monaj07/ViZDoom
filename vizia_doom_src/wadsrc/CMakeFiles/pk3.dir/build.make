# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.4

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ebony/vizia/vizia_doom_src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ebony/vizia/vizia_doom_src

# Utility rule file for pk3.

# Include the progress variables for this target.
include wadsrc/CMakeFiles/pk3.dir/progress.make

wadsrc/CMakeFiles/pk3: zdoom.pk3
	cd /home/ebony/vizia/vizia_doom_src/wadsrc && /usr/local/bin/cmake -E touch /home/ebony/vizia/vizia_doom_src/tools/zipdir/zipdir

zdoom.pk3: tools/zipdir/zipdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ebony/vizia/vizia_doom_src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../zdoom.pk3"
	cd /home/ebony/vizia/vizia_doom_src/wadsrc && ../tools/zipdir/zipdir -udf /home/ebony/vizia/vizia_doom_src/zdoom.pk3 /home/ebony/vizia/vizia_doom_src/wadsrc/static
	cd /home/ebony/vizia/vizia_doom_src/wadsrc && /usr/local/bin/cmake -E copy_if_different /home/ebony/vizia/vizia_doom_src/zdoom.pk3 /home/ebony/vizia/vizia_doom_src/zdoom.pk3

pk3: wadsrc/CMakeFiles/pk3
pk3: zdoom.pk3
pk3: wadsrc/CMakeFiles/pk3.dir/build.make

.PHONY : pk3

# Rule to build all files generated by this target.
wadsrc/CMakeFiles/pk3.dir/build: pk3

.PHONY : wadsrc/CMakeFiles/pk3.dir/build

wadsrc/CMakeFiles/pk3.dir/clean:
	cd /home/ebony/vizia/vizia_doom_src/wadsrc && $(CMAKE_COMMAND) -P CMakeFiles/pk3.dir/cmake_clean.cmake
.PHONY : wadsrc/CMakeFiles/pk3.dir/clean

wadsrc/CMakeFiles/pk3.dir/depend:
	cd /home/ebony/vizia/vizia_doom_src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ebony/vizia/vizia_doom_src /home/ebony/vizia/vizia_doom_src/wadsrc /home/ebony/vizia/vizia_doom_src /home/ebony/vizia/vizia_doom_src/wadsrc /home/ebony/vizia/vizia_doom_src/wadsrc/CMakeFiles/pk3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : wadsrc/CMakeFiles/pk3.dir/depend

