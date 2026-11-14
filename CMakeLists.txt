cmake_minimum_required(VERSION 3.18...3.31)
project(pynvjpeg LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Only the CUDA runtime headers and libnvjpeg are needed (no nvcc).
find_package(CUDAToolkit REQUIRED)

nanobind_add_module(pynvjpeg python-nvjpeg.cpp)
target_link_libraries(pynvjpeg PRIVATE CUDA::nvjpeg CUDA::cudart)

# Keep finding the toolkit's libnvjpeg after installation.
set_target_properties(pynvjpeg PROPERTIES INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

# Type stub (signatures + docstrings) for editors and type checkers.
nanobind_add_stub(
  pynvjpeg_stub
  MODULE pynvjpeg
  OUTPUT pynvjpeg.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:pynvjpeg>
  DEPENDS pynvjpeg)

install(TARGETS pynvjpeg LIBRARY DESTINATION .)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pynvjpeg.pyi DESTINATION .)
