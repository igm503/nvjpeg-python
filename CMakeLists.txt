cmake_minimum_required(VERSION 3.15...3.27)
project(pynvjpeg)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

nanobind_add_module(pynvjpeg python-nvjpeg.cpp)

target_link_libraries(nvjpeg_wrapper PRIVATE 
    CUDA::nvjpeg
    CUDA::cudart
)

target_include_directories(nvjpeg_wrapper PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
)
