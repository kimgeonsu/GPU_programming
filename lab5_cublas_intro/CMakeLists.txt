cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(gemm_cgm LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for gemm_cgm
add_executable(gemm_cgm gemm_cgm.cpp)

target_compile_options(gemm_cgm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(gemm_cgm PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(gemm_cgm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(gemm_cgm PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(gemm_cgm PRIVATE
    CUDA::cudart
    CUDA::cublas
)
