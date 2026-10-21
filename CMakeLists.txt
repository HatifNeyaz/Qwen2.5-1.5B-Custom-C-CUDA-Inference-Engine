cmake_minimum_required(VERSION 3.18)
project(practice_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -Wall")

include_directories(include)

find_package(CUDAToolkit REQUIRED)

set(KERNEL_SOURCES
    src/kernels/rmsnorm.cu
    src/kernels/rope_bias.cu
    src/kernels/swiglu.cu
    src/kernels/attention.cu
    src/kernels/sampling.cu
)

set(COMMON_SOURCES
    src/model.cpp
    src/tokenizer.cpp
    ${KERNEL_SOURCES}
)

# Main interactive engine binary
add_executable(qwen_infer
    src/main.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(qwen_infer PRIVATE
    CUDA::cudart
    CUDA::cublas
)

# Unit Tests
add_executable(test_rmsnorm tests/test_rmsnorm.cu src/kernels/rmsnorm.cu)
target_link_libraries(test_rmsnorm PRIVATE CUDA::cudart)

add_executable(test_rope_bias tests/test_rope_bias.cu src/kernels/rope_bias.cu)
target_link_libraries(test_rope_bias PRIVATE CUDA::cudart)

add_executable(test_swiglu tests/test_swiglu.cu src/kernels/swiglu.cu)
target_link_libraries(test_swiglu PRIVATE CUDA::cudart)

add_executable(test_attention tests/test_attention.cu src/kernels/attention.cu)
target_link_libraries(test_attention PRIVATE CUDA::cudart)

add_executable(test_forward
    tests/test_forward.cu
    ${COMMON_SOURCES}
)
target_link_libraries(test_forward PRIVATE
    CUDA::cudart
    CUDA::cublas
)