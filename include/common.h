#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <stdexcept>

// Production error checking macros
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err) << std::endl;      \
            throw std::runtime_error("CUDA call failed");                     \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = (call);                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "[cuBLAS ERROR] " << __FILE__ << ":" << __LINE__     \
                      << " -> Status code: " << status << std::endl;          \
            throw std::runtime_error("cuBLAS call failed");                   \
        }                                                                     \
    } while (0)

// High-resolution GPU device timer using CUDA events
struct GpuTimer {
    cudaEvent_t start_ev, stop_ev;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_ev, stream));
    }

    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_ev, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
        return ms;
    }
};

// Memory diagnostics helper
inline void print_gpu_memory(const std::string& label = "") {
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    double free_gb = static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0);
    double total_gb = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
    double used_gb = total_gb - free_gb;

    std::cout << "[VRAM Diagnostics " << (label.empty() ? "" : "(" + label + ")") 
              << "] Used: " << used_gb << " GB / " << total_gb 
              << " GB (Free: " << free_gb << " GB)" << std::endl;
}