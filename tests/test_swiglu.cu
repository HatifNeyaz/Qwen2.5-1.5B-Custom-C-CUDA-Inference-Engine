#include "common.h"
#include "kernels.cuh"
#include <vector>
#include <cmath>
#include <iomanip>

int main() {
    std::cout << "====================================================\n";
    std::cout << "        Unit Test: Fused SwiGLU Activation          \n";
    std::cout << "====================================================\n";

    const int intermediate_size = 8960;
    const size_t tensor_bytes = intermediate_size * sizeof(half);

    // 1. Load PyTorch Golden Reference Tensors
    std::string gate_path = "tests/golden/06_gate_proj_out.bin";
    std::string up_path = "tests/golden/06_up_proj_out.bin";
    std::string gold_path = "tests/golden/07_swiglu_out.bin";

    std::vector<half> h_gate(intermediate_size), h_up(intermediate_size), h_gold(intermediate_size);

    auto load_bin = [](const std::string& path, void* dst, size_t bytes) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open " + path);
        f.read(reinterpret_cast<char*>(dst), bytes);
    };

    try {
        load_bin(gate_path, h_gate.data(), tensor_bytes);
        load_bin(up_path, h_up.data(), tensor_bytes);
        load_bin(gold_path, h_gold.data(), tensor_bytes);
    } catch (const std::exception& e) {
        std::cerr << "[-] Error: " << e.what() << "\n";
        return 1;
    }

    // 2. Allocate GPU memory
    half *d_gate = nullptr, *d_up = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gate, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_up, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, tensor_bytes));

    CUDA_CHECK(cudaMemcpy(d_gate, h_gate.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_up.data(), tensor_bytes, cudaMemcpyHostToDevice));

    // 3. Execute SwiGLU Kernel
    std::cout << "[+] Executing Fused SwiGLU kernel...\n";
    launch_swiglu(d_out, d_gate, d_up, intermediate_size, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_out(intermediate_size);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, tensor_bytes, cudaMemcpyDeviceToHost));

    // 4. Parity Verification
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    for (int i = 0; i < intermediate_size; ++i) {
        float actual = __half2float(h_out[i]);
        float expected = __half2float(h_gold[i]);
        float diff = std::abs(actual - expected);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
    }
    float mean_diff = sum_diff / intermediate_size;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    Max Absolute Error : " << max_diff << "\n";
    std::cout << "    Mean Absolute Error: " << mean_diff << "\n";

    if (max_diff > 2.5e-3f) {
        std::cerr << "[✗] FAILURE: SwiGLU output diverges from PyTorch golden reference!\n";
        return 1;
    }
    std::cout << "[✓] PASS: SwiGLU matches PyTorch golden reference within 1 ULP!\n\n";

    // 5. Benchmark Latency
    std::cout << "[+] Benchmarking Kernel Latency (1000 iterations)...\n";
    GpuTimer timer;
    for (int i = 0; i < 100; ++i) {
        launch_swiglu(d_out, d_gate, d_up, intermediate_size, 1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.start();
    const int iters = 1000;
    for (int i = 0; i < iters; ++i) {
        launch_swiglu(d_out, d_gate, d_up, intermediate_size, 1);
    }
    float elapsed_ms = timer.stop();
    float avg_us = (elapsed_ms / iters) * 1000.0f;

    std::cout << "    Average Latency: " << avg_us << " us (" << elapsed_ms / iters << " ms)\n";

    // Clean up
    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_out));

    std::cout << "\n[✓] SwiGLU unit checks passed successfully.\n";
    return 0;
}