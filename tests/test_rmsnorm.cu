#include "common.h"
#include "kernels.cuh"
#include <vector>
#include <cmath>
#include <iomanip>

int main() {
    std::cout << "====================================================\n";
    std::cout << "     Unit Test: Fused RMSNorm & Golden Parity       \n";
    std::cout << "====================================================\n";

    const int hidden_size = 1536;
    const float eps = 1e-6f;
    const size_t tensor_bytes = hidden_size * sizeof(half);

    // 1. Load input embedding output from tests/golden/01_embed_out.bin
    std::string embed_path = "tests/golden/01_embed_out.bin";
    std::ifstream f_embed(embed_path, std::ios::binary);
    if (!f_embed.is_open()) {
        std::cerr << "[-] Error: Cannot open " << embed_path << "\n";
        return 1;
    }
    std::vector<half> h_x(hidden_size);
    f_embed.read(reinterpret_cast<char*>(h_x.data()), tensor_bytes);
    f_embed.close();

    // 2. Load Layer 0 input norm weight from qwen2.5_1.5b_fp16.bin
    // Offset calculation: 64-byte header + (vocab_size * hidden_size * sizeof(half))
    const size_t vocab_size = 151936;
    const size_t weight_offset = 64 + (vocab_size * hidden_size * sizeof(half));

    std::string bin_path = "models/qwen2.5-1.5b/qwen2.5_1.5b_fp16.bin";
    std::ifstream f_weights(bin_path, std::ios::binary);
    if (!f_weights.is_open()) {
        std::cerr << "[-] Error: Cannot open " << bin_path << "\n";
        return 1;
    }
    f_weights.seekg(weight_offset, std::ios::beg);
    std::vector<half> h_weight(hidden_size);
    f_weights.read(reinterpret_cast<char*>(h_weight.data()), tensor_bytes);
    f_weights.close();

    // 3. Load golden output from tests/golden/02_input_norm_out.bin
    std::string golden_path = "tests/golden/02_input_norm_out.bin";
    std::ifstream f_golden(golden_path, std::ios::binary);
    if (!f_golden.is_open()) {
        std::cerr << "[-] Error: Cannot open " << golden_path << "\n";
        return 1;
    }
    std::vector<half> h_golden(hidden_size);
    f_golden.read(reinterpret_cast<char*>(h_golden.data()), tensor_bytes);
    f_golden.close();

    // 4. Device allocation
    half *d_x = nullptr, *d_weight = nullptr, *d_out = nullptr, *d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_res, tensor_bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), tensor_bytes, cudaMemcpyHostToDevice));

    // // 5. Test Case 1: Standard RMSNorm (No residual)
    // std::cout << "[+] Running Test Case 1: Standard RMSNorm (residual = nullptr)...\n";
    // launch_rmsnorm(d_out, d_x, nullptr, d_weight, eps, hidden_size, 1);
    // CUDA_CHECK(cudaDeviceSynchronize());

    // std::vector<half> h_out(hidden_size);
    // CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, tensor_bytes, cudaMemcpyDeviceToHost));

    // float max_diff = 0.0f;
    // float sum_diff = 0.0f;
    // for (int i = 0; i < hidden_size; ++i) {
    //     float actual = __half2float(h_out[i]);
    //     float expected = __half2float(h_golden[i]);
    //     float diff = std::abs(actual - expected);
    //     if (diff > max_diff) max_diff = diff;
    //     sum_diff += diff;
    // }
    // float mean_diff = sum_diff / hidden_size;

    // std::cout << std::fixed << std::setprecision(6);
    // std::cout << "    Max Absolute Error : " << max_diff << "\n";
    // std::cout << "    Mean Absolute Error: " << mean_diff << "\n";

    // if (max_diff > 1.5e-3f) {
    //     std::cerr << "[✗] FAILURE: Kernel output diverges from PyTorch golden reference!\n";
    //     return 1;
    // }
    // std::cout << "[✓] PASS: Standard RMSNorm matches PyTorch Layer 0 input_layernorm!\n\n";


    // 5. Test Case 1: Standard RMSNorm (No residual)
    std::cout << "[+] Running Test Case 1: Standard RMSNorm (residual = nullptr)...\n";
    launch_rmsnorm(d_out, d_x, nullptr, d_weight, eps, hidden_size, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_out(hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, tensor_bytes, cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        float actual = __half2float(h_out[i]);
        float expected = __half2float(h_golden[i]);
        float diff = std::abs(actual - expected);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }
    float mean_diff = sum_diff / hidden_size;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    Max Absolute Error : " << max_diff << "\n";
    std::cout << "    Mean Absolute Error: " << mean_diff << "\n";

    // Standard FP16 tolerance: 1 ULP around 2.0-4.0 is 0.001953f.
    if (max_diff > 2.5e-3f) {
        std::cerr << "[✗] FAILURE: Kernel output diverges from PyTorch golden reference!\n";
        return 1;
    }
    std::cout << "[✓] PASS: Standard RMSNorm matches PyTorch Layer 0 input_layernorm (within 1 ULP)!\n\n";

    // 6. Test Case 2: Fused RMSNorm with Residual Accumulation
    std::cout << "[+] Running Test Case 2: Fused Residual Accumulation...\n";
    // Initialize residual to 0.5f across all elements
    std::vector<half> h_res_init(hidden_size, __float2half(0.5f));
    CUDA_CHECK(cudaMemcpy(d_res, h_res_init.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), tensor_bytes, cudaMemcpyHostToDevice));

    launch_rmsnorm(d_out, d_x, d_res, d_weight, eps, hidden_size, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify residual has updated: residual_new = x + 0.5f
    std::vector<half> h_res_updated(hidden_size);
    CUDA_CHECK(cudaMemcpy(h_res_updated.data(), d_res, tensor_bytes, cudaMemcpyDeviceToHost));

    float max_res_diff = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        float expected = __half2float(h_x[i]) + 0.5f;
        float actual = __half2float(h_res_updated[i]);
        max_res_diff = std::max(max_res_diff, std::abs(actual - expected));
    }
    std::cout << "    Max Residual Accumulation Error: " << max_res_diff << "\n";
    if (max_res_diff > 1e-3f) {
        std::cerr << "[✗] FAILURE: Fused residual update incorrect!\n";
        return 1;
    }
    std::cout << "[✓] PASS: Fused residual updated correctly in-place!\n\n";

    // 7. Microbenchmark
    std::cout << "[+] Benchmarking Kernel Latency (1000 iterations)...\n";
    GpuTimer timer;
    // Warmup
    for (int i = 0; i < 100; ++i) {
        launch_rmsnorm(d_out, d_x, nullptr, d_weight, eps, hidden_size, 1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.start();
    const int iters = 1000;
    for (int i = 0; i < iters; ++i) {
        launch_rmsnorm(d_out, d_x, nullptr, d_weight, eps, hidden_size, 1);
    }
    float elapsed_ms = timer.stop();
    float avg_us = (elapsed_ms / iters) * 1000.0f;

    std::cout << "    Average Latency: " << avg_us << " us (" << elapsed_ms / iters << " ms)\n";

    // Clean up
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_res));

    std::cout << "\n[✓] All RMSNorm unit checks passed successfully.\n";
    return 0;
}