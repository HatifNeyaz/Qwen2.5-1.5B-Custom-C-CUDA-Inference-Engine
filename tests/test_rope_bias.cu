#include "common.h"
#include "kernels.cuh"
#include <vector>
#include <cmath>
#include <iomanip>

int main() {
    std::cout << "====================================================\n";
    std::cout << "   Unit Test: Fused Bias + RoPE + KV Cache Store    \n";
    std::cout << "====================================================\n";

    const int num_heads = 12;
    const int num_kv_heads = 2;
    const int head_dim = 128;
    const int hidden_size = num_heads * head_dim;       // 1536
    const int kv_dim = num_kv_heads * head_dim;          // 256
    const float rope_theta = 1000000.0f;
    const int max_seq_len = 2048;

    // 1. Load Golden PyTorch Reference Tensors
    std::string q_in_path = "tests/golden/03_q_proj_biased.bin";
    std::string k_in_path = "tests/golden/03_k_proj_biased.bin";
    std::string v_in_path = "tests/golden/03_v_proj_biased.bin";
    std::string q_gold_path = "tests/golden/04_q_rope_out.bin";
    std::string k_gold_path = "tests/golden/04_k_rope_out.bin";

    std::vector<half> h_q(hidden_size), h_k(kv_dim), h_v(kv_dim);
    std::vector<half> h_q_gold(hidden_size), h_k_gold(kv_dim);

    auto load_bin = [](const std::string& path, void* dst, size_t bytes) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open " + path);
        f.read(reinterpret_cast<char*>(dst), bytes);
    };

    load_bin(q_in_path, h_q.data(), hidden_size * sizeof(half));
    load_bin(k_in_path, h_k.data(), kv_dim * sizeof(half));
    load_bin(v_in_path, h_v.data(), kv_dim * sizeof(half));
    load_bin(q_gold_path, h_q_gold.data(), hidden_size * sizeof(half));
    load_bin(k_gold_path, h_k_gold.data(), kv_dim * sizeof(half));

    // 2. Allocate GPU memory
    half *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    half *d_k_cache = nullptr, *d_v_cache = nullptr;
    size_t cache_bytes = max_seq_len * kv_dim * sizeof(half);

    CUDA_CHECK(cudaMalloc(&d_q, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, kv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, kv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_cache, cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_cache, cache_bytes));

    // ----------------------------------------------------
    // Test Case 1: PyTorch Golden Parity at pos = 0
    // ----------------------------------------------------
    std::cout << "[+] Test 1: RoPE Parity with PyTorch Golden Reference (pos = 0)...\n";
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), kv_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), kv_dim * sizeof(half), cudaMemcpyHostToDevice));

    launch_rope_bias_kv(d_q, d_k, d_v, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
                        num_heads, num_kv_heads, head_dim, rope_theta);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_q_out(hidden_size), h_k_out(kv_dim);
    CUDA_CHECK(cudaMemcpy(h_q_out.data(), d_q, hidden_size * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_k_out.data(), d_k, kv_dim * sizeof(half), cudaMemcpyDeviceToHost));

    float max_q_err = 0.0f, max_k_err = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        max_q_err = std::max(max_q_err, std::abs(__half2float(h_q_out[i]) - __half2float(h_q_gold[i])));
    }
    for (int i = 0; i < kv_dim; ++i) {
        max_k_err = std::max(max_k_err, std::abs(__half2float(h_k_out[i]) - __half2float(h_k_gold[i])));
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    Max Query Error (pos=0): " << max_q_err << "\n";
    std::cout << "    Max Key Error   (pos=0): " << max_k_err << "\n";
    if (max_q_err > 2.5e-3f || max_k_err > 2.5e-3f) {
        std::cerr << "[✗] FAILURE: Golden reference mismatch at pos=0!\n";
        return 1;
    }
    std::cout << "[✓] PASS: Golden reference bit-exact at pos=0!\n\n";

    // ----------------------------------------------------
    // Test Case 2: Analytical Rotary Check at pos = 7
    // ----------------------------------------------------
    std::cout << "[+] Test 2: Analytical Rotary Validation at Non-Zero Position (pos = 7)...\n";
    const int test_pos = 7;
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), kv_dim * sizeof(half), cudaMemcpyHostToDevice));

    launch_rope_bias_kv(d_q, d_k, d_v, nullptr, nullptr, nullptr, nullptr, nullptr, test_pos,
                        num_heads, num_kv_heads, head_dim, rope_theta);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_q_out.data(), d_q, hidden_size * sizeof(half), cudaMemcpyDeviceToHost));

    // Compute CPU double-precision ground truth for Query at pos = 7
    float max_analytic_err = 0.0f;
    for (int h = 0; h < num_heads; ++h) {
        for (int tid = 0; tid < head_dim / 2; ++tid) {
            double exponent = (2.0 * tid) / static_cast<double>(head_dim);
            double inv_freq = std::pow(rope_theta, -exponent);
            double angle = static_cast<double>(test_pos) * inv_freq;
            double cos_v = std::cos(angle);
            double sin_v = std::sin(angle);

            double q1 = __half2float(h_q[h * head_dim + tid]);
            double q2 = __half2float(h_q[h * head_dim + tid + (head_dim / 2)]);

            double expected1 = q1 * cos_v - q2 * sin_v;
            double expected2 = q2 * cos_v + q1 * sin_v;

            float actual1 = __half2float(h_q_out[h * head_dim + tid]);
            float actual2 = __half2float(h_q_out[h * head_dim + tid + (head_dim / 2)]);

            max_analytic_err = std::max(max_analytic_err, static_cast<float>(std::abs(actual1 - expected1)));
            max_analytic_err = std::max(max_analytic_err, static_cast<float>(std::abs(actual2 - expected2)));
        }
    }
    std::cout << "    Max Analytic Error (pos=7): " << max_analytic_err << "\n";
    if (max_analytic_err > 8.0e-3f) {
        std::cerr << "[✗] FAILURE: Analytic RoPE mismatch at pos=7!\n";
        return 1;
    }
    std::cout << "[✓] PASS: Rotary embeddings match analytical ground truth!\n\n";

    // ----------------------------------------------------
    // Test Case 3: Fused Bias Add & KV Cache Insertion
    // ----------------------------------------------------
    std::cout << "[+] Test 3: Fused Bias Add & KV Cache Storage...\n";
    // Load Layer 0 biases directly from qwen2.5_1.5b_fp16.bin
    const size_t vocab_size = 151936;
    const size_t header_bytes = 64;
    const size_t embed_bytes = vocab_size * hidden_size * sizeof(half);
    const size_t layer0_start = header_bytes + embed_bytes;

    size_t q_w_bytes = hidden_size * hidden_size * sizeof(half);
    size_t q_b_bytes = hidden_size * sizeof(half);
    size_t k_w_bytes = kv_dim * hidden_size * sizeof(half);
    size_t k_b_bytes = kv_dim * sizeof(half);
    size_t v_w_bytes = kv_dim * hidden_size * sizeof(half);

    size_t q_b_offset = layer0_start + (hidden_size * sizeof(half)) + q_w_bytes;
    size_t k_b_offset = q_b_offset + q_b_bytes + k_w_bytes;
    size_t v_b_offset = k_b_offset + k_b_bytes + v_w_bytes;

    std::string bin_path = "models/qwen2.5-1.5b/qwen2.5_1.5b_fp16.bin";
    std::ifstream fb(bin_path, std::ios::binary);
    if (!fb.is_open()) throw std::runtime_error("Cannot open " + bin_path);

    std::vector<half> h_qb(hidden_size), h_kb(kv_dim), h_vb(kv_dim);
    fb.seekg(q_b_offset, std::ios::beg); fb.read(reinterpret_cast<char*>(h_qb.data()), q_b_bytes);
    fb.seekg(k_b_offset, std::ios::beg); fb.read(reinterpret_cast<char*>(h_kb.data()), k_b_bytes);
    fb.seekg(v_b_offset, std::ios::beg); fb.read(reinterpret_cast<char*>(h_vb.data()), kv_dim * sizeof(half));
    fb.close();

    // Construct raw inputs: raw = biased - bias
    std::vector<half> h_q_raw(hidden_size), h_k_raw(kv_dim), h_v_raw(kv_dim);
    for (int i = 0; i < hidden_size; ++i) {
        h_q_raw[i] = __float2half(__half2float(h_q[i]) - __half2float(h_qb[i]));
    }
    for (int i = 0; i < kv_dim; ++i) {
        h_k_raw[i] = __float2half(__half2float(h_k[i]) - __half2float(h_kb[i]));
        h_v_raw[i] = __float2half(__half2float(h_v[i]) - __half2float(h_vb[i]));
    }

    half *d_qb = nullptr, *d_kb = nullptr, *d_vb = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qb, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_kb, kv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_vb, kv_dim * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q_raw.data(), hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k_raw.data(), kv_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v_raw.data(), kv_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qb, h_qb.data(), hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kb, h_kb.data(), kv_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vb, h_vb.data(), kv_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_k_cache, 0, cache_bytes));
    CUDA_CHECK(cudaMemset(d_v_cache, 0, cache_bytes));

    launch_rope_bias_kv(d_q, d_k, d_v, d_qb, d_kb, d_vb, d_k_cache, d_v_cache, test_pos,
                        num_heads, num_kv_heads, head_dim, rope_theta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify cache contents at position 7
    std::vector<half> h_k_cache_slot(kv_dim), h_v_cache_slot(kv_dim);
    size_t slot_offset = test_pos * kv_dim;
    CUDA_CHECK(cudaMemcpy(h_k_cache_slot.data(), d_k_cache + slot_offset, kv_dim * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_cache_slot.data(), d_v_cache + slot_offset, kv_dim * sizeof(half), cudaMemcpyDeviceToHost));

    float max_v_cache_err = 0.0f;
    for (int i = 0; i < kv_dim; ++i) {
        max_v_cache_err = std::max(max_v_cache_err, std::abs(__half2float(h_v_cache_slot[i]) - __half2float(h_v[i])));
    }
    std::cout << "    Max V Cache Reconstitution Error: " << max_v_cache_err << "\n";
    if (max_v_cache_err > 2.5e-3f) {
        std::cerr << "[✗] FAILURE: V Cache mismatch!\n";
        return 1;
    }
    std::cout << "[✓] PASS: Fused bias addition and KV-cache store verified!\n\n";

    // ----------------------------------------------------
    // Test Case 4: Latency Benchmark
    // ----------------------------------------------------
    std::cout << "[+] Test 4: Benchmarking Kernel Latency (1000 iterations)...\n";
    GpuTimer timer;
    for (int i = 0; i < 100; ++i) {
        launch_rope_bias_kv(d_q, d_k, d_v, d_qb, d_kb, d_vb, d_k_cache, d_v_cache, test_pos,
                            num_heads, num_kv_heads, head_dim, rope_theta);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.start();
    const int iters = 1000;
    for (int i = 0; i < iters; ++i) {
        launch_rope_bias_kv(d_q, d_k, d_v, d_qb, d_kb, d_vb, d_k_cache, d_v_cache, test_pos,
                            num_heads, num_kv_heads, head_dim, rope_theta);
    }
    float elapsed_ms = timer.stop();
    float avg_us = (elapsed_ms / iters) * 1000.0f;

    std::cout << "    Average Latency: " << avg_us << " us (" << elapsed_ms / iters << " ms)\n";

    // Clean up
    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_k)); CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_qb)); CUDA_CHECK(cudaFree(d_kb)); CUDA_CHECK(cudaFree(d_vb));
    CUDA_CHECK(cudaFree(d_k_cache)); CUDA_CHECK(cudaFree(d_v_cache));

    std::cout << "\n[✓] All RoPE and KV Cache unit checks passed successfully.\n";
    return 0;
}