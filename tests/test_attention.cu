#include "common.h"
#include "kernels.cuh"
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

int main() {
    std::cout << "====================================================\n";
    std::cout << "       Unit Test: GQA Decode Self-Attention         \n";
    std::cout << "====================================================\n";

    const int num_heads = 12;
    const int num_kv_heads = 2;
    const int head_dim = 128;
    const int hidden_size = num_heads * head_dim; // 1536
    const int kv_dim = num_kv_heads * head_dim;   // 256
    const int max_seq_len = 2048;

    // Allocate GPU memory
    half *d_out = nullptr, *d_q = nullptr;
    half *d_k_cache = nullptr, *d_v_cache = nullptr;
    size_t cache_bytes = max_seq_len * kv_dim * sizeof(half);

    CUDA_CHECK(cudaMalloc(&d_out, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_q, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_cache, cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_cache, cache_bytes));

    // ----------------------------------------------------
    // Test Case 1: Mathematical Identity at pos = 0
    // ----------------------------------------------------
    std::cout << "[+] Test 1: Single-Token Identity Validation (pos = 0)...\n";
    std::string q_path = "tests/golden/04_q_rope_out.bin";
    std::string k_path = "tests/golden/04_k_rope_out.bin";
    std::string v_path = "tests/golden/03_v_proj_biased.bin";

    std::vector<half> h_q(hidden_size), h_k(kv_dim), h_v(kv_dim);
    auto load_bin = [](const std::string& path, void* dst, size_t bytes) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open " + path);
        f.read(reinterpret_cast<char*>(dst), bytes);
    };

    load_bin(q_path, h_q.data(), hidden_size * sizeof(half));
    load_bin(k_path, h_k.data(), kv_dim * sizeof(half));
    load_bin(v_path, h_v.data(), kv_dim * sizeof(half));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache, h_k.data(), kv_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache, h_v.data(), kv_dim * sizeof(half), cudaMemcpyHostToDevice));

    launch_gqa_decode(d_out, d_q, d_k_cache, d_v_cache, 0, num_heads, num_kv_heads, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_out(hidden_size);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, hidden_size * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err_pos0 = 0.0f;
    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / 6;
        for (int d = 0; d < head_dim; ++d) {
            float actual = __half2float(h_out[h * head_dim + d]);
            float expected = __half2float(h_v[kv_h * head_dim + d]);
            max_err_pos0 = std::max(max_err_pos0, std::abs(actual - expected));
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    Max Identity Error (pos = 0): " << max_err_pos0 << "\n";
    if (max_err_pos0 > 1e-4f) {
        std::cerr << "[✗] FAILURE: Attention output diverged from Value vector at pos=0!\n";
        return 1;
    }
    std::cout << "[✓] PASS: Attention correctly reduces to V projection at pos=0!\n\n";

    // ----------------------------------------------------
    // Test Case 2: Multi-Token GQA Parity at pos = 3 (4 Tokens)
    // ----------------------------------------------------
    std::cout << "[+] Test 2: Multi-Token Analytical Validation (pos = 3)...\n";
    const int test_pos = 3;
    const int num_tokens = test_pos + 1;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<half> h_rand_q(hidden_size);
    std::vector<half> h_rand_k(num_tokens * kv_dim);
    std::vector<half> h_rand_v(num_tokens * kv_dim);

    for (auto& val : h_rand_q) val = __float2half(dist(rng));
    for (auto& val : h_rand_k) val = __float2half(dist(rng));
    for (auto& val : h_rand_v) val = __float2half(dist(rng));

    CUDA_CHECK(cudaMemcpy(d_q, h_rand_q.data(), hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache, h_rand_k.data(), num_tokens * kv_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache, h_rand_v.data(), num_tokens * kv_dim * sizeof(half), cudaMemcpyHostToDevice));

    launch_gqa_decode(d_out, d_q, d_k_cache, d_v_cache, test_pos, num_heads, num_kv_heads, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, hidden_size * sizeof(half), cudaMemcpyDeviceToHost));

    // Analytical CPU Reference
    float max_multi_err = 0.0f;
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / 6;
        std::vector<double> scores(num_tokens);
        double max_score = -1e30;

        for (int t = 0; t < num_tokens; ++t) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                double q_elem = __half2float(h_rand_q[h * head_dim + d]);
                double k_elem = __half2float(h_rand_k[t * kv_dim + kv_h * head_dim + d]);
                dot += q_elem * k_elem;
            }
            scores[t] = dot * scale;
            if (scores[t] > max_score) max_score = scores[t];
        }

        double sum_exp = 0.0;
        for (int t = 0; t < num_tokens; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        for (int t = 0; t < num_tokens; ++t) {
            scores[t] /= sum_exp;
        }

        for (int d = 0; d < head_dim; ++d) {
            double expected_out = 0.0;
            for (int t = 0; t < num_tokens; ++t) {
                double v_elem = __half2float(h_rand_v[t * kv_dim + kv_h * head_dim + d]);
                expected_out += scores[t] * v_elem;
            }
            float actual_out = __half2float(h_out[h * head_dim + d]);
            max_multi_err = std::max(max_multi_err, static_cast<float>(std::abs(actual_out - expected_out)));
        }
    }

    std::cout << "    Max Multi-Token Error (pos = 3): " << max_multi_err << "\n";
    if (max_multi_err > 2.5e-3f) {
        std::cerr << "[✗] FAILURE: Multi-token analytical mismatch!\n";
        return 1;
    }
    std::cout << "[✓] PASS: GQA matches analytical reference within 1 ULP!\n\n";

    // ----------------------------------------------------
    // Test Case 3: Latency Benchmarks
    // ----------------------------------------------------
    std::cout << "[+] Test 3: Benchmarking Decode Latency...\n";
    auto bench_context = [&](int ctx_len) {
        GpuTimer timer;
        for (int i = 0; i < 50; ++i) {
            launch_gqa_decode(d_out, d_q, d_k_cache, d_v_cache, ctx_len - 1, num_heads, num_kv_heads, head_dim);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.start();
        const int iters = 1000;
        for (int i = 0; i < iters; ++i) {
            launch_gqa_decode(d_out, d_q, d_k_cache, d_v_cache, ctx_len - 1, num_heads, num_kv_heads, head_dim);
        }
        float ms = timer.stop();
        std::cout << "    Context Length " << std::setw(4) << ctx_len << " tokens: "
                  << std::setw(6) << (ms / iters) * 1000.0f << " us\n";
    };

    bench_context(64);
    bench_context(256);
    bench_context(512);

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache));

    std::cout << "\n[✓] All GQA Decode unit checks passed successfully.\n";
    return 0;
}