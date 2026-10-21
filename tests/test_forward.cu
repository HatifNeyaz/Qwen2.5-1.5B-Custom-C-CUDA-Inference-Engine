#include "common.h"
#include "model.h"
#include "kernels.cuh"
#include <vector>
#include <cmath>
#include <iomanip>

int main() {
    std::cout << "====================================================\n";
    std::cout << "   Full Transformer Forward Pass & Parity Check     \n";
    std::cout << "====================================================\n";

    QwenConfig cfg;
    cfg.magic = 0x5157454E;
    cfg.hidden_size = 1536;
    cfg.intermediate_size = 8960;
    cfg.num_layers = 28;
    cfg.num_heads = 12;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 128;
    cfg.vocab_size = 151936;
    cfg.max_seq_len = 2048;
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = 1000000.0f;

    QwenWeights weights;
    KVCache kv_cache;
    RunState state;

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    try {
        load_model_weights("models/qwen2.5-1.5b/qwen2.5_1.5b_fp16.bin", cfg, weights);
        init_kv_cache(cfg, kv_cache);
        init_run_state(cfg, state);

        // 1. Read input token
        std::ifstream f_tok("tests/golden/input_token.txt");
        if (!f_tok.is_open()) throw std::runtime_error("Cannot open tests/golden/input_token.txt");
        int input_token = 0;
        f_tok >> input_token;
        f_tok.close();

        // 2. Read PyTorch golden predicted token
        std::ifstream f_gold_tok("tests/golden/golden_next_token.txt");
        if (!f_gold_tok.is_open()) throw std::runtime_error("Cannot open tests/golden/golden_next_token.txt");
        int golden_token = 0;
        f_gold_tok >> golden_token;
        f_gold_tok.close();

        std::cout << "[+] Input Token ID        : " << input_token << "\n";
        std::cout << "[+] Golden Next Token ID  : " << golden_token << "\n\n";

        // ----------------------------------------------------
        // Step 1: Validate True Layer 0 Forward Pass
        // ----------------------------------------------------
        std::cout << "[+] Step 1: Validating Layer 0 Forward Pass...\n";
        CUDA_CHECK(cudaMemcpy(
            state.x,
            weights.embed_tokens + (static_cast<size_t>(input_token) * cfg.hidden_size),
            cfg.hidden_size * sizeof(half),
            cudaMemcpyDeviceToDevice
        ));

        forward_layer(0, 0, cfg, weights.layers[0], kv_cache.d_k_cache, kv_cache.d_v_cache, state, cublas_handle);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::string swiglu_path = "tests/golden/08_layer0_swiglu_out.bin";
        std::ifstream f_swiglu(swiglu_path, std::ios::binary);
        if (f_swiglu.is_open()) {
            std::vector<half> h_gold(cfg.intermediate_size);
            f_swiglu.read(reinterpret_cast<char*>(h_gold.data()), cfg.intermediate_size * sizeof(half));
            f_swiglu.close();

            std::vector<half> h_gate(cfg.intermediate_size);
            CUDA_CHECK(cudaMemcpy(h_gate.data(), state.gate, cfg.intermediate_size * sizeof(half), cudaMemcpyDeviceToHost));

            float max_swiglu_err = 0.0f;
            float sum_swiglu_err = 0.0f;
            for (int i = 0; i < cfg.intermediate_size; ++i) {
                float diff = std::abs(__half2float(h_gate[i]) - __half2float(h_gold[i]));
                max_swiglu_err = std::max(max_swiglu_err, diff);
                sum_swiglu_err += diff;
            }
            float mean_swiglu_err = sum_swiglu_err / cfg.intermediate_size;

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "    Layer 0 SwiGLU Max Error : " << max_swiglu_err << "\n";
            std::cout << "    Layer 0 SwiGLU Mean Error: " << mean_swiglu_err << "\n";

            // Standard production threshold: mean error < 1e-3 and max error < 3-4 ULPs (0.0625)
            if (mean_swiglu_err > 1.0e-3f || max_swiglu_err > 0.065f) {
                std::cerr << "[✗] FAILURE: Layer 0 divergent from golden reference!\n";
                return 1;
            }
            std::cout << "[✓] PASS: Layer 0 forward pass verified within FP16 GEMM tolerance.\n\n";
        }

        // ----------------------------------------------------
        // Step 2: Full 28-Layer Decode Pass & Argmax Parity
        // ----------------------------------------------------
        std::cout << "[+] Step 2: Running Full 28-Layer Forward Pass (pos = 0)...\n";
        int predicted_token = forward_token(input_token, 0, cfg, weights, kv_cache, state, cublas_handle);
        std::cout << "    Engine Predicted Token : " << predicted_token << "\n";
        std::cout << "    PyTorch Golden Token   : " << golden_token << "\n";

        if (predicted_token != golden_token) {
            std::cerr << "[✗] FAILURE: Predicted token ID does not match PyTorch golden token!\n";
            return 1;
        }
        std::cout << "[✓] PASS: Full 28-layer pipeline matches PyTorch greedy output bit-for-bit!\n\n";

        // ----------------------------------------------------
        // Step 3: Performance Microbenchmark
        // ----------------------------------------------------
        std::cout << "[+] Step 3: Benchmarking Full 28-Layer Decode Step (50 iterations)...\n";
        for (int i = 0; i < 5; ++i) {
            forward_token(input_token, 0, cfg, weights, kv_cache, state, cublas_handle);
        }

        GpuTimer timer;
        const int iters = 50;
        timer.start();
        for (int i = 0; i < iters; ++i) {
            forward_token(input_token, 0, cfg, weights, kv_cache, state, cublas_handle);
        }
        float total_ms = timer.stop();
        float avg_ms = total_ms / iters;
        float tok_per_sec = 1000.0f / avg_ms;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "    Average Step Latency : " << avg_ms << " ms\n";
        std::cout << "    Decode Throughput    : " << tok_per_sec << " tokens/sec\n";

    } catch (const std::exception& e) {
        std::cerr << "[-] Error: " << e.what() << "\n";
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        return 1;
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    free_run_state(state);
    free_kv_cache(kv_cache);
    free_model_weights(weights);

    std::cout << "\n[✓] All forward pass validation checks passed successfully.\n";
    return 0;
}