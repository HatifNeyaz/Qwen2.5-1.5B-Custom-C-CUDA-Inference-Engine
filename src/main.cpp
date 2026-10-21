#include "common.h"
#include "model.h"
#include "tokenizer.h"
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

std::string format_chat_prompt(const std::string& user_query, const std::string& system_prompt = "You are a helpful assistant.") {
    return "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
         + "<|im_start|>user\n" + user_query + "<|im_end|>\n"
         + "<|im_start|>assistant\n";
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "   Qwen2.5-1.5B-Instruct Custom CUDA Inference CLI \n";
    std::cout << "====================================================\n";

    print_gpu_memory("Baseline");

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
    Tokenizer tokenizer;

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    try {
        GpuTimer init_timer;
        init_timer.start();

        load_model_weights("models/qwen2.5-1.5b/qwen2.5_1.5b_fp16.bin", cfg, weights);
        init_kv_cache(cfg, kv_cache);
        init_run_state(cfg, state);

        if (!tokenizer.load("models/qwen2.5-1.5b/tokenizer.json")) {
            throw std::runtime_error("Failed to load tokenizer.json");
        }

        float init_ms = init_timer.stop();
        std::cout << "\n[✓] Engine and Tokenizer ready in " << init_ms << " ms\n";
        print_gpu_memory("Engine Ready");

        std::cout << "\nType your prompt below. Commands: '/exit' to quit, '/reset' to clear KV cache.\n";
        std::cout << "--------------------------------------------------------------------\n";

        while (true) {
            std::cout << "\n\033[1;32mUser>\033[0m ";
            std::string user_input;
            if (!std::getline(std::cin, user_input)) break;

            if (user_input.empty()) continue;
            if (user_input == "/exit" || user_input == "exit" || user_input == "quit") break;
            if (user_input == "/reset") {
                CUDA_CHECK(cudaMemset(kv_cache.d_k_cache, 0, kv_cache.size_per_cache_bytes));
                CUDA_CHECK(cudaMemset(kv_cache.d_v_cache, 0, kv_cache.size_per_cache_bytes));
                std::cout << "[✓] KV Cache reset.\n";
                continue;
            }

            // 1. Format prompt and tokenize
            std::string formatted_prompt = format_chat_prompt(user_input);
            std::vector<int> prompt_tokens = tokenizer.encode(formatted_prompt);
            int prompt_len = static_cast<int>(prompt_tokens.size());

            if (prompt_len >= cfg.max_seq_len) {
                std::cerr << "[-] Error: Prompt exceeds maximum sequence length (" << cfg.max_seq_len << ").\n";
                continue;
            }

            // 2. Prefill Phase
            GpuTimer prefill_timer;
            prefill_timer.start();

            int curr_token = 0;
            for (int i = 0; i < prompt_len; ++i) {
                curr_token = forward_token(
                    prompt_tokens[i], i, cfg, weights, kv_cache, state, cublas_handle
                );
            }
            float prefill_ms = prefill_timer.stop();
            float prefill_tok_s = (prefill_ms > 0.0f) ? (prompt_len / (prefill_ms / 1000.0f)) : 0.0f;

            // 3. Decode / Autoregressive Generation
            std::cout << "\033[1;34mQwen>\033[0m ";

            GpuTimer decode_timer;
            decode_timer.start();

            int gen_count = 0;
            for (int pos = prompt_len; pos < cfg.max_seq_len; ++pos) {
                if (curr_token == tokenizer.im_end_id() || curr_token == tokenizer.eos_token_id()) {
                    break;
                }

                // Stream decoded piece to console
                std::string piece = tokenizer.decode(curr_token);
                std::cout << piece << std::flush;
                gen_count++;

                curr_token = forward_token(
                    curr_token, pos, cfg, weights, kv_cache, state, cublas_handle
                );
            }
            std::cout << "\n";

            float decode_ms = decode_timer.stop();
            float decode_tok_s = (gen_count > 0 && decode_ms > 0.0f) ? (gen_count / (decode_ms / 1000.0f)) : 0.0f;

            // 4. Performance Statistics
            std::cout << std::fixed << std::setprecision(1);
            std::cout << "\033[2m["
                      << "Prefill: " << prompt_len << " tok in " << prefill_ms << " ms (" << prefill_tok_s << " tok/s) | "
                      << "Decode: " << gen_count << " tok in " << decode_ms << " ms (" << decode_tok_s << " tok/s)"
                      << "]\033[0m\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[-] Runtime Exception: " << e.what() << "\n";
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        return 1;
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    free_run_state(state);
    free_kv_cache(kv_cache);
    free_model_weights(weights);

    std::cout << "\n[✓] Engine shutdown cleanly.\n";
    print_gpu_memory("Shutdown");

    return 0;
}