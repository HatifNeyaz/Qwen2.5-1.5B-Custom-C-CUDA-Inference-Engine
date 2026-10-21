#pragma once

#include "common.h"

struct QwenConfig {
    int32_t magic;               // 0x5157454E
    int32_t hidden_size;         // 1536
    int32_t intermediate_size;   // 8960
    int32_t num_layers;          // 28
    int32_t num_heads;           // 12
    int32_t num_kv_heads;        // 2
    int32_t head_dim;            // 128
    int32_t vocab_size;          // 151936
    int32_t max_seq_len;         // 2048
    float   rms_norm_eps;        // 1e-6f
    float   rope_theta;          // 1000000.0f
};

// Pointers for a single transformer block
struct LayerWeights {
    half* input_norm;       // [1536]
    half* q_weight;         // [1536, 1536]
    half* q_bias;           // [1536]
    half* k_weight;         // [256, 1536]
    half* k_bias;           // [256]
    half* v_weight;         // [256, 1536]
    half* v_bias;           // [256]
    half* o_weight;         // [1536, 1536]
    half* post_attn_norm;   // [1536]
    half* gate_weight;      // [8960, 1536]
    half* up_weight;        // [8960, 1536]
    half* down_weight;      // [1536, 8960]
};

// Complete parameter container
struct QwenWeights {
    half*        d_pool;         // Monolithic device allocation for all weights
    size_t       total_elements; // Total half-precision elements
    size_t       total_bytes;    // Total memory footprint

    half*        embed_tokens;   // [vocab_size, hidden_size]
    LayerWeights layers[28];
    half*        final_norm;     // [hidden_size]
    half*        lm_head;        // Tied to embed_tokens
};

// Pre-allocated static KV-cache
struct KVCache {
    half* d_k_cache; // [layers, max_seq_len, num_kv_heads, head_dim]
    half* d_v_cache; // [layers, max_seq_len, num_kv_heads, head_dim]
    size_t size_per_cache_bytes;
};

// Execution scratchpad buffers
struct RunState {
    half* x;                 // [hidden_size] - Residual stream
    half* xb;                // [hidden_size] - Normalized activation / projection target
    half* q;                 // [hidden_size]
    half* k;                 // [num_kv_heads * head_dim]
    half* v;                 // [num_kv_heads * head_dim]
    half* att_out;           // [hidden_size]
    half* gate;              // [intermediate_size]
    half* up;                // [intermediate_size]
    half* mlp_out;           // [hidden_size]
    half* logits;            // [vocab_size]
    int*  d_next_token_id;   // [1] Device int for argmax result

    void* d_scratch_pool;
    size_t scratch_pool_bytes;
};

// Memory and weight lifecycle functions
void load_model_weights(const std::string& bin_path, const QwenConfig& cfg, QwenWeights& weights);
void free_model_weights(QwenWeights& weights);
void init_kv_cache(const QwenConfig& cfg, KVCache& kv_cache);
void free_kv_cache(KVCache& kv_cache);
void init_run_state(const QwenConfig& cfg, RunState& state);
void free_run_state(RunState& state);

// cuBLAS Linear Projection Layer (Y = X * W^T)
void matmul_transB(
    cublasHandle_t handle,
    half* Y,
    const half* X,
    const half* W,
    int M, int N, int K
);

// Single Transformer Layer Execution
void forward_layer(
    int layer_idx,
    int pos,
    const QwenConfig& cfg,
    const LayerWeights& layer,
    half* k_cache_layer,
    half* v_cache_layer,
    RunState& state,
    cublasHandle_t cublas_handle,
    cudaStream_t stream = 0
);

// Full End-to-End Decode Step (1 token in -> next token ID out)
int forward_token(
    int input_token,
    int pos,
    const QwenConfig& cfg,
    const QwenWeights& weights,
    KVCache& kv_cache,
    RunState& state,
    cublasHandle_t cublas_handle,
    cudaStream_t stream = 0
);