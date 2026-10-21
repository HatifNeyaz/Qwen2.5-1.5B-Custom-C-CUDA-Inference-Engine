#include "model.h"
#include "kernels.cuh"
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>

void load_model_weights(const std::string& bin_path, const QwenConfig& cfg, QwenWeights& weights) {
    std::cout << "[+] Opening binary weight file: " << bin_path << "..." << std::endl;
    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open binary weight file: " + bin_path);
    }

    int32_t header[16];
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!file) throw std::runtime_error("Failed to read binary header");

    int32_t magic = header[0];
    if (magic != 0x5157454E) {
        throw std::runtime_error("Invalid binary magic number. Expected 0x5157454E.");
    }

    size_t vocab_size = cfg.vocab_size;
    size_t hidden = cfg.hidden_size;
    size_t inter = cfg.intermediate_size;
    size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;

    size_t embed_elems = vocab_size * hidden;
    size_t per_layer_elems = hidden +                      // input_norm
                             (hidden * hidden) + hidden +  // q_proj w + b
                             (kv_dim * hidden) + kv_dim +  // k_proj w + b
                             (kv_dim * hidden) + kv_dim +  // v_proj w + b
                             (hidden * hidden) +           // o_proj w
                             hidden +                      // post_attn_norm
                             (inter * hidden) +            // gate_proj
                             (inter * hidden) +            // up_proj
                             (hidden * inter);             // down_proj
    size_t final_norm_elems = hidden;

    weights.total_elements = embed_elems + (cfg.num_layers * per_layer_elems) + final_norm_elems;
    weights.total_bytes = weights.total_elements * sizeof(half);

    std::cout << "[+] Allocating " << (weights.total_bytes / (1024.0 * 1024.0 * 1024.0))
              << " GB monolithic device memory buffer..." << std::endl;
    CUDA_CHECK(cudaMalloc(&weights.d_pool, weights.total_bytes));

    const size_t chunk_bytes = 128 * 1024 * 1024;
    std::vector<char> host_chunk(chunk_bytes);
    size_t bytes_remaining = weights.total_bytes;
    size_t bytes_copied = 0;

    while (bytes_remaining > 0) {
        size_t current_read = std::min(bytes_remaining, chunk_bytes);
        file.read(host_chunk.data(), current_read);
        if (!file && bytes_remaining > current_read) {
            throw std::runtime_error("Unexpected EOF while reading weights binary.");
        }
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char*>(weights.d_pool) + bytes_copied,
                              host_chunk.data(),
                              current_read,
                              cudaMemcpyHostToDevice));
        bytes_copied += current_read;
        bytes_remaining -= current_read;
    }
    file.close();
    std::cout << "[+] Weights streamed directly to VRAM successfully." << std::endl;

    half* ptr = weights.d_pool;
    weights.embed_tokens = ptr;
    ptr += embed_elems;

    for (int i = 0; i < cfg.num_layers; ++i) {
        LayerWeights& l = weights.layers[i];
        l.input_norm = ptr; ptr += hidden;
        l.q_weight = ptr;   ptr += (hidden * hidden);
        l.q_bias = ptr;     ptr += hidden;
        l.k_weight = ptr;   ptr += (kv_dim * hidden);
        l.k_bias = ptr;     ptr += kv_dim;
        l.v_weight = ptr;   ptr += (kv_dim * hidden);
        l.v_bias = ptr;     ptr += kv_dim;
        l.o_weight = ptr;   ptr += (hidden * hidden);
        l.post_attn_norm = ptr; ptr += hidden;
        l.gate_weight = ptr; ptr += (inter * hidden);
        l.up_weight = ptr;   ptr += (inter * hidden);
        l.down_weight = ptr; ptr += (hidden * inter);
    }

    weights.final_norm = ptr;
    ptr += final_norm_elems;
    weights.lm_head = weights.embed_tokens;

    size_t assigned_elements = static_cast<size_t>(ptr - weights.d_pool);
    if (assigned_elements != weights.total_elements) {
        throw std::runtime_error("Pointer slicing mismatch in weight loader!");
    }
}

void free_model_weights(QwenWeights& weights) {
    if (weights.d_pool) {
        CUDA_CHECK(cudaFree(weights.d_pool));
        weights.d_pool = nullptr;
    }
}

void init_kv_cache(const QwenConfig& cfg, KVCache& kv_cache) {
    size_t elements = static_cast<size_t>(cfg.num_layers) * cfg.max_seq_len *
                      cfg.num_kv_heads * cfg.head_dim;
    kv_cache.size_per_cache_bytes = elements * sizeof(half);

    std::cout << "[+] Allocating KV Cache (" 
              << (2 * kv_cache.size_per_cache_bytes / (1024.0 * 1024.0)) 
              << " MB total)..." << std::endl;

    CUDA_CHECK(cudaMalloc(&kv_cache.d_k_cache, kv_cache.size_per_cache_bytes));
    CUDA_CHECK(cudaMalloc(&kv_cache.d_v_cache, kv_cache.size_per_cache_bytes));
    CUDA_CHECK(cudaMemset(kv_cache.d_k_cache, 0, kv_cache.size_per_cache_bytes));
    CUDA_CHECK(cudaMemset(kv_cache.d_v_cache, 0, kv_cache.size_per_cache_bytes));
}

void free_kv_cache(KVCache& kv_cache) {
    if (kv_cache.d_k_cache) CUDA_CHECK(cudaFree(kv_cache.d_k_cache));
    if (kv_cache.d_v_cache) CUDA_CHECK(cudaFree(kv_cache.d_v_cache));
    kv_cache.d_k_cache = nullptr;
    kv_cache.d_v_cache = nullptr;
}

void init_run_state(const QwenConfig& cfg, RunState& state) {
    size_t hidden = cfg.hidden_size;
    size_t inter = cfg.intermediate_size;
    size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
    size_t vocab = cfg.vocab_size;

    size_t total_elems = hidden +              // x
                         hidden +              // xb
                         hidden +              // q
                         kv_dim +              // k
                         kv_dim +              // v
                         hidden +              // att_out
                         inter +               // gate
                         inter +               // up
                         hidden +              // mlp_out
                         vocab;                // logits

    state.scratch_pool_bytes = total_elems * sizeof(half) + sizeof(int);
    CUDA_CHECK(cudaMalloc(&state.d_scratch_pool, state.scratch_pool_bytes));

    half* ptr = reinterpret_cast<half*>(state.d_scratch_pool);
    state.x = ptr;          ptr += hidden;
    state.xb = ptr;         ptr += hidden;
    state.q = ptr;          ptr += hidden;
    state.k = ptr;          ptr += kv_dim;
    state.v = ptr;          ptr += kv_dim;
    state.att_out = ptr;    ptr += hidden;
    state.gate = ptr;       ptr += inter;
    state.up = ptr;         ptr += inter;
    state.mlp_out = ptr;    ptr += hidden;
    state.logits = ptr;     ptr += vocab;

    state.d_next_token_id = reinterpret_cast<int*>(ptr);
}

void free_run_state(RunState& state) {
    if (state.d_scratch_pool) {
        CUDA_CHECK(cudaFree(state.d_scratch_pool));
        state.d_scratch_pool = nullptr;
    }
}

// -----------------------------------------------------------------------------
// cuBLAS Linear Projection Layer: Y = X * W^T
// -----------------------------------------------------------------------------
void matmul_transB(
    cublasHandle_t handle,
    half* Y,
    const half* X,
    const half* W,
    int M, int N, int K
) {
    const half alpha = __float2half(1.0f);
    const half beta  = __float2half(0.0f);

    CUBLAS_CHECK(cublasHgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, K,
        X, K,
        &beta,
        Y, N
    ));
}

// -----------------------------------------------------------------------------
// Single Transformer Layer Forward Pass
// -----------------------------------------------------------------------------
void forward_layer(
    int layer_idx,
    int pos,
    const QwenConfig& cfg,
    const LayerWeights& layer,
    half* k_cache_layer,
    half* v_cache_layer,
    RunState& state,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    int kv_dim = cfg.num_kv_heads * cfg.head_dim;

    // 1. Input RMSNorm with residual accumulation
    if (layer_idx == 0) {
        launch_rmsnorm(state.xb, state.x, nullptr, layer.input_norm,
                       cfg.rms_norm_eps, cfg.hidden_size, 1, stream);
    } else {
        launch_rmsnorm(state.xb, state.mlp_out, state.x, layer.input_norm,
                       cfg.rms_norm_eps, cfg.hidden_size, 1, stream);
    }

    // 2. QKV Linear Projections
    matmul_transB(cublas_handle, state.q, state.xb, layer.q_weight, 1, cfg.hidden_size, cfg.hidden_size);
    matmul_transB(cublas_handle, state.k, state.xb, layer.k_weight, 1, kv_dim, cfg.hidden_size);
    matmul_transB(cublas_handle, state.v, state.xb, layer.v_weight, 1, kv_dim, cfg.hidden_size);

    // 3. Fused Bias Add + RoPE + KV Cache Storage
    launch_rope_bias_kv(
        state.q, state.k, state.v,
        layer.q_bias, layer.k_bias, layer.v_bias,
        k_cache_layer, v_cache_layer,
        pos,
        cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
        cfg.rope_theta,
        stream
    );

    // 4. Vectorized Grouped-Query Attention (GQA) Decode
    launch_gqa_decode(
        state.att_out,
        state.q,
        k_cache_layer,
        v_cache_layer,
        pos,
        cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
        stream
    );

    // 5. Output Projection (W_o)
    matmul_transB(cublas_handle, state.xb, state.att_out, layer.o_weight, 1, cfg.hidden_size, cfg.hidden_size);

    // 6. Post-Attention RMSNorm with Fused Residual Accumulation
    launch_rmsnorm(state.xb, state.xb, state.x, layer.post_attn_norm,
                   cfg.rms_norm_eps, cfg.hidden_size, 1, stream);

    // 7. MLP Gate and Up Projections
    matmul_transB(cublas_handle, state.gate, state.xb, layer.gate_weight, 1, cfg.intermediate_size, cfg.hidden_size);
    matmul_transB(cublas_handle, state.up,   state.xb, layer.up_weight,   1, cfg.intermediate_size, cfg.hidden_size);

    // 8. Fused SwiGLU Activation
    launch_swiglu(state.gate, state.gate, state.up, cfg.intermediate_size, 1, stream);

    // 9. MLP Down Projection
    matmul_transB(cublas_handle, state.mlp_out, state.gate, layer.down_weight, 1, cfg.hidden_size, cfg.intermediate_size);
}

// -----------------------------------------------------------------------------
// Complete 28-Layer Transformer Forward Pass
// -----------------------------------------------------------------------------
int forward_token(
    int input_token,
    int pos,
    const QwenConfig& cfg,
    const QwenWeights& weights,
    KVCache& kv_cache,
    RunState& state,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    // 1. Embedding lookup
    const half* embed_src = weights.embed_tokens + (static_cast<size_t>(input_token) * cfg.hidden_size);
    CUDA_CHECK(cudaMemcpyAsync(
        state.x,
        embed_src,
        cfg.hidden_size * sizeof(half),
        cudaMemcpyDeviceToDevice,
        stream
    ));

    size_t layer_kv_elems = static_cast<size_t>(cfg.max_seq_len) * cfg.num_kv_heads * cfg.head_dim;

    // 2. Transformer layers (0 .. 27)
    for (int l = 0; l < cfg.num_layers; ++l) {
        half* k_cache_layer = kv_cache.d_k_cache + (l * layer_kv_elems);
        half* v_cache_layer = kv_cache.d_v_cache + (l * layer_kv_elems);

        forward_layer(
            l, pos, cfg,
            weights.layers[l],
            k_cache_layer, v_cache_layer,
            state,
            cublas_handle,
            stream
        );
    }

    // 3. Final RMSNorm
    launch_rmsnorm(
        state.xb,
        state.mlp_out,
        state.x,
        weights.final_norm,
        cfg.rms_norm_eps,
        cfg.hidden_size,
        1,
        stream
    );

    // 4. LM Head Projection
    matmul_transB(
        cublas_handle,
        state.logits,
        state.xb,
        weights.lm_head,
        1,
        cfg.vocab_size,
        cfg.hidden_size
    );

    // 5. Argmax Sampling
    launch_argmax(state.d_next_token_id, state.logits, cfg.vocab_size, stream);

    // 6. Copy next token ID to host
    int next_token = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &next_token,
        state.d_next_token_id,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return next_token;
}