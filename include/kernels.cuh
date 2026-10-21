#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void launch_rmsnorm(
    half* out,
    half* x,
    half* residual,
    const half* weight,
    float eps,
    int hidden_size,
    int num_tokens = 1,
    cudaStream_t stream = 0
);

void launch_rope_bias_kv(
    half* q,
    half* k,
    half* v,
    const half* q_bias,
    const half* k_bias,
    const half* v_bias,
    half* k_cache,
    half* v_cache,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rope_theta = 1000000.0f,
    cudaStream_t stream = 0
);

void launch_swiglu(
    half* out,
    const half* gate,
    const half* up,
    int intermediate_size,
    int num_tokens = 1,
    cudaStream_t stream = 0
);

void launch_gqa_decode(
    half* out,
    const half* q,
    const half* k_cache,
    const half* v_cache,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream = 0
);

void launch_argmax(
    int* d_out_idx,
    const half* logits,
    int vocab_size,
    cudaStream_t stream = 0
);