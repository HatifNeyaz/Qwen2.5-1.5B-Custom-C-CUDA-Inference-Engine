#include "kernels.cuh"
#include <cuda_fp16.h>
#include <cmath>

namespace {

__global__ void rope_bias_kv_kernel(
    half* __restrict__ q,
    half* __restrict__ k,
    half* __restrict__ v,
    const half* __restrict__ q_bias,
    const half* __restrict__ k_bias,
    const half* __restrict__ v_bias,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rope_theta
) {
    int tid = threadIdx.x;              // 0 .. 63
    int half_dim = head_dim / 2;        // 64
    int block_id = blockIdx.x;          // 0 .. 13

    // Rotary frequency: inv_freq = 1.0 / (rope_theta ^ (2 * tid / head_dim))
    float exponent = (2.0f * static_cast<float>(tid)) / static_cast<float>(head_dim);
    float inv_freq = powf(rope_theta, -exponent);
    float angle = static_cast<float>(pos) * inv_freq;
    float cos_val = __cosf(angle);
    float sin_val = __sinf(angle);

    if (block_id < num_heads) {
        // --- Query Heads (0 to 11) ---
        int h = block_id;
        int head_offset = h * head_dim;

        half q1 = q[head_offset + tid];
        half q2 = q[head_offset + tid + half_dim];

        if (q_bias != nullptr) {
            q1 = __hadd(q1, q_bias[head_offset + tid]);
            q2 = __hadd(q2, q_bias[head_offset + tid + half_dim]);
        }

        float q1_f = __half2float(q1);
        float q2_f = __half2float(q2);

        // NeoX 2D rotation
        float out1_f = q1_f * cos_val - q2_f * sin_val;
        float out2_f = q2_f * cos_val + q1_f * sin_val;

        q[head_offset + tid] = __float2half_rn(out1_f);
        q[head_offset + tid + half_dim] = __float2half_rn(out2_f);

    } else {
        // --- Key and Value Heads (0 to 1) ---
        int kv_h = block_id - num_heads;
        if (kv_h < num_kv_heads) {
            int kv_offset = kv_h * head_dim;

            // 1. Process Key (Bias + RoPE)
            half k1 = k[kv_offset + tid];
            half k2 = k[kv_offset + tid + half_dim];

            if (k_bias != nullptr) {
                k1 = __hadd(k1, k_bias[kv_offset + tid]);
                k2 = __hadd(k2, k_bias[kv_offset + tid + half_dim]);
            }

            float k1_f = __half2float(k1);
            float k2_f = __half2float(k2);

            float k_out1_f = k1_f * cos_val - k2_f * sin_val;
            float k_out2_f = k2_f * cos_val + k1_f * sin_val;

            half k_final1 = __float2half_rn(k_out1_f);
            half k_final2 = __float2half_rn(k_out2_f);

            k[kv_offset + tid] = k_final1;
            k[kv_offset + tid + half_dim] = k_final2;

            // Write to KV Cache slot: [pos, kv_head, :]
            if (k_cache != nullptr) {
                int cache_offset = pos * (num_kv_heads * head_dim) + kv_offset;
                k_cache[cache_offset + tid] = k_final1;
                k_cache[cache_offset + tid + half_dim] = k_final2;
            }

            // 2. Process Value (Bias only, NO RoPE)
            half v1 = v[kv_offset + tid];
            half v2 = v[kv_offset + tid + half_dim];

            if (v_bias != nullptr) {
                v1 = __hadd(v1, v_bias[kv_offset + tid]);
                v2 = __hadd(v2, v_bias[kv_offset + tid + half_dim]);
            }

            v[kv_offset + tid] = v1;
            v[kv_offset + tid + half_dim] = v2;

            // Write to KV Cache slot: [pos, kv_head, :]
            if (v_cache != nullptr) {
                int cache_offset = pos * (num_kv_heads * head_dim) + kv_offset;
                v_cache[cache_offset + tid] = v1;
                v_cache[cache_offset + tid + half_dim] = v2;
            }
        }
    }
}

} // namespace

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
    float rope_theta,
    cudaStream_t stream
) {
    int total_blocks = num_heads + num_kv_heads; // 12 + 2 = 14
    int block_threads = head_dim / 2;            // 64 threads

    rope_bias_kv_kernel<<<total_blocks, block_threads, 0, stream>>>(
        q, k, v,
        q_bias, k_bias, v_bias,
        k_cache, v_cache,
        pos,
        num_heads, num_kv_heads, head_dim,
        rope_theta
    );
}