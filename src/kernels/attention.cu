#include "kernels.cuh"
#include <cuda_fp16.h>
#include <cmath>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__global__ void gqa_decode_kernel(
    half* __restrict__ out,
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int h = blockIdx.x;              // Query head index: 0 .. 11
    int tid = threadIdx.x;           // Dimension index: 0 .. 127
    int lane_id = tid % 32;
    int warp_id = tid / 32;          // 4 warps per block

    // Map Query head to corresponding KV head (12 / 2 = 6 query heads per KV head)
    int kv_heads_per_q = num_heads / num_kv_heads; // 6
    int kv_h = h / kv_heads_per_q;                 // 0 or 1

    // 1. Cache current query vector in shared memory
    __shared__ float s_q[128];
    s_q[tid] = __half2float(q[h * head_dim + tid]);
    __syncthreads();

    float q_val = s_q[tid];

    // Shared memory for warp dot-product reduction
    __shared__ float s_warp_sums[4];
    __shared__ float s_score;

    // Online softmax tracking variables
    float m = -1e20f;  // Running max
    float l = 0.0f;    // Running sum of exponentials
    float acc = 0.0f;  // Running weighted V accumulator for dimension tid

    int kv_stride_token = num_kv_heads * head_dim; // 2 * 128 = 256

    // 2. Iterate across all cached tokens up to pos
    for (int t = 0; t <= pos; ++t) {
        int kv_offset = t * kv_stride_token + kv_h * head_dim + tid;

        float k_val = __half2float(k_cache[kv_offset]);
        float prod = q_val * k_val;

        // Warp reduction
        float warp_sum = warp_reduce_sum(prod);
        if (lane_id == 0) {
            s_warp_sums[warp_id] = warp_sum;
        }
        __syncthreads();

        // Block reduction performed by Warp 0
        if (warp_id == 0) {
            float b_sum = (lane_id < 4) ? s_warp_sums[lane_id] : 0.0f;
            b_sum = warp_reduce_sum(b_sum);
            if (lane_id == 0) {
                s_score = b_sum * scale;
            }
        }
        __syncthreads();

        float score = s_score;
        float v_val = __half2float(v_cache[kv_offset]);

        // Online Softmax numerical update
        if (t == 0) {
            m = score;
            l = 1.0f;
            acc = v_val;
        } else {
            float m_prev = m;
            m = fmaxf(m, score);
            float alpha = __expf(m_prev - m);
            float beta = __expf(score - m);
            l = l * alpha + beta;
            acc = acc * alpha + beta * v_val;
        }
    }

    // 3. Final normalization and store to global memory
    float out_val = (l > 0.0f) ? (acc / l) : 0.0f;
    out[h * head_dim + tid] = __float2half_rn(out_val);
}

} // namespace

void launch_gqa_decode(
    half* out,
    const half* q,
    const half* k_cache,
    const half* v_cache,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    dim3 grid(num_heads);  // 12 blocks (1 per Query head)
    dim3 block(head_dim);  // 128 threads (1 per channel dimension)

    gqa_decode_kernel<<<grid, block, 0, stream>>>(
        out, q, k_cache, v_cache,
        pos, num_heads, num_kv_heads, head_dim, scale
    );
}