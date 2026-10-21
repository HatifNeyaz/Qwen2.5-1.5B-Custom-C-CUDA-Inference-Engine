#include "kernels.cuh"
#include <cuda_fp16.h>

namespace {

// Warp-level reduction across 32 lanes using XOR shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Block-level reduction combining warp reductions via shared memory
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float s_warp_sums[32]; // Supports up to 1024 threads (32 warps)
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane_id == 0) {
        s_warp_sums[warp_id] = val;
    }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    float warp_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0f;
    warp_sum = warp_reduce_sum(warp_sum);

    return warp_sum;
}

__global__ void rmsnorm_kernel(
    half* __restrict__ out,
    half* __restrict__ x,
    half* __restrict__ residual,
    const half* __restrict__ weight,
    float eps,
    int hidden_size
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int half2_dim = hidden_size / 2;

    // Offset pointers to the current token's row
    half2* out_h2 = reinterpret_cast<half2*>(out + token_idx * hidden_size);
    half2* x_h2 = reinterpret_cast<half2*>(x + token_idx * hidden_size);
    half2* res_h2 = residual ? reinterpret_cast<half2*>(residual + token_idx * hidden_size) : nullptr;
    const half2* w_h2 = reinterpret_cast<const half2*>(weight);

    float sum_sq = 0.0f;

    // Phase 1: Accumulate residual (if present) and compute sum of squares in FP32
    for (int i = tid; i < half2_dim; i += blockDim.x) {
        half2 val = x_h2[i];
        if (res_h2 != nullptr) {
            val = __hadd2(val, res_h2[i]);
            res_h2[i] = val; // Store back to residual stream
        }
        float2 f = __half22float2(val);
        sum_sq += f.x * f.x + f.y * f.y;
    }

    // Phase 2: Compute RMS across block
    float block_sum = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf((block_sum / static_cast<float>(hidden_size)) + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    // Phase 3: Scale by inv_rms * gamma (weight)
    for (int i = tid; i < half2_dim; i += blockDim.x) {
        half2 val = (res_h2 != nullptr) ? res_h2[i] : x_h2[i];
        half2 w = w_h2[i];

        float2 v_f = __half22float2(val);
        float2 w_f = __half22float2(w);

        float2 out_f;
        out_f.x = v_f.x * inv_rms * w_f.x;
        out_f.y = v_f.y * inv_rms * w_f.y;

        out_h2[i] = __float22half2_rn(out_f);
    }
}

} // namespace

void launch_rmsnorm(
    half* out,
    half* x,
    half* residual,
    const half* weight,
    float eps,
    int hidden_size,
    int num_tokens,
    cudaStream_t stream
) {
    constexpr int block_threads = 256;
    dim3 grid(num_tokens);
    dim3 block(block_threads);

    rmsnorm_kernel<<<grid, block, 0, stream>>>(out, x, residual, weight, eps, hidden_size);
}