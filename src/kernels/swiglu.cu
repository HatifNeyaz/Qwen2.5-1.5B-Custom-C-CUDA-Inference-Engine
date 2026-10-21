#include "kernels.cuh"
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void swiglu_kernel(
    half* __restrict__ out,
    const half* __restrict__ gate,
    const half* __restrict__ up,
    int num_elements
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int half2_elements = num_elements / 2;

    const half2* gate_h2 = reinterpret_cast<const half2*>(gate);
    const half2* up_h2 = reinterpret_cast<const half2*>(up);
    half2* out_h2 = reinterpret_cast<half2*>(out);

    if (idx < half2_elements) {
        half2 g = gate_h2[idx];
        half2 u = up_h2[idx];

        float2 g_f = __half22float2(g);
        float2 u_f = __half22float2(u);

        float2 res_f;
        res_f.x = silu(g_f.x) * u_f.x;
        res_f.y = silu(g_f.y) * u_f.y;

        out_h2[idx] = __float22half2_rn(res_f);
    }
}

} // namespace

void launch_swiglu(
    half* out,
    const half* gate,
    const half* up,
    int intermediate_size,
    int num_tokens,
    cudaStream_t stream
) {
    int total_elements = num_tokens * intermediate_size;
    int half2_elements = total_elements / 2;

    constexpr int block_threads = 256;
    int grid_blocks = (half2_elements + block_threads - 1) / block_threads;

    swiglu_kernel<<<grid_blocks, block_threads, 0, stream>>>(
        out, gate, up, total_elements
    );
}