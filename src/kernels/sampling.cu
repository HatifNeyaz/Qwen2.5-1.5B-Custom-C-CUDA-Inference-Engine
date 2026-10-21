#include "kernels.cuh"
#include <cuda_fp16.h>

namespace {

__global__ void argmax_kernel(
    int* __restrict__ d_out_idx,
    const half* __restrict__ logits,
    int vocab_size
) {
    int tid = threadIdx.x;
    float max_val = -1e30f;
    int max_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __half2float(logits[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    __shared__ float s_max_vals[512];
    __shared__ int s_max_indices[512];

    s_max_vals[tid] = max_val;
    s_max_indices[tid] = max_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max_vals[tid + stride] > s_max_vals[tid]) {
                s_max_vals[tid] = s_max_vals[tid + stride];
                s_max_indices[tid] = s_max_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_out_idx = s_max_indices[0];
    }
}

} // namespace

void launch_argmax(
    int* d_out_idx,
    const half* logits,
    int vocab_size,
    cudaStream_t stream
) {
    constexpr int block_threads = 512;
    argmax_kernel<<<1, block_threads, 0, stream>>>(d_out_idx, logits, vocab_size);
}