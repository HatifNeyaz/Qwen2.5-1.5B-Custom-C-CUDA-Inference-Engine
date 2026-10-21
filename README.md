# Qwen2.5-1.5B Custom C++/CUDA Inference Engine

A bare-metal, dependency-light inference runtime for **Qwen2.5-1.5B-Instruct** written from scratch in C++17 and CUDA. Designed to run high-throughput autoregressive generation on consumer GPUs with limited VRAM (e.g., GTX 1660 Ti 6GB).

---

## Architecture & Features

- **Monolithic Device Allocation:** Weights are packed into a single contiguous FP16 binary and loaded directly into VRAM with zero runtime `cudaMalloc` allocations.
- **Tied Embedding Re-use:** Leverages Qwen's tied input/output vocabulary projection to map the `lm_head` directly to `embed_tokens`, saving ~466 MB of VRAM.
- **Custom Fused CUDA Kernels:**
  - **RMSNorm + Residual:** Vectorized `half2` kernel with warp-level reductions for fused activation normalization and residual stream updates.
  - **RoPE + Bias + KV Cache:** Adds $Q, K, V$ bias projections, computes Hugging Face-style NeoX rotary embeddings, and stores $K, V$ directly into the pre-allocated cache arena.
  - **SwiGLU:** Vectorized `half2` element-wise activation kernel.
  - **GQA Decode Attention:** Vectorized multi-head decode kernel utilizing register-level Flash-style online softmax and head grouping ($12:2$ Query-to-KV ratio).
  - **Argmax Reduction:** In-VRAM parallel tree reduction for greedy sampling, transferring only a single 4-byte token ID over PCIe per decode step.
- **cuBLAS Acceleration:** Linear projections mapped to Tensor Core-accelerated `cublasHgemm` (`CUBLAS_OP_T`).
- **Native BPE Tokenizer:** Standalone C++ tokenizer utilizing `nlohmann/json` to load `tokenizer.json`, handling byte-level fallback mappings, regex splits, and ChatML formatting.

---

## Memory Footprint (FP16)

| Component | VRAM Allocated |
| :--- | :--- |
| Model Weights | ~3.08 GB |
| Static KV Cache (2048 Context) | ~58.7 MB |
| Activation Scratchpads | ~1.2 MB |
| **Total Engine VRAM** | **~3.14 GB** |

---

## Directory Layout

```text
.
├── CMakeLists.txt          # Build configuration (sm_75+, cuBLAS, cudart)
├── include/                # Header declarations and utilities
├── src/                    # Model orchestrator, tokenizer, and main CLI
│   └── kernels/            # Custom CUDA kernels
├── scripts/                # Offline serialization and PyTorch parity scripts
├── tests/                  # Unit tests and end-to-end forward benchmarks
└── models/                 # Model assets (safetensors, tokenizer, binary)
```

# PrerequisitesNVIDIA 
* GPU with Compute Capability >= 7.5$ (Turing, Ampere, Ada, Hopper)
* CUDA Toolkit >= 11.8$ (nvcc, cublas)
* CMake >= 3.18$ and a C++17 compiler (g++ or clang++)
* Python >= 3.10$ with torch, transformers, and safetensors (for offline export only)

Quickstart
1. Download Model Assets

``` Bash
pip install huggingface_hub safetensors torch transformers
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  config.json tokenizer.json tokenizer_config.json model.safetensors \
  --local-dir ./models/qwen2.5-1.5b
```
2. Export Weights to Flat Binary
Convert the downloaded Hugging Face safetensors into a contiguous execution-ordered binary:

```Bash
python3 scripts/export_weights.py
```

This generates models/qwen2.5-1.5b/qwen2.5_1.5b_fp16.bin (~3.08 GB).

3. Build the Engine
``` Bash
mkdir -p build && cd build
cmake ..
make qwen_infer -j$(nproc)
```
4. Run Interactive Chat
``` Bash
./qwen_infer
```
# Kernel & Pipeline Unit Testing
Each kernel includes an isolated unit test validating numerical bit-parity against PyTorch golden activations:

```Bash
# Generate ground-truth activations
python3 scripts/generate_golden.py

# Build and run individual unit tests
cd build
make -j$(nproc)

./test_rmsnorm      # Validates Fused RMSNorm & Residual update
./test_rope_bias    # Validates Rotary Position Embeddings & Bias
./test_swiglu       # Validates Fused SwiGLU activation
./test_attention    # Validates GQA decode kernel against analytical reference
./test_forward      # Validates full 28-layer pipeline & benchmarks decode latency
```