import struct
import json
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
import torch

MODEL_DIR = Path("models/qwen2.5-1.5b")
CONFIG_PATH = MODEL_DIR / "config.json"
SAFETENSORS_PATH = MODEL_DIR / "model.safetensors"
OUTPUT_BIN_PATH = MODEL_DIR / "qwen2.5_1.5b_fp16.bin"

def export():
    print(f"[+] Loading config from {CONFIG_PATH}...")
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    vocab_size = cfg["vocab_size"]           # 151936
    hidden_size = cfg["hidden_size"]         # 1536
    intermediate_size = cfg["intermediate_size"] # 8960
    num_layers = cfg["num_hidden_layers"]   # 28
    num_heads = cfg["num_attention_heads"]   # 12
    num_kv_heads = cfg["num_key_value_heads"] # 2
    head_dim = hidden_size // num_heads      # 128
    max_seq_len = 2048                       # Inference budget context

    print(f"[+] Model Architecture:")
    print(f"    Layers: {num_layers}, Hidden Size: {hidden_size}, Inter Size: {intermediate_size}")
    print(f"    Heads: {num_heads}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}")

    print(f"[+] Loading weights from {SAFETENSORS_PATH}...")
    state_dict = load_file(str(SAFETENSORS_PATH), device="cpu")

    def write_tensor(f, tensor: torch.Tensor, expected_shape: tuple, name: str):
        assert tensor.shape == expected_shape, f"Shape mismatch for {name}: {tensor.shape} vs {expected_shape}"
        # Cast to float16 and ensure contiguous layout
        data = tensor.to(torch.float16).contiguous().numpy()
        data.tofile(f)

    print(f"[+] Writing binary weights to {OUTPUT_BIN_PATH}...")
    with open(OUTPUT_BIN_PATH, "wb") as f:
        # 1. Write Header (16 ints = 64 bytes)
        # Magic: 'QWEN' (0x5157454E)
        magic = 0x5157454E
        header = struct.pack(
            "16i",
            magic,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            0, 0, 0, 0, 0, 0, 0  # Padding for future metadata
        )
        f.write(header)

        # 2. Embedding weights: [vocab_size, hidden_size]
        print("[+] Writing embed_tokens...")
        write_tensor(f, state_dict["model.embed_tokens.weight"], 
                     (vocab_size, hidden_size), "embed_tokens")

        # 3. Layer Blocks
        for i in range(num_layers):
            prefix = f"model.layers.{i}."
            print(f"[+] Writing Layer {i}/{num_layers}...")

            # Input RMSNorm
            write_tensor(f, state_dict[f"{prefix}input_layernorm.weight"], 
                         (hidden_size,), f"L{i}.input_norm")

            # Q, K, V Projections (weights + biases)
            write_tensor(f, state_dict[f"{prefix}self_attn.q_proj.weight"], 
                         (hidden_size, hidden_size), f"L{i}.q_proj.weight")
            write_tensor(f, state_dict[f"{prefix}self_attn.q_proj.bias"], 
                         (hidden_size,), f"L{i}.q_proj.bias")

            write_tensor(f, state_dict[f"{prefix}self_attn.k_proj.weight"], 
                         (num_kv_heads * head_dim, hidden_size), f"L{i}.k_proj.weight")
            write_tensor(f, state_dict[f"{prefix}self_attn.k_proj.bias"], 
                         (num_kv_heads * head_dim,), f"L{i}.k_proj.bias")

            write_tensor(f, state_dict[f"{prefix}self_attn.v_proj.weight"], 
                         (num_kv_heads * head_dim, hidden_size), f"L{i}.v_proj.weight")
            write_tensor(f, state_dict[f"{prefix}self_attn.v_proj.bias"], 
                         (num_kv_heads * head_dim,), f"L{i}.v_proj.bias")

            # Out projection
            write_tensor(f, state_dict[f"{prefix}self_attn.o_proj.weight"], 
                         (hidden_size, hidden_size), f"L{i}.o_proj.weight")

            # Post-Attention RMSNorm
            write_tensor(f, state_dict[f"{prefix}post_attention_layernorm.weight"], 
                         (hidden_size,), f"L{i}.post_attention_norm")

            # MLP: Gate, Up, Down
            write_tensor(f, state_dict[f"{prefix}mlp.gate_proj.weight"], 
                         (intermediate_size, hidden_size), f"L{i}.gate_proj")
            write_tensor(f, state_dict[f"{prefix}mlp.up_proj.weight"], 
                         (intermediate_size, hidden_size), f"L{i}.up_proj")
            write_tensor(f, state_dict[f"{prefix}mlp.down_proj.weight"], 
                         (hidden_size, intermediate_size), f"L{i}.down_proj")

        # 4. Final RMSNorm
        print("[+] Writing final model.norm...")
        write_tensor(f, state_dict["model.norm.weight"], (hidden_size,), "final_norm")

    total_size = OUTPUT_BIN_PATH.stat().st_size
    print(f"\n[✓] Export complete!")
    print(f"[✓] Total binary size: {total_size / (1024**3):.3f} GB ({total_size:,} bytes)")

if __name__ == "__main__":
    export()