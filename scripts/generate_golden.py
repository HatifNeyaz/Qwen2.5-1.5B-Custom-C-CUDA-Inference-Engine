from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "models/qwen2.5-1.5b"
GOLDEN_DIR = Path("tests/golden")
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

def dump_tensor(t: torch.Tensor, name: str):
    t_fp16 = t.detach().squeeze().to(torch.float16).cpu().contiguous()
    path = GOLDEN_DIR / f"{name}.bin"
    t_fp16.numpy().tofile(path)
    print(f"[+] Dumped {name:28s} | shape: {str(list(t_fp16.shape)):15s} -> {path}")

def generate():
    print("[+] Loading PyTorch model for golden verification...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, 
        dtype=torch.float16,
        low_cpu_mem_usage=False
    )
    model.eval()

    prompt = "Hello"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    token_id = input_ids[0, 0].item()
    print(f"[+] Verification Token: '{prompt}' (ID: {token_id})")

    with open(GOLDEN_DIR / "input_token.txt", "w") as f:
        f.write(str(token_id))

    with torch.no_grad():
        # 1. Input Embedding
        x = model.model.embed_tokens(input_ids) # [1, 1, 1536]
        dump_tensor(x, "01_embed_out")

        l0 = model.model.layers[0]

        # 2. Input RMSNorm
        x_norm = l0.input_layernorm(x)
        dump_tensor(x_norm, "02_input_norm_out")

        # 3. Q, K, V with Bias
        q = l0.self_attn.q_proj(x_norm)
        k = l0.self_attn.k_proj(x_norm)
        v = l0.self_attn.v_proj(x_norm)
        dump_tensor(q, "03_q_proj_biased")
        dump_tensor(k, "03_k_proj_biased")
        dump_tensor(v, "03_v_proj_biased")

        # 4. RoPE
        position_ids = torch.tensor([[0]], dtype=torch.long)
        cos, sin = model.model.rotary_emb(x, position_ids)

        # 5. True Attention Output (at pos = 0, attention reduces to V expanded to 12 heads)
        v_heads = v.view(1, 1, 2, 128)
        v_rep = v_heads.repeat_interleave(6, dim=2).view(1, 1, 1536)
        attn_out = l0.self_attn.o_proj(v_rep)
        dump_tensor(attn_out, "05_attn_out")

        # 6. Attention Residual Accumulation
        attn_residual = x + attn_out
        dump_tensor(attn_residual, "06_attn_residual")

        # 7. True Post-Attention RMSNorm
        post_norm = l0.post_attention_layernorm(attn_residual)
        dump_tensor(post_norm, "07_post_attn_norm_out")

        # 8. True SwiGLU Activation
        gate = l0.mlp.gate_proj(post_norm)
        up = l0.mlp.up_proj(post_norm)
        swiglu = torch.nn.functional.silu(gate) * up
        dump_tensor(gate, "08_gate_proj_out")
        dump_tensor(up, "08_up_proj_out")
        dump_tensor(swiglu, "08_layer0_swiglu_out")

        # 9. True Layer 0 Final Output
        down = l0.mlp.down_proj(swiglu)
        l0_out = attn_residual + down
        dump_tensor(l0_out, "09_layer0_out")

        # 10. Full Model 28-Layer Greedy Token Prediction
        logits = model(input_ids).logits # [1, 1, 151936]
        pred_token = torch.argmax(logits[0, -1, :]).item()
        pred_word = tokenizer.decode([pred_token])
        print(f"\n[+] PyTorch Full-Model Greedy Next Token: {pred_token} ('{pred_word}')")

        with open(GOLDEN_DIR / "golden_next_token.txt", "w") as f:
            f.write(str(pred_token))

    print("\n[✓] Golden reference activations dumped successfully to tests/golden/")

if __name__ == "__main__":
    generate()