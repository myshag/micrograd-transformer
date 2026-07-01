"""
Запуск настоящей LLM (EleutherAI Pythia-70M, GPT-NeoX) на нашем движке.

Веса скачиваем с HuggingFace (transformers используем только как десериализатор
весов и токенизатор — это ввод-вывод). Сам forward считается на НАШЕМ Tensor:
matmul, softmax, LayerNorm, GELU. RoPE, раскладка голов, маска — numpy (это
перестановки данных, как и в настоящих фреймворках).

Особенности GPT-NeoX: объединённый QKV, rotary-эмбеддинги (RoPE, частичные),
параллельный residual (attn и mlp читают из одного x), GELU.

Сверяем логиты с эталоном HF, затем генерируем текст.

Требуется: transformers (pip). Запуск:  python3 pythia.py
"""

import numpy as np

from tensor import Tensor, no_grad

MODEL = "EleutherAI/pythia-70m"
C = dict(hid=512, nl=6, nh=8, hd=64, inter=2048, vocab=50304,
         rot=16, base=10000, eps=1e-5)   # rot = head_dim * rotary_pct(0.25)


def load_weights(name=MODEL):
    from transformers import AutoModelForCausalLM
    sd = AutoModelForCausalLM.from_pretrained(name).state_dict()
    return {k: v.numpy().astype(np.float32) for k, v in sd.items()}


# --- слои на нашем Tensor -----------------------------------------------------

def layernorm(h, g, b, eps):
    mu = h.mean(axis=-1, keepdims=True)
    xc = h - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    return xc / (var + eps) ** 0.5 * Tensor(g) + Tensor(b)


def linear(h, W, b=None):
    out = h @ Tensor(np.ascontiguousarray(W.T))    # W: (out,in) -> h@Wᵀ
    return out if b is None else out + Tensor(b)


# --- RoPE (numpy: это поворот данных) ----------------------------------------

def rope_cache(T, rot, base):
    inv = 1.0 / (base ** (np.arange(0, rot, 2) / rot))     # (rot/2,)
    freqs = np.outer(np.arange(T), inv)                    # (T, rot/2)
    emb = np.concatenate([freqs, freqs], axis=-1)          # (T, rot)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x, cos, sin, rot):        # x: (T, nh, hd)
    xr, xp = x[..., :rot], x[..., rot:]
    half = rot // 2
    rh = np.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
    return np.concatenate([xr * cos[:, None, :] + rh * sin[:, None, :], xp], axis=-1)


# --- forward всей модели ------------------------------------------------------

def forward(W, ids):
    c, T = C, len(ids)
    with no_grad():
        h = Tensor(W["gpt_neox.embed_in.weight"][ids])     # эмбеддинг (T, D)
        cos, sin = rope_cache(T, c["rot"], c["base"])
        mask = np.triu(np.full((T, T), -1e9, np.float32), 1)

        for i in range(c["nl"]):
            p = f"gpt_neox.layers.{i}."
            ln1 = layernorm(h, W[p + "input_layernorm.weight"],
                            W[p + "input_layernorm.bias"], c["eps"])
            # объединённый QKV -> (T, nh, 3*hd) -> q,k,v
            qkv = linear(ln1, W[p + "attention.query_key_value.weight"],
                         W[p + "attention.query_key_value.bias"]).data
            qkv = qkv.reshape(T, c["nh"], 3 * c["hd"])
            q, k, v = qkv[..., :c["hd"]], qkv[..., c["hd"]:2 * c["hd"]], qkv[..., 2 * c["hd"]:]
            q = apply_rope(q, cos, sin, c["rot"])
            k = apply_rope(k, cos, sin, c["rot"])
            # (T, nh, hd) -> (nh, T, hd); внимание батчево по головам
            q, k, v = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
            scores = (Tensor(np.ascontiguousarray(q)) @ Tensor(np.ascontiguousarray(k.transpose(0, 2, 1)))) \
                * (1.0 / np.sqrt(c["hd"]))
            scores = scores + Tensor(mask)                 # causal
            ctx = (scores.softmax(axis=-1) @ Tensor(np.ascontiguousarray(v))).data
            ctx = ctx.transpose(1, 0, 2).reshape(T, c["hid"])
            attn = linear(Tensor(ctx), W[p + "attention.dense.weight"],
                          W[p + "attention.dense.bias"])

            ln2 = layernorm(h, W[p + "post_attention_layernorm.weight"],
                            W[p + "post_attention_layernorm.bias"], c["eps"])
            m = linear(ln2, W[p + "mlp.dense_h_to_4h.weight"],
                       W[p + "mlp.dense_h_to_4h.bias"]).gelu()
            m = linear(m, W[p + "mlp.dense_4h_to_h.weight"],
                       W[p + "mlp.dense_4h_to_h.bias"])
            h = h + attn + m                               # параллельный residual

        h = layernorm(h, W["gpt_neox.final_layer_norm.weight"],
                      W["gpt_neox.final_layer_norm.bias"], c["eps"])
        return linear(h, W["embed_out.weight"]).data       # логиты (T, vocab)


def generate(W, tok, prompt, n=30):
    ids = tok(prompt)["input_ids"]
    for _ in range(n):
        logits = forward(W, ids)
        ids.append(int(logits[-1].argmax()))              # greedy
    return tok.decode(ids)


def main():
    from transformers import AutoTokenizer
    print("Загружаю Pythia-70M...")
    W = load_weights()
    tok = AutoTokenizer.from_pretrained(MODEL)

    # --- сверка логитов с эталоном HF ---
    prompt = "The capital of France is"
    ids = tok(prompt)["input_ids"]
    ours = forward(W, ids)
    import torch
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(MODEL)(
        torch.tensor([ids])).logits[0].detach().numpy()
    print(f"\n=== Сверка forward с PyTorch/HF ===")
    print(f"max|наш - HF| = {np.abs(ours - ref).max():.2e}")
    print(f"argmax next token: наш={ours[-1].argmax()}, HF={ref[-1].argmax()}, "
          f"совпал={ours[-1].argmax() == ref[-1].argmax()}")

    # --- генерация НА НАШЕМ ДВИЖКЕ ---
    print("\n=== Генерация на нашем движке ===")
    print(generate(W, tok, prompt, n=30))


if __name__ == "__main__":
    main()
