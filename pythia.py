"""
Настоящая LLM (Pythia-70M, GPT-NeoX) на нашем движке — ДИФФЕРЕНЦИРУЕМАЯ.

Forward собран целиком из наших Tensor-операций (без numpy посреди), поэтому
граф не рвётся и backward доходит до каждого веса. RoPE выражен через
дифференцируемые slice (`__getitem__`) и `cat`; эмбеддинг — `index_rows`;
разрез QKV — срезами; головы — `reshape`/`swapaxes`.

Веса грузим с HuggingFace (transformers — только десериализатор). Сверяем и
forward-логиты, и backward-ГРАДИЕНТЫ с torch.autograd HF, затем генерируем текст.

Требуется: transformers. Запуск:  python3 pythia.py
"""

import numpy as np

from tensor import Tensor, cat, no_grad

MODEL = "EleutherAI/pythia-70m"
C = dict(hid=512, nl=6, nh=8, hd=64, vocab=50304, rot=16, base=10000, eps=1e-5)


def load_params(name=MODEL):
    from transformers import AutoModelForCausalLM
    sd = AutoModelForCausalLM.from_pretrained(name).state_dict()
    return {k: Tensor(v.numpy().astype(np.float32)) for k, v in sd.items()}


# --- слои (всё на Tensor) -----------------------------------------------------

def layernorm(h, g, b):
    mu = h.mean(axis=-1, keepdims=True)
    xc = h - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    return xc / (var + C["eps"]) ** 0.5 * g + b


def linear(h, W, b=None):
    out = h @ W.T                       # W: (out,in) -> h @ Wᵀ
    return out if b is None else out + b


def rope_cache(T, rot, base):
    inv = 1.0 / (base ** (np.arange(0, rot, 2) / rot))
    freqs = np.outer(np.arange(T), inv)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)                  # (T, rot)
    return (Tensor(cos[:, None, :].astype(np.float32)),
            Tensor(sin[:, None, :].astype(np.float32)))  # (T,1,rot) для broadcast


def rope(x, cos, sin, rot):             # x: (T, nh, hd) — всё дифференцируемо
    xr, xp = x[:, :, :rot], x[:, :, rot:]
    half = rot // 2
    rh = cat([-xr[:, :, half:], xr[:, :, :half]], axis=2)   # rotate_half
    return cat([xr * cos + rh * sin, xp], axis=2)


# --- forward всей модели (дифференцируемый) -----------------------------------

def forward(P, ids):
    c, T = C, len(ids)
    h = P["gpt_neox.embed_in.weight"].index_rows(np.asarray(ids))     # (T, hid)
    cos, sin = rope_cache(T, c["rot"], c["base"])
    mask = Tensor(np.triu(np.full((T, T), -1e9, np.float32), 1))

    for i in range(c["nl"]):
        p = f"gpt_neox.layers.{i}."
        ln1 = layernorm(h, P[p + "input_layernorm.weight"], P[p + "input_layernorm.bias"])
        qkv = linear(ln1, P[p + "attention.query_key_value.weight"],
                     P[p + "attention.query_key_value.bias"]).reshape(T, c["nh"], 3 * c["hd"])
        q, k, v = qkv[:, :, :c["hd"]], qkv[:, :, c["hd"]:2 * c["hd"]], qkv[:, :, 2 * c["hd"]:]
        q, k = rope(q, cos, sin, c["rot"]), rope(k, cos, sin, c["rot"])
        q, k, v = q.swapaxes(0, 1), k.swapaxes(0, 1), v.swapaxes(0, 1)   # (nh,T,hd)
        scores = (q @ k.mT) * (1.0 / np.sqrt(c["hd"])) + mask            # (nh,T,T)
        ctx = (scores.softmax(axis=-1) @ v).swapaxes(0, 1).reshape(T, c["hid"])
        attn = linear(ctx, P[p + "attention.dense.weight"], P[p + "attention.dense.bias"])

        ln2 = layernorm(h, P[p + "post_attention_layernorm.weight"],
                        P[p + "post_attention_layernorm.bias"])
        m = linear(ln2, P[p + "mlp.dense_h_to_4h.weight"], P[p + "mlp.dense_h_to_4h.bias"]).gelu()
        m = linear(m, P[p + "mlp.dense_4h_to_h.weight"], P[p + "mlp.dense_4h_to_h.bias"])
        h = h + attn + m                                                 # parallel residual

    h = layernorm(h, P["gpt_neox.final_layer_norm.weight"], P["gpt_neox.final_layer_norm.bias"])
    return linear(h, P["embed_out.weight"])                             # (T, vocab)


def generate(P, tok, prompt, n=30):
    ids = tok(prompt)["input_ids"]
    with no_grad():
        for _ in range(n):
            ids.append(int(forward(P, ids).data[-1].argmax()))
    return tok.decode(ids)


def validate(tok):
    """Сверка forward И backward с torch.autograd HF — в float64, чтобы убрать
    float32-шум (иначе накопление через 6 слоёв даёт разницу до ~1 в логитах)."""
    import torch
    import tensor as _T
    from transformers import AutoModelForCausalLM
    _T.set_dtype(np.float64)
    hf = AutoModelForCausalLM.from_pretrained(MODEL).double()
    P = {k: Tensor(v.detach().numpy().astype(np.float64)) for k, v in hf.state_dict().items()}

    ids = tok("The capital of France is")["input_ids"]
    logits = forward(P, ids)                          # forward на нашем движке
    for pt in P.values():
        pt.grad = np.zeros_like(pt.data)
    loss = logits[:-1].softmax_cross_entropy(np.asarray(ids[1:], np.int64))
    loss.backward()                                  # backward на нашем движке

    inp = torch.tensor([ids])
    out = hf(input_ids=inp, labels=inp)
    out.loss.backward()                              # эталон torch.autograd
    hf_g = {n: pr.grad.numpy() for n, pr in hf.named_parameters()}

    print("=== Сверка с HF (float64) ===")
    print(f"forward логиты: max|наш - HF| = "
          f"{np.abs(logits.data - hf(inp).logits[0].detach().numpy()).max():.1e}")
    print(f"loss: наш = {float(loss.data):.5f}, HF = {out.loss.item():.5f}")
    print("backward (градиент через RoPE / attention / GELU / LayerNorm):")
    for k in ["gpt_neox.embed_in.weight",
              "gpt_neox.layers.0.attention.query_key_value.weight",
              "gpt_neox.layers.3.mlp.dense_h_to_4h.weight",
              "embed_out.weight"]:
        rel = np.abs(P[k].grad - hf_g[k]).max() / max(1e-12, np.abs(hf_g[k]).max())
        print(f"  {'.'.join(k.split('.')[-2:]):30} отн.расхождение = {rel:.1e}")
    _T.set_dtype(np.float32)


def main():
    from transformers import AutoTokenizer
    print("Загружаю Pythia-70M...")
    tok = AutoTokenizer.from_pretrained(MODEL)

    validate(tok)                                    # forward+backward == HF (float64)

    P = load_params()                                # float32 — генерация быстрее
    print("\n=== Генерация на нашем движке ===")
    print(generate(P, tok, "The capital of France is", n=30))


if __name__ == "__main__":
    main()
