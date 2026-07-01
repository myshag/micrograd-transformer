"""
Интерпретируемость LLM на нашем движке (Pythia-70M).

Три классических инструмента «заглянуть внутрь» модели:
  1) 2D-проекция эмбеддингов (PCA) — видны ли семантические кластеры;
  2) карта внимания — какие позиции на какие «смотрят»;
  3) logit lens — что модель «предсказывает» после каждого слоя (как решение
     формируется в глубину сети).

Всё считается на нашем Tensor (forward с захватом промежуточных активаций).
Требуется: transformers. Запуск:  python3 interpret.py
"""

import numpy as np

import pythia
from tensor import Tensor, no_grad


def run_capture(P, ids):
    """Forward Pythia с захватом активаций каждого слоя и карт внимания."""
    c, T = pythia.C, len(ids)
    hiddens, attns = [], []
    with no_grad():
        h = P["gpt_neox.embed_in.weight"].index_rows(np.asarray(ids))
        cos, sin = pythia.rope_cache(T, c["rot"], c["base"])
        mask = Tensor(np.triu(np.full((T, T), -1e9, np.float32), 1))
        for i in range(c["nl"]):
            p = f"gpt_neox.layers.{i}."
            ln1 = pythia.layernorm(h, P[p + "input_layernorm.weight"],
                                   P[p + "input_layernorm.bias"])
            qkv = pythia.linear(ln1, P[p + "attention.query_key_value.weight"],
                                P[p + "attention.query_key_value.bias"]).reshape(T, c["nh"], 3 * c["hd"])
            q, k, v = qkv[:, :, :c["hd"]], qkv[:, :, c["hd"]:2 * c["hd"]], qkv[:, :, 2 * c["hd"]:]
            q, k = pythia.rope(q, cos, sin, c["rot"]), pythia.rope(k, cos, sin, c["rot"])
            q, k, v = q.swapaxes(0, 1), k.swapaxes(0, 1), v.swapaxes(0, 1)
            scores = (q @ k.mT) * (1.0 / np.sqrt(c["hd"])) + mask
            aw = scores.softmax(axis=-1)                 # (nh, T, T) — карта внимания
            attns.append(aw.data)
            ctx = (aw @ v).swapaxes(0, 1).reshape(T, c["hid"])
            attn = pythia.linear(ctx, P[p + "attention.dense.weight"],
                                 P[p + "attention.dense.bias"])
            ln2 = pythia.layernorm(h, P[p + "post_attention_layernorm.weight"],
                                   P[p + "post_attention_layernorm.bias"])
            m = pythia.linear(ln2, P[p + "mlp.dense_h_to_4h.weight"],
                              P[p + "mlp.dense_h_to_4h.bias"]).gelu()
            m = pythia.linear(m, P[p + "mlp.dense_4h_to_h.weight"],
                              P[p + "mlp.dense_4h_to_h.bias"])
            h = h + attn + m
            hiddens.append(h.data.copy())                # состояние после слоя i
    return hiddens


def unembed(P, h_row):
    """logit lens: применить финальный LayerNorm + голову к одному состоянию."""
    with no_grad():
        ln = pythia.layernorm(Tensor(h_row[None, :]),
                              P["gpt_neox.final_layer_norm.weight"],
                              P["gpt_neox.final_layer_norm.bias"])
        return (ln @ P["embed_out.weight"].T).data[0]


# --- 1. 2D-проекция эмбеддингов (PCA) ---------------------------------------

def pca2(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T                                 # (N, 2)


def scatter(pts, labels, w=68, h=22):
    x, y = pts[:, 0], pts[:, 1]
    x = (x - x.min()) / (np.ptp(x) + 1e-9)
    y = (y - y.min()) / (np.ptp(y) + 1e-9)
    grid = [[" "] * w for _ in range(h)]
    for i, (xi, yi) in enumerate(zip(x, y)):
        cx, cy = int(xi * (w - 7)), int((1 - yi) * (h - 1))
        for k, ch in enumerate(labels[i].strip()[:6]):
            if cx + k < w:
                grid[cy][cx + k] = ch
    return "\n".join("".join(r) for r in grid)


def emb_map(P, tok):
    groups = {
        "звери": [" cat", " dog", " lion", " horse", " bird"],
        "числа": [" one", " two", " three", " four", " five"],
        "цвета": [" red", " blue", " green", " black", " white"],
        "страны": [" France", " Germany", " China", " Japan", " Russia"],
    }
    words, labels = [], []
    for g, ws in groups.items():
        for wd in ws:
            words.append(tok(wd, add_special_tokens=False)["input_ids"][0])
            labels.append(wd)
    E = P["gpt_neox.embed_in.weight"].data[words]         # (N, hid)
    print("=== 1. Эмбеддинги в 2D (PCA). Похожие слова должны быть рядом ===")
    print(scatter(pca2(E), labels))


# --- 2. Карта внимания -------------------------------------------------------

def attention_map(P, tok, layer=3):
    text = "The cat sat on the mat"
    ids = tok(text)["input_ids"]
    toks = [tok.decode([i]).strip() or "·" for i in ids]
    # усредним по головам выбранного слоя
    aw = run_capture_attn(P, ids)[layer].mean(0)         # (T, T)
    print(f"\n=== 2. Внимание, слой {layer} (среднее по головам) ===")
    print("      " + " ".join(f"{t[:4]:>4}" for t in toks) + "   (на кого)")
    for i, t in enumerate(toks):
        cells = " ".join(f"{aw[i, j]:4.2f}" if j <= i else "   ." for j in range(len(toks)))
        print(f"{t[:5]:>5} {cells}")
    print("   (нижний треугольник — каузальность: позиция видит только левое)")


def run_capture_attn(P, ids):
    c, T = pythia.C, len(ids)
    attns = []
    with no_grad():
        h = P["gpt_neox.embed_in.weight"].index_rows(np.asarray(ids))
        cos, sin = pythia.rope_cache(T, c["rot"], c["base"])
        mask = Tensor(np.triu(np.full((T, T), -1e9, np.float32), 1))
        for i in range(c["nl"]):
            p = f"gpt_neox.layers.{i}."
            ln1 = pythia.layernorm(h, P[p + "input_layernorm.weight"], P[p + "input_layernorm.bias"])
            qkv = pythia.linear(ln1, P[p + "attention.query_key_value.weight"],
                                P[p + "attention.query_key_value.bias"]).reshape(T, c["nh"], 3 * c["hd"])
            q, k, v = qkv[:, :, :c["hd"]], qkv[:, :, c["hd"]:2 * c["hd"]], qkv[:, :, 2 * c["hd"]:]
            q, k = pythia.rope(q, cos, sin, c["rot"]), pythia.rope(k, cos, sin, c["rot"])
            q, k, v = q.swapaxes(0, 1), k.swapaxes(0, 1), v.swapaxes(0, 1)
            aw = ((q @ k.mT) * (1.0 / np.sqrt(c["hd"])) + mask).softmax(axis=-1)
            attns.append(aw.data)
            ctx = (aw @ v).swapaxes(0, 1).reshape(T, c["hid"])
            attn = pythia.linear(ctx, P[p + "attention.dense.weight"], P[p + "attention.dense.bias"])
            ln2 = pythia.layernorm(h, P[p + "post_attention_layernorm.weight"], P[p + "post_attention_layernorm.bias"])
            m = pythia.linear(ln2, P[p + "mlp.dense_h_to_4h.weight"], P[p + "mlp.dense_h_to_4h.bias"]).gelu()
            m = pythia.linear(m, P[p + "mlp.dense_4h_to_h.weight"], P[p + "mlp.dense_4h_to_h.bias"])
            h = h + attn + m
    return attns


# --- 3. Logit lens (что творится на разных слоях) ---------------------------

def logit_lens(P, tok):
    text = "The capital of France is"
    ids = tok(text)["input_ids"]
    hs = run_capture(P, ids)                              # состояние после каждого слоя
    print(f"\n=== 3. Logit lens: топ-предсказание next-token ПОСЛЕ каждого слоя ===")
    print(f"промпт: {text!r}")
    for i, h in enumerate(hs):
        logits = unembed(P, h[-1])                        # позиция последнего токена
        top = logits.argsort()[::-1][:3]
        print(f"  слой {i}: " + ", ".join(f"{tok.decode([t]).strip()!r}" for t in top))
    print("  (видно, как предсказание формируется от ранних слоёв к поздним)")


def main():
    from transformers import AutoTokenizer
    print("Загружаю Pythia-70M...")
    P = pythia.load_params()
    tok = AutoTokenizer.from_pretrained(pythia.MODEL)
    emb_map(P, tok)
    attention_map(P, tok)
    logit_lens(P, tok)


if __name__ == "__main__":
    main()
