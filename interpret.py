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


EMB_GROUPS = {
    "звери": [" cat", " dog", " lion", " horse", " bird"],
    "числа": [" one", " two", " three", " four", " five"],
    "цвета": [" red", " blue", " green", " black", " white"],
    "страны": [" France", " Germany", " China", " Japan", " Russia"],
}


def _emb_points(P, tok):
    words, labels, groups = [], [], []
    for g, ws in EMB_GROUPS.items():
        for wd in ws:
            words.append(tok(wd, add_special_tokens=False)["input_ids"][0])
            labels.append(wd.strip())
            groups.append(g)
    E = P["gpt_neox.embed_in.weight"].data[words]         # (N, hid)
    return pca2(E), labels, groups


def emb_map(P, tok):
    pts, labels, _ = _emb_points(P, tok)
    print("=== 1. Эмбеддинги в 2D (PCA). Похожие слова должны быть рядом ===")
    print(scatter(pts, labels))


def emb_map_png(P, tok, path="docs/images/emb_pca.png"):
    """Настоящая картинка PCA-проекции: цветные кластеры категорий с подписями."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, labels, groups = _emb_points(P, tok)
    colors = {"звери": "#e4572e", "числа": "#17a398",
              "цвета": "#4059ad", "страны": "#8e44ad"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    seen = set()
    for (x, y), lab, g in zip(pts, labels, groups):
        ax.scatter(x, y, c=colors[g], s=90, zorder=3,
                   label=g if g not in seen else None)
        seen.add(g)
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_title("Эмбеддинги Pythia-70M в 2D (PCA)\nкластеры возникли сами из предсказания токена")
    ax.set_xlabel("главная компонента 1")
    ax.set_ylabel("главная компонента 2")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"=== 1b. PNG PCA-проекции сохранён: {path} ===")


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


def attention_map_png(P, tok, layer=3, path="docs/images/attention.png"):
    """Тепловая карта внимания: настоящий heatmap слоя (среднее по головам)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text = "The cat sat on the mat"
    ids = tok(text)["input_ids"]
    toks = [tok.decode([i]).strip() or "·" for i in ids]
    aw = run_capture_attn(P, ids)[layer].mean(0)         # (T, T)
    masked = np.where(np.triu(np.ones_like(aw), 1) > 0, np.nan, aw)  # скрыть будущее

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
    im = ax.imshow(masked, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks)
    ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks)
    ax.set_xlabel("на кого смотрит (ключи)")
    ax.set_ylabel("кто смотрит (запросы)")
    ax.set_title(f"Внимание Pythia-70M, слой {layer} (среднее по головам)\n"
                 f"виден attention sink на первом токене")
    for i in range(len(toks)):
        for j in range(i + 1):
            ax.text(j, i, f"{aw[i, j]:.2f}", ha="center", va="center",
                    color="white" if aw[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="вес внимания")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"=== 2b. PNG карты внимания сохранён: {path} ===")


def forward_logits(P, ids, ablate=None):
    """Полный forward до логитов. ablate=(layer, head) или список таких пар —
    зануляет вклад голов (их ctx перед выходной проекцией) = zero-ablation."""
    c, T = pythia.C, len(ids)
    if ablate is not None and len(ablate) and isinstance(ablate[0], int):
        ablate = [ablate]                                # одиночную пару -> список
    abl = set(map(tuple, ablate)) if ablate else set()
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
            ctx = (aw @ v).swapaxes(0, 1)                # (T, nh, hd)
            heads_here = [hh for (ll, hh) in abl if ll == i]
            if heads_here:
                keep = np.ones((1, c["nh"], 1), np.float32)
                for hh in heads_here:
                    keep[0, hh, 0] = 0.0                 # зануляем указанные головы
                ctx = ctx * Tensor(keep)
            ctx = ctx.reshape(T, c["hid"])
            attn = pythia.linear(ctx, P[p + "attention.dense.weight"], P[p + "attention.dense.bias"])
            ln2 = pythia.layernorm(h, P[p + "post_attention_layernorm.weight"], P[p + "post_attention_layernorm.bias"])
            mm = pythia.linear(ln2, P[p + "mlp.dense_h_to_4h.weight"], P[p + "mlp.dense_h_to_4h.bias"]).gelu()
            mm = pythia.linear(mm, P[p + "mlp.dense_4h_to_h.weight"], P[p + "mlp.dense_4h_to_h.bias"])
            h = h + attn + mm
        ln = pythia.layernorm(h, P["gpt_neox.final_layer_norm.weight"], P["gpt_neox.final_layer_norm.bias"])
        return (ln @ P["embed_out.weight"].T).data      # (T, vocab)


def find_induction_head(P, n=8, seed=0):
    """Каноничный тест: случайная последовательность, повторённая дважды.
    Индукц. балл головы = внимание позиции i (2-я половина) на (i-n)+1 —
    токен, что СЛЕДОВАЛ за прошлым вхождением. Возвращает (слой, голова, балл, ids)."""
    rng = np.random.default_rng(seed)
    half = rng.integers(200, 4000, n).tolist()
    ids = half + half
    A = run_capture_attn(P, ids)
    best = (-1.0, 0, 0)
    for l, aw in enumerate(A):
        for h in range(aw.shape[0]):
            score = float(np.mean([aw[h, i, (i - n) + 1] for i in range(n, 2 * n - 1)]))
            if score > best[0]:
                best = (score, l, h)
    return best[1], best[2], best[0], ids


def induction_head_png(P, tok, n=8, path="docs/images/induction_head.png"):
    """Карта найденной индукционной головы — видна диагональ сдвига во 2-й половине."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer, head, score, ids = find_induction_head(P, n=n)
    aw = run_capture_attn(P, ids)[layer][head]           # (T,T) конкретной головы
    T = len(ids)
    masked = np.where(np.triu(np.ones_like(aw), 1) > 0, np.nan, aw)
    labels = [f"t{i}" for i in range(n)] + [f"t{i}'" for i in range(n)]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=120)
    im = ax.imshow(masked, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(T)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(range(T)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("на кого смотрит (ключи)")
    ax.set_ylabel("кто смотрит (запросы)")
    # обвести индукционную диагональ: запрос i' (2-я копия) -> ключ (i+1) 1-й копии
    for i in range(n, 2 * n - 1):
        ax.add_patch(plt.Rectangle(((i - n) + 1 - .5, i - .5), 1, 1,
                     fill=False, edgecolor="cyan", lw=1.8))
    ax.set_title(f"Индукционная голова: слой {layer}, голова {head}\n"
                 f"induction score = {score:.2f} (случайно ~{1/T:.2f})\n"
                 f"голубым — сдвиг +1: t_i' смотрит на того, кто шёл за t_i")
    fig.colorbar(im, ax=ax, fraction=0.046, label="вес внимания")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"=== 2d. PNG индукционной головы (L{layer}H{head}) сохранён: {path} ===")


def _induction_prob(P, ablate, n=8, seeds=8):
    """Средняя P(правильного индукц. токена) во 2-й копии при данной абляции."""
    def sm(x):
        e = np.exp(x - x.max()); return e / e.sum()
    ps = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        half = rng.integers(200, 4000, n).tolist()
        full = half + half
        L = forward_logits(P, full, ablate=ablate)
        for i in range(n, 2 * n - 1):
            ps.append(sm(L[i])[full[i + 1]])
    return float(np.mean(ps))


def ablation_png(P, tok, n=8, path="docs/images/ablation.png"):
    """Причинный тест: зануляем индукционные головы и меряем просадку предсказания."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    agg = defaultdict(list)                               # ранжируем головы по индукц. баллу
    for seed in range(8):
        rng = np.random.default_rng(seed)
        half = rng.integers(200, 4000, n).tolist(); full = half + half
        A = run_capture_attn(P, full)
        for l, aw in enumerate(A):
            for h in range(aw.shape[0]):
                agg[(l, h)].append(np.mean([aw[h, i, (i - n) + 1] for i in range(n, 2 * n - 1)]))
    rank = sorted(agg, key=lambda k: -np.mean(agg[k]))

    base = _induction_prob(P, None, n)
    d1 = _induction_prob(P, rank[:1], n)
    d2 = _induction_prob(P, rank[:2], n)
    d3 = _induction_prob(P, rank[:3], n)
    allh = [(l, h) for l in range(6) for h in range(8)]
    rc = np.random.default_rng(123)
    ctrl = np.mean([_induction_prob(P, [allh[j] for j in rc.choice(len(allh), 3, replace=False)], n)
                    for _ in range(6)])

    labels = ["база", "-L3H5", "-топ2", "-топ3", "3 случ.\n(контроль)"]
    vals = [base, d1, d2, d3, ctrl]
    colors = ["#4059ad", "#e4572e", "#e4572e", "#e4572e", "#17a398"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=120)
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(base, ls="--", c="gray", alpha=0.6)
    ax.set_ylabel("P(правильный индукц. токен)")
    ax.set_title("Абляция индукционных голов: причинный тест\n"
                 "зануление L3H5 и соседей рушит индукцию; случайные головы — нет")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"=== 2e. PNG абляции сохранён: {path}  "
          f"(база {base:.3f} -> -L3H5 {d1:.3f} -> -топ3 {d3:.3f}; контроль {ctrl:.3f}) ===")


def attention_layers_png(P, tok, text="The cat sat on the mat",
                         path="docs/images/attention_layers.png"):
    """Сетка тепловых карт внимания по ВСЕМ слоям — видно разделение труда."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = tok(text)["input_ids"]
    toks = [tok.decode([i]).strip() or "·" for i in ids]
    A = run_capture_attn(P, ids)                          # list[nl] (nh,T,T)
    nl = len(A)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=120)
    for i, ax in enumerate(axes.flat):
        m = A[i].mean(0)
        masked = np.where(np.triu(np.ones_like(m), 1) > 0, np.nan, m)
        im = ax.imshow(masked, cmap="viridis", vmin=0, vmax=1)
        sink = m[1:, 0].mean()
        ax.set_title(f"слой {i}  (sink→0: {sink:.2f})", fontsize=10)
        ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks, fontsize=7, rotation=45)
        ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks, fontsize=7)
    fig.suptitle(f"Внимание Pythia-70M по слоям (среднее по головам): {text!r}\n"
                 f"ранние слои — локальное смешивание, средние — attention sink, "
                 f"поздний — сбор к предсказанию")
    fig.colorbar(im, ax=axes, fraction=0.02, label="вес внимания")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"=== 2c. PNG внимания по слоям сохранён: {path} ===")


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


# --- 4. Путь токена по слоям (logit lens во времени) ------------------------

def _journey_lens(P, tok, prompt, n_track):
    """Стадии эмбеддинг+слои → (имена, вероятности logit lens, топ-кандидаты)."""
    ids = tok(prompt)["input_ids"]
    emb = P["gpt_neox.embed_in.weight"].data[ids[-1]]
    stages = [("эмб", emb)] + [(f"сл{i}", h[-1]) for i, h in enumerate(run_capture(P, ids))]
    names, probs = [], []
    for name, h in stages:
        lo = unembed(P, h)
        e = np.exp(lo - lo.max())
        names.append(name)
        probs.append(e / e.sum())
    tracked = probs[-1].argsort()[::-1][:n_track]
    return names, probs, tracked


def token_journey(P, tok, prompt="The capital of France is", n_track=4):
    names, probs_all, tracked = _journey_lens(P, tok, prompt, n_track)
    lens = list(zip(names, probs_all))
    final = lens[-1][1]
    win = tracked[0]
    wname = tok.decode([win]).strip()

    print(f"\n=== 4. Путь токена {wname!r} к вершине по слоям (logit lens) ===")
    print(f"промпт: {prompt!r}\n")
    for name, probs in lens:
        pr = probs[win]
        rank = int((probs > pr).sum()) + 1
        bar = "█" * int(round(pr / final[win] * 26))
        print(f"  {name}  |{bar:<26}| p={pr:.3f}  ранг #{rank}")
    print(f"  ↑ {wname!r} поднимается из глубин словаря к #1 сквозь слои\n")

    # заодно — как соперничают несколько кандидатов (их ранг по слоям)
    print("  Конкуренция кандидатов (ранг по слоям):")
    print("        " + "  ".join(f"{n:>4}" for n, _ in lens))
    for t in tracked:
        ranks = [int((probs > probs[t]).sum()) + 1 for _, probs in lens]
        print(f"  {tok.decode([t]).strip()[:6]:>6}  " +
              "  ".join(f"{r:>4}" for r in ranks))


def token_journey_png(P, tok, prompt="The capital of France is", n_track=4,
                      path="docs/images/token_journey.png"):
    """Две панели: ранг кандидатов по слоям (лог-шкала) и вероятность победителя."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, probs_all, tracked = _journey_lens(P, tok, prompt, n_track)
    x = range(len(names))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    for t in tracked:
        ranks = [int((p > p[t]).sum()) + 1 for p in probs_all]
        ax1.plot(x, ranks, marker="o", label=tok.decode([t]).strip())
    ax1.set_yscale("log"); ax1.invert_yaxis()            # #1 сверху
    ax1.set_xticks(list(x)); ax1.set_xticklabels(names)
    ax1.set_ylabel("ранг (лог-шкала, #1 сверху)")
    ax1.set_title("Конкуренция кандидатов по слоям")
    ax1.grid(True, alpha=0.3); ax1.legend()

    win = tracked[0]
    pwin = [p[win] for p in probs_all]
    ax2.plot(x, pwin, marker="o", color="#e4572e")
    ax2.fill_between(x, pwin, alpha=0.2, color="#e4572e")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(names)
    ax2.set_ylabel("вероятность")
    ax2.set_title(f"Как всплывает победитель {tok.decode([win]).strip()!r}")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Путь токена сквозь слои (logit lens): {prompt!r}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"=== 4b. PNG пути токена сохранён: {path} ===")


def main():
    from transformers import AutoTokenizer
    print("Загружаю Pythia-70M...")
    P = pythia.load_params()
    tok = AutoTokenizer.from_pretrained(pythia.MODEL)
    emb_map(P, tok)
    emb_map_png(P, tok)
    attention_map(P, tok)
    attention_map_png(P, tok)
    attention_layers_png(P, tok)
    induction_head_png(P, tok)
    ablation_png(P, tok)
    logit_lens(P, tok)
    token_journey(P, tok)
    token_journey_png(P, tok)


if __name__ == "__main__":
    main()
