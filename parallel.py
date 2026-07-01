"""
Мульти-GPU параллелизм на основе современных подходов (в симуляторе).

Реальных GPU нет — «устройства» это просто списки массивов, а обмен между ними
мы моделируем на CPU. Но АЛГОРИТМЫ и СЕМАНТИКА — настоящие, те же, что в больших
тренировках (Megatron-LM, GPipe, DeepSpeed ZeRO, PyTorch DDP/FSDP).

Реализовано:
  - коллективы: broadcast, all_reduce (ring — как в NCCL/Horovod), all_gather,
    reduce_scatter;
  - data parallelism  — реплики модели, шардируем батч, all_reduce градиентов;
  - tensor parallelism — колоночный параллелизм линейного слоя (Megatron);
  - pipeline parallelism — стадии на разных устройствах + микробатчи (GPipe).

Каждый режим сверяется с однокарточным эталоном (должно совпасть до бита/1e-6).

Запуск:  python3 parallel.py
"""

import numpy as np

from tensor import Tensor


# --- Коллективные операции --------------------------------------------------
# Работают со списком массивов (по одному на «устройство»).

def broadcast(x, n):
    """Разослать x на n устройств (у всех одинаковая копия)."""
    return [x.copy() for _ in range(n)]


def all_reduce(shards, op="sum"):
    """Ring all-reduce: у каждого устройства оказывается сумма по всем.

    Настоящий алгоритм NCCL/Horovod: reduce-scatter по кольцу, затем all-gather.
    Пропускная способность на устройство ~2·(N-1)/N·|data| и НЕ растёт с N —
    поэтому так и синхронизируют градиенты на тысячах GPU.
    """
    n = len(shards)
    if n == 1:
        return [shards[0].copy() if op == "sum" else shards[0].copy()]

    flats = [s.ravel().copy() for s in shards]
    size = flats[0].size
    edges = [size * i // n for i in range(n + 1)]     # границы n чанков

    def ch(dev, c):
        return flats[dev][edges[c]:edges[c + 1]]

    # reduce-scatter: N-1 шагов по кольцу, каждый добавляет полученный чанк
    for s in range(n - 1):
        sends = [ch(i, (i - s) % n).copy() for i in range(n)]
        for i in range(n):
            src = (i - 1) % n
            ch(i, (src - s) % n)[:] += sends[src]
    # теперь у устройства i полностью просуммирован чанк (i+1)%n

    # all-gather: N-1 шагов, разносим просуммированные чанки всем
    for s in range(n - 1):
        sends = [ch(i, (i + 1 - s) % n).copy() for i in range(n)]
        for i in range(n):
            src = (i - 1) % n
            ch(i, (src + 1 - s) % n)[:] = sends[src]

    out = [f.reshape(sh.shape) for f, sh in zip(flats, shards)]
    if op == "mean":
        out = [o / n for o in out]
    return out


def all_gather(shards):
    """Каждому устройству — конкатенация кусков со всех устройств."""
    full = np.concatenate([s.ravel() for s in shards])
    return [full.copy() for _ in shards]


def reduce_scatter(shards):
    """Сумма по устройствам, но каждому достаётся только своя часть результата."""
    n = len(shards)
    total = np.add.reduce([s.ravel() for s in shards])
    size = total.size
    edges = [size * i // n for i in range(n + 1)]
    return [total[edges[i]:edges[i + 1]].copy() for i in range(n)]


# --- Проверка коллективов ---------------------------------------------------

def _check_collectives():
    rng = np.random.default_rng(0)
    n = 4
    shards = [rng.standard_normal(10).astype(np.float32) for _ in range(n)]
    ref = np.add.reduce(shards)                       # эталонная сумма
    ar = all_reduce(shards, "sum")
    worst = max(np.abs(a - ref).max() for a in ar)
    same = all(np.allclose(a, ar[0]) for a in ar)     # у всех одинаково
    print(f"ring all_reduce: max|ring - sum| = {worst:.2e}, "
          f"у всех устройств одинаково = {same}")


# --- 1. Data parallelism (PyTorch DDP) --------------------------------------
# Реплика модели на каждом GPU. Батч режем на шарды, каждый считает градиент по
# своему шарду, затем all_reduce(mean) усредняет градиенты — как будто считали
# по всему батчу. Так масштабируют по данным на тысячах GPU.

def demo_data_parallel(n=4):
    rng = np.random.default_rng(1)
    M, K = 4 * n, 5                       # батч делится на n поровну
    X = rng.standard_normal((M, K)).astype(np.float32)
    Y = rng.standard_normal((M, 1)).astype(np.float32)
    W = rng.standard_normal((K, 1)).astype(np.float32)
    b = rng.standard_normal(1).astype(np.float32)

    def grads(xb, yb, W, b):
        tW, tb = Tensor(W), Tensor(b)
        diff = Tensor(xb) @ tW + tb - Tensor(yb)
        loss = (diff * diff).mean()       # MSE
        loss.backward()
        return tW.grad, tb.grad

    gW_full, gb_full = grads(X, Y, W, b)                       # эталон: весь батч

    xs = np.split(X, n); ys = np.split(Y, n)                  # шарды по устройствам
    gWs, gbs = zip(*[grads(xs[i], ys[i], W, b) for i in range(n)])
    gW = all_reduce(list(gWs), "mean")[0]                     # синхронизация градиентов
    gb = all_reduce(list(gbs), "mean")[0]

    d = max(np.abs(gW - gW_full).max(), np.abs(gb - gb_full).max())
    print(f"data parallel (n={n}): max|DDP - однокарточный градиент| = {d:.2e}")


# --- 2. Tensor parallelism (Megatron-LM) ------------------------------------
# Один большой линейный слой не делят по данным, а РАЗРЕЗАЮТ саму матрицу весов
# по устройствам. Колоночный вариант: W = [W0 | W1 | ...], каждое устройство
# считает свой кусок выхода, затем all_gather склеивает столбцы обратно.

def demo_tensor_parallel(n=2):
    rng = np.random.default_rng(2)
    M, K, N = 6, 8, 4 * n
    X = rng.standard_normal((M, K)).astype(np.float32)
    W = rng.standard_normal((K, N)).astype(np.float32)
    ref = X @ W

    Ws = np.split(W, n, axis=1)                               # столбцы по устройствам
    # каждый «GPU» держит свой срез весов и считает свой кусок выхода
    partial = [(Tensor(X).to(f"cuda:{i}") @ Tensor(Ws[i]).to(f"cuda:{i}")).data
               for i in range(n)]
    full = np.concatenate(partial, axis=1)                   # all_gather по столбцам
    print(f"tensor parallel (n={n}): max|Megatron - целый слой| = "
          f"{np.abs(full - ref).max():.2e}")


# --- 3. Pipeline parallelism (GPipe) ----------------------------------------
# Модель режут на СТАДИИ (группы слоёв), каждая на своём устройстве. Батч режут
# на микробатчи, они текут по конвейеру: пока стадия 2 считает микробатч 1,
# стадия 1 уже считает микробатч 2 — так заняты все устройства.

def demo_pipeline(n_micro=4):
    rng = np.random.default_rng(3)
    M, K, H, O = 8, 6, 10, 3
    X = rng.standard_normal((M, K)).astype(np.float32)
    W1 = rng.standard_normal((K, H)).astype(np.float32)      # стадия 1 -> cuda:0
    W2 = rng.standard_normal((H, O)).astype(np.float32)      # стадия 2 -> cuda:1

    tW1 = Tensor(W1).cuda(0); tW2 = Tensor(W2).cuda(1)
    outs = []
    for mb in np.split(X, n_micro):                          # микробатчи
        h = (Tensor(mb).cuda(0) @ tW1).relu()               # стадия 1 на cuda:0
        h = h.to("cuda:1")                                   # перенос между устройствами
        outs.append((h @ tW2).data)                          # стадия 2 на cuda:1
    pipe = np.concatenate(outs, axis=0)

    ref = np.maximum(0, X @ W1) @ W2                         # эталон на одном устройстве
    print(f"pipeline parallel ({n_micro} микробатчей): max|GPipe - эталон| = "
          f"{np.abs(pipe - ref).max():.2e}")


# --- 4. ZeRO / FSDP (DeepSpeed / PyTorch) -----------------------------------
# В DDP каждая карта хранит ПОЛНУЮ копию параметров, градиентов и состояний
# оптимизатора (Adam: m и v — ещё +2×параметров). Избыточно. ZeRO/FSDP это всё
# ШАРДИРУЮТ: карта i держит 1/N. Для forward/backward нужные веса собирает
# all_gather'ом и тут же освобождает; градиенты reduce_scatter — каждому его
# шард; оптимизатор обновляет только свою 1/N. Так тренируют модели, которые в
# одну карту не влезают в принципе.

def _grads_from_flat(pflat, xb, yb, K, Nout):
    W = Tensor(pflat[:K * Nout].reshape(K, Nout).astype(np.float32))
    b = Tensor(pflat[K * Nout:].astype(np.float32))
    diff = Tensor(xb) @ W + b - Tensor(yb)
    (diff * diff).mean().backward()                  # MSE
    return np.concatenate([W.grad.ravel(), b.grad.ravel()]).astype(np.float64)


def demo_zero(n=4):
    rng = np.random.default_rng(7)
    M, K, Nout = 4 * n, 5, 3
    X = rng.standard_normal((M, K)).astype(np.float32)
    Y = rng.standard_normal((M, Nout)).astype(np.float32)
    p0 = rng.standard_normal(K * Nout + Nout)         # плоский вектор параметров
    P = p0.size
    lr, b1, b2, eps = 1e-2, 0.9, 0.999, 1e-8

    def adam_step(p, m, v, g):                        # один шаг Adam (t=1)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        return p - lr * (m / (1 - b1)) / (np.sqrt(v / (1 - b2)) + eps), m, v

    # эталон: один «GPU», полный батч, полные m, v
    g_full = _grads_from_flat(p0, X, Y, K, Nout)
    p_ref, _, _ = adam_step(p0, np.zeros(P), np.zeros(P), g_full)

    # ZeRO-3: p, m, v, grad — всё шардировано по n устройствам
    edges = [P * i // n for i in range(n + 1)]
    p_sh = [p0[edges[i]:edges[i + 1]].copy() for i in range(n)]
    m_sh = [np.zeros(edges[i + 1] - edges[i]) for i in range(n)]
    v_sh = [np.zeros(edges[i + 1] - edges[i]) for i in range(n)]

    full_p = np.concatenate(p_sh)                     # all_gather параметров
    xs, ys = np.split(X, n), np.split(Y, n)
    gs = [_grads_from_flat(full_p, xs[i], ys[i], K, Nout) for i in range(n)]
    g_sh = [g / n for g in reduce_scatter(gs)]        # reduce_scatter -> шард mean-градиента

    new_p = []
    for i in range(n):                                # каждый обновляет ТОЛЬКО свою 1/N
        pi, _, _ = adam_step(p_sh[i], m_sh[i], v_sh[i], g_sh[i])
        new_p.append(pi)
    p_zero = np.concatenate(new_p)                    # all_gather обновлённых весов

    ddp_mem = 4 * P                                   # p+grad+m+v на КАЖДОЙ карте (DDP)
    zero_mem = 4 * P / n                              # то же, но 1/N (ZeRO-3)
    print(f"ZeRO-3 (n={n}): max|ZeRO - однокарточный Adam| = "
          f"{np.abs(p_zero - p_ref).max():.2e}")
    print(f"   память на карту: DDP={ddp_mem:.0f} ед., ZeRO-3={zero_mem:.0f} ед. "
          f"(в {n}x меньше)")


if __name__ == "__main__":
    print("=== Коллективы (ring all-reduce, как в NCCL) ===")
    _check_collectives()
    print("\n=== Стратегии параллелизма (сверка с однокарточным) ===")
    demo_data_parallel()
    demo_tensor_parallel()
    demo_pipeline()
    demo_zero()
    print("\nВсё считает симулятор на CPU, но алгоритмы и семантика — настоящие\n"
          "(Megatron-LM, GPipe, PyTorch DDP, ring all-reduce из NCCL/Horovod).")
