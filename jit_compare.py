"""
Наш C+BLAS против PyTorch eager и torch.compile (JIT) — на одном MLP forward.

Наш compile_blas убирает Python-обвязку (граф -> C+BLAS). Ровно это делает и
torch.compile (граф -> C++/Triton). Поэтому здесь разрыв eager-vs-наш должен
резко сократиться: оба компилятора уходят от интерпретатора к нативному коду.

Запуск:  python3 jit_compare.py
"""

import time

import numpy as np
import torch

from tensor import Tensor, no_grad
import compile_blas

B, K, H, O = 64, 256, 512, 256
REPS = 200


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((B, K)).astype(np.float32)
    W1 = rng.standard_normal((K, H)).astype(np.float32); b1 = rng.standard_normal(H).astype(np.float32)
    W2 = rng.standard_normal((H, O)).astype(np.float32); b2 = rng.standard_normal(O).astype(np.float32)

    # --- наш C+BLAS: компилируем граф forward и меряем в тугом C-цикле ---
    tX, tW1, tb1 = Tensor(X), Tensor(W1), Tensor(b1)
    tW2, tb2 = Tensor(W2), Tensor(b2)
    graph = (tX @ tW1 + tb1).relu() @ tW2 + tb2
    c_ms = compile_blas.compile_and_time(graph, REPS)

    # --- PyTorch: та же сеть, те же веса ---
    m = torch.nn.Sequential(torch.nn.Linear(K, H), torch.nn.ReLU(), torch.nn.Linear(H, O))
    m[0].weight.data = torch.tensor(W1.T.copy()); m[0].bias.data = torch.tensor(b1)
    m[2].weight.data = torch.tensor(W2.T.copy()); m[2].bias.data = torch.tensor(b2)
    m.eval()
    tx = torch.tensor(X)

    def bench(fn):
        with torch.no_grad():
            for _ in range(10):
                fn()
            t0 = time.perf_counter()
            for _ in range(REPS):
                fn()
            return (time.perf_counter() - t0) / REPS * 1e3

    eager_ms = bench(lambda: m(tx))

    mc = torch.compile(m)                     # JIT: граф -> C++/Triton
    comp_ms = bench(lambda: mc(tx))           # (первый вызов компилирует — прогрев в bench)

    # сверка корректности (наш C vs torch)
    c_out, _ = compile_blas.compile_and_run(graph)
    with torch.no_grad():
        t_out = m(tx).numpy()
    diff = np.abs(c_out - t_out).max()

    print(f"MLP forward {B}x{K} -> {H} -> {O}, {REPS} прогонов\n")
    print(f"наш C + BLAS:        {c_ms:6.3f} мс/проход")
    print(f"PyTorch eager:       {eager_ms:6.3f} мс/проход")
    print(f"PyTorch torch.compile:{comp_ms:6.3f} мс/проход")
    print(f"\nсверка нашего C с PyTorch: max diff = {diff:.2e}")


if __name__ == "__main__":
    main()
