"""
Наш C+BLAS vs PyTorch eager vs torch.compile — развёртка по размеру модели.

Идея: torch.compile выигрывает не «на масштабе», а где узкое место — ПАМЯТЬ
(фьюжн поэлементных операций) или запуски ядер (GPU). На matmul-bound MLP он на
CPU не быстрее eager (matmul и там, и там — BLAS). См. функцию elementwise().

Наш compile_blas читает данные из бинарного файла (не литералы), поэтому
масштабируется на любые размеры — включён во все строки.

Запуск:  python3 jit_compare.py
"""

import time

import numpy as np
import torch

from tensor import Tensor
import compile_blas

SIZES = [                     # (batch, in, hidden, out)
    (64, 256, 512, 256),
    (128, 512, 2048, 512),
    (256, 1024, 4096, 1024),
]


def torch_model(W1, b1, W2, b2):
    K, H = W1.shape
    O = W2.shape[1]
    m = torch.nn.Sequential(torch.nn.Linear(K, H), torch.nn.ReLU(), torch.nn.Linear(H, O))
    m[0].weight.data = torch.tensor(W1.T.copy()); m[0].bias.data = torch.tensor(b1)
    m[2].weight.data = torch.tensor(W2.T.copy()); m[2].bias.data = torch.tensor(b2)
    return m.eval()


def bench(fn, reps):
    with torch.no_grad():
        for _ in range(8):
            fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3


def main():
    rng = np.random.default_rng(0)
    print(f"{'модель':22} {'наш C+BLAS':>12} {'torch eager':>12} "
          f"{'torch.compile':>14} {'compile/eager':>13}")
    for (B, K, H, O) in SIZES:
        X = rng.standard_normal((B, K)).astype(np.float32)
        W1 = rng.standard_normal((K, H)).astype(np.float32); b1 = rng.standard_normal(H).astype(np.float32)
        W2 = rng.standard_normal((H, O)).astype(np.float32); b2 = rng.standard_normal(O).astype(np.float32)
        reps = max(20, int(2e8 / (B * K * H)))       # меньше прогонов на крупных

        m = torch_model(W1, b1, W2, b2)
        tx = torch.tensor(X)
        eager = bench(lambda: m(tx), reps)
        mc = torch.compile(m)
        comp = bench(lambda: mc(tx), reps)

        # наш C — теперь на любом размере (данные читаются из файла, не литералы)
        g = (Tensor(X) @ Tensor(W1) + Tensor(b1)).relu() @ Tensor(W2) + Tensor(b2)
        our_c = f"{compile_blas.compile_and_time(g, reps):.3f} мс"

        tag = f"{B}x{K}->{H}->{O}"
        print(f"{tag:22} {our_c:>12} {eager:9.3f} мс {comp:11.3f} мс "
              f"{comp/eager:12.2f}x")


def elementwise():
    """Поэлементно-тяжёлый случай: цепочка pointwise-операций над большим
    тензором. Тут узкое место — ПАМЯТЬ (каждая op — полный проход), и фьюжн
    torch.compile сливает всё в один проход. Здесь JIT и должен обгонять eager."""
    x = torch.randn(4096, 4096)

    def f(t):                                   # ~10 поэлементных операций
        for _ in range(5):
            t = torch.tanh(t * 1.001 + 0.5).relu()
        return t

    reps = 30
    eager = bench(lambda: f(x), reps)
    fc = torch.compile(f)
    comp = bench(lambda: fc(x), reps)
    print("\n=== Поэлементно-тяжёлый случай (узкое место — память) ===")
    print(f"torch eager:        {eager:7.2f} мс")
    print(f"torch.compile:      {comp:7.2f} мс   -> {eager/comp:.2f}x быстрее eager "
          f"(фьюжн: 10 проходов по памяти -> 1)")


if __name__ == "__main__":
    main()
    elementwise()
