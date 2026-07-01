"""
Эталон PyTorch: сверка нашего движка с настоящим фреймворком.

Строим ОДИН И ТОТ ЖЕ MLP (784 -> 256 -> 10) в нашем движке и в PyTorch с
идентичными весами и сравниваем:
  - точность: совпадают ли forward-логиты и все градиенты после backward;
  - скорость: сколько занимает forward+backward в каждом.

PyTorch — доверенный оракул: если наши градиенты совпали с ним, backprop верен
(строже, чем численная проверка). А заодно видно честную разницу в скорости.

Запуск:  python3 torch_compare.py
"""

import time

import numpy as np
import torch

import nn
import optim
from tensor import Tensor

BATCH, DIN, H, DOUT = 64, 784, 256, 10


def build_ours(seed=0):
    return nn.Sequential(nn.Linear(DIN, H, seed=seed),
                         nn.ReLU(),
                         nn.Linear(H, DOUT, seed=seed + 1))


def build_torch_from(ours):
    """Torch-модель с ТЕМИ ЖЕ весами (наш weight (out,in) == torch weight)."""
    lin = [l for l in ours.layers if isinstance(l, nn.Linear)]
    m = torch.nn.Sequential(torch.nn.Linear(DIN, H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H, DOUT))
    tl = [m[0], m[2]]
    for our_l, t_l in zip(lin, tl):
        t_l.weight.data = torch.tensor(our_l.weight.data, dtype=torch.float32)
        t_l.bias.data = torch.tensor(our_l.bias.data, dtype=torch.float32)
    return m


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((BATCH, DIN)).astype(np.float32)
    y = rng.integers(0, DOUT, size=BATCH).astype(np.int64)

    ours = build_ours()
    tm = build_torch_from(ours)

    # --- точность: forward ---
    our_logits = ours(Tensor(X)).data
    t_logits = tm(torch.tensor(X)).detach().numpy()
    print("=== Точность (сверка с PyTorch) ===")
    print(f"forward логиты:  max|наш - torch| = {np.abs(our_logits - t_logits).max():.2e}")

    # --- точность: backward (градиенты) ---
    loss = nn.CrossEntropyLoss()(ours(Tensor(X)), y)
    ours.zero_grad(); loss.backward()

    t_loss = torch.nn.CrossEntropyLoss()(tm(torch.tensor(X)), torch.tensor(y))
    tm.zero_grad(); t_loss.backward()

    print(f"loss:            наш = {float(loss.data):.6f},  torch = {t_loss.item():.6f}")
    lin = [l for l in ours.layers if isinstance(l, nn.Linear)]
    worst = 0.0
    for i, (our_l, t_l) in enumerate(zip(lin, [tm[0], tm[2]])):
        dW = np.abs(our_l.weight.grad - t_l.weight.grad.numpy()).max()
        db = np.abs(our_l.bias.grad - t_l.bias.grad.numpy()).max()
        worst = max(worst, dW, db)
        print(f"слой {i}: max|grad(наш) - grad(torch)| = {max(dW, db):.2e}")
    print("Итог:", "OK ✓ градиенты совпали с PyTorch" if worst < 1e-3
          else "РАСХОЖДЕНИЕ")

    # --- скорость: forward + backward ---
    print("\n=== Скорость (forward + backward, mean over 50 iters) ===")
    reps = 50

    opt = optim.Adam(ours.parameters(), lr=1e-3)
    for _ in range(3):                                   # прогрев
        loss = nn.CrossEntropyLoss()(ours(Tensor(X)), y)
        opt.zero_grad(); loss.backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(reps):
        loss = nn.CrossEntropyLoss()(ours(Tensor(X)), y)
        opt.zero_grad(); loss.backward(); opt.step()
    ours_ms = (time.perf_counter() - t0) / reps * 1e3

    topt = torch.optim.Adam(tm.parameters(), lr=1e-3)
    tX, ty = torch.tensor(X), torch.tensor(y)
    tcrit = torch.nn.CrossEntropyLoss()
    for _ in range(3):
        tl = tcrit(tm(tX), ty); topt.zero_grad(); tl.backward(); topt.step()
    t0 = time.perf_counter()
    for _ in range(reps):
        tl = tcrit(tm(tX), ty); topt.zero_grad(); tl.backward(); topt.step()
    torch_ms = (time.perf_counter() - t0) / reps * 1e3

    print(f"наш движок (numpy float32): {ours_ms:.2f} мс/шаг")
    print(f"PyTorch (CPU float32):      {torch_ms:.2f} мс/шаг")
    print(f"PyTorch быстрее в {ours_ms / torch_ms:.1f}x")


if __name__ == "__main__":
    main()
