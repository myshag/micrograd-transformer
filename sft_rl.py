"""
SFT + RL — промышленный конвейер обучения (тёплый старт, потом RL).

Чистый RL-с-нуля нестабилен и коллапсирует (см. math_rl.py). Реальные системы
сначала делают SFT (supervised fine-tuning) — учат на ПРАВИЛЬНЫХ ответах, это
стабильно и быстро, — а потом RL шлифует под награду. Здесь это на задаче
(a+b) mod 10:

  Фаза 1 (SFT):  loss = CE(логиты, ПРАВИЛЬНЫЙ ответ)     ← знаем ответ, имитируем
  Фаза 2 (RL):   GRPO + entropy по проверяемой награде   ← дошлифовать до ~1.0

Смысл: SFT выводит модель из случайного состояния (где RL коллапсировал) в
рабочее, а RL добивает. Тёплый старт — вот чего не хватало чистому RL.

Запуск:  python3 sft_rl.py
"""

import numpy as np

import nn
import optim
from tensor import Tensor, no_grad
from math_rl import TinyGPT, problems, VOCAB


def greedy_acc(model):
    c = 0
    with no_grad():
        for a in range(10):
            for b in range(10):
                pred = int(model(Tensor(np.array([[a, b]]))).data[0, -1].argmax())
                c += pred == (a + b) % 10
    return c / 100


def main():
    rng = np.random.default_rng(0)
    model = TinyGPT()
    crit = nn.CrossEntropyLoss()

    # --- Фаза 1: SFT (короткая — чтобы RL было что добивать) ---
    opt = optim.Adam(model.parameters(), lr=3e-3)
    print("Фаза 1 — SFT (supervised на ПРАВИЛЬНЫЙ ответ):")
    for s in range(1, 13):
        a, b = problems(64)
        lg = model(Tensor(np.stack([a, b], axis=1)))[:, -1, :]
        loss = crit(lg, (a + b) % 10)                 # цель = правильный ответ
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 4 == 0:
            print(f"  SFT шаг {s:2d}: greedy acc = {greedy_acc(model):.2f}")
    print(f"после SFT: {greedy_acc(model):.2f}  (стабильно, без коллапса)\n")

    # --- Фаза 2: RLVR (GRPO + entropy) поверх тёплого старта ---
    opt = optim.Adam(model.parameters(), lr=1e-3)
    K, G, STEPS = 16, 16, 120
    print("Фаза 2 — RLVR (GRPO + entropy) поверх SFT:")
    for step in range(1, STEPS + 1):
        a0, b0 = problems(K)
        a, b = np.repeat(a0, G), np.repeat(b0, G)
        inp = np.stack([a, b], axis=1)
        with no_grad():
            last = model(Tensor(inp)).data[:, -1, :]
            e = np.exp(last - last.max(1, keepdims=True))
            p = e / e.sum(1, keepdims=True)
            ans = np.array([rng.choice(VOCAB, p=p[i]) for i in range(K * G)])
        R = (ans == (a + b) % 10).astype(np.float32)
        Rg = R.reshape(K, G)
        adv = ((Rg - Rg.mean(1, keepdims=True)) / (Rg.std(1, keepdims=True) + 1e-4)).reshape(-1)

        lg = model(Tensor(inp))[:, -1, :]
        loss = None
        for i in range(K * G):
            ce = lg[i:i + 1].softmax_cross_entropy(np.array([ans[i]], np.int64))
            term = ce * float(adv[i])
            loss = term if loss is None else loss + term
        loss = loss * (1.0 / (K * G))
        beta = 0.01 * (1.0 - step / STEPS) + 0.0005    # мягкий entropy-бонус с отжигом
        pol = lg.softmax(axis=-1)
        loss = loss - (pol * pol.log() * -1).sum(axis=-1).mean() * beta
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"  RL шаг {step:3d}: greedy acc = {greedy_acc(model):.2f}")

    print(f"\nИтог: greedy acc = {greedy_acc(model):.2f} "
          f"(случайно 0.10). SFT дал старт, RL дошлифовал.")


if __name__ == "__main__":
    main()
