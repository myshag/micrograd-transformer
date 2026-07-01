"""
REINFORCE вживую на 3 сэмплах — как награда двигает log π.

Показывает суть policy gradient: у каждого сэмпла есть награда R; вес (R−baseline)
задаёт, тянуть его вероятность вверх (лучше среднего) или вниз (хуже). После шагов
видно: хорошие сэмплы становятся вероятнее, плохие — реже, пропорционально весу.

Награда тут игрушечная (число токенов '7'), но механизм тот же, что в RLHF.
Запуск:  python3 rl_demo.py
"""

import numpy as np

import nn
import optim
from tensor import Tensor
from rl import TinyGPT, VOCAB, BLOCK, reward

NAMES = ["A (много 7)", "B (средне) ", "C (ноль 7) "]
SAMPLES = np.array([
    [7, 7, 7, 7, 7, 7, 3, 1, 2, 0, 4, 5],   # A: высокая награда
    [1, 7, 2, 7, 3, 0, 4, 5, 6, 8, 9, 1],   # B: средняя
    [1, 2, 3, 4, 5, 6, 8, 9, 0, 1, 2, 3],   # C: нулевая
], dtype=np.int64)


def main():
    model = TinyGPT(seed=3)
    opt = optim.Adam(model.parameters(), lr=1e-3)   # маленький lr — RL нестабилен
    crit = nn.CrossEntropyLoss()

    def logp(seqs):
        # средний log π(выбранного токена) по позициям = −CE, для каждого сэмпла
        return np.array([
            -float(crit(model(Tensor(np.concatenate([[0], s[:-1]])[None, :]))
                        .reshape(BLOCK, VOCAB), s).data) for s in seqs])

    R = reward(SAMPLES)
    base = R.mean()
    lp0 = logp(SAMPLES)

    print("СЭМПЛ          R    log π(ср)   R-baseline   действие")
    for i, nm in enumerate(NAMES):
        w = R[i] - base
        print(f"{nm}   {R[i]:3.0f}   {lp0[i]:8.3f}    {w:+6.2f}     "
              f"{'ПОДНЯТЬ ↑' if w > 0 else 'опустить ↓'}")
    print(f"baseline (среднее R) = {base:.2f}")

    # несколько маленьких REINFORCE-шагов: loss = mean_i( CE_i · (R_i − baseline) )
    for _ in range(5):
        loss = None
        for i, s in enumerate(SAMPLES):
            ce = crit(model(Tensor(np.concatenate([[0], s[:-1]])[None, :]))
                      .reshape(BLOCK, VOCAB), s)
            term = ce * float(R[i] - base)      # R — КОНСТАНТА-множитель
            loss = term if loss is None else loss + term
        loss = loss * (1.0 / len(SAMPLES))
        opt.zero_grad()
        loss.backward()
        opt.step()

    lp1 = logp(SAMPLES)
    print("\nПосле шагов — как сдвинулась log π каждого сэмпла:")
    for i, nm in enumerate(NAMES):
        d = lp1[i] - lp0[i]
        print(f"{nm}   log π: {lp0[i]:+.3f} -> {lp1[i]:+.3f}   Δ {d:+.3f}   "
              f"{'вероятнее ↑' if d > 0 else 'реже ↓'}")
    print("\nΔlog π ∝ (R − baseline): лучше среднего — вверх, хуже — вниз. "
          "Это и есть policy gradient.")


if __name__ == "__main__":
    main()
