"""
RLVR: обучение считать через RL с ПРОВЕРЯЕМОЙ наградой (математика-судья).

Так учат reasoning-модели: награда не от reward-модели, а от ВЕРИФИКАТОРА —
объективная проверка. Здесь судья — арифметика. Модель видит [a, b] и выдаёт
ответ; награда = 1, если ответ == (a+b) mod 10 (мы просто вычисляем). Никаких
размеченных примеров — модель открывает сложение сама, из проб и награды.

Путь набитых шишек (всё воспроизводимо в истории проекта):
  - наивный REINFORCE (baseline на весь батч) СХЛОПЫВАЕТСЯ в один токен;
  - GRPO (G ответов на промпт, baseline по группе) даёт честное преимущество
    для каждого промпта, но всё ещё коллапсирует;
  - + entropy-бонус с отжигом (держать исследование, потом эксплуатация) —
    модель перестаёт коллапсировать и учится (~0.5-0.6 greedy vs 0.10 случайно).

Ответ mod 10 равномерен, поэтому «всегда X» даёт лишь 10% — нет лазейки,
модель вынуждена реально считать. RL-с-нуля нестабилен: отсюда в реальных
системах сначала SFT (тёплый старт), потом RL.

Запуск:  python3 math_rl.py
"""

import numpy as np

import nn
import optim
from tensor import Tensor, no_grad

VOCAB = 10
rng = np.random.default_rng(0)


class TinyGPT(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, seed=0):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d, seed=seed)
        self.pos = nn.Embedding(2, d, seed=seed + 1)
        self.dec = nn.TransformerEncoder(d, heads, layers, causal=True, seed=seed)
        self.head = nn.Linear(d, VOCAB, seed=seed + 7)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(np.arange(idx.shape[1]))
        return self.head(self.dec(x))                    # (B, T, vocab)


def problems(K):
    return rng.integers(0, 10, K), rng.integers(0, 10, K)   # ответ = (a+b) mod 10


def main():
    model = TinyGPT()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    K, G, STEPS = 16, 16, 200

    print("Судья — арифметика. GRPO + entropy-бонус с отжигом (explore -> exploit).\n")
    for step in range(1, STEPS + 1):
        a0, b0 = problems(K)
        a, b = np.repeat(a0, G), np.repeat(b0, G)        # каждый промпт по G раз
        inp = np.stack([a, b], axis=1)                   # (K*G, 2)
        with no_grad():                                  # сэмплируем G ответов на промпт
            last = model(Tensor(inp)).data[:, -1, :]
            e = np.exp(last - last.max(1, keepdims=True))
            p = e / e.sum(1, keepdims=True)
            ans = np.array([rng.choice(VOCAB, p=p[i]) for i in range(K * G)])
        R = (ans == (a + b) % 10).astype(np.float32)            # ВЕРИФИКАТОР
        Rg = R.reshape(K, G)                             # преимущество ОТНОСИТЕЛЬНО группы
        adv = ((Rg - Rg.mean(1, keepdims=True)) /
               (Rg.std(1, keepdims=True) + 1e-4)).reshape(-1)

        lg = model(Tensor(inp))[:, -1, :]                # (K*G, vocab) — с градиентом
        loss = None
        for i in range(K * G):
            ce = lg[i:i + 1].softmax_cross_entropy(np.array([ans[i]], np.int64))
            term = ce * float(adv[i])                    # advantage — множитель
            loss = term if loss is None else loss + term
        loss = loss * (1.0 / (K * G))
        # entropy-бонус с отжигом: вначале много исследования, к концу — эксплуатация
        beta = 0.03 * (1.0 - step / STEPS) + 0.001
        pol = lg.softmax(axis=-1)
        entropy = (pol * pol.log() * -1).sum(axis=-1).mean()
        loss = loss - entropy * beta                     # максимизируем энтропию
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 15 == 0:
            print(f"шаг {step:3d}: точность (верных ответов) = {R.mean():.2f}")

    # финальная проверка greedy на всех 100 парах (a+b) mod 10
    print("\nПроверка (greedy):")
    correct = tot = 0
    with no_grad():
        for a in range(10):
            for b in range(10):
                pred = int(model(Tensor(np.array([[a, b]]))).data[0, -1].argmax())
                correct += pred == (a + b) % 10
                tot += 1
                if (a, b) in [(3, 4), (2, 2), (5, 4), (7, 8), (6, 9)]:
                    tr = (a + b) % 10
                    print(f"   ({a}+{b})%10={pred} {'✓' if pred==tr else '✗'}  (верно {tr})")
    print(f"\nИтоговая точность: {correct}/{tot} = {correct / tot:.2f} "
          f"— выучил сложение ТОЛЬКО из награды.")


if __name__ == "__main__":
    main()
