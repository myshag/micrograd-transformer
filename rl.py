"""
RL-дообучение языковой модели на нашем движке (REINFORCE / policy gradient).

Это основа RLHF/PPO в простейшем виде. У нас есть маленький GPT (полностью на
нашем Tensor — значит backward работает). Дообучаем его под скалярную НАГРАДУ,
без размеченных данных:

  1. сэмплируем K последовательностей текущей моделью (политикой);
  2. считаем награду каждой (здесь — сколько раз встретился целевой токен);
  3. REINFORCE: увеличиваем вероятность токенов удачных последовательностей:
        loss = среднее по (R - baseline) · (−log π(выбранные токены))
     градиентный спуск по этому loss = подъём ожидаемой награды.

Награда тут игрушечная (счётчик токена), но механизм ТОТ ЖЕ, что в RLHF —
только там награду даёт обученная reward-модель по человеческим предпочтениям.

Запуск:  python3 rl.py
"""

import numpy as np

import nn
import optim
from tensor import Tensor, no_grad

VOCAB, BLOCK, TARGET = 10, 12, 7        # словарь цифр, длина, целевой токен
rng = np.random.default_rng(0)


class TinyGPT(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, seed=0):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d, seed=seed)
        self.pos = nn.Embedding(BLOCK + 1, d, seed=seed + 1)
        self.dec = nn.TransformerEncoder(d, heads, layers, causal=True, seed=seed)
        self.head = nn.Linear(d, VOCAB, seed=seed + 7)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(np.arange(T))
        return self.head(self.dec(x))


def sample(model, K):
    """Сэмплируем K последовательностей (политикой), без градиента."""
    seqs = np.zeros((K, BLOCK), dtype=np.int64)
    ctx = np.zeros((K, 1), dtype=np.int64)          # BOS = 0
    with no_grad():
        for t in range(BLOCK):
            logits = model(Tensor(ctx)).data[:, -1, :]
            e = np.exp(logits - logits.max(1, keepdims=True))
            p = e / e.sum(1, keepdims=True)
            nxt = np.array([rng.choice(VOCAB, p=p[i]) for i in range(K)])
            seqs[:, t] = nxt
            ctx = np.concatenate([ctx, nxt[:, None]], axis=1)
    return seqs


def reward(seqs):
    return (seqs == TARGET).sum(axis=1).astype(np.float32)   # счётчик целевого токена


def train(steps=120, K=32, lr=1e-2):
    model = TinyGPT()
    opt = optim.Adam(model.parameters(), lr)
    crit = nn.CrossEntropyLoss()

    print(f"Награда = сколько раз токен '{TARGET}' в {BLOCK} шагах (макс {BLOCK}).")
    print("До RL, примеры сэмплов:", *[list(s) for s in sample(model, 3)], sep="\n  ")

    for s in range(1, steps + 1):
        seqs = sample(model, K)                     # политика
        R = reward(seqs)
        base = R.mean()                             # baseline снижает дисперсию
        inp = np.concatenate([np.zeros((K, 1), np.int64), seqs[:, :-1]], axis=1)

        loss = None
        for i in range(K):                          # REINFORCE-loss по батчу
            logits = model(Tensor(inp[i:i + 1]))    # (1, BLOCK, vocab)
            ce = crit(logits.reshape(BLOCK, VOCAB), seqs[i])   # среднее −log π
            term = ce * float(R[i] - base)          # взвешиваем наградой
            loss = term if loss is None else loss + term
        loss = loss * (1.0 / K)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if s % 15 == 0 or s == 1:
            print(f"шаг {s:3d}: средняя награда = {R.mean():.2f} / {BLOCK}")

    print("\nПосле RL, примеры сэмплов:", *[list(s) for s in sample(model, 3)], sep="\n  ")
    return model


if __name__ == "__main__":
    train()
