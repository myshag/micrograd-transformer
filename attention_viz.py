"""
Визуализация матрицы внимания softmax(Q·Kᵀ) трансформера на простых числах.

Обучаем компактный классификатор простое/составное на числах < 10000 (каждое —
4 цифры), затем для примеров показываем, на какие цифры «смотрит» каждая позиция.
Каждая строка матрицы — распределение внимания одной позиции по всем позициям
(сумма = 1). Ожидаем фокус на последней цифре: именно она решает делимость на 2 и 5.

Запуск:  python3 attention_viz.py
"""

import numpy as np

import nn
import optim
from tensor import Tensor, no_grad

LIMIT = 10000
SEQ_LEN = 4
SHADES = " .:-=+*#%@"        # градации «тепла» от слабого к сильному


def sieve(n):
    p = np.ones(n, dtype=bool); p[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if p[i]:
            p[i * i::i] = False
    return p


def to_digits(n):
    return [(n // 10 ** k) % 10 for k in reversed(range(SEQ_LEN))]


def make_dataset(seed=0):
    is_prime = sieve(LIMIT)
    primes = np.nonzero(is_prime)[0]
    comps = np.array([x for x in range(2, LIMIT) if not is_prime[x]])
    rng = np.random.default_rng(seed)
    comp = rng.choice(comps, size=len(primes), replace=False)
    nums = np.concatenate([primes, comp])
    y = np.concatenate([np.ones(len(primes)), np.zeros(len(comp))]).astype(np.int64)
    X = np.array([to_digits(int(n)) for n in nums], dtype=np.int64)
    perm = rng.permutation(len(nums))
    return X[perm], y[perm], is_prime


class Classifier(nn.Module):
    def __init__(self, d_model=32, n_heads=4, n_layers=2, seed=0):
        super().__init__()
        self.tok_emb = nn.Embedding(10, d_model, seed=seed)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model, seed=seed + 1)
        self.encoder = nn.TransformerEncoder(d_model, n_heads, n_layers, seed=seed)
        self.head = nn.Linear(d_model, 2, seed=seed + 7)

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb(np.arange(SEQ_LEN))
        x = self.encoder(x)
        return self.head(x.mean(axis=1))


def train(model, X, y, epochs=20, batch=64, lr=1e-3):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            idx = order[i:i + batch]
            loss = crit(model(Tensor(X[idx])), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0:
            with no_grad():
                acc = (model(Tensor(X)).data.argmax(1) == y).mean()
            print(f"  эпоха {ep:2d}: train acc = {acc:.3f}")


def heatmap(matrix, digits):
    """Печатает T×T матрицу внимания тепловой картой. Строка i — куда смотрит
    позиция i; столбец j — на какую позицию (цифру) она смотрит."""
    labels = [f"d{k}={d}" for k, d in enumerate(digits)]
    print("            " + "  ".join(f"{l:>5s}" for l in labels) + "   (на кого)")
    for i, row in enumerate(matrix):
        cells = []
        for w in row:
            sh = SHADES[min(len(SHADES) - 1, int(w * len(SHADES)))]
            cells.append(f"{sh}{w:4.2f}")
        print(f"  {labels[i]:>8s}  " + " ".join(cells))


def show_number(model, n, is_prime, layer=-1):
    digits = to_digits(n)
    with no_grad():
        model(Tensor(np.array([digits])))          # прогон -> запишутся веса
    # веса слоя: (B=1, heads, T, T) -> среднее по головам
    W = model.encoder.layers[layer].attn.attn_weights[0]
    real = "простое" if is_prime[n] else "составное"
    print(f"\n### Число {n} ({real}) — внимание (среднее по головам), слой {layer}:")
    heatmap(W.mean(axis=0), digits)


def main():
    X, y, is_prime = make_dataset()
    model = Classifier()
    print(f"Обучаем компактный классификатор ({sum(p.data.size for p in model.parameters()):,} параметров):")
    train(model, X, y)

    # Примеры: составные с «говорящей» последней цифрой и простые.
    for n in [1234, 9995, 9973, 7919]:
        show_number(model, n, is_prime)

    # Покажем ещё разбивку по головам для одного числа.
    n = 9973
    with no_grad():
        model(Tensor(np.array([to_digits(n)])))
    heads = model.encoder.layers[-1].attn.attn_weights[0]   # (heads, T, T)
    print(f"\n### Число {n}: внимание по каждой голове (последний слой):")
    for h in range(heads.shape[0]):
        print(f"-- голова {h}:")
        heatmap(heads[h], to_digits(n))


if __name__ == "__main__":
    main()
