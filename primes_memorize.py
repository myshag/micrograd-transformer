"""
Достигаем 100% точности на простых числах — через ЗАПОМИНАНИЕ (overfitting).

Важный урок. Цифры однозначно задают число, поэтому датасет
«последовательность цифр -> простое/составное» непротиворечив: одинаковый вход
всегда имеет одну метку. Значит трансформер с достаточной ёмкостью может
идеально ЗАПОМНИТЬ обучающую выборку и дать на ней 100%.

Но это не «понимание» простоты: на отложенном тесте точность застревает намного
ниже. Так наглядно видно различие между запоминанием (train -> 100%) и
обобщением (test -> плато). Это классический overfitting.

Запуск:  python3 primes_memorize.py
"""

import numpy as np

import nn
import optim
from tensor import Tensor, no_grad

LIMIT = 10000      # числа из [2, LIMIT)
SEQ_LEN = 4        # 9999 -> 4 цифры


def sieve(n):
    is_prime = np.ones(n, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = False
    return is_prime


def to_digits(n):
    return [(n // 10 ** k) % 10 for k in reversed(range(SEQ_LEN))]


def make_dataset(seed=0):
    is_prime = sieve(LIMIT)
    primes = np.nonzero(is_prime)[0]
    composites = np.array([x for x in range(2, LIMIT) if not is_prime[x]])
    rng = np.random.default_rng(seed)
    comp = rng.choice(composites, size=len(primes), replace=False)

    nums = np.concatenate([primes, comp])
    labels = np.concatenate([np.ones(len(primes)), np.zeros(len(comp))]).astype(np.int64)
    X = np.array([to_digits(int(n)) for n in nums], dtype=np.int64)

    perm = rng.permutation(len(nums))
    X, labels = X[perm], labels[perm]
    split = int(0.8 * len(nums))
    return (X[:split], labels[:split]), (X[split:], labels[split:])


class PrimeClassifier(nn.Module):
    def __init__(self, d_model=96, n_heads=6, n_layers=3, seed=0):
        super().__init__()
        self.tok_emb = nn.Embedding(10, d_model, seed=seed)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model, seed=seed + 1)
        self.encoder = nn.TransformerEncoder(d_model, n_heads, n_layers, seed=seed)
        self.head = nn.Linear(d_model, 2, seed=seed + 7)

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb(np.arange(SEQ_LEN))
        x = self.encoder(x)
        x = x.mean(axis=1)
        return self.head(x)


def accuracy(model, X, y, batch=512):
    correct = 0
    with no_grad():
        for i in range(0, len(X), batch):
            logits = model(Tensor(X[i:i + batch]))
            correct += (logits.data.argmax(1) == y[i:i + batch]).sum()
    return correct / len(X)


def train(epochs=200, batch_size=64, lr=3e-4):
    (Xtr, ytr), (Xte, yte) = make_dataset()
    model = PrimeClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    n = len(Xtr)

    print(f"Обучающих: {n}, тест: {len(Xte)}")
    print(f"Параметров: {sum(p.data.size for p in model.parameters()):,}")
    print("Цель: train -> 100% (запоминание). Следим и за test.\n")

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        running, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            logits = model(Tensor(Xtr[idx]))
            loss = criterion(logits, ytr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.data
            nb += 1

        tr = accuracy(model, Xtr, ytr)
        te = accuracy(model, Xte, yte)
        if epoch % 5 == 0 or tr == 1.0:
            print(f"эпоха {epoch:3d}: loss={running/nb:.4f}  "
                  f"train={tr:.4f}  test={te:.4f}")
        if tr == 1.0:
            print(f"\n>>> Достигнута 100% точность на train за {epoch} эпох.")
            print(f">>> При этом на тесте только {te:.3f} — модель ЗАПОМНИЛА, "
                  f"а не научилась проверять простоту.")
            break

    return model


if __name__ == "__main__":
    train()
