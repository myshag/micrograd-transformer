"""
Генеративный трансформер, который ПРЕДСКАЗЫВАЕТ (порождает) простые числа.

Это авторегрессионная языковая модель на уровне цифр (как мини-GPT):
число записывается как последовательность из 5 цифр, и модель учится
предсказывать следующую цифру по предыдущим. Обучаемся на всех простых
числах до 100000. Затем генерируем новые числа цифра за цифрой и смотрим,
какая доля из них действительно простые.

Ключевой механизм — каузальное (causal) внимание: при предсказании цифры
модель видит только уже сгенерированные цифры слева, но не «будущие».

Чему модель реально может научиться: как «выглядят» простые числа —
последняя цифра из {1,3,7,9}, типичные распределения разрядов и т.п.
Истинную простоту из цифр не вывести, но эти закономерности поднимают
долю простых среди сгенерированных НАМНОГО выше случайной (~9.6%).

Запуск:  python3 primes.py
"""

import numpy as np

import nn
import optim
from tensor import Tensor, no_grad

LIMIT = 100000     # простые числа из диапазона [2, LIMIT)
SEQ_LEN = 5        # 99991 -> 5 цифр
BOS = 10           # спец-токен «начало последовательности» (цифры это 0..9)
VOCAB = 11         # 10 цифр + BOS


def sieve(n):
    """Решето Эратосфена -> булев массив простоты."""
    is_prime = np.ones(n, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = False
    return is_prime


def to_digits(n):
    return [(n // 10 ** k) % 10 for k in reversed(range(SEQ_LEN))]


def make_dataset():
    """Все простые < LIMIT как входы [BOS,d0..d3] и цели [d0..d4]."""
    primes = np.nonzero(sieve(LIMIT))[0]
    digits = np.array([to_digits(int(p)) for p in primes], dtype=np.int64)  # (M,5)
    # Вход: BOS + первые 4 цифры. Цель: все 5 цифр (сдвиг на одну позицию).
    inputs = np.concatenate(
        [np.full((len(digits), 1), BOS), digits[:, :-1]], axis=1)          # (M,5)
    targets = digits                                                        # (M,5)
    return inputs, targets, set(primes.tolist())


# --- Модель (мини-GPT) ------------------------------------------------------

class PrimeLM(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2, seed=0):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, d_model, seed=seed)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model, seed=seed + 1)
        # causal=True — авторегрессионный режим.
        self.decoder = nn.TransformerEncoder(d_model, n_heads, n_layers,
                                             causal=True, seed=seed)
        self.head = nn.Linear(d_model, 10, seed=seed + 7)   # предсказываем цифру 0..9

    def forward(self, tokens):
        T = tokens.shape[1]
        x = self.tok_emb(tokens) + self.pos_emb(np.arange(T))
        x = self.decoder(x)
        return self.head(x)        # (B, T, 10) — логиты следующей цифры


# --- Генерация --------------------------------------------------------------

def generate(model, count, seed=0, temperature=1.0):
    """Сэмплируем `count` чисел цифра за цифрой из обученной модели."""
    rng = np.random.default_rng(seed)
    tokens = np.full((count, 1), BOS, dtype=np.int64)   # старт: только BOS
    with no_grad():
        for _ in range(SEQ_LEN):
            logits = model(Tensor(tokens))              # (count, t, 10)
            last = logits.data[:, -1, :] / temperature  # логиты следующей цифры
            # softmax по каждой строке и сэмплирование цифры
            e = np.exp(last - last.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            nxt = np.array([rng.choice(10, p=probs[i]) for i in range(count)])
            tokens = np.concatenate([tokens, nxt[:, None]], axis=1)
    digits = tokens[:, 1:]                              # убираем BOS
    numbers = digits @ (10 ** np.arange(SEQ_LEN - 1, -1, -1))
    return numbers


# --- Обучение и оценка ------------------------------------------------------

def evaluate(model, train_primes, is_prime, n=3000):
    nums = generate(model, n)
    prime_mask = np.array([is_prime[x] for x in nums])
    frac_prime = prime_mask.mean()
    uniq = len(set(nums.tolist()))
    # Сколько РАЗНЫХ простых сгенерировано (мера разнообразия выдачи).
    distinct_primes = len({int(x) for x in nums[prime_mask]})
    last_digit = np.bincount(nums % 10, minlength=10)
    return frac_prime, uniq, distinct_primes, last_digit


def train(epochs=15, batch_size=64, lr=1e-3):
    inputs, targets, train_primes = make_dataset()
    is_prime = sieve(LIMIT)
    model = PrimeLM()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    n = len(inputs)
    base = is_prime.sum() / LIMIT

    print(f"Обучающих простых: {n}")
    print(f"Параметров: {sum(p.data.size for p in model.parameters()):,}")
    print(f"Доля простых среди случайных чисел (база): {base:.3f}\n")

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        running, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            logits = model(Tensor(inputs[idx]))               # (B,5,10)
            B = len(idx)
            loss = criterion(logits.reshape(B * SEQ_LEN, 10),
                             targets[idx].reshape(B * SEQ_LEN))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.data
            nb += 1
        frac, uniq, distinct_primes, _ = evaluate(model, train_primes, is_prime, n=1000)
        print(f"эпоха {epoch:2d}: loss = {running / nb:.4f}, "
              f"простых среди сгенерированных = {frac:.3f}  "
              f"(уникальных чисел {uniq}, разных простых {distinct_primes})")

    return model, train_primes, is_prime


if __name__ == "__main__":
    model, train_primes, is_prime = train()

    print("\nГенерируем 3000 чисел обученной моделью:")
    frac, uniq, distinct_primes, last_digit = evaluate(model, train_primes, is_prime, n=3000)
    print(f"  доля простых: {frac:.3f}  (случайно было бы ~{is_prime.sum()/LIMIT:.3f})")
    print(f"  уникальных чисел: {uniq}, из них разных простых: {distinct_primes}")
    print(f"  распределение последней цифры: {last_digit.tolist()}")
    print("  (у простых >5 последняя цифра только 1,3,7,9 — модель это выучивает)")

    print("\nПримеры сгенерированных простых чисел:")
    nums = generate(model, 4000, seed=7)
    primes_out = sorted({int(x) for x in nums if is_prime[x]})
    print("  ", primes_out[:20])
