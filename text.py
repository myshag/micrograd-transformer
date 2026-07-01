"""
Char-level языковая модель (мини-GPT) на реальном тексте.

Тот же авторегрессионный трансформер, что генерировал простые числа, но теперь
словарь — символы текста, а контекст — целое «окно» символов. Модель
предсказывает следующий символ по предыдущим и учится структуре языка.

Корпус и размер модели — параметры. По умолчанию — English Shakespeare;
для русского запустите train_pushkin() (крупная модель на текстах Пушкина).

Запуск:  python3 text.py                 # Shakespeare, компактная модель
         python3 text.py pushkin         # Пушкин, крупная модель
"""

import os
import sys
import urllib.request

import numpy as np

import nn
import optim
from tensor import Tensor, no_grad

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path, url=None):
    if not os.path.exists(path) and url:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        open(path, "wb").write(urllib.request.urlopen(req, timeout=60).read())
    return open(path, encoding="utf-8").read()


class CharGPT(nn.Module):
    def __init__(self, vocab, block_size, d_model=64, n_heads=4, n_layers=3, seed=0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab, d_model, seed=seed)
        self.pos_emb = nn.Embedding(block_size, d_model, seed=seed + 1)
        self.decoder = nn.TransformerEncoder(d_model, n_heads, n_layers,
                                             causal=True, seed=seed)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, seed=seed + 9)

    def forward(self, tokens):
        T = tokens.shape[1]
        x = self.tok_emb(tokens) + self.pos_emb(np.arange(T))
        x = self.decoder(x)
        return self.head(self.norm(x))     # (B, T, vocab)


def get_batch(data, block_size, batch_size, rng):
    """Случайные окна текста: x = кусок, y = тот же кусок, сдвинутый на 1."""
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def generate(model, stoi, itos, prompt="\n", length=400, seed=0, temperature=0.8):
    rng = np.random.default_rng(seed)
    ctx = [stoi.get(c, 0) for c in prompt]
    out = list(prompt)
    with no_grad():
        for _ in range(length):
            window = np.array([ctx[-model.block_size:]])
            logits = model(Tensor(window)).data[0, -1] / temperature
            e = np.exp(logits - logits.max())
            p = e / e.sum()
            nxt = rng.choice(len(itos), p=p)
            ctx.append(nxt)
            out.append(itos[nxt])
    return "".join(out)


def train(text, block_size=32, d_model=64, n_heads=4, n_layers=3,
          iters=3000, batch_size=32, lr=3e-4, eval_every=250, sample_len=300):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)

    model = CharGPT(len(chars), block_size, d_model, n_heads, n_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)

    print(f"Символов: {len(data):,}, словарь: {len(chars)}")
    print(f"Модель: d_model={d_model}, heads={n_heads}, layers={n_layers}, "
          f"контекст={block_size}")
    print(f"Параметров: {sum(p.data.size for p in model.parameters()):,}\n")

    running = 0.0
    for it in range(1, iters + 1):
        x, y = get_batch(data, block_size, batch_size, rng)
        logits = model(Tensor(x))
        B, T, V = logits.data.shape
        loss = criterion(logits.reshape(B * T, V), y.reshape(B * T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.data

        if it % eval_every == 0:
            print(f"iter {it:4d}: loss = {running / eval_every:.4f}")
            running = 0.0
            print("--- образец ---")
            print(generate(model, stoi, itos, length=sample_len))
            print("---------------\n")

    return model, stoi, itos


def train_shakespeare():
    text = load_text("data/shakespeare.txt", SHAKESPEARE_URL)
    return train(text, block_size=32, d_model=64, n_heads=4, n_layers=3, iters=3000)


def train_pushkin():
    # Крупная модель (~820k параметров) на текстах Пушкина.
    text = load_text("data/pushkin.txt")
    return train(text, block_size=48, d_model=128, n_heads=8, n_layers=4,
                 iters=4000, batch_size=16, lr=3e-4, sample_len=400)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pushkin":
        train_pushkin()
    else:
        train_shakespeare()
