"""
Char-level языковая модель (мини-GPT) на реальном тексте.

Тот же авторегрессионный трансформер, что генерировал простые числа, но теперь
словарь — символы текста, а контекст — целое «окно» из block_size символов.
Модель предсказывает следующий символ по предыдущим и так учится структуре
языка: словам, пробелам, пунктуации, именам персонажей.

Данные: data/shakespeare.txt (скачивается автоматически).
Запуск:  python3 text.py
"""

import os
import urllib.request

import numpy as np

import nn
import optim
from tensor import Tensor, no_grad

BLOCK_SIZE = 32     # длина контекста (сколько символов видит модель)
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path="data/shakespeare.txt"):
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        open(path, "wb").write(urllib.request.urlopen(req, timeout=60).read())
    return open(path, encoding="utf-8").read()


class CharGPT(nn.Module):
    def __init__(self, vocab, d_model=64, n_heads=4, n_layers=3, seed=0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model, seed=seed)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, d_model, seed=seed + 1)
        self.decoder = nn.TransformerEncoder(d_model, n_heads, n_layers,
                                             causal=True, seed=seed)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, seed=seed + 9)

    def forward(self, tokens):
        T = tokens.shape[1]
        x = self.tok_emb(tokens) + self.pos_emb(np.arange(T))
        x = self.decoder(x)
        return self.head(self.norm(x))     # (B, T, vocab)


def get_batch(data, batch_size, rng):
    """Случайные окна текста: x = кусок, y = тот же кусок, сдвинутый на 1."""
    ix = rng.integers(0, len(data) - BLOCK_SIZE - 1, size=batch_size)
    x = np.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = np.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x, y


def generate(model, stoi, itos, prompt="\n", length=300, seed=0, temperature=0.8):
    rng = np.random.default_rng(seed)
    ctx = [stoi[c] for c in prompt]
    out = list(prompt)
    with no_grad():
        for _ in range(length):
            window = np.array([ctx[-BLOCK_SIZE:]])          # последние BLOCK_SIZE
            logits = model(Tensor(window)).data[0, -1] / temperature
            e = np.exp(logits - logits.max())
            p = e / e.sum()
            nxt = rng.choice(len(itos), p=p)
            ctx.append(nxt)
            out.append(itos[nxt])
    return "".join(out)


def train(iters=3000, batch_size=32, lr=3e-4, eval_every=250):
    text = load_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)

    model = CharGPT(vocab=len(chars))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)

    print(f"Символов в корпусе: {len(data):,}, словарь: {len(chars)}")
    print(f"Параметров модели: {sum(p.data.size for p in model.parameters()):,}")
    print(f"Контекст: {BLOCK_SIZE} символов\n")

    running = 0.0
    for it in range(1, iters + 1):
        x, y = get_batch(data, batch_size, rng)
        logits = model(Tensor(x))                          # (B, T, vocab)
        B, T, V = logits.data.shape
        loss = criterion(logits.reshape(B * T, V), y.reshape(B * T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.data

        if it % eval_every == 0:
            print(f"iter {it:4d}: loss = {running / eval_every:.4f}")
            running = 0.0
            sample = generate(model, stoi, itos, length=200)
            print("--- образец ---")
            print(sample.replace("\n", "\\n"))
            print("---------------\n")

    return model, stoi, itos


if __name__ == "__main__":
    train()
