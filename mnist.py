"""
Обучение MLP на MNIST через мини-API (nn.py + optim.py).

Сравните с автономной версией: модель теперь собирается из кубиков,
а обновление параметров делает оптимизатор. Меньше ручного кода, та же суть.

Запуск:  python3 mnist.py
"""

import numpy as np

import nn
import optim
from tensor import Tensor


# --- Данные -----------------------------------------------------------------

def load_mnist(path="data/mnist.npz"):
    """Загружает MNIST и нормализует пиксели в диапазон [0, 1]."""
    d = np.load(path)
    x_train = d["x_train"].astype(np.float64) / 255.0   # (60000, 784)
    y_train = d["y_train"].astype(np.int64)             # (60000,)
    x_test = d["x_test"].astype(np.float64) / 255.0
    y_test = d["y_test"].astype(np.int64)
    return x_train, y_train, x_test, y_test


# --- Модель: собираем из кубиков --------------------------------------------

def build_model():
    return nn.Sequential(
        nn.Linear(784, 128, seed=42),
        nn.ReLU(),
        nn.Linear(128, 10, seed=43),
    )


# --- Оценка точности --------------------------------------------------------

def accuracy(model, x, y, batch=1000):
    correct = 0
    for i in range(0, len(x), batch):
        logits = model(Tensor(x[i:i + batch]))
        pred = logits.data.argmax(axis=1)
        correct += (pred == y[i:i + batch]).sum()
    return correct / len(x)


# --- Цикл обучения ----------------------------------------------------------

def train(epochs=5, batch_size=64, lr=0.001):
    x_train, y_train, x_test, y_test = load_mnist()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    n = len(x_train)

    print(f"Параметров: {sum(p.data.size for p in model.parameters()):,}")
    print(f"Оптимизатор: Adam(lr={lr})")
    print(f"Точность до обучения: {accuracy(model, x_test, y_test):.4f}\n")

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        running_loss, n_batches = 0.0, 0

        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = Tensor(x_train[idx])
            yb = y_train[idx]

            # Канонический цикл PyTorch: forward -> zero_grad -> backward -> step.
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            n_batches += 1

        acc = accuracy(model, x_test, y_test)
        print(f"эпоха {epoch}: средний loss = {running_loss / n_batches:.4f}, "
              f"точность на тесте = {acc:.4f}")

    return model


if __name__ == "__main__":
    train()
