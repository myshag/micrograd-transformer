"""
Свёрточная сеть (CNN) на MNIST поверх нашего движка.

Архитектура (классическая LeNet-подобная):
  вход (1,28,28)
    Conv2d(1->8, 3x3, pad1) -> ReLU -> MaxPool2d(2)   # 28x28 -> 14x14
    Conv2d(8->16,3x3, pad1) -> ReLU -> MaxPool2d(2)   # 14x14 ->  7x7
    Flatten -> Linear(16*7*7 -> 10)

Свёртки учитывают локальную структуру изображения (соседние пиксели), поэтому
дают заметно выше точность, чем полносвязный MLP. Код — снова чистый PyTorch.

Запуск:  python3 mnist_conv.py
"""

import time

import numpy as np

import nn
import optim
from tensor import Tensor, no_grad


def load_mnist(path="data/mnist.npz"):
    d = np.load(path)
    # Картинки приводим к форме (N, 1, 28, 28) — 1 канал (оттенки серого).
    x_train = d["x_train"].astype(np.float64).reshape(-1, 1, 28, 28) / 255.0
    x_test = d["x_test"].astype(np.float64).reshape(-1, 1, 28, 28) / 255.0
    return x_train, d["y_train"].astype(np.int64), x_test, d["y_test"].astype(np.int64)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, seed=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, seed=2)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 7 * 7, 10, seed=3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # -> (N, 8, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))   # -> (N, 16, 7, 7)
        return self.fc(self.flatten(x))           # -> (N, 10)


def accuracy(model, x, y, batch=200):
    correct = 0
    with no_grad():                       # инференс без построения графа
        for i in range(0, len(x), batch):
            logits = model(Tensor(x[i:i + batch]))
            correct += (logits.data.argmax(1) == y[i:i + batch]).sum()
    return correct / len(x)


def train(epochs=2, batch_size=64, lr=1e-3):
    x_train, y_train, x_test, y_test = load_mnist()
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    n = len(x_train)

    print(f"Параметров: {sum(p.data.size for p in model.parameters()):,}")
    print(f"Точность до обучения: {accuracy(model, x_test, y_test):.4f}\n")

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        running_loss, n_batches = 0.0, 0
        t0 = time.time()

        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            logits = model(Tensor(x_train[idx]))
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            n_batches += 1

        acc = accuracy(model, x_test, y_test)
        print(f"эпоха {epoch}: loss = {running_loss / n_batches:.4f}, "
              f"точность = {acc:.4f}  ({time.time() - t0:.0f} c)")

    return model


if __name__ == "__main__":
    train()
