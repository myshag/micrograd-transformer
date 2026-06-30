"""
Обучение MLP на MNIST поверх нашего тензорного autograd.

Сеть:  784 -> 128 -> ReLU -> 10  (softmax + кросс-энтропия)

Запуск:  python3 mnist.py
"""

import numpy as np

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


# --- Модель -----------------------------------------------------------------

class MLP:
    """Двухслойный перцептрон: вход -> скрытый слой (ReLU) -> выход."""

    def __init__(self, n_in=784, n_hidden=128, n_out=10, seed=42):
        rng = np.random.default_rng(seed)

        # Инициализация He для слоёв с ReLU: масштаб ~ sqrt(2/n_in).
        # Хороший масштаб важен: слишком большой -> взрыв, малый -> затухание.
        self.W1 = Tensor(rng.standard_normal((n_in, n_hidden)) * np.sqrt(2 / n_in))
        self.b1 = Tensor(np.zeros(n_hidden))
        self.W2 = Tensor(rng.standard_normal((n_hidden, n_out)) * np.sqrt(2 / n_hidden))
        self.b2 = Tensor(np.zeros(n_out))

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        # x: Tensor формы (N, 784).  Возвращаем «сырые» логиты (N, 10).
        h = (x @ self.W1 + self.b1).relu()
        logits = h @ self.W2 + self.b2
        return logits


# --- Оценка точности --------------------------------------------------------

def accuracy(model, x, y, batch=1000):
    """Доля верных предсказаний на (x, y). Считаем батчами для скорости."""
    correct = 0
    for i in range(0, len(x), batch):
        logits = model.forward(Tensor(x[i:i + batch]))
        pred = logits.data.argmax(axis=1)
        correct += (pred == y[i:i + batch]).sum()
    return correct / len(x)


# --- Цикл обучения ----------------------------------------------------------

def train(epochs=5, batch_size=64, lr=0.1):
    x_train, y_train, x_test, y_test = load_mnist()
    model = MLP()
    rng = np.random.default_rng(0)
    n = len(x_train)

    print(f"Старт. Параметров: "
          f"{sum(p.data.size for p in model.parameters()):,}")
    print(f"Точность до обучения: {accuracy(model, x_test, y_test):.4f}\n")

    for epoch in range(1, epochs + 1):
        # Перемешиваем порядок примеров каждую эпоху.
        order = rng.permutation(n)
        running_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = Tensor(x_train[idx])
            yb = y_train[idx]

            # 1) forward: считаем логиты и loss
            logits = model.forward(xb)
            loss = logits.softmax_cross_entropy(yb)

            # 2) обнуляем градиенты (иначе они накопятся с прошлого шага)
            for p in model.parameters():
                p.grad = np.zeros_like(p.data)

            # 3) backward: считаем градиенты по всем параметрам
            loss.backward()

            # 4) шаг SGD: двигаем параметры против градиента
            for p in model.parameters():
                p.data -= lr * p.grad

            running_loss += loss.data
            n_batches += 1

        acc = accuracy(model, x_test, y_test)
        print(f"эпоха {epoch}: средний loss = {running_loss / n_batches:.4f}, "
              f"точность на тесте = {acc:.4f}")

    return model


if __name__ == "__main__":
    train()
