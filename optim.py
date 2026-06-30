"""
Оптимизаторы — правила обновления параметров по их градиентам.

Оптимизатор получает список параметров (Tensor'ов) и на каждом шаге метода
step() двигает их data, используя накопленный grad. backward() считает
градиенты, optimizer.step() их применяет — разделение как в PyTorch.
"""

import numpy as np


class SGD:
    """Стохастический градиентный спуск, опционально с моментом.

    Без момента:     p -= lr * grad
    С моментом:      v = momentum*v + grad;  p -= lr * v
    Момент сглаживает шаги и помогает быстрее проходить «овраги» функции потерь.
    """

    def __init__(self, params, lr=0.1, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        # Накопленная «скорость» для каждого параметра (для момента).
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocities):
            if self.momentum:
                v *= self.momentum
                v += p.grad
                p.data -= self.lr * v
            else:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class Adam:
    """Adam — адаптивный шаг для каждого параметра.

    Хранит две скользящие средние градиента:
      m — среднее самого градиента (момент 1-го порядка),
      v — среднее квадрата градиента (момент 2-го порядка).
    Шаг масштабируется как m / (√v + eps): по «шумным» направлениям (большой v)
    шаг меньше, по стабильным — больше. Обычно сходится быстрее SGD.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0   # номер шага — нужен для коррекции смещения

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            # Обновляем скользящие средние.
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            # Коррекция смещения (в начале m и v занижены, т.к. стартуют с нуля).
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
