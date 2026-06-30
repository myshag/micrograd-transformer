"""
Тензорный autograd на numpy.

Это «старший брат» скалярного движка из autograd.py. Идея та же —
строим граф вычислений и считаем градиенты обратным проходом, — но теперь
узел графа хранит не одно число, а целый массив numpy. Благодаря этому
операции выполняются над матрицами сразу, и можно реально обучать сеть
на MNIST.

Главное новое усложнение по сравнению со скаляром — broadcasting.
Когда numpy «растягивает» массивы разной формы (например, прибавляет
вектор-смещение b к матрице X@W), градиент нужно «сжать» обратно к исходной
форме, просуммировав по растянутым осям. За это отвечает _unbroadcast.
"""

import numpy as np


def _unbroadcast(grad, shape):
    """Привести grad к форме shape, суммируя по осям, добавленным broadcasting'ом.

    Пример: b формы (10,) прибавили к матрице (N, 10). При forward numpy
    растянул b до (N, 10). При backward нам нужно вернуть градиент формы (10,),
    значит суммируем по оси N.
    """
    # 1) numpy мог добавить новые оси слева — суммируем их.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) Там, где исходный размер был 1 (а растянули до большего) — тоже суммируем.
    for axis, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """Узел графа: массив numpy + его градиент той же формы."""

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    # --- Операции -----------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # Градиент сложения проходит как есть, но с поправкой на broadcasting.
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def matmul(self, other):
        """Матричное умножение self @ other — основа линейного слоя."""
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # Для C = A @ B:  dA = dC @ B^T,  dB = A^T @ dC
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            # производная: 1 там, где вход был > 0, иначе 0
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            # d/dx sigmoid = sigmoid * (1 - sigmoid)
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def transpose(self):
        """Транспонирование. Нужно линейному слою (веса хранятся как (out, in))."""
        out = Tensor(self.data.T, (self,), "T")

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            # Градиент суммы — единица в каждую ячейку входа (с учётом формы).
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        # mean = sum / N, поэтому переиспользуем sum и делим на число элементов.
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def softmax_cross_entropy(self, targets):
        """Совмещённые softmax + кросс-энтропия -> скаляр-loss.

        targets — массив индексов правильных классов, форма (N,).

        Почему совмещаем: по отдельности softmax и log численно неустойчивы,
        а их комбинация даёт простой и стабильный градиент: (p - y_onehot) / N.
        Это стандартный приём во всех фреймворках.
        """
        logits = self.data
        N = logits.shape[0]

        # Численно устойчивый softmax: вычитаем максимум по строке.
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        probs = exp / exp.sum(axis=1, keepdims=True)        # (N, classes)

        # Кросс-энтропия: -log(вероятность правильного класса), усреднённая.
        log_likelihood = -np.log(probs[np.arange(N), targets] + 1e-12)
        loss = log_likelihood.mean()

        out = Tensor(loss, (self,), "softmax_xent")

        def _backward():
            grad = probs.copy()
            grad[np.arange(N), targets] -= 1      # p - y_onehot
            grad /= N                             # усреднение по батчу
            self.grad += grad * out.grad          # out.grad обычно = 1

        out._backward = _backward
        # probs пригодятся снаружи для подсчёта точности
        out.probs = probs
        return out

    # --- Backward (как в скалярном движке) ----------------------------------

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)   # d(self)/d(self) = 1
        for v in reversed(topo):
            v._backward()

    # --- Сахар --------------------------------------------------------------

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Tensor(shape={self.data.shape})"
