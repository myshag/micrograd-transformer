"""
Минимальный autograd-движок (скалярный).

Идея: каждое число мы оборачиваем в объект Value. Когда мы выполняем
операции (+, *, **, tanh ...), мы не только считаем результат, но и
запоминаем, ИЗ ЧЕГО он получился. Так строится граф вычислений.

Зная граф, мы можем посчитать производную итогового значения по любому
из входов — это и есть автоматическое дифференцирование (backprop).
"""

import math


class Value:
    """Узел графа вычислений: хранит одно число и его градиент."""

    def __init__(self, data, _children=(), _op=""):
        # data  — само значение (forward-проход).
        self.data = data
        # grad  — производная итогового результата ПО этому узлу.
        #         Накапливается во время backward. По умолчанию 0.
        self.grad = 0.0

        # _backward — функция, которая «протолкнёт» градиент от этого узла
        #             к его родителям (children). Для листа делать нечего.
        self._backward = lambda: None
        # _prev — родители: узлы, из которых получился текущий.
        self._prev = set(_children)
        # _inputs — те же родители, но упорядоченные и с повторами (например,
        # для a*a это (a, a)). Множество _prev это теряет, а компилятору графа
        # в C (compile_graph.py) нужен точный порядок и кратность операндов.
        self._inputs = tuple(_children)
        # _op — какая операция породила узел (только для наглядности/отладки).
        self._op = _op

    # --- Базовые операции ---------------------------------------------------
    # В каждой операции мы:
    #   1) считаем data результата (это forward),
    #   2) создаём узел-результат out,
    #   3) определяем out._backward — как раздать градиент родителям
    #      по правилам дифференцирования.

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  ->  градиент проходит без изменений.
            # Используем += , т.к. узел может участвовать в нескольких местах
            # (правило суммы градиентов из multivariable chain rule).
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        # Поддерживаем только постоянную степень (int/float).
        assert isinstance(other, (int, float)), "степень должна быть числом"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d(a**n)/da = n * a**(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        # Нелинейность — без неё сеть из линейных слоёв остаётся линейной.
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)**2
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

        def _backward():
            # производная ReLU: 0 при x<0, иначе 1
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # --- Главный механизм: обратное распространение --------------------------

    def backward(self):
        """Посчитать grad всех узлов относительно self (обычно это loss)."""

        # 1) Топологическая сортировка графа: каждый узел должен идти ПОСЛЕ
        #    своих родителей, чтобы к моменту его обработки его собственный
        #    grad был уже полностью накоплен.
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 2) Производная результата по самому себе равна 1 — точка старта.
        self.grad = 1.0

        # 3) Идём от конца графа к началу и на каждом узле «раздаём»
        #    его градиент родителям через локальные производные.
        for v in reversed(topo):
            v._backward()

    # --- Удобства: чтобы a + 1, 2 * a, a - b и т.п. тоже работали ------------

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
