"""
Демонстрация autograd-движка.

Запуск:  python3 demo.py
"""

from autograd import Value


def demo_basic():
    """Проверим градиенты на простом выражении и сверим с математикой."""
    print("=== Пример 1: f = a*b + c ===")
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    f = a * b + c          # forward: 2*(-3) + 10 = 4
    f.backward()           # backward: посчитать df/da, df/db, df/dc

    print(f"f = {f.data}")        # 4.0
    print(f"df/da = {a.grad}")    # = b = -3
    print(f"df/db = {b.grad}")    # = a =  2
    print(f"df/dc = {c.grad}")    # = 1
    print()


def demo_neuron():
    """Один нейрон: out = tanh(w1*x1 + w2*x2 + bias)."""
    print("=== Пример 2: один нейрон с tanh ===")
    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    bias = Value(6.8813735870195432)

    out = (x1 * w1 + x2 * w2 + bias).tanh()
    out.backward()

    print(f"out  = {out.data:.4f}")
    print(f"grad по w1 = {w1.grad:.4f}")
    print(f"grad по x1 = {x1.grad:.4f}")
    print()


def demo_gradient_descent():
    """Мини-обучение: подгоняем x так, чтобы (x - 3)**2 -> минимум (x≈3)."""
    print("=== Пример 3: градиентный спуск, минимизируем (x-3)^2 ===")
    x = Value(0.0)
    lr = 0.1  # шаг обучения

    for step in range(25):
        loss = (x - 3) ** 2     # forward
        x.grad = 0.0            # обнуляем градиент перед новым backward
        loss.backward()         # считаем dloss/dx
        x.data -= lr * x.grad   # шаг против градиента
        if step % 5 == 0:
            print(f"шаг {step:2d}: x = {x.data:.4f}, loss = {loss.data:.4f}")

    print(f"итог: x = {x.data:.4f} (ожидаем ~3.0)")
    print()


if __name__ == "__main__":
    demo_basic()
    demo_neuron()
    demo_gradient_descent()
