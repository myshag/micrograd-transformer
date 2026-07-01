"""
Граф тензорных операций: значения и градиенты на каждом шаге.

Строит маленький «слой сети»  L = sum(relu(X @ W + b))  на читаемых матрицах,
печатает дерево графа, а затем — data (forward) и grad (backward) каждого узла.

Запуск:  python3 tensor_graph.py
"""

import numpy as np

from tensor import Tensor


def print_tree(v, prefix="", is_last=True):
    """Дерево графа: операция и форма тензора в каждом узле."""
    op = v._op if v._op else "input"
    conn = "└── " if is_last else "├── "
    print(prefix + conn + f"{op:8s} shape={v.data.shape}")
    children = list(v._prev)
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(children):
        print_tree(child, child_prefix, i == len(children) - 1)


def show(name, arr):
    text = np.array2string(arr)
    indented = "\n".join("      " + line for line in text.splitlines())
    print(f"  {name} =\n{indented}")


def main():
    np.set_printoptions(precision=0, suppress=True)

    # Входы (именуем через _op, чтобы их было видно в дереве и распечатке).
    X = Tensor([[1., 2., 3.], [4., 5., 6.]]); X._op = "X"     # (2,3)
    W = Tensor([[1., 0.], [0., 1.], [1., 1.]]); W._op = "W"   # (3,2)
    b = Tensor([-6., -2.]); b._op = "b"                       # (2,)

    # Forward: строим граф.
    Z = X @ W          # (2,2)   матричное умножение
    H = Z + b          # (2,2)   + смещение (broadcasting по строкам)
    A = H.relu()       # (2,2)   нелинейность
    L = A.sum()        # скаляр  итоговое значение

    print("ДЕРЕВО ГРАФА (корень L сверху):")
    print_tree(L)

    print("\n=== FORWARD (data на каждом шаге) ===")
    show("X (вход)", X.data)
    show("W (вход)", W.data)
    show("b (вход)", b.data)
    show("Z = X @ W", Z.data)
    show("H = Z + b", H.data)
    show("A = relu(H)", A.data)
    print(f"  L = sum(A) = {L.data}")

    # Backward: считаем градиенты.
    L.backward()

    print("\n=== BACKWARD (grad = dL/dузел) ===")
    show("dL/dA", A.grad)          # ones: L = сумма всех элементов A
    show("dL/dH", H.grad)          # маска ReLU: 1 где H>0, иначе 0
    show("dL/dZ", Z.grad)          # то же, что dL/dH (сложение проходит как есть)
    show("dL/db", b.grad)          # сумма dH по строкам (обратно к broadcasting)
    show("dL/dX", X.grad)          # dZ @ Wᵀ
    show("dL/dW", W.grad)          # Xᵀ @ dZ


if __name__ == "__main__":
    main()
