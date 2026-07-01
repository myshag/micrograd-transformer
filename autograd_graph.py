"""
Наглядная демонстрация графа вычислений и backprop на скалярном движке.

Печатает дерево выражения L = (a*b + c) * f, значения (forward) и
градиенты (backward), а также показывает накопление градиента (+=),
когда узел используется несколько раз.

Запуск:  python3 autograd_graph.py
"""

from autograd import Value


def print_tree(v, prefix="", is_last=True):
    """Печатает граф вычислений как дерево (корень — результат — сверху).

    ВНИМАНИЕ: v._prev это множество (set), поэтому порядок веток может
    отличаться от запуска к запуску — на смысл это не влияет.
    """
    op = v._op if v._op else "input"
    conn = "└── " if is_last else "├── "
    print(prefix + conn + f"{op:6s} data={v.data:+.1f} grad={v.grad:+.1f}")
    children = list(v._prev)
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(children):
        print_tree(child, child_prefix, i == len(children) - 1)


def demo_graph():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    f = Value(-2.0)

    e = a * b        # -6
    d = e + c        #  4
    L = d * f        # -8

    print("FORWARD (значения):")
    print(f"  e = a*b   = {e.data}")
    print(f"  d = e+c   = {d.data}")
    print(f"  L = d*f   = {L.data}\n")

    L.backward()

    print("BACKWARD (grad = dL/dузел):")
    for name, v in [("a", a), ("b", b), ("c", c), ("f", f),
                    ("e", e), ("d", d), ("L", L)]:
        print(f"  dL/d{name} = {v.grad:+.1f}")

    print("\nДЕРЕВО ГРАФА:")
    print_tree(L)


def demo_reuse():
    print("\nНакопление градиента при повторном использовании узла:")
    a = Value(3.0)
    y = a * a            # a входит в граф ДВАЖДЫ
    y.backward()
    print(f"  y = a*a при a=3 -> y={y.data}, dy/da={a.grad} (верно: 2a=6)")


if __name__ == "__main__":
    demo_graph()
    demo_reuse()
