"""
Как мы «захватываем граф» — на примерах.

Главная мысль: у нас нет отдельной фазы «захвата графа». Граф строится сам,
когда выполняется обычный Python-код на наших Tensor-ах, потому что каждая
операция запоминает свои входы. Поэтому trace() — это буквально «вызвать
функцию». Ниже это видно, включая случай ветвления по данным, ради которого
torch.compile городит Dynamo/guards, — и почему нам он не нужен.

Запуск:  python3 trace_demo.py
"""

import numpy as np

from tensor import Tensor, no_grad
from compile_blas import topo_sort, compile_and_run


def trace(f, *inputs):
    """«Трассировка» = просто вызвать f на Tensor-ах. Граф уже построен в
    ссылках _inputs; возвращаем корень и его топологический порядок."""
    out = f(*inputs)
    return out, topo_sort(out)


def show_graph(topo):
    idx = {id(n): i for i, n in enumerate(topo)}
    for i, n in enumerate(topo):
        op = n._op or "input"
        args = [f"v{idx[id(c)]}" for c in n._inputs]
        print(f"  v{i:<2} = {op:6} {', '.join(args):10} shape={n.data.shape}")


rng = np.random.default_rng(0)


def banner(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


# ---------------------------------------------------------------------------
banner("1. Обычная на вид функция -> граф строится сам")

def mlp(x, W1, b1, W2, b2):
    h = (x @ W1 + b1).relu()
    return h @ W2 + b2

x = Tensor(rng.standard_normal((2, 4)))
W1 = Tensor(rng.standard_normal((4, 8))); b1 = Tensor(rng.standard_normal(8))
W2 = Tensor(rng.standard_normal((8, 3))); b2 = Tensor(rng.standard_normal(3))

print("Пишем обычный Python:  h = (x @ W1 + b1).relu();  return h @ W2 + b2")
out, topo = trace(mlp, x, W1, b1, W2, b2)
print("\ntrace(mlp, ...) вернул готовый граф (никакого перехвата байткода):")
show_graph(topo)

print("\n...и этот же граф сразу компилируется в C+BLAS:")
c_out, _ = compile_and_run(out)
print(f"  max|python - C| = {np.abs(out.data - c_out).max():.2e}  (совпало)")


# ---------------------------------------------------------------------------
banner("2. Python-цикл разворачивается в граф (как в JAX)")

def deep(x, W):
    for _ in range(3):            # обычный питоновский for
        x = (x @ W).relu()
    return x

xv = Tensor(rng.standard_normal((2, 4)))
Wv = Tensor(rng.standard_normal((4, 4)))
_, topo2 = trace(deep, xv, Wv)
print("for _ in range(3): x = (x @ W).relu()")
print("Цикл исчез — в графе три развёрнутых пары @/relu:")
show_graph(topo2)


# ---------------------------------------------------------------------------
banner("3. Ветвление ПО ДАННЫМ — тут torch.compile нужен guard, а нам нет")

def cond(x, W):
    if float(x.data.sum()) > 0:          # ветка зависит от значений!
        return (x @ W).relu()
    else:
        return x @ W

W = Tensor(rng.standard_normal((4, 4)))
x_pos = Tensor(np.abs(rng.standard_normal((2, 4))))      # сумма > 0
x_neg = Tensor(-np.abs(rng.standard_normal((2, 4))))     # сумма < 0

_, t_pos = trace(cond, x_pos, W)
_, t_neg = trace(cond, x_neg, W)
ops_pos = [n._op for n in t_pos if n._op]
ops_neg = [n._op for n in t_neg if n._op]
print("сумма>0 -> в графе операции:", ops_pos)
print("сумма<0 -> в графе операции:", ops_neg, " (relu исчез — взята другая ветка)")
print("""
Мы просто ЗАПЕКАЕМ ту ветку, что реально выполнилась (специализация под вход).
torch.compile же обязан работать на любом входе, поэтому вставляет guard
'x.sum() > 0' и ПЕРЕКОМПИЛИРУЕТ, если условие поменяется. Вся сложность Dynamo —
ровно из-за таких мест. Наш контракт проще: 'дай граф для этого прогона'.""")


if __name__ == "__main__":
    pass
