"""
Компиляция графа вычислений в C-код.

Берём граф скалярного autograd (Value), топологически его сортируем и
генерируем прямолинейный C-код: каждый узел -> одна переменная double, каждая
операция -> одно выражение. Отдельно генерируем backward (тоже как прямой C).
Затем компилируем gcc, запускаем и сверяем результат с Python-версией.

Это и есть суть «компиляции графа» (как в torch.compile / XLA), только в самом
наглядном виде: динамический граф превращается в статический код без Python и
без накладных расходов на объекты.

Замечание о масштабе: здесь компилируется СКАЛЯРНЫЙ граф. Для тензоров узкое
место — большие матричные умножения, которые numpy и так считает через BLAS
(тот же C); там выигрыш дал бы не наивный цикл, а слияние операций и та же BLAS.
Идея компиляции при этом ровно та же.

Запуск:  python3 compile_graph.py
"""

import os
import subprocess
import tempfile

from autograd import Value


def topo_sort(root):
    """Узлы в порядке «входы раньше узла» (по идентичности объектов)."""
    topo, seen = [], set()

    def visit(v):
        if v in seen:
            return
        seen.add(v)
        for child in v._inputs:
            visit(child)
        topo.append(v)

    visit(root)
    return topo


def _forward_expr(node, name):
    """C-выражение для значения узла через имена его входов."""
    op, ins = node._op, node._inputs
    if op == "+":
        return " + ".join(name[c] for c in ins)
    if op == "*":
        return " * ".join(name[c] for c in ins)
    if op.startswith("**"):
        return f"pow({name[ins[0]]}, {op[2:]})"
    if op == "tanh":
        return f"tanh({name[ins[0]]})"
    if op == "relu":
        return f"fmax(0.0, {name[ins[0]]})"
    raise ValueError(f"неизвестная операция: {op!r}")


def _backward_lines(node, val, grad):
    """C-строки, раздающие градиент узла его входам (те же правила, что в autograd)."""
    op, ins = node._op, node._inputs
    gn = grad[node]
    out = []
    if op == "+":                                  # d(a+b)/d* = 1
        for c in ins:
            out.append(f"  {grad[c]} += {gn};")
    elif op == "*":                                # d(∏)/dcᵢ = ∏ остальных
        for i, c in enumerate(ins):
            others = [val[o] for j, o in enumerate(ins) if j != i] or ["1.0"]
            out.append(f"  {grad[c]} += ({' * '.join(others)}) * {gn};")
    elif op.startswith("**"):                      # d(aⁿ)/da = n·aⁿ⁻¹
        n, a = op[2:], ins[0]
        out.append(f"  {grad[a]} += ({n} * pow({val[a]}, {n} - 1)) * {gn};")
    elif op == "tanh":                             # d/dx tanh = 1 - tanh²
        a = ins[0]
        out.append(f"  {grad[a]} += (1.0 - {val[node]}*{val[node]}) * {gn};")
    elif op == "relu":                             # 1 при x>0, иначе 0
        a = ins[0]
        out.append(f"  {grad[a]} += ({val[node]} > 0.0 ? 1.0 : 0.0) * {gn};")
    return out


def generate_c(root, named_inputs):
    """Сгенерировать C-исходник, считающий forward и градиенты по named_inputs.

    named_inputs: dict {имя: Value} — листья, по которым хотим градиент.
    """
    topo = topo_sort(root)
    val = {n: f"v{i}" for i, n in enumerate(topo)}
    grad = {n: f"g{i}" for i, n in enumerate(topo)}

    fwd = []
    for n in topo:
        if not n._inputs:                          # лист (вход или константа)
            fwd.append(f"  double {val[n]} = {float(n.data)!r};")
        else:
            fwd.append(f"  double {val[n]} = {_forward_expr(n, val)};")

    # объявляем градиенты нулями, у корня = 1, дальше — обратный проход
    gdecl = "  double " + ", ".join(f"{grad[n]} = 0.0" for n in topo) + ";"
    bwd = [f"  {grad[root]} = 1.0;"]
    for n in reversed(topo):
        bwd += _backward_lines(n, val, grad)

    prints = [f'  printf("forward %.12g\\n", {val[root]});']
    for label, node in named_inputs.items():
        prints.append(f'  printf("grad {label} %.12g\\n", {grad[node]});')

    return "\n".join([
        "#include <math.h>",
        "#include <stdio.h>",
        "int main(void) {",
        "  /* --- forward --- */",
        *fwd,
        "  /* --- backward --- */",
        gdecl,
        *bwd,
        "  /* --- вывод --- */",
        *prints,
        "  return 0;",
        "}",
        "",
    ])


def compile_and_run(root, named_inputs, keep_source=False):
    """Скомпилировать граф в C, запустить, вернуть {'forward':..., 'grad <имя>':...}."""
    src = generate_c(root, named_inputs)
    tmp = tempfile.mkdtemp()
    cpath, epath = os.path.join(tmp, "graph.c"), os.path.join(tmp, "graph")
    open(cpath, "w").write(src)
    subprocess.run(["gcc", "-O2", "-o", epath, cpath, "-lm"], check=True)
    out = subprocess.run([epath], capture_output=True, text=True, check=True).stdout

    result = {}
    for line in out.strip().splitlines():
        key, _, num = line.rpartition(" ")
        result[key] = float(num)
    if keep_source:
        result["_source"] = src
    return result


def demo():
    # Выражение с несколькими операциями: L = tanh(a*b + c) * (a ** 2)
    a, b, c = Value(0.7), Value(-1.3), Value(2.0)
    L = (a * b + c).tanh() * (a ** 2)

    # 1) Python autograd
    L.backward()
    print("=== Python autograd ===")
    print(f"forward  = {L.data:.12g}")
    print(f"grad a   = {a.grad:.12g}")
    print(f"grad b   = {b.grad:.12g}")
    print(f"grad c   = {c.grad:.12g}")

    # 2) Компиляция того же графа в C
    res = compile_and_run(L, {"a": a, "b": b, "c": c}, keep_source=True)
    print("\n=== Сгенерированный C-код ===")
    print(res.pop("_source"))
    print("=== Запуск скомпилированного C ===")
    for k, v in res.items():
        print(f"{k} = {v:.12g}")

    # 3) Сверка
    ok = (abs(res["forward"] - L.data) < 1e-9
          and abs(res["grad a"] - a.grad) < 1e-9
          and abs(res["grad b"] - b.grad) < 1e-9
          and abs(res["grad c"] - c.grad) < 1e-9)
    print("\nСовпадение Python и C:", "OK ✓" if ok else "РАСХОЖДЕНИЕ ✗")


if __name__ == "__main__":
    demo()
