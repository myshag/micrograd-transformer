"""
Компиляция тензорного графа (forward) в C с вызовом BLAS.

Это прототип того, что делают torch.compile / XLA: динамический граф
превращается в статический C-код, где большие матричные умножения идут через
BLAS (cblas_sgemm), а поэлементные операции (+ смещение, relu, tanh) — в циклы.
Затем компилируем gcc + OpenBLAS, запускаем и сверяем с numpy-версией.

Поддержаны операции forward-графа: вход, '@' (2D matmul), '+' (в т.ч.
broadcast смещения по последней оси), 'relu', 'tanh'. Этого хватает на MLP.
Backward и прочие операции опущены намеренно — цель показать идею BLAS+fusion.

Требуется: gcc и libopenblas-dev.
Запуск:  python3 compile_blas.py
"""

import os
import subprocess
import tempfile

import numpy as np

import tensor
from tensor import Tensor


def topo_sort(root):
    topo, seen = [], set()

    def visit(v):
        if v in seen:
            return
        seen.add(v)
        for c in v._inputs:
            visit(c)
        topo.append(v)

    visit(root)
    return topo


_EW = ("+", "relu", "tanh")     # поэлементные операции (кандидаты на слияние)


def _apply_ew(node, idx):
    """C-выражение, обновляющее регистр v результатом поэлементной op узла."""
    op, ins = node._op, node._inputs
    if op == "+":
        b = idx[ins[1]]
        if ins[1].data.ndim == 1:                 # broadcast смещения по колонкам
            return f"v+=t{b}[i%{ins[1].data.shape[0]}];"
        return f"v+=t{b}[i];"
    if op == "relu":
        return "v=v>0?v:0;"
    if op == "tanh":
        return "v=tanhf(v);"
    raise ValueError(op)


def generate_c(root, reps=None, profile=False, fuse=False):
    """Сгенерировать C-исходник forward-прохода графа.

    reps=None            — вычислить один раз и напечатать выход (для сверки).
    reps=N               — прогнать N раз с таймером, напечатать среднее время.
    reps=N, profile=True — таймер ВОКРУГ КАЖДОЙ операции (профилирование).
    fuse=True            — слить цепочки поэлементных операций в один цикл.
    """
    topo = topo_sort(root)
    idx = {n: i for i, n in enumerate(topo)}

    # Кто кого потребляет (для определения безопасного слияния).
    consumers = {n: [] for n in topo}
    for n in topo:
        for c in n._inputs:
            consumers[c].append(n)

    def is_ew(n):
        return n._op in _EW

    # «Внутренние» поэлементные узлы: единственный потребитель — тоже
    # поэлементная op, берущая их как ГЛАВНЫЙ вход. Их буфер не нужен —
    # значение проживёт в регистре внутри цикла потребителя.
    internal = set()
    if fuse:
        for n in topo:
            if is_ew(n) and len(consumers[n]) == 1:
                c = consumers[n][0]
                if is_ew(c) and c._inputs[0] is n:
                    internal.add(n)

    decls, body, labels = [], [], []
    for n in topo:
        size = max(1, int(np.prod(n.data.shape)))
        if not n._inputs:                                   # входной лист
            vals = ", ".join(f"{float(x):.9g}f" for x in n.data.ravel(order="C"))
            decls.append(f"static float t{idx[n]}[{size}] = {{ {vals} }};")
            continue
        if n in internal:                                   # слит в потребителя
            continue
        decls.append(f"static float t{idx[n]}[{size}];")

        op, ins = n._op, n._inputs
        a = idx[ins[0]]
        if op == "@":                                       # matmul -> BLAS
            M, K = ins[0].data.shape
            _, N = ins[1].data.shape
            b = idx[ins[1]]
            body.append(
                f"cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,"
                f"{M},{N},{K},1.0f,t{a},{K},t{b},{N},0.0f,t{idx[n]},{N});")
            labels.append(f"matmul {M}x{K}@{K}x{N}")
        elif op in _EW:                                     # поэлементная (терминал цепочки)
            # Пройти назад по внутренним узлам до реального источника-буфера.
            chain = [n]
            src = ins[0]
            while src in internal:
                chain.append(src)
                src = src._inputs[0]
            chain.reverse()                                 # от источника к терминалу
            steps = "".join(_apply_ew(m, idx) for m in chain)
            body.append(f"for(int i=0;i<{n.data.size};i++)"
                        f"{{float v=t{idx[src]}[i];{steps}t{idx[n]}[i]=v;}}")
            names = "→".join(m._op for m in chain)
            prefix = "fused " if len(chain) > 1 else ""
            labels.append(f"{prefix}{names} [{n.data.size}]")
        else:
            raise ValueError(f"компилятор не умеет операцию {op!r}")

    headers = ["#include <stdio.h>", "#include <math.h>", "#include <cblas.h>"]
    timer = ("static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);"
             "return t.tv_sec+t.tv_nsec*1e-9;}")

    if reps is None:
        main = ["int main(void){",
                *[f"  {line}" for line in body],
                f"  for(int i=0;i<{root.data.size};i++) printf(\"%.7g \", t{idx[root]}[i]);",
                "  printf(\"\\n\");  return 0;", "}"]
    elif not profile:
        headers.append("#include <time.h>")
        main = [timer, "int main(void){",
                "  double _start=now();",
                f"  for(int r=0;r<{reps};r++){{",
                *[f"    {line}" for line in body],
                "  }",
                f"  double ms=(now()-_start)/{reps}*1e3;",
                f"  printf(\"%.4f %.7g\\n\", ms, t{idx[root]}[0]);",
                "  return 0;", "}"]
    else:
        # режим профилирования: таймер вокруг каждой операции
        headers.append("#include <time.h>")
        n_ops = len(body)
        loop = [f"  for(int r=0;r<{reps};r++){{"]
        for i, stmt in enumerate(body):
            loop.append(f"    _s=now(); {stmt} prof[{i}]+=now()-_s;")
        loop.append("  }")
        prints = ['  double total=0; for(int i=0;i<%d;i++) total+=prof[i];' % n_ops]
        prints.append('  printf("  %-24s %10s  %6s\\n", "операция", "мс/проход", "доля");')
        for i, lab in enumerate(labels):
            prints.append(
                f'  printf("  %-24s %10.5f  %5.1f%%\\n", "{lab}", '
                f'prof[{i}]/{reps}*1e3, 100.0*prof[{i}]/total);')
        prints.append(f'  printf("  %-24s %10.5f\\n", "ВСЕГО", total/{reps}*1e3);')
        prints.append(f'  printf("anchor %g\\n", t{idx[root]}[0]);')
        main = [timer, "int main(void){",
                f"  double prof[{n_ops}]={{0}}, _s;",
                *loop, *prints, "  return 0;", "}"]

    return "\n".join(headers + decls + main + [""])


def _build(src):
    tmp = tempfile.mkdtemp()
    cpath, epath = os.path.join(tmp, "g.c"), os.path.join(tmp, "g")
    open(cpath, "w").write(src)
    subprocess.run(["gcc", "-O2", "-o", epath, cpath, "-lopenblas", "-lm"], check=True)
    return epath


def compile_and_run(root, fuse=False):
    """Скомпилировать граф в C (с BLAS), запустить, вернуть выход как numpy-массив."""
    src = generate_c(root, fuse=fuse)
    out = subprocess.run([_build(src)], capture_output=True, text=True, check=True).stdout
    arr = np.array([float(x) for x in out.split()], dtype=np.float32)
    return arr.reshape(root.data.shape), src


def compile_and_time(root, reps, fuse=False):
    """Скомпилировать и замерить среднее время forward в C (мс/проход)."""
    out = subprocess.run([_build(generate_c(root, reps, fuse=fuse))],
                         capture_output=True, text=True, check=True).stdout
    return float(out.split()[0])


def compile_and_profile(root, reps, fuse=False):
    """Скомпилировать с таймером вокруг каждой операции; вернуть текст-разбивку."""
    out = subprocess.run([_build(generate_c(root, reps, profile=True, fuse=fuse))],
                         capture_output=True, text=True, check=True).stdout
    return out


def demo():
    rng = np.random.default_rng(0)
    # Двухслойный MLP forward: Y = relu(X @ W1 + b1) @ W2 + b2
    X = Tensor(rng.standard_normal((4, 8)))
    W1 = Tensor(rng.standard_normal((8, 16)))
    b1 = Tensor(rng.standard_normal(16))
    W2 = Tensor(rng.standard_normal((16, 5)))
    b2 = Tensor(rng.standard_normal(5))

    Y = (X @ W1 + b1).relu() @ W2 + b2      # строим граф в нашем движке

    c_out, src = compile_and_run(Y)         # компилируем этот же граф в C+BLAS

    print("=== Фрагмент сгенерированного C ===")
    for line in src.splitlines():
        if "cblas_sgemm" in line or "relu" in line[:12] or "i%16" in line:
            print(line.strip())
    print("\n=== Сверка ===")
    print("numpy (наш Tensor) [0] =", Y.data[0])
    print("C + BLAS           [0] =", c_out[0])
    diff = np.abs(Y.data - c_out).max()
    print(f"\nmax|numpy - C| = {diff:.2e}  ->",
          "OK ✓" if diff < 1e-3 else "РАСХОЖДЕНИЕ ✗")


def bench():
    """Сравнить forward нашего Python-движка и скомпилированного C+BLAS."""
    import time
    from tensor import no_grad
    rng = np.random.default_rng(0)
    X = Tensor(rng.standard_normal((32, 128)))
    W1 = Tensor(rng.standard_normal((128, 256)))
    b1 = Tensor(rng.standard_normal(256))
    W2 = Tensor(rng.standard_normal((256, 64)))
    b2 = Tensor(rng.standard_normal(64))

    def forward():
        return (X @ W1 + b1).relu() @ W2 + b2

    reps = 500
    with no_grad():                      # честный инференс-forward
        forward()
        t = time.perf_counter()
        for _ in range(reps):
            forward()
        py_ms = (time.perf_counter() - t) / reps * 1e3

    c_ms = compile_and_time(forward(), reps)
    print("\n=== Замер forward (MLP 32x128 -> 256 -> 64) ===")
    print(f"наш Python-движок (numpy):  {py_ms:.3f} мс/проход")
    print(f"скомпилированный C + BLAS:   {c_ms:.3f} мс/проход")
    print(f"ускорение: {py_ms / c_ms:.1f}x  (matmul тот же BLAS; "
          f"выигрыш — от снятия Python-обвязки и fusion)")


def profile():
    """Профилирование скомпилированного C: сколько времени на какой операции."""
    rng = np.random.default_rng(0)
    # Трёхслойный MLP — чтобы в разбивке было несколько gemm и поэлементных op.
    x = Tensor(rng.standard_normal((64, 256)))
    layers = [(256, 512), (512, 512), (512, 128)]
    ins = []
    for i, (nin, nout) in enumerate(layers):
        W = Tensor(rng.standard_normal((nin, nout)))
        b = Tensor(rng.standard_normal(nout))
        ins += [W, b]
    W1, b1, W2, b2, W3, b3 = ins
    Y = ((x @ W1 + b1).relu() @ W2 + b2).relu() @ W3 + b3

    print("\n=== Профиль forward (MLP 256->512->512->128) ===")
    print("--- БЕЗ fusion ---")
    print(compile_and_profile(Y, reps=1000).strip())
    print("\n--- С fusion (+bias и relu слиты в один цикл) ---")
    print(compile_and_profile(Y, reps=1000, fuse=True).strip())


def fusion_demo():
    """Показать, что fusion не меняет результат и убирает промежуточные буферы."""
    rng = np.random.default_rng(1)
    X = Tensor(rng.standard_normal((8, 32)))
    W1 = Tensor(rng.standard_normal((32, 64))); b1 = Tensor(rng.standard_normal(64))
    W2 = Tensor(rng.standard_normal((64, 16))); b2 = Tensor(rng.standard_normal(16))
    Y = (X @ W1 + b1).relu() @ W2 + b2

    plain, _ = compile_and_run(Y, fuse=False)
    fused, src = compile_and_run(Y, fuse=True)
    print("\n=== Fusion: корректность и код ===")
    print("Слитый цикл (+bias -> relu за один проход):")
    for line in src.splitlines():
        if "float v=" in line:
            print("  " + line.strip())
    d1 = np.abs(Y.data - plain).max()
    d2 = np.abs(Y.data - fused).max()
    print(f"max|numpy - C(без fusion)| = {d1:.2e}")
    print(f"max|numpy - C(с fusion)|   = {d2:.2e}  ->",
          "OK ✓" if d2 < 1e-3 else "РАСХОЖДЕНИЕ ✗")


if __name__ == "__main__":
    demo()
    bench()
    fusion_demo()
    profile()
