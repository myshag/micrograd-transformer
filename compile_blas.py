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


def generate_c(root, reps=None):
    """Сгенерировать C-исходник forward-прохода графа.

    reps=None — вычислить один раз и напечатать выход (для сверки).
    reps=N    — прогнать N раз с таймером и напечатать среднее время (замер).
    """
    topo = topo_sort(root)
    idx = {n: i for i, n in enumerate(topo)}

    decls, body = [], []
    for n in topo:
        size = max(1, int(np.prod(n.data.shape)))
        if not n._inputs:                                   # входной лист
            vals = ", ".join(f"{float(x):.9g}f" for x in n.data.ravel(order="C"))
            decls.append(f"static float t{idx[n]}[{size}] = {{ {vals} }};")
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
        elif op == "+":                                     # сложение (+ bias)
            b = idx[ins[1]]
            sz = n.data.size
            if ins[1].data.ndim == 1:                       # broadcast смещения
                cols = ins[1].data.shape[0]
                body.append(f"for(int i=0;i<{sz};i++) "
                            f"t{idx[n]}[i]=t{a}[i]+t{b}[i%{cols}];")
            else:
                body.append(f"for(int i=0;i<{sz};i++) t{idx[n]}[i]=t{a}[i]+t{b}[i];")
        elif op == "relu":
            body.append(f"for(int i=0;i<{n.data.size};i++)"
                        f"{{float x=t{a}[i];t{idx[n]}[i]=x>0?x:0;}}")
        elif op == "tanh":
            body.append(f"for(int i=0;i<{n.data.size};i++) "
                        f"t{idx[n]}[i]=tanhf(t{a}[i]);")
        else:
            raise ValueError(f"компилятор не умеет операцию {op!r}")

    headers = ["#include <stdio.h>", "#include <math.h>", "#include <cblas.h>"]

    if reps is None:
        # режим сверки: посчитать один раз, напечатать выход
        main = ["int main(void){",
                *[f"  {line}" for line in body],
                f"  for(int i=0;i<{root.data.size};i++) printf(\"%.7g \", t{idx[root]}[i]);",
                "  printf(\"\\n\");",
                "  return 0;", "}"]
    else:
        # режим замера: прогнать reps раз с таймером
        headers.append("#include <time.h>")
        main = ["static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);"
                "return t.tv_sec+t.tv_nsec*1e-9;}",
                "int main(void){",
                "  double _start=now();",
                f"  for(int r=0;r<{reps};r++){{",
                *[f"    {line}" for line in body],
                "  }",
                f"  double ms=(now()-_start)/{reps}*1e3;",
                f"  printf(\"%.4f %.7g\\n\", ms, t{idx[root]}[0]);",  # время + якорь, чтобы цикл не выкинули
                "  return 0;", "}"]

    return "\n".join(headers + decls + main + [""])


def _build(src):
    tmp = tempfile.mkdtemp()
    cpath, epath = os.path.join(tmp, "g.c"), os.path.join(tmp, "g")
    open(cpath, "w").write(src)
    subprocess.run(["gcc", "-O2", "-o", epath, cpath, "-lopenblas", "-lm"], check=True)
    return epath


def compile_and_run(root):
    """Скомпилировать граф в C (с BLAS), запустить, вернуть выход как numpy-массив."""
    src = generate_c(root)
    out = subprocess.run([_build(src)], capture_output=True, text=True, check=True).stdout
    arr = np.array([float(x) for x in out.split()], dtype=np.float32)
    return arr.reshape(root.data.shape), src


def compile_and_time(root, reps):
    """Скомпилировать и замерить среднее время forward в C (мс/проход)."""
    out = subprocess.run([_build(generate_c(root, reps))],
                         capture_output=True, text=True, check=True).stdout
    return float(out.split()[0])


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


if __name__ == "__main__":
    demo()
    bench()
