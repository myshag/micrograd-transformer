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


def _cf(x):
    """Число -> корректный C-литерал float (всегда с точкой/экспонентой)."""
    s = f"{float(x):.9g}"
    if not any(c in s for c in ".eEnf"):     # '0' -> '0.0', '2' -> '2.0'
        s += ".0"
    return s + "f"


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
            vals = ", ".join(_cf(x) for x in n.data.ravel(order="C"))
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


def _fwd_stmt(n, idx):
    """C-инструкция forward для одного узла (без слияния — буферы сохраняются)."""
    op, ins = n._op, n._inputs
    a = idx[ins[0]]
    if op == "@":
        M, K = ins[0].data.shape
        _, N = ins[1].data.shape
        return (f"cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,"
                f"{M},{N},{K},1.0f,t{a},{K},t{idx[ins[1]]},{N},0.0f,t{idx[n]},{N});")
    if op == "+":
        b, sz = idx[ins[1]], n.data.size
        if ins[1].data.ndim == 1:
            return f"for(int i=0;i<{sz};i++) t{idx[n]}[i]=t{a}[i]+t{b}[i%{ins[1].data.shape[0]}];"
        return f"for(int i=0;i<{sz};i++) t{idx[n]}[i]=t{a}[i]+t{b}[i];"
    if op == "relu":
        return f"for(int i=0;i<{n.data.size};i++){{float x=t{a}[i];t{idx[n]}[i]=x>0?x:0;}}"
    if op == "tanh":
        return f"for(int i=0;i<{n.data.size};i++) t{idx[n]}[i]=tanhf(t{a}[i]);"
    raise ValueError(op)


def _bwd_stmts(n, idx):
    """C-инструкции backward: раздать градиент g{n} входам узла (с накоплением)."""
    op, ins = n._op, n._inputs
    gn = f"g{idx[n]}"
    if op == "@":
        X, W = ins
        M, K = X.data.shape          # X: (M,K),  W: (K,N),  out: (M,N)
        _, N = W.data.shape
        # dX += dZ @ Wᵀ ;  dW += Xᵀ @ dZ  — оба через sgemm с транспонированием
        return [
            f"cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,{M},{K},{N},"
            f"1.0f,{gn},{N},t{idx[W]},{N},1.0f,g{idx[X]},{K});",
            f"cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,{K},{N},{M},"
            f"1.0f,t{idx[X]},{K},{gn},{N},1.0f,g{idx[W]},{N});"]
    if op == "+":
        A, B = ins
        sz = n.data.size
        out = [f"for(int i=0;i<{sz};i++) g{idx[A]}[i]+={gn}[i];"]
        if B.data.ndim == 1:         # смещение: db[j] = сумма по строкам
            out.append(f"for(int i=0;i<{sz};i++) g{idx[B]}[i%{B.data.shape[0]}]+={gn}[i];")
        else:
            out.append(f"for(int i=0;i<{sz};i++) g{idx[B]}[i]+={gn}[i];")
        return out
    if op == "relu":                 # маска по forward-выходу (relu>0 <=> вход>0)
        return [f"for(int i=0;i<{n.data.size};i++) "
                f"g{idx[ins[0]]}[i]+=(t{idx[n]}[i]>0?1.0f:0.0f)*{gn}[i];"]
    if op == "tanh":                 # d/dx = 1 - tanh(x)^2
        return [f"for(int i=0;i<{n.data.size};i++) "
                f"g{idx[ins[0]]}[i]+=(1.0f-t{idx[n]}[i]*t{idx[n]}[i])*{gn}[i];"]
    raise ValueError(op)


def generate_backward_c(root, named):
    """C-исходник: forward + backward графа, печать градиентов листьев `named`.

    named: dict {имя: Tensor-лист}. Выходной градиент засевается единицами
    (как self.grad = ones в нашем backward).
    """
    topo = topo_sort(root)
    idx = {n: i for i, n in enumerate(topo)}

    decls, fwd, bwd = [], [], []
    for n in topo:
        size = max(1, int(np.prod(n.data.shape)))
        if not n._inputs:
            vals = ", ".join(_cf(x) for x in n.data.ravel(order="C"))
            decls.append(f"static float t{idx[n]}[{size}] = {{ {vals} }};")
        else:
            decls.append(f"static float t{idx[n]}[{size}];")
            fwd.append(_fwd_stmt(n, idx))
        decls.append(f"static float g{idx[n]}[{size}];")   # градиент (0 по умолч.)

    # backward: засеять выход и пройти в обратном топологическом порядке
    bwd.append(f"for(int i=0;i<{root.data.size};i++) g{idx[root]}[i]=1.0f;")
    for n in reversed(topo):
        if n._inputs:
            bwd += _bwd_stmts(n, idx)

    prints = []
    for name, node in named.items():
        prints.append(f'  printf("{name}");')
        prints.append(f'  for(int i=0;i<{node.data.size};i++) printf(" %.7g", g{idx[node]}[i]);')
        prints.append('  printf("\\n");')

    return "\n".join(
        ["#include <stdio.h>", "#include <math.h>", "#include <cblas.h>"]
        + decls + ["int main(void){"]
        + [f"  {s}" for s in fwd] + [f"  {s}" for s in bwd] + prints
        + ["  return 0;", "}", ""])


def compile_backward(root, named):
    """Скомпилировать backward в C+BLAS, вернуть {имя: numpy-градиент}."""
    src = generate_backward_c(root, named)
    out = subprocess.run([_build(src)], capture_output=True, text=True, check=True).stdout
    grads = {}
    for line in out.strip().splitlines():
        parts = line.split()
        grads[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return grads, src


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


def backward_demo():
    """Скомпилировать backward в C+BLAS и сверить градиенты с Python autograd."""
    rng = np.random.default_rng(2)
    X = Tensor(rng.standard_normal((6, 12)))
    W1 = Tensor(rng.standard_normal((12, 20))); b1 = Tensor(rng.standard_normal(20))
    W2 = Tensor(rng.standard_normal((20, 5)));  b2 = Tensor(rng.standard_normal(5))
    Y = (X @ W1 + b1).relu() @ W2 + b2
    named = {"X": X, "W1": W1, "b1": b1, "W2": W2, "b2": b2}

    # 1) сначала компилируем backward в C (граф ещё цел)
    c_grads, src = compile_backward(Y, named)

    # 2) затем Python autograd (backward освобождает граф — поэтому после C)
    Y.backward()
    py_grads = {k: v.grad.ravel() for k, v in named.items()}

    print("=== Backward: сверка C+BLAS с Python autograd ===")
    print("Фрагмент сгенерированного backward (градиенты matmul — тоже sgemm):")
    for line in src.splitlines():
        if "CblasTrans" in line:
            print("  " + line.strip())
    print()
    worst = 0.0
    for k in named:
        d = np.abs(py_grads[k] - c_grads[k]).max()
        worst = max(worst, d)
        print(f"  grad {k:3s}: max|py - C| = {d:.2e}")
    print("\nИтог:", "OK ✓" if worst < 1e-3 else "РАСХОЖДЕНИЕ ✗",
          f"(худшее расхождение {worst:.2e}, точность float32)")


if __name__ == "__main__":
    demo()
    bench()
    fusion_demo()
    profile()
    print()
    backward_demo()
