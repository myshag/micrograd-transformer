"""
Полный C-обучатель одного слоя: forward + loss + backward + SGD в C+BLAS.

Компилируем ВЕСЬ цикл обучения линейного слоя (регрессия с MSE) в один
самодостаточный C-бинарник. Внутри цикла на каждом шаге:
  1. forward:   P = X @ W + b            (cblas_sgemm + смещение)
  2. loss-grad: dP = 2/(M·N) · (P − Y)   (градиент MSE)
  3. backward:  gW = Xᵀ @ dP,  gb = Σ dP (cblas_sgemm с CblasTrans + сумма)
  4. SGD-шаг:   W −= lr·gW,  b −= lr·gb
Градиент по X не нужен — X это данные, а не параметр.

Правильность проверяем эталоном на numpy (та же математика, float32):
кривые loss и итоговые W/b должны совпасть.

Требуется: gcc, libopenblas-dev.
Запуск:  python3 train_c.py
"""

import os
import subprocess
import tempfile

import numpy as np


def _cf(x):
    """Число -> корректный C-литерал float (всегда с точкой/экспонентой)."""
    s = f"{float(x):.9g}"
    if not any(c in s for c in ".eEnf"):     # '0' -> '0.0', '1' -> '1.0'
        s += ".0"
    return s + "f"


def _lit(a):
    return ", ".join(_cf(x) for x in np.asarray(a, dtype=np.float32).ravel())


def generate_trainer_c(X, Y, W0, b0, lr, steps, log_every):
    M, K = X.shape
    _, N = W0.shape
    MN, KN = M * N, K * N
    return f"""#include <stdio.h>
#include <cblas.h>
static float X[{M * K}]  = {{ {_lit(X)} }};
static float Yt[{MN}] = {{ {_lit(Y)} }};
static float W[{KN}]  = {{ {_lit(W0)} }};
static float b[{N}]   = {{ {_lit(b0)} }};
static float P[{MN}], dP[{MN}], gW[{KN}], gb[{N}];

int main(void) {{
  float lr = {lr}f;
  for (int s = 0; s < {steps}; s++) {{
    /* 1. forward: P = X @ W + b */
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,{M},{N},{K},
                1.0f, X,{K}, W,{N}, 0.0f, P,{N});
    for (int i = 0; i < {MN}; i++) P[i] += b[i % {N}];

    /* 2. градиент MSE по выходу: dP = 2/(M*N) * (P - Y) */
    for (int i = 0; i < {MN}; i++) dP[i] = (2.0f/{MN}) * (P[i] - Yt[i]);

    /* 3. backward: gW = X^T @ dP,  gb = сумма dP по строкам */
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,{K},{N},{M},
                1.0f, X,{K}, dP,{N}, 0.0f, gW,{N});
    for (int j = 0; j < {N}; j++) gb[j] = 0.0f;
    for (int i = 0; i < {MN}; i++) gb[i % {N}] += dP[i];

    /* 4. SGD-шаг */
    for (int i = 0; i < {KN}; i++) W[i] -= lr * gW[i];
    for (int j = 0; j < {N}; j++)  b[j] -= lr * gb[j];

    if (s % {log_every} == 0 || s == {steps} - 1) {{
      float L = 0; for (int i = 0; i < {MN}; i++) {{ float e = P[i]-Yt[i]; L += e*e; }}
      printf("step %5d  loss %.6f\\n", s, L/{MN});
    }}
  }}
  printf("FINAL_W"); for (int i = 0; i < {KN}; i++) printf(" %.6g", W[i]); printf("\\n");
  printf("FINAL_B"); for (int j = 0; j < {N};  j++) printf(" %.6g", b[j]); printf("\\n");
  return 0;
}}
"""


def train_in_c(X, Y, W0, b0, lr, steps, log_every):
    src = generate_trainer_c(X, Y, W0, b0, lr, steps, log_every)
    tmp = tempfile.mkdtemp()
    cpath, epath = os.path.join(tmp, "train.c"), os.path.join(tmp, "train")
    open(cpath, "w").write(src)
    subprocess.run(["gcc", "-O2", "-o", epath, cpath, "-lopenblas", "-lm"], check=True)
    out = subprocess.run([epath], capture_output=True, text=True, check=True).stdout
    W, b = None, None
    for line in out.splitlines():
        if line.startswith("FINAL_W"):
            W = np.array(line.split()[1:], dtype=np.float32).reshape(W0.shape)
        elif line.startswith("FINAL_B"):
            b = np.array(line.split()[1:], dtype=np.float32)
    return out, W, b, src


def train_numpy(X, Y, W0, b0, lr, steps):
    """Тот же алгоритм на numpy (эталон для сверки)."""
    W, b = W0.astype(np.float32).copy(), b0.astype(np.float32).copy()
    M, N = Y.shape
    for _ in range(steps):
        P = X @ W + b
        dP = (2.0 / (M * N)) * (P - Y)
        W -= lr * (X.T @ dP)
        b -= lr * dP.sum(axis=0)
    return W.astype(np.float32), b.astype(np.float32)


def main():
    rng = np.random.default_rng(0)
    M, K, N = 200, 4, 3
    # Синтетическая линейная зависимость + небольшой шум.
    X = rng.standard_normal((M, K)).astype(np.float32)
    W_true = rng.standard_normal((K, N)).astype(np.float32)
    b_true = rng.standard_normal(N).astype(np.float32)
    Y = (X @ W_true + b_true + 0.05 * rng.standard_normal((M, N))).astype(np.float32)

    W0 = np.zeros((K, N), np.float32)
    b0 = np.zeros(N, np.float32)
    lr, steps, log = 0.1, 4000, 500

    out, cW, cb, src = train_in_c(X, Y, W0, b0, lr, steps, log)

    print("=== Обучение линейного слоя ЦЕЛИКОМ в C (forward+backward+SGD) ===")
    print(out.strip().split("FINAL_W")[0].strip())

    nW, nb = train_numpy(X, Y, W0, b0, lr, steps)
    print("\n=== Сверка с numpy-эталоном (та же математика) ===")
    print(f"max|W_C - W_numpy| = {np.abs(cW - nW).max():.2e}")
    print(f"max|b_C - b_numpy| = {np.abs(cb - nb).max():.2e}")
    print(f"max|W_C - W_true|  = {np.abs(cW - W_true).max():.2e} "
          f"(восстановили истинные веса)")
    ok = np.abs(cW - nW).max() < 1e-3 and np.abs(cb - nb).max() < 1e-3
    print("Итог:", "OK ✓ C-обучатель считает то же, что numpy" if ok else "РАСХОЖДЕНИЕ ✗")


if __name__ == "__main__":
    main()
