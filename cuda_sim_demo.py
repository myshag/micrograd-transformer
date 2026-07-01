"""
GPU-код без GPU: CUDA-кернелы в симуляторе Numba (NUMBA_ENABLE_CUDASIM=1).

Пишем ровно те же операции, что компилировали в C (matmul и слитый +bias→relu),
но в CUDA-стиле: один поток GPU на один элемент выхода. Numba исполняет эти
кернелы на CPU в чистом Python (симулятор), поэтому видеокарта не нужна — можно
отлаживать логику. Скорости GPU тут нет (каждый поток гоняется интерпретатором),
только корректность. Сверяем с numpy.

Запуск:  python3 cuda_sim_demo.py
"""

import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"      # ВКЛючить симулятор ДО импорта cuda

import numpy as np
from numba import cuda


@cuda.jit
def matmul_kernel(C, A, B):
    """C = A @ B. Один поток вычисляет один элемент C[i,j]."""
    i, j = cuda.grid(2)                        # глобальные индексы потока (2D-сетка)
    if i < C.shape[0] and j < C.shape[1]:
        acc = 0.0
        for k in range(A.shape[1]):
            acc += A[i, k] * B[k, j]
        C[i, j] = acc


@cuda.jit
def bias_relu_kernel(out, x, bias):
    """Слитый +bias→relu (наш fusion), но на потоках GPU: поток на элемент."""
    i, j = cuda.grid(2)
    if i < x.shape[0] and j < x.shape[1]:
        v = x[i, j] + bias[j]                  # + смещение
        out[i, j] = v if v > 0.0 else 0.0      # relu — всё в одном ядре


def launch_2d(kernel, shape, *args):
    """Запуск кернела на 2D-сетке блоков (как на настоящем CUDA)."""
    tpb = (8, 8)                               # threads per block
    bpg = (int(np.ceil(shape[0] / tpb[0])),    # blocks per grid
           int(np.ceil(shape[1] / tpb[1])))
    kernel[bpg, tpb](*args)
    return bpg, tpb


def main():
    print("Режим CUDA:", "СИМУЛЯТОР (CPU)" if cuda.is_available() and
          os.environ.get("NUMBA_ENABLE_CUDASIM") == "1" else "?")
    rng = np.random.default_rng(0)

    # Линейный слой forward: A = relu(X @ W + b) — но на «GPU»
    X = rng.standard_normal((4, 8)).astype(np.float32)
    W = rng.standard_normal((8, 6)).astype(np.float32)
    b = rng.standard_normal(6).astype(np.float32)

    Z = np.zeros((4, 6), np.float32)
    A = np.zeros((4, 6), np.float32)

    bpg1, tpb1 = launch_2d(matmul_kernel, Z.shape, Z, X, W)      # Z = X @ W
    bpg2, tpb2 = launch_2d(bias_relu_kernel, A.shape, A, Z, b)   # A = relu(Z + b)

    print(f"\nmatmul-кернел:   сетка {bpg1} блоков x {tpb1} потоков "
          f"= {bpg1[0]*bpg1[1]*tpb1[0]*tpb1[1]} потоков на выход {Z.shape}")
    print(f"bias_relu-кернел: сетка {bpg2} блоков x {tpb2} потоков (слитый +bias→relu)")

    ref = np.maximum(0, X @ W + b)              # эталон numpy
    print(f"\nвыход GPU-симулятора [0] = {A[0]}")
    print(f"эталон numpy         [0] = {ref[0]}")
    diff = np.abs(A - ref).max()
    print(f"\nmax|sim - numpy| = {diff:.2e}  ->",
          "OK ✓ логика CUDA-кернелов верна" if diff < 1e-4 else "РАСХОЖДЕНИЕ ✗")
    print("\nЭто исполнилось на CPU: каждый 'поток GPU' — итерация в Python-симуляторе.\n"
          "Тот же код (без CUDASIM) запустился бы на настоящем GPU без изменений.")


if __name__ == "__main__":
    main()
