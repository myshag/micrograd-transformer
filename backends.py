"""
Бэкенды вычислений за диспетчером — как в PyTorch.

Tensor должен отвечать только за граф автодиффа (что из чего получилось и как
течёт градиент). А *как именно* складывать/умножать числа на конкретном
устройстве — дело бэкенда. Поэтому вычисления вынесены сюда:

    NumpyBackend    — 'cpu', считает через numpy/BLAS
    CudaSimBackend  — 'cuda', те же операции через CUDA-кернелы в СИМУЛЯТОРЕ
                      Numba (без GPU; наследует numpy для того, чего не переопределил)

    BACKENDS = {'cpu': ..., 'cuda': ...}
    get_backend(device).add(a, b)   # диспетч по устройству

Добавить новое устройство (Metal, OpenCL, настоящий CUDA) = добавить класс
бэкенда и строчку в BACKENDS. Ни одной операции в Tensor править не нужно —
это и есть смысл диспетчера.
"""

import math
import os

import numpy as np


class NumpyBackend:
    """Эталонный бэкенд на numpy. Все операции над host-массивами."""

    def matmul(self, a, b):
        return a @ b

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# --- CUDA-бэкенд (симулятор Numba) ------------------------------------------
# Ядра грузятся лениво: без обращения к 'cuda' numba не импортируется вовсе.
# _nbcuda — модульная ссылка: kernel резолвит имена в симуляторе через globals
# функции, а не через замыкание.
_nbcuda = None
_kernels = None


def _load_kernels():
    global _nbcuda, _kernels
    if _kernels is None:
        os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")   # ДО импорта cuda
        from numba import cuda
        _nbcuda = cuda

        @cuda.jit
        def _mm(C, A, B):                      # matmul: поток на элемент C[i,j]
            i, j = _nbcuda.grid(2)
            if i < C.shape[0] and j < C.shape[1]:
                acc = 0.0
                for k in range(A.shape[1]):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc

        @cuda.jit
        def _add(o, a, b):                     # поэлементные: 1D-сетка
            i = _nbcuda.grid(1)
            if i < o.size:
                o[i] = a[i] + b[i]

        @cuda.jit
        def _mul(o, a, b):
            i = _nbcuda.grid(1)
            if i < o.size:
                o[i] = a[i] * b[i]

        @cuda.jit
        def _relu(o, x):
            i = _nbcuda.grid(1)
            if i < o.size:
                v = x[i]
                o[i] = v if v > 0.0 else 0.0

        @cuda.jit
        def _tanh(o, x):
            i = _nbcuda.grid(1)
            if i < o.size:
                o[i] = math.tanh(x[i])

        @cuda.jit
        def _sigmoid(o, x):
            i = _nbcuda.grid(1)
            if i < o.size:
                o[i] = 1.0 / (1.0 + math.exp(-x[i]))

        _kernels = dict(mm=_mm, add=_add, mul=_mul,
                        relu=_relu, tanh=_tanh, sigmoid=_sigmoid)
    return _kernels


def _launch_unary(name, x):
    k = _load_kernels()
    # np.array(copy) -> записываемый плоский массив (симулятор копирует
    # аргументы обратно на хост; read-only view его роняет).
    xf = np.array(x, dtype=x.dtype).reshape(-1)
    o = np.empty_like(xf)
    tpb = 64
    k[name][(o.size + tpb - 1) // tpb, tpb](o, xf)
    return o.reshape(x.shape)


def _launch_binary(name, a, b):
    shape = np.broadcast_shapes(a.shape, b.shape)      # broadcasting на хосте
    af = np.array(np.broadcast_to(a, shape), dtype=a.dtype).reshape(-1)
    bf = np.array(np.broadcast_to(b, shape), dtype=a.dtype).reshape(-1)
    k = _load_kernels()
    o = np.empty_like(af)
    tpb = 64
    k[name][(o.size + tpb - 1) // tpb, tpb](o, af, bf)
    return o.reshape(shape)


class CudaSimBackend(NumpyBackend):
    """'cuda' через CUDA-кернелы (симулятор). Чего не переопределили —
    наследуем от NumpyBackend (считается на хосте)."""

    def matmul(self, a, b):
        if a.ndim != 2 or b.ndim != 2:         # батчи — на хосте (упрощение)
            return super().matmul(a, b)
        k = _load_kernels()
        C = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)
        tpb = (8, 8)
        bpg = (int(np.ceil(C.shape[0] / tpb[0])), int(np.ceil(C.shape[1] / tpb[1])))
        k["mm"][bpg, tpb](C, np.ascontiguousarray(a), np.ascontiguousarray(b))
        return C

    def add(self, a, b):
        return _launch_binary("add", a, b)

    def mul(self, a, b):
        return _launch_binary("mul", a, b)

    def relu(self, x):
        return _launch_unary("relu", x)

    def tanh(self, x):
        return _launch_unary("tanh", x)

    def sigmoid(self, x):
        return _launch_unary("sigmoid", x)


BACKENDS = {
    "cpu": NumpyBackend(),
    "cuda": CudaSimBackend(),
}


def get_backend(device):
    """Диспетч: вернуть бэкенд для устройства."""
    if device not in BACKENDS:
        raise ValueError(f"неизвестное устройство: {device!r} "
                         f"(есть: {list(BACKENDS)})")
    return BACKENDS[device]
