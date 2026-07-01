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


def _sum_to(grad, shape, xp):
    """Свернуть grad к форме shape по осям, размноженным broadcasting'ом
    (обобщение _unbroadcast для любого backend-модуля: numpy или cupy)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class NumpyBackend:
    """Эталонный бэкенд на numpy. Все операции над host-массивами.

    forward и backward — обе стороны на устройстве: методы *_grad считают
    градиент активаций, transpose_last2/sum_to нужны backward'у matmul и +/*.
    """

    xp = np

    # --- forward ---
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

    # --- backward (тоже на устройстве) ---
    def transpose_last2(self, x):
        return self.xp.swapaxes(x, -1, -2)

    def sum_to(self, grad, shape):
        return _sum_to(grad, shape, self.xp)

    def relu_grad(self, x, gy):
        return (x > 0) * gy

    def tanh_grad(self, t, gy):
        return (1 - t * t) * gy

    def sigmoid_grad(self, s, gy):
        return s * (1 - s) * gy


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

        # backward-кернелы активаций: o = grad_вход по значению forward и gy
        @cuda.jit
        def _grelu(o, x, gy):
            i = _nbcuda.grid(1)
            if i < o.size:
                o[i] = gy[i] if x[i] > 0.0 else 0.0

        @cuda.jit
        def _gtanh(o, t, gy):
            i = _nbcuda.grid(1)
            if i < o.size:
                o[i] = (1.0 - t[i] * t[i]) * gy[i]

        @cuda.jit
        def _gsigmoid(o, s, gy):
            i = _nbcuda.grid(1)
            if i < o.size:
                o[i] = s[i] * (1.0 - s[i]) * gy[i]

        _kernels = dict(mm=_mm, add=_add, mul=_mul,
                        relu=_relu, tanh=_tanh, sigmoid=_sigmoid,
                        grelu=_grelu, gtanh=_gtanh, gsigmoid=_gsigmoid)
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

    # backward активаций — через grad-кернелы (x и gy одной формы)
    def relu_grad(self, x, gy):
        return _launch_binary("grelu", x, gy)

    def tanh_grad(self, t, gy):
        return _launch_binary("gtanh", t, gy)

    def sigmoid_grad(self, s, gy):
        return _launch_binary("gsigmoid", s, gy)
    # transpose_last2 / sum_to наследуем от NumpyBackend (для симулятора это
    # тот же numpy; настоящий GPU переопределил бы их через cupy).


# --- Настоящий CUDA-бэкенд через .cu-ядра (нужен GPU + CuPy) -----------------
# Демонстрирует главный смысл диспетчера: чтобы задействовать реальный GPU,
# добавляется ОДИН класс — ни строчки в Tensor. Здесь не запускается (нет GPU),
# но на машине с NVIDIA GPU это drop-in: use_real_cuda() подменяет бэкенд 'cuda'.
_real_mod = None


def _real_kernels():
    global _real_mod
    if _real_mod is None:
        import cupy as cp                      # требует GPU + CUDA toolkit
        src = open("kernels.cu").read()
        _real_mod = cp.RawModule(code=src)     # NVRTC компилирует .cu -> PTX
    return _real_mod


class RealCudaBackend(NumpyBackend):
    """'cuda' на НАСТОЯЩЕМ GPU: ядра из kernels.cu через CuPy (NVRTC).

    Для наглядности на каждой операции копируем host<->device. В боевом варианте
    данные жили бы на GPU (cupy-массивы) постоянно, без копий на каждый шаг.
    """

    def _ew(self, fname, *arrays):
        import cupy as cp
        import numpy as _np
        mod = _real_kernels()
        n = arrays[0].size
        dev = [cp.asarray(a, dtype=cp.float32).reshape(-1) for a in arrays]
        o = cp.empty(n, dtype=cp.float32)
        tpb = 256
        mod.get_function(fname)(((n + tpb - 1) // tpb,), (tpb,),
                                (*dev, o, _np.int32(n)))
        return cp.asnumpy(o).reshape(arrays[0].shape)

    def matmul(self, a, b):
        import cupy as cp
        import numpy as _np
        if a.ndim != 2 or b.ndim != 2:
            return super().matmul(a, b)
        mod = _real_kernels()
        A = cp.asarray(a, dtype=cp.float32); B = cp.asarray(b, dtype=cp.float32)
        M, K = A.shape; N = B.shape[1]
        C = cp.zeros((M, N), dtype=cp.float32)
        tpb = (8, 8)
        bpg = ((N + 7) // 8, (M + 7) // 8)     # (x=cols, y=rows)
        mod.get_function("matmul")(bpg, tpb,
            (A, B, C, _np.int32(M), _np.int32(N), _np.int32(K)))
        return cp.asnumpy(C)

    def add(self, a, b):
        return self._ew("ew_add", np.broadcast_to(a, np.broadcast_shapes(a.shape, b.shape)),
                        np.broadcast_to(b, np.broadcast_shapes(a.shape, b.shape)))

    def mul(self, a, b):
        s = np.broadcast_shapes(a.shape, b.shape)
        return self._ew("ew_mul", np.broadcast_to(a, s), np.broadcast_to(b, s))

    def relu(self, x):
        return self._ew("ew_relu", x)

    def tanh(self, x):
        return self._ew("ew_tanh", x)

    def sigmoid(self, x):
        return self._ew("ew_sigmoid", x)

    def relu_grad(self, x, gy):
        return self._ew("grad_relu", x, gy)

    def tanh_grad(self, t, gy):
        return self._ew("grad_tanh", t, gy)

    def sigmoid_grad(self, s, gy):
        return self._ew("grad_sigmoid", s, gy)


BACKENDS = {
    "cpu": NumpyBackend(),
    "cuda": CudaSimBackend(),      # по умолчанию — симулятор (работает без GPU)
}


def use_real_cuda():
    """Переключить 'cuda' на настоящие .cu-ядра (нужен GPU + CuPy).

    Весь смысл диспетчера: одна строка — и весь код, использующий device='cuda',
    начинает считать на реальном GPU. Ни Tensor, ни модели не меняются.
    """
    BACKENDS["cuda"] = RealCudaBackend()


def get_backend(device):
    """Диспетч: вернуть бэкенд для устройства.

    Бэкенд выбирается по БАЗОВОМУ типу: 'cuda:0' и 'cuda:1' используют один
    бэкенд 'cuda' (у нас — симулятор). Различие устройств хранится в метке
    Tensor.device и проверяется на совместимость операндов (мульти-GPU).
    """
    base = device.split(":")[0]               # 'cuda:1' -> 'cuda'
    if base not in BACKENDS:
        raise ValueError(f"неизвестное устройство: {device!r} "
                         f"(есть: {list(BACKENDS)})")
    return BACKENDS[base]
