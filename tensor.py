"""
Тензорный autograd на numpy.

Это «старший брат» скалярного движка из autograd.py. Идея та же —
строим граф вычислений и считаем градиенты обратным проходом, — но теперь
узел графа хранит не одно число, а целый массив numpy. Благодаря этому
операции выполняются над матрицами сразу, и можно реально обучать сеть
на MNIST.

Главное новое усложнение по сравнению со скаляром — broadcasting.
Когда numpy «растягивает» массивы разной формы (например, прибавляет
вектор-смещение b к матрице X@W), градиент нужно «сжать» обратно к исходной
форме, просуммировав по растянутым осям. За это отвечает _unbroadcast.
"""

import math
import os

import numpy as np


def _unbroadcast(grad, shape):
    """Привести grad к форме shape, суммируя по осям, добавленным broadcasting'ом.

    Пример: b формы (10,) прибавили к матрице (N, 10). При forward numpy
    растянул b до (N, 10). При backward нам нужно вернуть градиент формы (10,),
    значит суммируем по оси N.
    """
    # 1) numpy мог добавить новые оси слева — суммируем их.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) Там, где исходный размер был 1 (а растянули до большего) — тоже суммируем.
    for axis, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


# --- im2col / col2im: ядро эффективной свёртки -------------------------------
# Свёртка — это «приложить ядро к каждому окну изображения». Наивно это
# вложенные циклы. Трюк im2col: вытащить ВСЕ окна и разложить их по столбцам
# одной большой матрицы. Тогда свёртка превращается в одно матричное
# умножение (его numpy/BLAS считают очень быстро). col2im — обратная операция,
# нужна в backward, чтобы «разложить» градиент столбцов обратно по пикселям.

def im2col(x, kh, kw, stride, pad):
    """x: (N, C, H, W) -> матрица окон (N*out_h*out_w, C*kh*kw)."""
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))

    cols = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = xp[:, :, i:i_max:stride, j:j_max:stride]

    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return cols, out_h, out_w


def col2im(cols, x_shape, kh, kw, stride, pad, out_h, out_w):
    """Обратная к im2col: (N*out_h*out_w, C*kh*kw) -> (N, C, H, W)."""
    N, C, H, W = x_shape
    cols = cols.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            # += потому что одно и то же место входит в несколько окон.
            xp[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]
    return xp[:, :, pad:pad + H, pad:pad + W]


# Глобальный переключатель построения графа. Когда выключен (режим no_grad),
# операции не запоминают родителей и не заводят backward — так инференс не
# держит в памяти весь граф и промежуточные карты сразу освобождаются.
_grad_enabled = True

# Тип чисел движка. float32 вдвое быстрее и легче по памяти, чем float64
# (это же дефолт в PyTorch). Для строгой численной проверки градиентов можно
# временно переключить на float64: tensor.set_dtype(np.float64).
_dtype = np.float32


def set_dtype(dt):
    global _dtype
    _dtype = dt


# Используются в backward() для разрыва ссылочных циклов и освобождения графа.
def _noop():
    return None


_EMPTY = frozenset()


class no_grad:
    """Контекст инференса, как torch.no_grad():

        with no_grad():
            logits = model(x)   # граф не строится, память минимальна
    """

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._prev
        return False


# --- Диспетчер по устройству (device) ---------------------------------------
# 'cpu' считает через numpy. 'cuda' демонстрационно исполняет ту же операцию
# через CUDA-кернел в СИМУЛЯТОРЕ Numba (без настоящего GPU): видно, как device
# выбирает бэкенд вычислений — как в PyTorch. Скорости GPU здесь нет.
_cuda_cache = None
_nbcuda = None      # модульная ссылка на numba.cuda: kernel резолвит имена в
                    # симуляторе через globals функции, а не через замыкание.


def _get_cuda():
    """Лениво собрать набор CUDA-кернелов (в симуляторе). Возвращает (cuda, dict)."""
    global _cuda_cache, _nbcuda
    if _cuda_cache is None:
        os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")   # ДО импорта cuda
        from numba import cuda
        _nbcuda = cuda

        @cuda.jit
        def _mm(C, A, B):                      # matmul: один поток на элемент C[i,j]
            i, j = _nbcuda.grid(2)
            if i < C.shape[0] and j < C.shape[1]:
                acc = 0.0
                for k in range(A.shape[1]):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc

        # Поэлементные кернелы: 1D-сетка, поток на элемент (данные уже плоские).
        @cuda.jit
        def _add(o, a, b):
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

        _cuda_cache = (cuda, dict(mm=_mm, add=_add, mul=_mul,
                                  relu=_relu, tanh=_tanh, sigmoid=_sigmoid))
    return _cuda_cache


def _cuda_matmul(A, B):
    if A.ndim != 2 or B.ndim != 2:             # батчи — на хосте (упрощение)
        return A @ B
    _, k = _get_cuda()
    C = np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
    tpb = (8, 8)
    bpg = (int(np.ceil(C.shape[0] / tpb[0])), int(np.ceil(C.shape[1] / tpb[1])))
    k["mm"][bpg, tpb](C, np.ascontiguousarray(A), np.ascontiguousarray(B))
    return C


def _cuda_unary(name, x):
    _, k = _get_cuda()
    # np.array(copy=True) -> записываемый плоский массив (симулятор Numba
    # копирует аргументы обратно на хост, read-only view его роняет).
    xf = np.array(x, dtype=x.dtype).reshape(-1)
    o = np.empty_like(xf)
    n = o.size
    tpb = 64
    k[name][(n + tpb - 1) // tpb, tpb](o, xf)
    return o.reshape(x.shape)


def _cuda_binary(name, a, b):
    # broadcasting делаем на хосте, кернел работает над плоскими массивами
    shape = np.broadcast_shapes(a.shape, b.shape)
    af = np.array(np.broadcast_to(a, shape), dtype=a.dtype).reshape(-1)
    bf = np.array(np.broadcast_to(b, shape), dtype=a.dtype).reshape(-1)
    _, k = _get_cuda()
    o = np.empty_like(af)
    n = o.size
    tpb = 64
    k[name][(n + tpb - 1) // tpb, tpb](o, af, bf)
    return o.reshape(shape)


class Tensor:
    """Узел графа: массив numpy + его градиент той же формы."""

    def __init__(self, data, _children=(), _op="", device=None):
        self.data = np.asarray(data, dtype=_dtype)
        # Под no_grad не аллоцируем grad (экономия памяти) и не держим детей.
        self.grad = np.zeros_like(self.data) if _grad_enabled else None
        self._backward = lambda: None
        self._prev = set(_children) if _grad_enabled else set()
        # Упорядоченные входы (с порядком и кратностью) — нужны компилятору
        # графа в C (compile_blas.py). Множество _prev это теряет.
        self._inputs = tuple(_children)
        self._op = _op
        # Устройство: по умолчанию наследуем от первого входа (так device сам
        # распространяется по всему графу), для листа — 'cpu', либо задано явно.
        self.device = device if device is not None else (
            _children[0].device if _children else "cpu")

    def _set_backward(self, fn):
        # Регистрируем backward только если граф включён. Иначе замыкание
        # нигде не сохраняется и тут же освобождается вместе со ссылками
        # на входные тензоры — это и есть экономия памяти при инференсе.
        if _grad_enabled:
            self._backward = fn

    @property
    def shape(self):
        return self.data.shape

    # --- Устройство (как в PyTorch) -----------------------------------------

    def to(self, device):
        """Переместить тензор на устройство. Значения те же, меняется backend."""
        if device == self.device:
            return self
        out = Tensor(self.data, (self,), f"to:{device}", device=device)

        def _backward():           # перенос — тождественная операция
            self.grad += out.grad

        out._set_backward(_backward)
        return out

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    # --- Операции -----------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        assert self.device == other.device, (
            f"тензоры на разных устройствах: {self.device} и {other.device}")
        result = (_cuda_binary("add", self.data, other.data)
                  if self.device == "cuda" else self.data + other.data)
        out = Tensor(result, (self, other), "+")

        def _backward():
            # Градиент сложения проходит как есть, но с поправкой на broadcasting.
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._set_backward(_backward)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        assert self.device == other.device, (
            f"тензоры на разных устройствах: {self.device} и {other.device}")
        result = (_cuda_binary("mul", self.data, other.data)
                  if self.device == "cuda" else self.data * other.data)
        out = Tensor(result, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._set_backward(_backward)
        return out

    def matmul(self, other):
        """Матричное умножение self @ other.

        Поддерживает батчи (любые ведущие оси, как в numpy): нужно для
        multi-head attention, где тензоры имеют форму (B, heads, T, d).
        device выбирает бэкенд: 'cpu' -> numpy/BLAS, 'cuda' -> CUDA-кернел
        (в симуляторе). Операнды должны быть на одном устройстве, как в PyTorch.
        """
        assert self.device == other.device, (
            f"тензоры на разных устройствах: {self.device} и {other.device}")
        result = (_cuda_matmul(self.data, other.data)
                  if self.device == "cuda" else self.data @ other.data)
        out = Tensor(result, (self, other), "@")

        def _backward():
            # Для C = A @ B:  dA = dC @ Bᵀ,  dB = Aᵀ @ dC.
            # Транспонируем только две последние оси (батчи не трогаем),
            # а _unbroadcast сворачивает оси, размноженные broadcasting'ом.
            ga = out.grad @ np.swapaxes(other.data, -1, -2)
            gb = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += _unbroadcast(ga, self.data.shape)
            other.grad += _unbroadcast(gb, other.data.shape)

        out._set_backward(_backward)
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def relu(self):
        data = (_cuda_unary("relu", self.data)
                if self.device == "cuda" else np.maximum(0, self.data))
        out = Tensor(data, (self,), "relu")

        def _backward():
            # производная: 1 там, где вход был > 0, иначе 0
            self.grad += (self.data > 0) * out.grad

        out._set_backward(_backward)
        return out

    def tanh(self):
        t = (_cuda_unary("tanh", self.data)
             if self.device == "cuda" else np.tanh(self.data))
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._set_backward(_backward)
        return out

    def sigmoid(self):
        s = (_cuda_unary("sigmoid", self.data)
             if self.device == "cuda" else 1 / (1 + np.exp(-self.data)))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            # d/dx sigmoid = sigmoid * (1 - sigmoid)
            self.grad += s * (1 - s) * out.grad

        out._set_backward(_backward)
        return out

    def transpose(self):
        """Транспонирование. Нужно линейному слою (веса хранятся как (out, in))."""
        out = Tensor(self.data.T, (self,), "T")

        def _backward():
            self.grad += out.grad.T

        out._set_backward(_backward)
        return out

    @property
    def T(self):
        return self.transpose()

    def swapaxes(self, axis1, axis2):
        """Поменять местами две оси (для перестановки голов в attention)."""
        out = Tensor(np.swapaxes(self.data, axis1, axis2), (self,), "swapaxes")

        def _backward():
            self.grad += np.swapaxes(out.grad, axis1, axis2)

        out._set_backward(_backward)
        return out

    @property
    def mT(self):
        """Транспонирование двух последних осей — для Q @ Kᵀ в attention."""
        return self.swapaxes(-1, -2)

    def __pow__(self, p):
        """Возведение в постоянную степень (нужно для LayerNorm: √, 1/x)."""
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += (p * self.data ** (p - 1)) * out.grad

        out._set_backward(_backward)
        return out

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return self * other ** -1
        return self * (1.0 / other)

    def __rtruediv__(self, other):
        return (self ** -1) * other

    def softmax(self, axis=-1):
        """Softmax вдоль оси — превращает «сырые» оценки в веса внимания."""
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(p, (self,), "softmax")

        def _backward():
            # Якобиан softmax: dx = p ⊙ (g − Σ(g⊙p)).
            s = (out.grad * p).sum(axis=axis, keepdims=True)
            self.grad += p * (out.grad - s)

        out._set_backward(_backward)
        return out

    def index_rows(self, idx):
        """Выбрать строки по индексам (основа Embedding). idx — numpy-массив."""
        out = Tensor(self.data[idx], (self,), "embed")

        def _backward():
            # Несколько позиций могут ссылаться на одну строку — копим через add.at.
            np.add.at(self.grad, idx, out.grad)

        out._set_backward(_backward)
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            # Градиент суммы — единица в каждую ячейку входа (с учётом формы).
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad

        out._set_backward(_backward)
        return out

    def mean(self, axis=None, keepdims=False):
        # mean = sum / N, поэтому переиспользуем sum и делим на число элементов.
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            # Просто возвращаем градиент к исходной форме.
            self.grad += out.grad.reshape(self.data.shape)

        out._set_backward(_backward)
        return out

    # --- Свёрточные операции ------------------------------------------------

    def conv2d(self, weight, bias=None, stride=1, padding=0):
        """Свёртка. self: (N, C_in, H, W), weight: (C_out, C_in, kh, kw).

        Реализована через im2col: окна -> столбцы -> одно матричное умножение.
        Градиенты текут и во вход (self), и в веса, и в смещение.
        """
        N, C_in, H, W = self.data.shape
        C_out, _, kh, kw = weight.data.shape

        cols, out_h, out_w = im2col(self.data, kh, kw, stride, padding)
        W_row = weight.data.reshape(C_out, -1)          # (C_out, C_in*kh*kw)

        out_data = cols @ W_row.T                        # (N*out_h*out_w, C_out)
        if bias is not None:
            out_data = out_data + bias.data
        # вернуть форму (N, C_out, out_h, out_w)
        out_data = out_data.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)

        children = (self, weight) if bias is None else (self, weight, bias)
        out = Tensor(out_data, children, "conv2d")

        def _backward():
            # dout в форму столбцов: (N*out_h*out_w, C_out)
            dout = out.grad.transpose(0, 2, 3, 1).reshape(-1, C_out)
            if bias is not None:
                bias.grad += dout.sum(axis=0)
            # градиент весов: dW = colsᵀ @ dout
            weight.grad += (cols.T @ dout).T.reshape(weight.data.shape)
            # градиент входа: раскладываем dcols обратно по пикселям через col2im
            dcols = dout @ W_row
            self.grad += col2im(dcols, self.data.shape, kh, kw,
                                stride, padding, out_h, out_w)

        out._set_backward(_backward)
        return out

    def maxpool2d(self, kernel=2, stride=None):
        """Max-pooling. self: (N, C, H, W). Берёт максимум в каждом окне kxk.

        Для простоты поддержан типичный случай: stride == kernel, без паддинга
        и H, W кратны kernel. Уменьшает карту вдвое (при kernel=2), оставляя
        самые сильные отклики и делая сеть устойчивой к сдвигам.
        """
        k = kernel
        s = stride or kernel
        assert s == k, "поддержан только stride == kernel"
        N, C, H, W = self.data.shape
        assert H % k == 0 and W % k == 0, "H и W должны делиться на kernel"
        out_h, out_w = H // k, W // k

        # Разрезаем на окна kxk и берём максимум по ним.
        x = self.data.reshape(N, C, out_h, k, out_w, k)
        out_data = x.max(axis=(3, 5))
        out = Tensor(out_data, (self,), "maxpool2d")

        def _backward():
            # Градиент идёт только в ту ячейку окна, где был максимум.
            mask = (x == out_data[:, :, :, None, :, None])
            # если в окне несколько одинаковых максимумов — делим поровну
            mask = mask / mask.sum(axis=(3, 5), keepdims=True)
            grad = out.grad[:, :, :, None, :, None] * mask
            self.grad += grad.reshape(N, C, H, W)

        out._set_backward(_backward)
        return out

    def softmax_cross_entropy(self, targets):
        """Совмещённые softmax + кросс-энтропия -> скаляр-loss.

        targets — массив индексов правильных классов, форма (N,).

        Почему совмещаем: по отдельности softmax и log численно неустойчивы,
        а их комбинация даёт простой и стабильный градиент: (p - y_onehot) / N.
        Это стандартный приём во всех фреймворках.
        """
        logits = self.data
        N = logits.shape[0]

        # Численно устойчивый softmax: вычитаем максимум по строке.
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        probs = exp / exp.sum(axis=1, keepdims=True)        # (N, classes)

        # Кросс-энтропия: -log(вероятность правильного класса), усреднённая.
        log_likelihood = -np.log(probs[np.arange(N), targets] + 1e-12)
        loss = log_likelihood.mean()

        out = Tensor(loss, (self,), "softmax_xent")

        def _backward():
            grad = probs.copy()
            grad[np.arange(N), targets] -= 1      # p - y_onehot
            grad /= N                             # усреднение по батчу
            self.grad += grad * out.grad          # out.grad обычно = 1

        out._set_backward(_backward)
        # probs пригодятся снаружи для подсчёта точности
        out.probs = probs
        return out

    # --- Backward (как в скалярном движке) ----------------------------------

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)   # d(self)/d(self) = 1
        for v in reversed(topo):
            v._backward()

        # Освобождаем граф (как PyTorch по умолчанию). Замыкание _backward
        # ссылается на свой же узел -> ссылочный цикл, который обычный счётчик
        # ссылок Python не освобождает. Разрываем циклы, обнуляя _backward и
        # _prev: тогда промежуточные узлы удаляются сразу, а не копятся до
        # срабатывания сборщика циклов (иначе при больших графах — рост памяти).
        # .grad и .data узлов остаются доступны тем, кто держит на них ссылку.
        for v in topo:
            v._backward = _noop
            v._prev = _EMPTY
            v._inputs = ()

    # --- Сахар --------------------------------------------------------------

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        dev = "" if self.device == "cpu" else f", device='{self.device}'"
        return f"Tensor(shape={self.data.shape}{dev})"
