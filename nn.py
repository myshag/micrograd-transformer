"""
Мини-API нейросетей, базово совместимый с PyTorch (torch.nn).

Цель: код модели должен выглядеть как настоящий PyTorch. Совпадают имена
классов и сигнатуры (`Linear(in, out)`, `ReLU`, `Sequential`,
`CrossEntropyLoss`), имена параметров (`weight`, `bias`), форма весов
`(out_features, in_features)` и формула `x @ weight.T + bias`. Поэтому
учебник по PyTorch почти дословно переносится на наш движок.

Отличие только под капотом: вместо torch.Tensor используется наш Tensor
из tensor.py, а градиенты считает наш autograd.
"""

import numpy as np

from tensor import Tensor


class Module:
    """Аналог torch.nn.Module: базовый класс слоёв и моделей."""

    def parameters(self):
        """Все обучаемые Tensor'ы — рекурсивно по полям (как в PyTorch)."""
        return [p for _, p in self.named_parameters()]

    def named_parameters(self, prefix=""):
        """Пары (имя, Tensor). Имена в стиле PyTorch: 'layers.0.weight'."""
        seen, out = set(), []

        def visit(module, pfx):
            for name, value in module.__dict__.items():
                full = f"{pfx}{name}"
                if isinstance(value, Tensor):
                    if id(value) not in seen:
                        seen.add(id(value))
                        out.append((full, value))
                elif isinstance(value, Module):
                    visit(value, full + ".")
                elif isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        if isinstance(item, Module):
                            visit(item, f"{full}.{i}.")
                        elif isinstance(item, Tensor):
                            out.append((f"{full}.{i}", item))

        visit(self, prefix)
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def train(self, mode=True):
        # Заглушка для совместимости (у нас пока нет dropout/batchnorm).
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        """Словарь {имя: numpy-массив} — как torch.state_dict()."""
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state):
        for name, p in self.named_parameters():
            p.data = np.array(state[name], dtype=np.float64)

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError


class Linear(Module):
    """torch.nn.Linear: y = x @ weightᵀ + bias.

    Как в PyTorch: weight имеет форму (out_features, in_features), bias — (out,).
    Инициализация тоже как в PyTorch по умолчанию: равномерно в ±1/√in_features.
    """

    def __init__(self, in_features, out_features, bias=True, seed=None):
        rng = np.random.default_rng(seed)
        bound = 1.0 / np.sqrt(in_features)
        self.weight = Tensor(rng.uniform(-bound, bound, (out_features, in_features)))
        self.bias = Tensor(rng.uniform(-bound, bound, out_features)) if bias else None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Conv2d(Module):
    """torch.nn.Conv2d: свёрточный слой.

    weight: (out_channels, in_channels, kernel, kernel) — как в PyTorch.
    Вход и выход — карты признаков формы (N, C, H, W).
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, seed=None):
        rng = np.random.default_rng(seed)
        k = kernel_size
        # Инициализация как в PyTorch: ±1/√(in_channels·k·k).
        bound = 1.0 / np.sqrt(in_channels * k * k)
        self.weight = Tensor(
            rng.uniform(-bound, bound, (out_channels, in_channels, k, k)))
        self.bias = Tensor(rng.uniform(-bound, bound, out_channels)) if bias else None
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return x.conv2d(self.weight, self.bias, self.stride, self.padding)


class MaxPool2d(Module):
    """torch.nn.MaxPool2d: уменьшает карту, беря максимум в окнах kxk."""

    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return x.maxpool2d(self.kernel_size, self.stride)


class Flatten(Module):
    """torch.nn.Flatten: (N, C, H, W) -> (N, C*H*W) перед полносвязным слоём."""

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Sequential(Module):
    """torch.nn.Sequential: прогон входа через слои по порядку."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CrossEntropyLoss(Module):
    """torch.nn.CrossEntropyLoss: принимает логиты (N, C) и индексы классов (N,).

    Совмещает log-softmax и NLL, как в PyTorch, и возвращает скаляр-loss.
    """

    def forward(self, logits, targets):
        targets = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
        return logits.softmax_cross_entropy(targets.astype(np.int64))


# --- Слои для трансформера --------------------------------------------------

class Embedding(Module):
    """torch.nn.Embedding: таблица векторов. По индексу токена выдаёт его вектор.

    weight: (num_embeddings, embedding_dim). forward(idx) -> (..., embedding_dim).
    """

    def __init__(self, num_embeddings, embedding_dim, seed=None):
        rng = np.random.default_rng(seed)
        self.weight = Tensor(rng.standard_normal((num_embeddings, embedding_dim)) * 0.1)

    def forward(self, idx):
        idx = idx.data.astype(np.int64) if isinstance(idx, Tensor) else np.asarray(idx)
        return self.weight.index_rows(idx)


class LayerNorm(Module):
    """torch.nn.LayerNorm: нормирует последнюю ось к среднему 0 и дисперсии 1,
    затем масштабирует обучаемыми weight (γ) и bias (β).

    Стабилизирует обучение глубоких сетей — без неё трансформер плохо сходится.
    Реализован композицией примитивов, backward считает autograd.
    """

    def __init__(self, dim, eps=1e-5):
        self.weight = Tensor(np.ones(dim))
        self.bias = Tensor(np.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        xc = x - mean
        var = (xc * xc).mean(axis=-1, keepdims=True)
        norm = xc / (var + self.eps) ** 0.5
        return norm * self.weight + self.bias


class MultiheadAttention(Module):
    """torch.nn.MultiheadAttention (self-attention, batch_first).

    Идея внимания: каждая позиция формирует запрос Q и сравнивает его с ключами K
    всех позиций. Близость Q·K даёт веса (softmax), которыми усредняются значения V.
    Так позиция «собирает» информацию с других позиций. «Multi-head» — делаем это
    параллельно в нескольких подпространствах (головах) и склеиваем.

    Вход/выход: (B, T, d_model).
    """

    def __init__(self, d_model, n_heads, causal=False, seed=None):
        assert d_model % n_heads == 0, "d_model должен делиться на n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # causal=True -> позиция видит только себя и предыдущие (для генерации).
        self.causal = causal
        # Проекции Q, K, V и выходная — обычные Linear (как в PyTorch).
        s = (seed or 0)
        self.Wq = Linear(d_model, d_model, seed=s + 1)
        self.Wk = Linear(d_model, d_model, seed=s + 2)
        self.Wv = Linear(d_model, d_model, seed=s + 3)
        self.Wo = Linear(d_model, d_model, seed=s + 4)

    def _split_heads(self, x, B, T):
        # (B, T, d_model) -> (B, n_heads, T, d_head)
        return x.reshape(B, T, self.n_heads, self.d_head).swapaxes(1, 2)

    def forward(self, x):
        B, T, D = x.shape
        Q = self._split_heads(self.Wq(x), B, T)
        K = self._split_heads(self.Wk(x), B, T)
        V = self._split_heads(self.Wv(x), B, T)

        # Внимание: softmax(Q·Kᵀ / √d) · V
        scores = (Q @ K.mT) * (1.0 / np.sqrt(self.d_head))   # (B, h, T, T)
        if self.causal:
            # Запрещаем смотреть «в будущее»: добавляем -inf выше диагонали,
            # тогда после softmax эти веса станут нулевыми.
            mask = np.triu(np.full((T, T), -1e9), k=1)
            scores = scores + Tensor(mask)
        weights = scores.softmax(axis=-1)
        # Сохраняем веса внимания последнего прохода (для визуализации/анализа).
        self.attn_weights = weights.data                     # (B, h, T, T)
        context = weights @ V                                # (B, h, T, d_head)

        # Склеиваем головы обратно: (B, h, T, d_head) -> (B, T, d_model)
        context = context.swapaxes(1, 2).reshape(B, T, D)
        return self.Wo(context)


class TransformerEncoderLayer(Module):
    """torch.nn.TransformerEncoderLayer: блок «внимание + MLP» с остаточными
    связями и LayerNorm (вариант pre-norm — устойчивее при обучении).

        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(self, d_model, n_heads, d_ff=None, causal=False, seed=None):
        d_ff = d_ff or 4 * d_model
        self.attn = MultiheadAttention(d_model, n_heads, causal=causal, seed=seed)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ff = Sequential(
            Linear(d_model, d_ff, seed=(seed or 0) + 11),
            ReLU(),
            Linear(d_ff, d_model, seed=(seed or 0) + 12),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # остаточная связь вокруг внимания
        x = x + self.ff(self.norm2(x))     # остаточная связь вокруг MLP
        return x


class TransformerEncoder(Module):
    """torch.nn.TransformerEncoder: стопка одинаковых блоков-энкодеров."""

    def __init__(self, d_model, n_heads, n_layers, d_ff=None, causal=False, seed=None):
        self.layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, causal=causal,
                                    seed=(seed or 0) + 100 * i)
            for i in range(n_layers)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
