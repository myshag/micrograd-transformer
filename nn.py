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
