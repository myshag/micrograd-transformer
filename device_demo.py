"""
Параметр device в нашем Tensor — как в PyTorch.

'cpu'  -> вычисления через numpy/BLAS.
'cuda' -> та же операция через CUDA-кернел в СИМУЛЯТОРЕ Numba (без GPU):
          видно, как device выбирает бэкенд. Скорости GPU здесь нет.

device распространяется по графу автоматически, операнды обязаны быть на одном
устройстве (иначе ошибка — как в torch), .to()/.cpu()/.cuda() перемещают тензор.

Запуск:  python3 device_demo.py
"""

import numpy as np

from tensor import Tensor

rng = np.random.default_rng(0)


def line(t):
    print("\n" + t)


line("1. По умолчанию тензор на cpu")
x = Tensor(rng.standard_normal((4, 8)).astype(np.float32))
print("  ", x, "| device =", x.device)

line("2. .cuda() / .to('cuda') перемещает на устройство (как в PyTorch)")
xc = x.cuda()
print("  ", xc, "| device =", xc.device)

line("3. Операция на cuda исполняется CUDA-кернелом (в симуляторе)")
W = Tensor(rng.standard_normal((8, 6)).astype(np.float32)).cuda()
y = xc @ W                        # matmul -> CUDA-кернел, поток на элемент
print("   результат:", y, "| device распространился сам:", y.device)
print("   сверка с cpu: max diff =", np.abs(y.data - (x.cpu() @ W.cpu()).data).max())

line("4. device течёт через весь граф (мини-MLP forward на 'cuda')")
W1 = Tensor(rng.standard_normal((8, 16)).astype(np.float32)).cuda()
b1 = Tensor(rng.standard_normal(16).astype(np.float32)).cuda()
h = (xc @ W1 + b1).relu()
print("   после '@ + relu':", h, "| device =", h.device)

line("5. Разные устройства -> ошибка (как в torch)")
try:
    _ = x @ W                     # x на cpu, W на cuda
except AssertionError as e:
    print("   поймали:", e)

print("\nИтог: device — это выбор бэкенда вычислений. У нас cpu=numpy, "
      "cuda=CUDA-кернел\nв симуляторе; в PyTorch cuda — настоящий GPU. API тот же.")
