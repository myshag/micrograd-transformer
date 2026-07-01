"""
Мульти-GPU: размещение тензоров на разных устройствах и перенос между ними.

Настоящих двух GPU у нас нет — оба 'cuda:0' и 'cuda:1' считает один симулятор на
CPU. Но СЕМАНТИКА мульти-GPU здесь настоящая и именно она — суть программирования
на нескольких видеокартах:
  - у каждого тензора есть устройство ('cuda:0', 'cuda:1', ...);
  - операции над тензорами на РАЗНЫХ устройствах запрещены (как в PyTorch);
  - чтобы их совместить, данные надо ЯВНО перенести через .to(...).

Ниже — «model parallelism»: слои сети живут на разных GPU, активация
переносится между ними. Так и обучают модели, не влезающие в одну карту.

Запуск:  python3 multi_gpu_demo.py
"""

import numpy as np

from tensor import Tensor

rng = np.random.default_rng(0)


def line(t):
    print("\n" + t)


line("1. Размещение на разных GPU")
x = Tensor(rng.standard_normal((2, 4)).astype(np.float32))
a0 = x.cuda(0)                     # .cuda()/.cuda(0) -> cuda:0
a1 = x.cuda(1)                     # .cuda(1)         -> cuda:1
print("  ", a0, "и", a1)
print("   ('cuda' == 'cuda:0':", x.cuda().device == "cuda:0", "— как в PyTorch)")

line("2. Операция над тензорами на РАЗНЫХ GPU -> ошибка")
W1 = Tensor(rng.standard_normal((4, 3)).astype(np.float32)).cuda(1)
try:
    _ = a0 @ W1                    # cuda:0 @ cuda:1
except AssertionError as e:
    print("  ", e)

line("3. Явный перенос .to('cuda:0') -> теперь можно")
y = a0 @ W1.to("cuda:0")
print("   ok, результат на", y.device)

line("4. Model parallelism: слой 1 на cuda:0, слой 2 на cuda:1")
# Веса каждого слоя лежат на своей карте.
W1 = Tensor(rng.standard_normal((4, 8)).astype(np.float32)).cuda(0)
b1 = Tensor(rng.standard_normal(8).astype(np.float32)).cuda(0)
W2 = Tensor(rng.standard_normal((8, 3)).astype(np.float32)).cuda(1)
b2 = Tensor(rng.standard_normal(3).astype(np.float32)).cuda(1)

xin = x.cuda(0)
h = (xin @ W1 + b1).relu()         # слой 1 — целиком на cuda:0
print("   слой 1 посчитан на", h.device)
h = h.to("cuda:1")                 # ПЕРЕНОС активации между картами
y = h @ W2 + b2                    # слой 2 — на cuda:1
print("   активация перенесена на cuda:1, слой 2 на", y.device)

# сверка с одним устройством
ref = ((x.cuda() @ W1.to("cuda:0") + b1).relu().to("cuda:0")
       @ W2.to("cuda:0") + b2.to("cuda:0"))
print("   сверка результата с одним GPU: max diff =",
      np.abs(y.data - ref.data).max())

print("\nИтог: device у тензора задаёт, на какой карте он живёт; перенос — явный "
      "(.to).\nЭто и есть основа мульти-GPU: разложить модель/данные по картам "
      "и гонять\nактивации между ними. Диспетчер выбирает бэкенд по типу "
      "('cuda' -> симулятор).")
