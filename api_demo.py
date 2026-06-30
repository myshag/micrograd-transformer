"""
Демонстрация совместимости с PyTorch.

Весь код ниже — это обычный код PyTorch. Чтобы запустить его на настоящем
torch, достаточно поменять две строки импорта:

    # наш движок:                         # настоящий PyTorch:
    import nn, optim                       import torch.nn as nn
    from tensor import Tensor              import torch.optim as optim
                                           from torch import tensor as Tensor

Всё остальное — определение модели, forward, criterion, optimizer, цикл
обучения, state_dict — совпадает один в один.

Запуск:  python3 api_demo.py
"""

import numpy as np

import nn
import optim
from tensor import Tensor


# --- 1. Кастомная модель как подкласс Module (канонический стиль PyTorch) ---

class Net(nn.Module):
    def __init__(self):
        super().__init__()                 # как в PyTorch
        self.fc1 = nn.Linear(20, 16, seed=1)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 3, seed=2)

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)


def main():
    rng = np.random.default_rng(0)
    # Игрушечный датасет: 200 примеров, 20 признаков, 3 класса.
    X = rng.standard_normal((200, 20))
    y = rng.integers(0, 3, size=200)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    print("Параметры модели (named_parameters):")
    for name, p in model.named_parameters():
        print(f"  {name:14s} {p.shape}")
    print()

    # --- 2. Цикл обучения — дословно как в PyTorch ---
    for epoch in range(1, 51):
        logits = model(Tensor(X))          # forward
        loss = criterion(logits, y)        # loss
        optimizer.zero_grad()              # обнулить градиенты
        loss.backward()                    # backward
        optimizer.step()                   # шаг оптимизатора
        if epoch % 10 == 0:
            acc = (logits.data.argmax(1) == y).mean()
            print(f"эпоха {epoch:2d}: loss = {loss.data:.4f}, acc = {acc:.3f}")

    # --- 3. Сохранение и загрузка весов (state_dict) ---
    state = model.state_dict()
    print(f"\nstate_dict содержит {len(state)} тензоров: {list(state)}")

    fresh = Net()
    fresh.load_state_dict(state)           # загрузили веса в новую модель
    same = np.allclose(fresh(Tensor(X)).data, model(Tensor(X)).data)
    print(f"после load_state_dict выходы совпадают: {same}")


if __name__ == "__main__":
    main()
