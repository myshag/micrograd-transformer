"""
RL-дообучение НАСТОЯЩЕЙ LLM (Pythia-70M) на нашем движке (REINFORCE).

Раз Pythia у нас дифференцируема, её можно RL-дообучать. Для скорости замораживаем
тело (frozen, no_grad — считается быстро без графа) и обучаем только выходную
голову embed_out — она сдвигает распределение токенов. Награда — сколько раз в
продолжении встретился целевой токен (мини-аналог RLHF-стирки под предпочтение).

Смотрим, как выход СДВИГАЕТСЯ ЗА pretrain-распределение: до RL модель почти не
говорит целевое слово, после — вставляет его.

Требуется: transformers. Запуск:  python3 rl_pythia.py
"""

import numpy as np

import optim
import pythia
from tensor import Tensor, no_grad

PROMPT_TEXT = "The weather today is"
TARGET_TEXT = "!"
CONT, K, STEPS = 5, 6, 25


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pythia.MODEL)
    P = pythia.load_params()
    head = P["embed_out.weight"]                      # единственное обучаемое
    opt = optim.Adam([head], lr=5e-3)

    prompt = tok(PROMPT_TEXT)["input_ids"]
    target = tok(TARGET_TEXT, add_special_tokens=False)["input_ids"][0]
    lp = len(prompt)
    print(f"Цель RL: чаще генерировать {TARGET_TEXT!r} (токен {target}).")

    def body(ids):
        with no_grad():
            return pythia.hidden(P, ids).data         # (T, hid) numpy, тело заморожено

    def sample_cont():
        ids = list(prompt)
        with no_grad():
            for _ in range(CONT):
                p = softmax(body(ids)[-1] @ head.data.T)
                ids.append(int(np.random.choice(len(p), p=p)))
        return ids

    def reward(full):
        return float(sum(1 for t in full[lp:] if t == target))

    def show(tag):
        print(tag)
        for _ in range(3):
            s = sample_cont()
            print("   ", repr(tok.decode(s[lp:])))

    show("До RL, продолжения:")
    for step in range(1, STEPS + 1):
        fulls = [sample_cont() for _ in range(K)]
        R = np.array([reward(f) for f in fulls])
        base = R.mean()
        loss = None
        for i, f in enumerate(fulls):
            logits = Tensor(body(f)) @ head.T         # frozen body -> trainable head
            lg = logits[lp - 1:len(f) - 1]            # позиции, предсказывающие продолжение
            ce = lg.softmax_cross_entropy(np.asarray(f[lp:], np.int64))
            term = ce * float(R[i] - base)            # REINFORCE-вес (R — константа)
            loss = term if loss is None else loss + term
        loss = loss * (1.0 / K)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"шаг {step:2d}: средняя награда = {R.mean():.2f} / {CONT}")

    show("После RL, продолжения:")


if __name__ == "__main__":
    main()
