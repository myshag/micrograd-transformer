"""
Скачивает MNIST и сохраняет в data/mnist.npz.

Берём официальные IDX-файлы с зеркала PyTorch (S3) и парсим их формат.
Запуск:  python3 download_mnist.py
"""

import gzip
import os
import urllib.request

import numpy as np

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def _fetch(name):
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0"})
    return gzip.decompress(urllib.request.urlopen(req, timeout=60).read())


def main():
    os.makedirs("data", exist_ok=True)
    raw = {k: _fetch(fn) for k, fn in FILES.items()}

    # IDX-формат: изображения — заголовок 16 байт, метки — 8 байт, дальше uint8.
    def images(b):
        return np.frombuffer(b, np.uint8, offset=16).reshape(-1, 28 * 28)

    def labels(b):
        return np.frombuffer(b, np.uint8, offset=8)

    np.savez_compressed(
        "data/mnist.npz",
        x_train=images(raw["train_x"]), y_train=labels(raw["train_y"]),
        x_test=images(raw["test_x"]), y_test=labels(raw["test_y"]),
    )
    print("Готово: data/mnist.npz (train 60000, test 10000)")


if __name__ == "__main__":
    main()
