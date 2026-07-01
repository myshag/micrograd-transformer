"""
Скачивает и чистит русский корпус (тексты Пушкина) -> data/pushkin.txt.

Берём «Евгения Онегина» (стихи) и «Капитанскую дочку» (проза) из открытых
GitHub-репозиториев, объединяем и нормализуем символы, чтобы словарь был
компактным (~83 символа: кириллица + базовая пунктуация).

Запуск:  python3 download_pushkin.py
"""

import base64
import os
import re
import subprocess
import urllib.request

ONEGIN_URL = "https://raw.githubusercontent.com/zzmicer/RNN-Pushkin/master/onegin.txt"
# «Капитанская дочка» лежит по пути с кириллицей — берём через GitHub API (base64).
CAPTAIN_REPO = "OlegDurandin/AuthorStyle"
CAPTAIN_PATH = ("Input/RUS_AA/Train/"
                "%D0%9F%D1%83%D1%88%D0%BA%D0%B8%D0%BD_"
                "%D0%9A%D0%B0%D0%BF%D0%B8%D1%82%D0%B0%D0%BD%D1%81%D0%BA%D0%B0%D1%8F"
                "%20%D0%B4%D0%BE%D1%87%D0%BA%D0%B0.txt")

ALLOWED = set("абвгдежзийклмнопрстуфхцчшщъыьэюяёАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЁ")
ALLOWED |= set("0123456789")
ALLOWED |= set(" \n.,!?:;-\"'()")


def clean(text):
    text = (text.replace("\xa0", " ").replace("–", "-").replace("—", "-")
                .replace("“", '"').replace("”", '"').replace("„", '"')
                .replace("’", "'"))
    text = "".join(ch for ch in text if ch in ALLOWED)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def main():
    os.makedirs("data", exist_ok=True)

    req = urllib.request.Request(ONEGIN_URL, headers={"User-Agent": "Mozilla/5.0"})
    onegin = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")

    # gh api вернёт содержимое файла в base64 (обходит проблему пути с кириллицей).
    out = subprocess.run(
        ["gh", "api", f"repos/{CAPTAIN_REPO}/contents/{CAPTAIN_PATH}", "--jq", ".content"],
        capture_output=True, text=True, check=True)
    captain = base64.b64decode(out.stdout).decode("utf-8", "ignore")

    text = clean(onegin + "\n\n" + captain)
    open("data/pushkin.txt", "w", encoding="utf-8").write(text)
    print(f"Готово: data/pushkin.txt — {len(text):,} символов, "
          f"словарь {len(set(text))}")


if __name__ == "__main__":
    main()
