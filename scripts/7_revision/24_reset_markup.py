"""
Сброс разметки предыдущего раунда ревизии в manuscript/main.tex (CPU).

Подсвеченная версия должна показывать дельту ТЕКУЩЕГО раунда. Правки прошлого
раунда стали обычным текстом статьи, поэтому их разметка снимается:

  \rev{X}, \revm{X}, \revf{X}, \revmf{X}  →  X          (содержимое остаётся)
  \del{X}, \delpar{X}                     →  (удаляются вместе с содержимым)

Макросы в преамбуле НЕ трогаются: они нужны для разметки нового раунда.
Скрипт идемпотентен — повторный запуск на уже очищенном файле ничего не меняет.

Запуск:  python scripts/7_revision/24_reset_markup.py [--file main.tex] [--dry-run]
"""
import argparse, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT

KEEP = ["rev", "revm", "revf", "revmf"]   # развернуть, содержимое сохранить
DROP = ["del", "delpar"]                  # удалить вместе с содержимым

# Граница преамбулы: до \title{...} лежат определения самих макросов, их
# разворачивать нельзя. Резать по \begin{document} нельзя — в классе wlscirep
# абстракт стоит ВЫШЕ него и тоже размечается.
BODY_START = r"\title{"


def find_close(s, open_idx):
    """Индекс парной закрывающей скобки к s[open_idx] == '{'."""
    depth = 0
    i = open_idx
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2                      # экранированный символ: \{ \} \\
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"не найдена закрывающая скобка от позиции {open_idx}")


def strip_macro(body, name, keep_content):
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])\s*\{")
    n = 0
    while True:
        m = pat.search(body)
        if not m:
            break
        close = find_close(body, m.end() - 1)
        inner = body[m.end():close]
        body = body[:m.start()] + (inner if keep_content else "") + body[close + 1:]
        n += 1
    return body, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="main.tex")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = f"{ROOT}/manuscript/{args.file}"
    src = open(path, encoding="utf-8").read()
    cut = src.index(BODY_START)
    head, body = src[:cut], src[cut:]

    total = {}
    for name in KEEP:
        body, n = strip_macro(body, name, keep_content=True)
        total[name] = n
    for name in DROP:
        body, n = strip_macro(body, name, keep_content=False)
        total[name] = n

    # После удаления \del{...} остаются двойные пробелы и пустые строки подряд.
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"([^\n]) {2,}", r"\1 ", body)
    body = re.sub(r"\n{4,}", "\n\n\n", body)

    print("  снято:", ", ".join(f"\\{k} × {v}" for k, v in total.items()))
    if args.dry_run:
        print("  --dry-run: файл не изменён")
        return
    open(path, "w", encoding="utf-8").write(head + body)
    print(f"  → {path}")


if __name__ == "__main__":
    main()
