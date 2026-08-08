r"""
Сборка подсвеченной версии рукописи (CPU).

manuscript/main_marked.tex порождается как ПОЛНАЯ КОПИЯ main.tex, в которой
переключён единственный флаг подсветки. Копия, а не \input-обёртка, потому что
обёртка требует смены главного документа в Overleaf и молча собирает чистую
версию, если этого не сделать.

Файл помечен как сгенерированный: править руками нельзя, любые изменения
вносятся в main.tex, после чего скрипт запускается заново.
"""
import os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT

SRC = f"{ROOT}/manuscript/main.tex"
DST = f"{ROOT}/manuscript/main_marked.tex"

FLAG_RE = re.compile(r"^\\ifdefined\\MARKUPVERSION.*$", re.M)

HEADER = r"""%% ─────────────────────────────────────────────────────────────────
%% СГЕНЕРИРОВАННЫЙ ФАЙЛ — НЕ ПРАВИТЬ РУКАМИ.
%%
%% Подсвеченная версия рукописи (marked-up copy) для related files.
%% Это полная копия main.tex с включённым флагом \markuptrue.
%% Любые правки вносятся в main.tex, после чего:
%%     python scripts/7_revision/20_make_marked.py
%%
%% Синий — новый или переписанный текст
%% Красный зачёркнутый — удалённый фрагмент
%% [REMOVED: ...] — удалённый крупный фрагмент
%% [TODO: ...] — незаполненный слот, виден в обеих версиях
%% ─────────────────────────────────────────────────────────────────
"""


def main():
    src = open(SRC, encoding="utf-8").read()

    if not FLAG_RE.search(src):
        raise SystemExit("не найдена строка флага \\ifdefined\\MARKUPVERSION в main.tex")

    out = FLAG_RE.sub(r"\\markuptrue  % включено скриптом 20_make_marked.py", src)

    if "\\markuptrue" not in out:
        raise SystemExit("подстановка флага не сработала")

    open(DST, "w", encoding="utf-8").write(HEADER + out)
    print(f"  → {DST}")
    print(f"  строк: {out.count(chr(10)) + 1}, подсвеченных фрагментов: "
          f"{out.count(chr(92) + 'rev{') + out.count(chr(92) + 'revm{')}")


if __name__ == "__main__":
    main()
