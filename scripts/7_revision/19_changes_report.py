r"""
Отчёт об изменениях рукописи в виде HTML (CPU).

Проходит по manuscript/main.tex, вытаскивает каждое вхождение \rev{}, \del{},
\delpar{} и \todo{} вместе с разделом, в котором оно стоит, и рендерит их
с цветовой разметкой. Нужен, чтобы просмотреть правки без сборки LaTeX.

Это не замена подсвеченной версии для подачи (её даёт main_marked.tex),
а средство просмотра.
"""
import os, re, sys, html

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT

SRC = f"{ROOT}/manuscript/main.tex"
OUT = f"{ROOT}/submission/changes_report.html"


def match_brace(s, i):
    """i указывает на '{'; вернуть индекс парной '}'."""
    depth = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def latex_to_text(s):
    """Грубая, но достаточная очистка LaTeX для чтения глазами."""
    s = re.sub(r"\\(cite|ref|label)\{[^}]*\}", "", s)
    s = re.sub(r"\\(textbf|textit|emph|rev|revm|revf|revmf)\{", "{", s)
    s = re.sub(r"\\todo\{", "{TODO: ", s)
    s = s.replace("---", "—").replace("--", "–")
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    s = s.replace("{", "").replace("}", "").replace("$", "")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


KINDS = {
    "rev":    ("added",   "добавлено или переписано"),
    "del":    ("deleted", "удалено"),
    "delpar": ("deleted", "удалён крупный фрагмент"),
    "todo":   ("todo",    "не заполнено"),
}


def main():
    src = open(SRC, encoding="utf-8").read()

    section = "Преамбула"
    items, i = [], 0
    sec_re = re.compile(r"\\(section|subsection)\*?\{")

    while i < len(src):
        m = sec_re.match(src, i)
        if m:
            j = src.index("{", i)
            k = match_brace(src, j)
            section = latex_to_text(src[j + 1:k])
            i = k + 1
            continue

        hit = None
        for kind in ("delpar", "del", "revmf", "revm", "revf", "rev", "todo"):
            tok = "\\" + kind + "{"
            if src.startswith(tok, i):
                hit = kind
                break
        if hit:
            j = i + len(hit) + 1
            k = match_brace(src, j)
            if k == -1:
                i += 1
                continue
            body = src[j + 1:k]
            kind = "rev" if hit in ("revm", "revf", "revmf") else hit
            text = latex_to_text(body)
            if len(text) > 2:
                items.append((section, kind, text))
            i = j + 1          # заходим внутрь: вложенные \todo не теряются
            continue
        i += 1

    counts = {}
    for _, kind, _ in items:
        counts[kind] = counts.get(kind, 0) + 1

    MARKS = {"rev": "+", "del": "−", "delpar": "−", "todo": "?"}
    rows, seen = [], None
    for section, kind, text in items:
        if section != seen:
            n = sum(1 for s, _, _ in items if s == section)
            rows.append(f'<h2>{html.escape(section)}'
                        f'<span class="n">{n}</span></h2>')
            seen = section
        cls, label = KINDS[kind]
        rows.append(
            f'<div class="e {cls}"><span class="m" aria-hidden="true">{MARKS[kind]}</span>'
            f'<div class="t"><span class="tag">{label}</span>'
            f'<p>{html.escape(text)}</p></div></div>')

    n_del = counts.get("del", 0) + counts.get("delpar", 0)
    doc = f"""<title>Правки рукописи — ревизия Scientific Reports</title>
<style>
:root {{
  --paper:#fbfcfd; --ink:#14181d; --muted:#5d6873; --rule:#dde3e9;
  --add:#0b6b4f; --del:#a3271e; --todo:#8a5a00; --todo-bg:#fdf6e6;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --paper:#101418; --ink:#e4e9ee; --muted:#8c98a4; --rule:#242c34;
    --add:#4cc79b; --del:#e8837a; --todo:#e0aa48; --todo-bg:#241c0c;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#101418; --ink:#e4e9ee; --muted:#8c98a4; --rule:#242c34;
  --add:#4cc79b; --del:#e8837a; --todo:#e0aa48; --todo-bg:#241c0c;
}}
:root[data-theme="light"] {{
  --paper:#fbfcfd; --ink:#14181d; --muted:#5d6873; --rule:#dde3e9;
  --add:#0b6b4f; --del:#a3271e; --todo:#8a5a00; --todo-bg:#fdf6e6;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0 auto; padding:1.6rem 1.15rem 4rem; max-width:44rem;
  background:var(--paper); color:var(--ink);
  font:400 16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-text-size-adjust:100%;
}}
header {{ border-bottom:2px solid var(--ink); padding-bottom:1rem; margin-bottom:.2rem; }}
.eyebrow {{
  font-size:.68rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); margin-bottom:.5rem;
}}
h1 {{
  font:600 1.5rem/1.25 Georgia,"Iowan Old Style","Times New Roman",serif;
  margin:0 0 .5rem; text-wrap:balance;
}}
.sub {{ color:var(--muted); font-size:.88rem; max-width:34rem; margin:0; }}
.tally {{
  display:flex; gap:1.6rem; flex-wrap:wrap;
  padding:.9rem 0 1.2rem; border-bottom:1px solid var(--rule); margin-bottom:.4rem;
}}
.tally div {{ display:flex; flex-direction:column; gap:.1rem; }}
.tally b {{
  font:600 1.35rem/1 Georgia,serif; font-variant-numeric:tabular-nums;
}}
.tally span {{ font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }}
.c-add b {{ color:var(--add); }}
.c-del b {{ color:var(--del); }}
.c-todo b {{ color:var(--todo); }}
h2 {{
  display:flex; align-items:baseline; justify-content:space-between; gap:1rem;
  font:600 .74rem/1.3 ui-sans-serif,sans-serif; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted);
  margin:2.4rem 0 .2rem; padding-bottom:.45rem; border-bottom:1px solid var(--rule);
}}
h2 .n {{ font-variant-numeric:tabular-nums; opacity:.65; letter-spacing:0; }}
.e {{
  display:grid; grid-template-columns:1.35rem 1fr; gap:.7rem;
  padding:.85rem 0; border-bottom:1px solid var(--rule);
}}
.m {{
  font:600 1rem/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
  text-align:center; user-select:none;
}}
.added .m {{ color:var(--add); }}
.deleted .m {{ color:var(--del); }}
.todo .m {{ color:var(--todo); }}
.tag {{
  display:block; font-size:.66rem; letter-spacing:.1em; text-transform:uppercase;
  font-weight:600; margin-bottom:.2rem;
}}
.added .tag {{ color:var(--add); }}
.deleted .tag {{ color:var(--del); }}
.todo .tag {{ color:var(--todo); }}
.t p {{
  margin:0; font:400 .96rem/1.62 Georgia,"Iowan Old Style","Times New Roman",serif;
}}
.deleted .t p {{ text-decoration:line-through; text-decoration-thickness:1px; color:var(--muted); }}
.todo {{ background:var(--todo-bg); margin-inline:-.6rem; padding-inline:.6rem; }}
footer {{ margin-top:2.5rem; padding-top:1rem; border-top:1px solid var(--rule);
  font-size:.8rem; color:var(--muted); }}
</style>

<header>
  <div class="eyebrow">Scientific Reports · ревизия</div>
  <h1>Правки рукописи</h1>
  <p class="sub">Two-Stage Synthetic-to-Real Transfer Learning for Automated
  Mammography Report Generation Using Vision-Language Models. Каждый фрагмент ниже —
  отдельная правка в <code>main.tex</code>, сгруппированы по разделам статьи
  в порядке следования.</p>
</header>

<div class="tally">
  <div class="c-add"><b>{counts.get('rev', 0)}</b><span>добавлено</span></div>
  <div class="c-del"><b>{n_del}</b><span>удалено</span></div>
  <div class="c-todo"><b>{counts.get('todo', 0)}</b><span>не заполнено</span></div>
</div>

{chr(10).join(rows)}

<footer>Отчёт построен автоматически из разметки <code>\\rev</code>,
<code>\\del</code>, <code>\\delpar</code> и <code>\\todo</code> в исходнике.
Для подачи подсвеченную версию даёт <code>main_marked.tex</code>.</footer>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(doc)
    print(f"  фрагментов: {len(items)}  {counts}")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
