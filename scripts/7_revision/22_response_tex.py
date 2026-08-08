r"""
Сборка ответа рецензентам в LaTeX (CPU).

Scientific Reports требует загрузить point-by-point отдельным PDF; своей формы
у журнала нет. Скрипт переводит submission/POINT_BY_POINT_RESPONSE_EN.md
в manuscript/response_to_reviewers.tex, который компилируется в Overleaf рядом
с рукописью.

Редакторская пометка в начале markdown-файла переносится в красную рамку,
чтобы её нельзя было подать по недосмотру.
"""
import os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT

SRC = f"{ROOT}/submission/POINT_BY_POINT_RESPONSE_EN.md"
OUT = f"{ROOT}/manuscript/response_to_reviewers.tex"

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[left=2.2cm,right=2.2cm,top=2.2cm,bottom=2.2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,array,longtable,xcolor,enumitem}
\usepackage[colorlinks=true,allcolors=blue]{hyperref}
\usepackage{titlesec}
\usepackage{framed}

\definecolor{rvhead}{RGB}{20,60,110}
\definecolor{rvnote}{RGB}{170,40,30}

\titleformat{\section}{\large\bfseries\color{rvhead}}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
\titlespacing{\section}{0pt}{16pt}{6pt}
\titlespacing{\subsection}{0pt}{12pt}{4pt}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\renewcommand{\arraystretch}{1.15}
\pagestyle{plain}

\newenvironment{ednote}%
  {\par\color{rvnote}\begin{leftbar}\small\textbf{EDITORIAL NOTE --- delete before submission.}\par}%
  {\end{leftbar}\par}

\begin{document}
"""


def esc(s):
    """Экранировать спецсимволы вне математики и \verb-фрагментов."""
    parts = re.split(r'(\$\$.*?\$\$|\$[^$]*\$)', s, flags=re.S)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append(p)
            continue
        p = p.replace('\\', r'\textbackslash{}')
        for a, b in [('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_'),
                     ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'),
                     ('^', r'\textasciicircum{}')]:
            p = p.replace(a, b)
        out.append(p)
    return ''.join(out)


def inline(s):
    s = esc(s)
    s = re.sub(r'`([^`]+)`', lambda m: r'\texttt{' + m.group(1) + '}', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', s)
    s = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\\emph{\1}', s)
    s = s.replace('—', '---').replace('–', '--')
    s = s.replace('→', r'$\rightarrow$').replace('×', r'$\times$')
    s = s.replace('≥', r'$\geq$').replace('≤', r'$\leq$').replace('±', r'$\pm$')
    s = s.replace('"', "``", 1) if s.count('"') % 2 == 0 else s
    return s


def table(block):
    rows = [r.strip().strip('|').split('|') for r in block if r.strip()]
    rows = [r for r in rows if not re.match(r'^[\s:|-]+$', '|'.join(r))]
    ncol = max(len(r) for r in rows)
    body = []
    for i, r in enumerate(rows):
        cells = [inline(c.strip()) for c in r] + [''] * (ncol - len(r))
        body.append(' & '.join(cells) + r' \\')
        if i == 0:
            body.append(r'\midrule')
    return ('{\\small\\begin{longtable}{@{}' + 'l' * ncol + '@{}}\n\\toprule\n'
            + '\n'.join(body) + '\n\\bottomrule\n\\end{longtable}}\n')


def main():
    md = open(SRC, encoding="utf-8").read()
    blocks = re.split(r"\n\s*\n", md)
    out = []

    for blk in blocks:
        lines = [l for l in blk.split("\n") if l.strip() != ""]
        if not lines:
            continue
        first = lines[0]

        if first.startswith("> "):
            txt = " ".join(l[2:].strip() if l.startswith("> ") else l.strip() for l in lines)
            txt = re.sub(r"^\*\*EDITORIAL NOTE[^*]*\*\*\s*", "", txt)
            out.append(r"\begin{ednote}" + "\n" + inline(txt) + "\n" + r"\end{ednote}")
            continue

        if first.startswith("|"):
            out.append(table(lines))
            continue

        if first.startswith("$$"):
            out.append("\\[\n" + blk.strip().strip("$") + "\n\\]")
            continue

        if first.startswith("### "):
            out.append(r"\subsection*{" + inline(first[4:]) + "}")
            continue
        if first.startswith("## "):
            out.append(r"\section*{" + inline(first[3:]) + "}")
            continue
        if first.startswith("# "):
            out.append(r"\begin{center}{\LARGE\bfseries " + inline(first[2:]) + r"}\end{center}")
            continue
        if first.strip() == "---":
            out.append(r"\vspace{4pt}\hrule\vspace{4pt}")
            continue

        if re.match(r"^\d+\.\s", first) or first.startswith("- "):
            ordered = bool(re.match(r"^\d+\.\s", first))
            items, cur = [], []
            for l in lines:
                if re.match(r"^(\d+\.|-)\s", l):
                    if cur:
                        items.append(" ".join(cur))
                    cur = [re.sub(r"^(\d+\.|-)\s", "", l)]
                else:
                    cur.append(l.strip())
            if cur:
                items.append(" ".join(cur))
            env = "enumerate" if ordered else "itemize"
            out.append("\\begin{%s}[leftmargin=*]\n" % env
                       + "\n".join(r"\item " + inline(x) for x in items)
                       + "\n\\end{%s}" % env)
            continue

        out.append(inline(" ".join(l.strip() for l in lines)))

    open(OUT, "w", encoding="utf-8").write(PREAMBLE + "\n\n".join(out) + "\n\\end{document}\n")
    print(f"  \u2192 {OUT}")


if __name__ == "__main__":
    main()
