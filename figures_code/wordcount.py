"""Approximate Scientific Reports word counts: main text (Intro+Results+Discussion without tables, figures, captions)."""
import os, re

SREP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "srep")


def words(fn):
    t = open(os.path.join(SREP, fn), encoding="utf-8").read()
    t = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}", " ", t)
    t = re.sub(r"\$[^$]*\$", " x ", t)
    t = re.sub(r"\\(sub)*section\*?\{([^}]*)\}", r" \2 ", t)
    t = re.sub(r"\\[A-Za-z]+\{\}", " x ", t)
    t = re.sub(r"\\textbf\{|\\emph\{|\\textit\{|\\textsubscript\{[^}]*\}", " ", t)
    t = re.sub(r"\\[A-Za-z]+", " x ", t)
    t = re.sub(r"[{}~]", " ", t)
    return len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])


tot = 0
for fn in ("sec_intro.tex", "sec_results.tex", "sec_discussion.tex"):
    n = words(fn); tot += n; print(f"{fn:22s} {n}")
print("main text total", tot, "(limit 4500)")
print("abstract", words("sec_abstract.tex"), "(limit 200)")
print("methods", words("sec_methods.tex"))
