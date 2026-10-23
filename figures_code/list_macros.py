"""Print generated macros whose names start with any of the given prefixes (helper for writing the text)."""
import re, sys

prefixes = tuple(sys.argv[1:])
for line in open(__file__.replace("figs\\list_macros.py", "srep\\numbers.tex").replace("figs/list_macros.py", "srep/numbers.tex"), encoding="utf-8"):
    m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", line.strip())
    if m and m.group(1).startswith(prefixes):
        print(m.group(1), "=", m.group(2))
