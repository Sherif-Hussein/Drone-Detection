"""List placeholder (\\pend) macros that are actually used in the manuscript sources."""
import glob, os, re

SREP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "srep")
nums = open(os.path.join(SREP, "numbers.tex"), encoding="utf-8").read()
pend = re.findall(r"\\newcommand\{\\(\w+)\}\{\\pend", nums)
txt = "".join(open(f, encoding="utf-8").read() for f in glob.glob(os.path.join(SREP, "sec_*.tex")) + [os.path.join(SREP, "main.tex")])
used = [m for m in pend if re.search(r"\\" + m + r"(?![A-Za-z])", txt)]
print("pending macros:", pend)
print("USED in text:", used)
print("literal \\pend{ in text:", len(re.findall(r"\\pend\{", txt)))
