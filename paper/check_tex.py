"""Static sanity checks for main.tex (no LaTeX install needed): environment nesting, braces, refs, cites."""
import re
import sys

src = open(sys.argv[1] if len(sys.argv) > 1 else "main.tex", encoding="utf8").read()
body = re.sub(r"(?<!\\)%.*", "", src)            # strip comments
ok = True

stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
    kind, env = m.groups()
    line = body.count("\n", 0, m.start()) + 1
    if kind == "begin":
        stack.append((env, line))
    elif not stack or stack[-1][0] != env:
        print(f"line {line}: \\end{{{env}}} does not match {stack[-1] if stack else 'nothing'}")
        ok = False
    else:
        stack.pop()
if stack:
    print("unclosed environments:", stack); ok = False

depth = 0
def escaped(i):
    n = 0
    while i - 1 - n >= 0 and body[i - 1 - n] == "\\":
        n += 1
    return n % 2 == 1


for i, ch in enumerate(body):
    if ch in "{}" and not escaped(i):
        depth += 1 if ch == "{" else -1
        if depth < 0:
            print("unbalanced } at line", body.count("\n", 0, i) + 1); ok = False; depth = 0
if depth:
    print("unbalanced braces, depth", depth); ok = False

labels = set(re.findall(r"\\label\{([^}]+)\}", body))
for r in set(re.findall(r"\\ref\{([^}]+)\}", body)) - labels:
    print("undefined ref:", r); ok = False
bib = open("refs.bib", encoding="utf8").read() if len(sys.argv) < 3 else open(sys.argv[2], encoding="utf8").read()
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
for c in set(k.strip() for grp in re.findall(r"\\cite\{([^}]+)\}", body) for k in grp.split(",")) - keys:
    print("undefined cite:", c); ok = False
for bad in re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", src):
    print("control character", repr(bad)); ok = False
print("OK" if ok else "problems found")
sys.exit(0 if ok else 1)
