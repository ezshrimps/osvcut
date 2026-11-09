"""Find math-only commands used outside $...$ in main.tex (the CI build fails with "Missing $ inserted").

    python check_math.py [main.tex]
Checks prose lines only (skips tikz/pgfplots, tables' math is inline anyway); prints line, column and context.
"""
import re
import sys

MATH_ONLY = re.compile(r"\\(times|approx|pm|circ|le|ge|leq|geq|tfrac|frac|cdot|lambda|alpha|beta|kappa|theta|phi|nabla|bar|mathrm)\b|\^")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "main.tex"
    lines = open(path, encoding="utf8").read().split("\n")
    bad = 0
    in_env = False
    for n, line in enumerate(lines, 1):
        if re.search(r"\\begin\{(tikzpicture|axis|equation|aligned|align)\}", line):
            in_env = True
        if in_env:
            if re.search(r"\\end\{(tikzpicture|equation|align)\}", line):
                in_env = False
            continue
        if line.lstrip().startswith("%"):
            continue
        # strip escaped dollars, then walk math / text segments
        parts = re.split(r"(?<!\\)\$", line)
        for k, seg in enumerate(parts):
            if k % 2 == 1:
                continue
            for m in MATH_ONLY.finditer(seg):
                bad += 1
                print(f"{path}:{n}: '{m.group(0)}' outside math: ...{seg[max(0, m.start() - 50):m.end() + 30]}...")
    print("OK" if not bad else f"{bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
