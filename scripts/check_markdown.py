"""Check Markdown files against the repo's doc rules: no tabs, and no hard line breaks in prose.

Usage: check_markdown.py FILE...

Each paragraph and list item stays on one source line. A line that continues the one before it (both non-blank, outside
fenced and indented code blocks and tables, and the second not starting a new block) is a hard line break.
"""

import re
import sys
from pathlib import Path

BLOCK_START = re.compile(r"\s*([-*+]\s|\d+[.)]\s|#|>|\||```|~~~|<)")
# A list item's marker; its text (and the blocks inside the item) starts after the marker and its spaces
LIST_ITEM = re.compile(r"( *)([-*+]|\d{1,9}[.)])( {1,4})(?=\S)")
# A code fence: three or more backticks (no backtick in the info string) or tildes, indented at most three spaces past
# the enclosing list item's text (indented further, it is an indented code line)
FENCE = re.compile(r" {0,3}(`{3,}(?=[^`]*$)|~{3,})(.*)")


def indent(line):
    return len(line) - len(line.lstrip(" "))


def problems(path, text):
    # items: text columns of the enclosing list items, innermost last; fence: the open fence and its item's column
    found, fence, previous = [], None, ""
    items: list[int] = []
    for number, line in enumerate(text.splitlines(), 1):
        if "\t" in line:
            found.append(f"{path}:{number}: tab")
        if fence is not None:
            marker, column = fence
            if not line.strip() or indent(line) >= column:
                # Closed only by the same character, at least as long, with nothing after it
                closing = FENCE.match(line[column:])
                if closing and closing[1][0] == marker[0] and len(closing[1]) >= len(marker) and not closing[2].strip():
                    fence = None
                continue
            fence = None  # a line less indented than the fence's list item text ends the item, and the fence with it
        if line.strip():
            # A non-blank line less indented than an item's text ends that item
            items = [c for c in items if c <= indent(line)]
        column = items[-1] if items else 0
        if indent(line) >= column + 4 and not previous.strip():
            previous = ""  # an indented code block: starts after a blank line, continues to its end
            continue
        item = LIST_ITEM.match(line)
        if item:
            items = [c for c in items if c <= len(item[1])] + [item.end()]
            column = item.end()  # the item's text, which may itself open a fence
        opening = FENCE.match(line[column:])
        if opening:
            fence, previous = (opening[1], column), ""
            continue
        continues = previous.strip() and line.strip() and not BLOCK_START.match(line)
        if continues and not previous.lstrip().startswith(("#", "|", "<")):
            found.append(f"{path}:{number}: hard line break (join with line {number - 1})")
        previous = line
    return found


def main(paths):
    found = [p for path in paths for p in problems(path, Path(path).read_text(encoding="utf-8"))]
    print("\n".join(found) or f"Markdown OK: {', '.join(paths)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
