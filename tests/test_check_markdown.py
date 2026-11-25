"""scripts/check_markdown.py, which `make lint` runs on the repo's Markdown."""

import check_markdown

OK = """# Title

One paragraph on one line.

- A list item.
- Another, with `code`.
  - Nested.
1. Numbered.

> A quote.

| a | b |
|---|---|
| 1 | 2 |

```bash
make build
  indented continuation
```

<details>
</details>
"""


def test_clean_document():
    assert check_markdown.problems("x.md", OK) == []


def test_hard_line_break():
    text = "# Title\n\nA sentence that wraps\nonto a second line.\n"
    assert check_markdown.problems("x.md", text) == ["x.md:4: hard line break (join with line 3)"]


def test_wrapped_list_item():
    text = "- A list item that wraps\n  onto a second line.\n"
    assert check_markdown.problems("x.md", text) == ["x.md:2: hard line break (join with line 1)"]


def test_tabs():
    assert check_markdown.problems("x.md", "Text\twith a tab.\n") == ["x.md:1: tab"]
    assert check_markdown.problems("x.md", "```\n\tindented\n```\n") == ["x.md:2: tab"]


def test_tilde_fence():
    assert check_markdown.problems("x.md", "~~~python\nline one\nline two\n~~~\n") == []


def test_longer_fence_contains_shorter():
    text = "````markdown\n```\nline one\nline two\n```\n````\n\nAfter.\n"
    assert check_markdown.problems("x.md", text) == []


def test_fence_closed_only_by_its_own_character():
    text = "~~~\n```\nline one\nline two\n~~~\nA sentence that wraps\nonto a second line.\n"
    assert check_markdown.problems("x.md", text) == ["x.md:7: hard line break (join with line 6)"]


def test_indented_code_block():
    text = "Run:\n\n    make build\n    make check\n\n    make clean\nAfter.\n"
    assert check_markdown.problems("x.md", text) == []


def test_indented_line_continuing_prose_is_a_break():
    text = "A sentence that wraps\n    onto an indented line.\n"
    assert check_markdown.problems("x.md", text) == ["x.md:2: hard line break (join with line 1)"]


def test_fence_indented_four_spaces_is_indented_code():
    text = "Intro.\n\n    ~~~\n\nA sentence that wraps\nonto a second line.\n"
    assert check_markdown.problems("x.md", text) == ["x.md:6: hard line break (join with line 5)"]


def test_closing_fence_indented_four_spaces_stays_in_the_block():
    text = "~~~\n    ~~~\nline one\nline two\n~~~\nA sentence that wraps\nonto a second line.\n"
    assert check_markdown.problems("x.md", text) == ["x.md:7: hard line break (join with line 6)"]


def test_fence_in_list_item():
    text = "- Run:\n  ```bash\n  make build\n  make check\n  ```\n"
    assert check_markdown.problems("x.md", text) == []


def test_fence_aligned_with_numbered_item_text():
    text = "100. Run these commands:\n\n     ```bash\n     make build\n     make check\n     ```\n\nAfter.\n"
    assert check_markdown.problems("x.md", text) == []


def test_fence_in_nested_list_item():
    text = "- Outer.\n  - Inner:\n\n    ~~~\n    line one\n    line two\n    ~~~\n- Next.\n"
    assert check_markdown.problems("x.md", text) == []


def test_fence_indented_four_past_item_text_is_indented_code():
    text = "- Item.\n\n      ~~~\n\n  A sentence that wraps\n  onto a second line.\n"
    assert check_markdown.problems("x.md", text) == ["x.md:6: hard line break (join with line 5)"]


def test_list_ends_at_less_indented_line():
    text = "- Item.\n\nAfter the list.\n\n   ~~~\nline one\nline two\n~~~\n"
    assert check_markdown.problems("x.md", text) == []


def test_fence_on_list_marker_line():
    text = "- ~~~sh\n  make build\n  make check\n  ~~~\n\nA sentence that wraps\nonto a second line.\n"
    assert check_markdown.problems("x.md", text) == ["x.md:7: hard line break (join with line 6)"]


def test_list_marker_in_indented_code():
    text = "Example:\n\n    - example\n      continued\n"
    assert check_markdown.problems("x.md", text) == []


def test_unclosed_fence_in_list_item_ends_with_the_item():
    text = "- Item:\n\n  ~~~\n  code\n\nA sentence that wraps\nonto a second line.\n"
    assert check_markdown.problems("x.md", text) == ["x.md:7: hard line break (join with line 6)"]
