"""Unit tests for the shared file-excerpt helper."""
from __future__ import annotations

import re

import pytest

from app.agents._shared.excerpts import (
    error_line_anchors,
    excerpt_text,
    gap_marker,
    read_text,
)

_MARKER_RE = re.compile(r"^\.\.\. \[lines? (\d+)(?:-(\d+))? not shown\] \.\.\.$")


def _big_file(n: int = 1000, overrides: dict[int, str] | None = None) -> str:
    """`n` lines of filler; `overrides` maps 1-based line numbers to text."""
    overrides = overrides or {}
    lines = [overrides.get(i, f"    filler_{i} = {i}") for i in range(1, n + 1)]
    return "\n".join(lines) + "\n"


def _shown(text: str) -> list[str]:
    return [line for line in text.split("\n") if not _MARKER_RE.match(line)]


def _rebuild(excerpt: str) -> dict[int, str]:
    """Map 1-based line number -> text for every shown line, by walking the
    markers. Lets tests check that shown lines sit where they belong."""
    out: dict[int, str] = {}
    n = 1
    for line in excerpt.split("\n"):
        m = _MARKER_RE.match(line)
        if m:
            start = int(m.group(1))
            end = int(m.group(2) or m.group(1))
            assert start == n, f"marker starts at {start}, expected {n}"
            n = end + 1
        else:
            out[n] = line
            n += 1
    return out


@pytest.mark.unit
def test_small_file_is_returned_whole():
    text = "import os\n\ndef f():\n    return 1\n"
    ex = excerpt_text(text, keywords=["f"], max_chars=10_000)
    assert ex.text == text.rstrip("\n")
    assert ex.truncated is False
    assert ex.shown_lines == ex.total_lines == 4


@pytest.mark.unit
def test_match_deep_in_file_is_shown_with_head_and_markers():
    text = _big_file(overrides={
        1: "import os",
        700: "def compute_total(items):",
        701: "    return sum(i.price for i in items)  # BUG",
    })
    ex = excerpt_text(text, keywords=["compute_total"], max_chars=2_000)
    assert ex.truncated is True
    assert len(ex.text) <= 2_000
    shown = _rebuild(ex.text)
    assert shown[1] == "import os"
    assert shown[700] == "def compute_total(items):"
    assert shown[701] == "    return sum(i.price for i in items)  # BUG"
    assert "not shown" in ex.text


@pytest.mark.unit
def test_shown_lines_are_verbatim_and_in_place():
    text = _big_file(overrides={500: "    special_marker()"})
    ex = excerpt_text(text, keywords=["special_marker"], max_chars=1_500)
    original = text.split("\n")
    for n, line in _rebuild(ex.text).items():
        assert original[n - 1] == line


@pytest.mark.unit
def test_rare_keyword_beats_common_one():
    overrides = {i: f"    user = load_user({i})" for i in range(10, 900, 5)}
    overrides[950] = "    reset_token = user.make_reset_token()"
    text = _big_file(overrides=overrides)
    ex = excerpt_text(
        text, keywords=["user", "reset_token"], max_chars=1_200,
    )
    assert "reset_token = user.make_reset_token()" in ex.text


@pytest.mark.unit
def test_anchor_line_is_shown_without_any_keyword():
    text = _big_file(overrides={812: "    x = (1 +"})
    ex = excerpt_text(text, anchor_lines=[812], max_chars=1_500)
    assert _rebuild(ex.text)[812] == "    x = (1 +"


@pytest.mark.unit
def test_no_hits_falls_back_to_head():
    text = _big_file()
    ex = excerpt_text(text, keywords=["nothing_matches_this"], max_chars=600)
    lines = ex.text.split("\n")
    assert lines[0] == "    filler_1 = 1"
    assert _MARKER_RE.match(lines[-1])
    assert len(ex.text) <= 600


@pytest.mark.unit
def test_enclosing_headers_come_along():
    overrides = {
        400: "class Cart:",
        401: "    def total(self):",
        402: "        n = 0",
    }
    for i in range(403, 520):
        overrides[i] = f"        n += {i}"
    overrides[515] = "        n -= discount_rate"
    text = _big_file(overrides=overrides)
    ex = excerpt_text(text, keywords=["discount_rate"], max_chars=1_200)
    shown = _rebuild(ex.text)
    assert shown[515] == "        n -= discount_rate"
    assert shown[401] == "    def total(self):"
    assert shown[400] == "class Cart:"


@pytest.mark.unit
@pytest.mark.parametrize("budget", [200, 500, 900, 1_700, 4_000, 9_000])
def test_char_budget_is_never_exceeded(budget):
    overrides = {i: f"    hit_word_{i % 7} = {i}" for i in range(1, 1000, 37)}
    text = _big_file(overrides=overrides)
    ex = excerpt_text(text, keywords=["hit_word_3", "hit_word_5"], max_chars=budget)
    assert len(ex.text) <= budget


@pytest.mark.unit
def test_line_budget_counts_markers():
    text = _big_file(overrides={300: "  needle()", 600: "  needle()"})
    ex = excerpt_text(text, keywords=["needle"], max_lines=20, head_lines=4)
    assert len(ex.text.split("\n")) <= 20
    shown = _rebuild(ex.text)
    assert shown[300] == "  needle()"


@pytest.mark.unit
def test_tiny_gaps_are_filled_not_marked():
    # Two hits 3 lines apart with no context: the 2-line gap between their
    # windows is shown instead of a marker.
    text = _big_file(overrides={500: "  alpha_hit()", 503: "  beta_hit()"})
    ex = excerpt_text(
        text, keywords=["alpha_hit", "beta_hit"], max_chars=5_000,
        head_lines=0, before=0, after=0,
    )
    shown = _rebuild(ex.text)
    assert {500, 501, 502, 503} <= set(shown)


@pytest.mark.unit
def test_read_text_normalizes_crlf(tmp_path):
    # The applier matches `search` text against a universal-newline read, so
    # excerpts must show the file the same way.
    p = tmp_path / "win.py"
    p.write_bytes(b"a = 1\r\nb = 2\r\n")
    ex = excerpt_text(read_text(p), max_chars=100)
    assert ex.text == "a = 1\nb = 2"


@pytest.mark.unit
def test_empty_text():
    ex = excerpt_text("", keywords=["x"], max_chars=100)
    assert ex.text == ""
    assert ex.truncated is False


@pytest.mark.unit
def test_gap_marker_wording():
    assert gap_marker(5, 5) == "... [line 5 not shown] ..."
    assert gap_marker(5, 9) == "... [lines 5-9 not shown] ..."


# ---------------------------------------------------------------------------
# error_line_anchors
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_anchors_from_python_traceback():
    err = 'Traceback:\n  File "pkg/mod.py", line 523\n    x = (\nSyntaxError: ...'
    assert error_line_anchors(err, "pkg/mod.py") == [523]


@pytest.mark.unit
def test_anchors_from_colon_style_and_absolute_paths():
    err = (
        "/workspace/repo/src/app.ts:41:7 - error TS2304\n"
        "pkg/mod.py:12: in <module>\n"
    )
    assert error_line_anchors(err, "src/app.ts") == [41]
    assert error_line_anchors(err, "pkg/mod.py") == [12]


@pytest.mark.unit
def test_anchors_from_subdir_relative_paths():
    # Test Runner works inside a monorepo sub-dir, so errors say
    # `pkg/mod.py` while the candidate path is `backend/pkg/mod.py`.
    err = 'File "pkg/mod.py", line 7'
    assert error_line_anchors(err, "backend/pkg/mod.py") == [7]


@pytest.mark.unit
def test_bare_basename_does_not_match_nested_file():
    err = 'File "__init__.py", line 3'
    assert error_line_anchors(err, "pkg/sub/__init__.py") == []
    assert error_line_anchors(err, "__init__.py") == [3]


@pytest.mark.unit
def test_anchors_ignore_other_files_and_dedupe():
    err = 'File "a.py", line 3\nFile "b.py", line 9\na.py:3: boom'
    assert error_line_anchors(err, "a.py") == [3]
    assert error_line_anchors("", "a.py") == []
