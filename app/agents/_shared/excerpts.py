"""Pick the parts of a source file that matter for an issue.

The Code Explorer and the Patch Writer both show file contents to an LLM
under a tight size budget (free-tier provider TPM caps). Showing only the
head of each file spends that budget on imports and licence headers, and
hides everything below the cut. For the Patch Writer that is fatal: it
can't produce an exact `search` string for code it never saw, so a fix
deep inside a large file could never apply.

`excerpt_text` keeps the whole file when it fits. Otherwise it keeps the
file head (imports, module docstring) plus windows around the lines that
mention the issue's keywords, or that an error message points at. Lines are
ranked so a keyword that appears on few lines beats one that appears
everywhere, and each window brings its enclosing `def` / `class` /
`function` lines along. Every skipped stretch becomes ONE marker line:

    ... [lines 120-480 not shown] ...

Shown lines are verbatim, so `search` text copied from an excerpt still
matches the file byte-for-byte.

Pure functions plus a small file-reading helper; no LLM, no network.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# A skipped stretch this short costs about as much as its marker line, and
# showing it keeps the excerpt readable. Shorter gaps are filled in.
_MAX_FILLED_GAP = 2

# Anchors (lines an error message points at) outrank any keyword match.
_ANCHOR_SCORE = 100.0

# How far up to look for the enclosing block header of a matched line, and
# how many nesting levels (method -> class -> ...) to bring along.
_HEADER_LOOKBACK = 400
_MAX_HEADER_LEVELS = 3

# Stop trying new windows after this many matched lines — the budget is
# almost always spent long before.
_MAX_WINDOWS = 60


@dataclass(frozen=True)
class Excerpt:
    """What `excerpt_text` returns."""
    text: str
    shown_lines: int
    total_lines: int

    @property
    def truncated(self) -> bool:
        return self.shown_lines < self.total_lines


def gap_marker(start: int, end: int) -> str:
    """Placeholder for skipped lines `start`..`end` (1-based, inclusive)."""
    if start == end:
        return f"... [line {start} not shown] ..."
    return f"... [lines {start}-{end} not shown] ..."


def read_text(path: Path) -> str:
    """Read a file the way the excerpts expect: UTF-8 with replacement for
    bad bytes, universal newlines (so CRLF files come back with `\\n`, which
    is also how the applier matches `search` text). Raises OSError."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read()


def excerpt_text(
    text: str,
    *,
    keywords: Sequence[str] = (),
    anchor_lines: Iterable[int] = (),
    max_chars: int | None = None,
    max_lines: int | None = None,
    head_lines: int = 15,
    before: int = 8,
    after: int = 16,
) -> Excerpt:
    """Cut `text` down to `max_chars` characters and/or `max_lines` lines.

    Args:
        keywords: lowercase issue keywords (see `extract_keywords`); a line
            matches a keyword when it contains it, case-insensitively.
        anchor_lines: 1-based line numbers to keep at all costs (e.g. the
            line a failed attempt's SyntaxError points at).
        max_chars / max_lines: the budget for the rendered excerpt, marker
            lines included. None means unlimited.
        head_lines: how much of the file head to keep when there's room.
        before / after: context lines kept around each matched line.

    With no keyword or anchor hits this falls back to the head of the file,
    which is what the agents showed before excerpts existed.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # a trailing newline isn't an extra line
    total = len(lines)
    if total == 0:
        return Excerpt(text="", shown_lines=0, total_lines=0)

    def fits(included: list[bool]) -> bool:
        rendered = _render(lines, included)
        if max_lines is not None and len(rendered) > max_lines:
            return False
        if max_chars is not None:
            chars = sum(len(s) for s in rendered) + len(rendered) - 1
            if chars > max_chars:
                return False
        return True

    everything = [True] * total
    if fits(everything):
        return _finish(lines, everything)

    included = [False] * total

    def try_add(indices: Iterable[int]) -> bool:
        nonlocal included
        candidate = included.copy()
        for i in indices:
            if 0 <= i < total:
                candidate[i] = True
        candidate = _fill_gaps(candidate)
        if fits(candidate):
            included = candidate
            return True
        return False

    # File head, one line at a time so a tiny budget still gets what fits.
    for i in range(min(head_lines, total)):
        if not try_add([i]):
            break

    ranked = _rank_lines(lines, keywords, anchor_lines)
    if not ranked:
        # Nothing matched: keep filling from the top, like the old
        # head-of-file behaviour.
        for i in range(total):
            if not included[i] and not try_add([i]):
                break
        return _finish(lines, included)

    for i in ranked[:_MAX_WINDOWS]:
        headers = _enclosing_headers(lines, i)
        wide = [*headers, *range(i - before, i + after + 1)]
        narrow = [*headers, *range(i - 2, i + 5)]
        if not try_add(wide) and not try_add(narrow):
            try_add([*headers, i])

    return _finish(lines, included)


def error_line_anchors(error_text: str, path: str) -> list[int]:
    """Line numbers (1-based) that `error_text` reports for the file `path`.

    Understands Python tracebacks / py_compile (`File "pkg/mod.py", line 12`)
    and the `pkg/mod.py:12` / `src/x.ts:12:5` style most test runners and
    compilers print. Paths in error output may be absolute (`/workspace/...`)
    or relative to a sub-directory, so either side may be a suffix of the
    other — as long as the shorter one still has a directory in it, or the
    two are equal. A bare `__init__.py` must not match every package.
    """
    if not error_text or not path:
        return []
    want = path.replace("\\", "/").removeprefix("./")
    anchors: list[int] = []
    for pattern in (_TRACEBACK_RE, _COLON_LINE_RE):
        for raw_path, raw_line in pattern.findall(error_text):
            got = raw_path.replace("\\", "/")
            if _same_file(got, want):
                n = int(raw_line)
                if n > 0 and n not in anchors:
                    anchors.append(n)
    return anchors


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_TRACEBACK_RE = re.compile(r'File "([^"]+)", line (\d+)')
_COLON_LINE_RE = re.compile(r"([\w./\\-]+\.[A-Za-z0-9]+):(\d+)")


def _same_file(got: str, want: str) -> bool:
    got = got.removeprefix("./")
    if got == want:
        return True
    if got.endswith("/" + want):
        return True
    return "/" in got and want.endswith("/" + got)


def _rank_lines(
    lines: list[str], keywords: Sequence[str], anchor_lines: Iterable[int],
) -> list[int]:
    """Indices of lines worth showing, best first.

    A keyword's weight is 1 / (number of lines containing it), so a rare
    identifier like `reset_token` outranks a word like `user` that shows up
    on every other line. Ties go to the earlier line.
    """
    lowered = [line.lower() for line in lines]
    terms = []
    for kw in keywords:
        k = kw.lower()
        if len(k) >= 3 and k not in terms:
            terms.append(k)

    hits_by_term: dict[str, list[int]] = {}
    for k in terms:
        idx = [i for i, line in enumerate(lowered) if k in line]
        if idx:
            hits_by_term[k] = idx

    scores: dict[int, float] = {}
    for idx in hits_by_term.values():
        weight = 1.0 / len(idx)
        for i in idx:
            scores[i] = scores.get(i, 0.0) + weight
    for n in anchor_lines:
        i = n - 1
        if 0 <= i < len(lines):
            scores[i] = scores.get(i, 0.0) + _ANCHOR_SCORE

    return sorted(scores, key=lambda i: (-scores[i], i))


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _enclosing_headers(lines: list[str], i: int) -> list[int]:
    """The nearest less-indented non-blank lines above `i`, innermost first —
    usually the `def` / `class` / `function` / `if` the line lives in."""
    if not lines[i].strip():
        return []
    level = _indent(lines[i])
    headers: list[int] = []
    j = i - 1
    stop = max(0, i - _HEADER_LOOKBACK)
    while level > 0 and j >= stop and len(headers) < _MAX_HEADER_LEVELS:
        if lines[j].strip() and _indent(lines[j]) < level:
            headers.append(j)
            level = _indent(lines[j])
        j -= 1
    return headers


def _fill_gaps(included: list[bool]) -> list[bool]:
    """Show skipped stretches of `_MAX_FILLED_GAP` lines or fewer, but only
    between shown lines — the file's head and tail stay as they are."""
    out = included.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        if i > 0 and j < n and j - i <= _MAX_FILLED_GAP:
            for k in range(i, j):
                out[k] = True
        i = j
    return out


def _render(lines: list[str], included: list[bool]) -> list[str]:
    out: list[str] = []
    n = len(lines)
    i = 0
    while i < n:
        if included[i]:
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < n and not included[j]:
            j += 1
        out.append(gap_marker(i + 1, j))
        i = j
    return out


def _finish(lines: list[str], included: list[bool]) -> Excerpt:
    if not any(included):
        return Excerpt(text="", shown_lines=0, total_lines=len(lines))
    return Excerpt(
        text="\n".join(_render(lines, included)),
        shown_lines=sum(included),
        total_lines=len(lines),
    )
