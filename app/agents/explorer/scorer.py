"""Pure scoring functions for the Code Explorer.

No I/O, no LLM, no GitHub — everything in here is a deterministic function
of (issue text, file path). That makes the whole layer trivially testable
and lets us treat scoring weights as a tuning surface independent of the
rest of the agent.

Score components (all 0-1):
  - referenced_score : 1.0 if the file path appears verbatim in the issue
  - filename_score   : keyword overlap with the *filename*
  - dir_score        : keyword overlap with the *directory* components
  - test_affinity    : nudge toward / away from test files based on whether
                       the issue is bug-flavored or feature-flavored

Combined into a single 0-1 score by `combine_scores`.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

# Identifier-ish tokens. Catches snake_case, camelCase, dotted (foo.bar),
# slashed (src/foo/bar.py), and backticked bits. Length ≥3 to skip noise.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-./]{2,}")

# Words too generic to score anything (would match every file).
_STOPWORDS: frozenset[str] = frozenset({
    "and", "the", "for", "with", "from", "this", "that", "when", "then",
    "you", "your", "our", "have", "has", "had", "but", "not", "are", "was",
    "were", "will", "would", "could", "should", "can", "may", "might",
    "issue", "bug", "feature", "request", "fix", "fixes", "fixed", "add",
    "adds", "added", "use", "using", "used", "new", "old", "any", "all",
    "some", "into", "out", "over", "more", "less", "very", "much", "many",
    "doesn", "isn", "wasn", "don", "won", "github", "com", "org", "https",
    "http", "www", "see", "also", "like", "way", "ways", "make", "made",
    "get", "got", "set", "put", "take",
})

# A path-shaped token "looks like" `something/with.slashes` or
# `something/with.multiple.dots.ts` (including leading-dot dirs like `.github/`).
_PATHISH_RE = re.compile(
    r"(?:\.?[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*[/\\])+"
    r"[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*",
)

# A filename-shaped token "looks like" `something.ext` or `foo.test.ts`.
_FILENAME_RE = re.compile(
    r"[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*\.[A-Za-z0-9]{1,6}",
)


def extract_keywords(text: str | None, *, limit: int = 40) -> list[str]:
    """Pull lowercase keyword candidates from issue text.

    Splits camelCase into halves so `UserController` also matches files
    named `user` or `controller`.
    """
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(text):
        low = raw.lower()
        if low in _STOPWORDS:
            continue
        if low not in seen:
            seen.add(low)
            found.append(low)
        # Split camelCase / PascalCase pieces.
        for piece in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z])", raw):
            p = piece.lower()
            if len(p) >= 3 and p not in _STOPWORDS and p not in seen:
                seen.add(p)
                found.append(p)
        if len(found) >= limit:
            break
    return found[:limit]


def extract_referenced_paths(text: str | None) -> list[str]:
    """Find file/path tokens the issue body explicitly mentions.

    A direct mention is the strongest signal we have. We collect both
    path-shaped tokens (`src/foo/bar.py`) and bare-filename tokens
    (`bar.py`) — the scorer treats the former as a verbatim match and the
    latter as a basename match.
    """
    if not text:
        return []
    refs: list[str] = []
    seen: set[str] = set()
    for m in _PATHISH_RE.findall(text) + _FILENAME_RE.findall(text):
        # Normalize Windows-style slashes; keep the rest of the casing alone
        # since case-sensitive matching against the workspace is meaningful
        # on case-sensitive filesystems.
        norm = m.replace("\\", "/")
        if norm not in seen:
            seen.add(norm)
            refs.append(norm)
    return refs


# ---------------------------------------------------------------------------
# Per-component scorers
# ---------------------------------------------------------------------------

def referenced_score(path: str, references: list[str]) -> tuple[float, list[str]]:
    """1.0 if the file is referenced by full path; 0.7 if just by basename."""
    if not references:
        return 0.0, []
    basename = path.rsplit("/", 1)[-1]
    matched: list[str] = []
    score = 0.0
    for ref in references:
        # A ref is "full-path-ish" only when it contains a slash. Otherwise
        # it's a bare filename, which is a real but weaker signal — lots of
        # files might be named `handlers.py`; the issue says SOME
        # `handlers.py` matters, not necessarily this one.
        has_dir = "/" in ref
        if has_dir and (ref == path or path.endswith("/" + ref)):
            matched.append(f"ref:{ref}")
            score = max(score, 1.0)
        elif not has_dir and ref == basename:
            matched.append(f"refname:{ref}")
            score = max(score, 0.7)
    return score, matched


def filename_score(path: str, keywords: list[str]) -> tuple[float, list[str]]:
    """How many keywords appear in the *filename* (sans extension)?

    Score = (matches / total keywords) capped at 1.0, weighted to favor
    longer-keyword matches (more specific).
    """
    if not keywords:
        return 0.0, []
    basename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    matched: list[str] = []
    weight = 0.0
    for kw in keywords:
        if kw in basename:
            matched.append(f"keyword:{kw}")
            # Longer keyword = stronger signal (4+ chars get full weight)
            weight += min(1.0, len(kw) / 4)
    if not matched:
        return 0.0, []
    # Normalize by keyword count so a 50-keyword issue doesn't always saturate.
    score = min(1.0, weight / max(3, len(keywords) / 3))
    return score, matched


def dir_score(path: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Lighter version of `filename_score` over the directory components."""
    if not keywords or "/" not in path:
        return 0.0, []
    dirs = path.rsplit("/", 1)[0].lower().split("/")
    matched: list[str] = []
    for d in dirs:
        if not d:
            continue
        for kw in keywords:
            if kw in d:
                matched.append(f"dir:{d}")
                break
    if not matched:
        return 0.0, []
    return min(1.0, 0.3 * len(matched)), matched


_TEST_DIR_RE = re.compile(r"(?:^|/)(tests?|__tests__|spec|specs)/")
_TEST_FILE_RE = re.compile(r"(?:^|/)(test_|_test\.|\.test\.|\.spec\.)")
_BUG_HINTS = ("bug", "crash", "error", "broken", "regression", "wrong",
              "incorrect", "fail", "exception", "panic")


def test_affinity(path: str, issue_text: str) -> tuple[float, list[str]]:
    """Small nudge toward test files when the issue smells like a bug.

    Reasoning: bug reports often imply "and there should be a test for this".
    Feature requests usually need source changes more than test changes.
    """
    is_test = bool(_TEST_DIR_RE.search(path) or _TEST_FILE_RE.search(path))
    bug_flavored = any(h in (issue_text or "").lower() for h in _BUG_HINTS)
    if is_test and bug_flavored:
        return 0.1, ["test:bug-related"]
    if is_test and not bug_flavored:
        return -0.05, []  # tiny demerit for feature requests
    return 0.0, []


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------

# Weights chosen by intuition; revisit once we have eval data.
_W_REFERENCED = 0.55
_W_FILENAME = 0.30
_W_DIR = 0.10
_W_TEST = 0.05


def combine_scores(
    *,
    path: str,
    keywords: list[str],
    references: list[str],
    issue_text: str,
) -> tuple[float, list[str]]:
    """Compute the final 0-1 score plus the list of signals that fired."""
    ref_s, ref_sig = referenced_score(path, references)
    fn_s, fn_sig = filename_score(path, keywords)
    dr_s, dr_sig = dir_score(path, keywords)
    tst_s, tst_sig = test_affinity(path, issue_text)

    raw = (
        _W_REFERENCED * ref_s
        + _W_FILENAME * fn_s
        + _W_DIR * dr_s
        + _W_TEST * tst_s
    )
    # Clamp — test_affinity can pull the score slightly negative.
    final = max(0.0, min(1.0, raw))

    signals: list[str] = []
    for batch in (ref_sig, fn_sig, dr_sig, tst_sig):
        for s in batch:
            if s not in signals:
                signals.append(s)
    return final, signals
