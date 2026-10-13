"""Walks a cloned workspace, scores every interesting file, returns the top-K.

This is the deterministic pre-rank step. The LLM rerank step in
`explorer.py` only ever sees the top-K from here — so the K must be big
enough to give the LLM real choice, and the K must be small enough that
the prompt fits in the model's context window once we attach snippets.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import structlog

from app.agents._shared.excerpts import excerpt_text, read_text
from app.agents._shared.file_filters import (
    MAX_BLOB_SIZE_BYTES,
    is_interesting_path,
)
from app.agents.explorer.schemas import ScannedFile
from app.agents.explorer.scorer import combine_scores

log = structlog.get_logger(__name__)

# Don't walk into these — saves time on real-world repos that ship with
# huge node_modules / .venv / target directories that survived the shallow
# clone (rare but possible if the user committed them).
_PRUNE_DIRS: frozenset[str] = frozenset({
    "node_modules", ".git", ".venv", "venv", "__pycache__",
    "dist", "build", "target", "out", ".next", ".nuxt", "vendor",
    ".tox", ".pytest_cache", ".mypy_cache", ".ruff_cache",
})

# Absolute ceiling on files we'll evaluate even before the top-K cut.
# Real-world repos can have 10k+ source files; we score all of them but
# don't want to walk a billion entries on a pathological input.
_HARD_SCAN_LIMIT = 25_000


def scan_workspace(
    repo_path: Path,
    *,
    keywords: list[str],
    references: list[str],
    issue_text: str,
    top_k: int = 25,
) -> tuple[list[ScannedFile], int]:
    """Walk `repo_path`, score every interesting file, return (top_K, total_scanned).

    `total_scanned` is the count BEFORE the top-K cut, so callers can
    surface it in telemetry ("looked at 1,243 files, considering 25").
    """
    if not repo_path.is_dir():
        raise FileNotFoundError(f"repo_path is not a directory: {repo_path}")

    candidates: list[ScannedFile] = []
    examined = 0

    for root, dirs, files in os.walk(repo_path):
        # Prune in-place so os.walk doesn't recurse into them.
        dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]

        rel_root = os.path.relpath(root, repo_path).replace(os.sep, "/")
        if rel_root == ".":
            rel_root = ""

        for filename in files:
            examined += 1
            if examined > _HARD_SCAN_LIMIT:
                log.warning(
                    "explorer_scan_hard_limit_hit",
                    repo_path=str(repo_path),
                    limit=_HARD_SCAN_LIMIT,
                )
                break

            rel_path = f"{rel_root}/{filename}" if rel_root else filename
            if not is_interesting_path(rel_path):
                continue

            full = Path(root) / filename
            try:
                size = full.stat().st_size
            except OSError:
                continue
            if size == 0 or size > MAX_BLOB_SIZE_BYTES:
                continue

            score, signals = combine_scores(
                path=rel_path,
                keywords=keywords,
                references=references,
                issue_text=issue_text,
            )
            if score <= 0:
                continue
            candidates.append(
                ScannedFile(
                    path=rel_path, size_bytes=size, score=score, signals=signals,
                ),
            )

        if examined > _HARD_SCAN_LIMIT:
            break

    candidates.sort(key=lambda c: c.score, reverse=True)
    log.info(
        "explorer_scan_done",
        examined=examined,
        scored=len(candidates),
        top_k=top_k,
    )
    return candidates[:top_k], examined


def read_snippet(
    repo_path: Path,
    rel_path: str,
    *,
    max_lines: int = 80,
    keywords: Sequence[str] | None = None,
) -> str:
    """At most `max_lines` lines of the file, decoded with replacement on bad
    bytes.

    Without `keywords` that's the head of the file. With them, a file too
    long to show whole becomes an excerpt: its first few lines plus the lines
    that mention the keywords (see `_shared/excerpts.py`), so the reranker
    sees the code the issue is about rather than just the imports.
    """
    full = repo_path / rel_path
    if keywords:
        try:
            text = read_text(full)
        except OSError as e:
            return f"(snippet read failed: {e})"
        return excerpt_text(
            text, keywords=keywords, max_lines=max_lines,
            head_lines=8, before=3, after=8,
        ).text
    try:
        with full.open("r", encoding="utf-8", errors="replace") as f:
            lines: list[str] = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines)
    except OSError as e:
        return f"(snippet read failed: {e})"
