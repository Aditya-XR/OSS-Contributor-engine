"""Code Explorer — given an issue + a cloned workspace, picks the files most
likely to need editing to resolve the issue.

Two-stage:
  1. Deterministic pre-rank over the whole workspace (`scanner.py`).
  2. LLM rerank: hand the top-K (with content snippets) to the LLM, ask it
     to pick the final N with confidences and one-line rationales.

The LLM step is optional. When called with `router=None` we skip it and
return the deterministic top-N — handy for tests, offline runs, and
sanity-checking the scoring layer in isolation.
"""
from __future__ import annotations

import time
from pathlib import Path

import structlog
from sqlalchemy.orm import Session

from app.agents.explorer.scanner import read_snippet, scan_workspace
from app.agents.explorer.schemas import (
    ExplorationResult,
    FileCandidate,
    ScannedFile,
    _LLMRankOutput,
)
from app.agents.explorer.scorer import (
    extract_keywords,
    extract_referenced_paths,
)
from app.llm import call_llm

log = structlog.get_logger(__name__)

# Prompt-size budget. The previous 20×80 produced ~14k-token prompts, which
# blow past free-tier provider TPM limits (Groq 8b caps at 6k TPM, 70b ~12k)
# — so when Gemini is down, every fallback rejects the request as "too
# large". 12×50 lands around ~5-6k tokens, which the fallbacks can actually
# accept while still giving the LLM real per-file context to rerank on.
DEFAULT_TOP_K_SCAN = 12         # how many files survive the deterministic step
DEFAULT_MAX_CANDIDATES = 8      # how many final candidates we return
DEFAULT_SNIPPET_LINES = 50      # per-file excerpt we hand the LLM, in lines


SYSTEM_PROMPT = """You are mapping a GitHub issue to the specific source files
a developer would need to read or modify to resolve it.

You're given:
- The issue (title + body + labels)
- A pre-ranked candidate list (the deterministic pre-rank used path and
  keyword heuristics — your job is to confirm or correct it using the
  snippet content). A long file's snippet shows its first lines plus the
  lines that mention the issue's keywords; `... [lines 120-480 not shown] ...`
  marks code that was skipped.

Return ONE JSON object:
  "candidates": up to {max_candidates} entries, ordered by confidence desc.
    Each entry:
      "path":       must match a path from the candidate list exactly.
      "confidence": 0.0-1.0 — how sure are you this file matters here.
      "reason":    ≤30 words explaining WHY (cite something concrete from
                   the snippet, not just the path).

Rules:
- Paths must be verbatim from the input list. Don't invent or shorten.
- If a snippet clearly has nothing to do with the issue, drop it — don't
  pad to fill the quota.
- Lower confidence (0.3-0.5) is fine and useful for "probably read this".
- No markdown fences. JSON only."""


async def explore(
    *,
    repo: str,
    repo_path: Path,
    issue_title: str,
    issue_body: str | None,
    issue_labels: list[str] | None = None,
    router=None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    top_k_scan: int = DEFAULT_TOP_K_SCAN,
    snippet_lines: int = DEFAULT_SNIPPET_LINES,
    investigation_id: str | None = None,
    user_id: int | None = None,
    session: Session | None = None,
) -> ExplorationResult:
    """Top-level entry point.

    Args:
        repo: "owner/name" — only used for the result + logs.
        repo_path: cloned repo root on disk. Must contain the project.
        issue_title, issue_body, issue_labels: the GitHub issue.
        router: LiteLLM router. None to skip the LLM rerank step.
        max_candidates, top_k_scan, snippet_lines: tuning knobs (see module
            constants for sane defaults).
        investigation_id, user_id, session: passed through to `call_llm` for
            agent-run telemetry.

    Returns:
        ExplorationResult — `.candidates` is sorted by confidence desc and
        capped at `max_candidates`.
    """
    started = time.monotonic()
    full_text = " ".join([issue_title, issue_body or ""])
    keywords = extract_keywords(full_text)
    references = extract_referenced_paths(issue_body)

    log.info(
        "explorer_start",
        repo=repo,
        repo_path=str(repo_path),
        keyword_count=len(keywords),
        ref_count=len(references),
    )

    scanned, examined = scan_workspace(
        repo_path,
        keywords=keywords,
        references=references,
        issue_text=full_text,
        top_k=top_k_scan,
    )

    if not scanned:
        log.info("explorer_no_candidates", repo=repo)
        return ExplorationResult(
            repo=repo,
            issue_title=issue_title,
            candidates=[],
            files_scanned=examined,
            files_pre_ranked=0,
            used_llm_rerank=False,
            elapsed_s=time.monotonic() - started,
        )

    if router is None:
        # Skip LLM — return deterministic top-N as-is.
        final = [
            FileCandidate(
                path=s.path,
                confidence=s.score,
                reason="",
                signals=s.signals,
                size_bytes=s.size_bytes,
            )
            for s in scanned[:max_candidates]
        ]
        log.info("explorer_done_no_llm", repo=repo, returned=len(final))
        return ExplorationResult(
            repo=repo,
            issue_title=issue_title,
            candidates=final,
            files_scanned=examined,
            files_pre_ranked=len(scanned),
            used_llm_rerank=False,
            elapsed_s=time.monotonic() - started,
        )

    # ---- LLM rerank ----
    user_msg = _build_user_message(
        repo=repo,
        repo_path=repo_path,
        issue_title=issue_title,
        issue_body=issue_body,
        issue_labels=issue_labels or [],
        scanned=scanned,
        snippet_lines=snippet_lines,
        max_candidates=max_candidates,
        keywords=keywords,
    )
    _result, parsed = call_llm(
        router,
        messages=[
            {"role": "system",
             "content": SYSTEM_PROMPT.format(max_candidates=max_candidates)},
            {"role": "user", "content": user_msg},
        ],
        agent_name="code_explorer",
        response_model=_LLMRankOutput,
        investigation_id=investigation_id,
        user_id=user_id,
        session=session,
        max_tokens=1500,
    )

    final: list[FileCandidate]
    if parsed is None:
        log.warning("explorer_llm_parse_failed", repo=repo)
        # Fall back to deterministic rather than returning nothing.
        final = [
            FileCandidate(
                path=s.path, confidence=s.score, reason="",
                signals=s.signals, size_bytes=s.size_bytes,
            )
            for s in scanned[:max_candidates]
        ]
    else:
        signals_by_path = {s.path: s for s in scanned}
        final = []
        for item in parsed.candidates:
            scanned_match = signals_by_path.get(item.path)
            if scanned_match is None:
                # Hallucinated path — drop it (this is exactly why we tell
                # the LLM "paths must be verbatim from the input list").
                log.warning(
                    "explorer_llm_hallucinated_path",
                    repo=repo, path=item.path,
                )
                continue
            final.append(FileCandidate(
                path=item.path,
                confidence=item.confidence,
                reason=item.reason.strip(),
                signals=scanned_match.signals,
                size_bytes=scanned_match.size_bytes,
            ))
        final.sort(key=lambda c: c.confidence, reverse=True)
        final = final[:max_candidates]

    elapsed = time.monotonic() - started
    log.info(
        "explorer_done",
        repo=repo,
        examined=examined,
        scored=len(scanned),
        returned=len(final),
        elapsed_s=round(elapsed, 3),
    )
    return ExplorationResult(
        repo=repo,
        issue_title=issue_title,
        candidates=final,
        files_scanned=examined,
        files_pre_ranked=len(scanned),
        used_llm_rerank=True,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

# Truncate issue body to keep the prompt size sane. Long bodies (template
# fillouts) blow past context windows otherwise.
_MAX_BODY_CHARS = 4_000


def _build_user_message(
    *,
    repo: str,
    repo_path: Path,
    issue_title: str,
    issue_body: str | None,
    issue_labels: list[str],
    scanned: list[ScannedFile],
    snippet_lines: int,
    max_candidates: int,
    keywords: list[str] | None = None,
) -> str:
    body_excerpt = (issue_body or "").strip()
    if len(body_excerpt) > _MAX_BODY_CHARS:
        body_excerpt = body_excerpt[:_MAX_BODY_CHARS] + "\n…(truncated)"

    labels_line = ", ".join(issue_labels) if issue_labels else "(none)"

    file_blocks: list[str] = []
    for s in scanned:
        snippet = read_snippet(
            repo_path, s.path, max_lines=snippet_lines, keywords=keywords,
        )
        signals_txt = ", ".join(s.signals) if s.signals else "(none)"
        file_blocks.append(
            f"--- {s.path}  "
            f"(size={s.size_bytes}B, prerank={s.score:.2f}, signals=[{signals_txt}])\n"
            f"{snippet}\n",
        )

    return (
        f"REPO: {repo}\n"
        f"TITLE: {issue_title}\n"
        f"LABELS: {labels_line}\n"
        f"BODY:\n{body_excerpt}\n\n"
        f"CANDIDATE FILES (you may return up to {max_candidates}):\n\n"
        + "\n".join(file_blocks)
    )
