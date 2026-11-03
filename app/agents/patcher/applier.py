"""Apply CodeEdits to a workspace and capture the resulting diff.

Pure-ish module: I/O is just file reads/writes + a single `git diff` call.
No LLM, no network. Trivially unit-testable against a tmp_path fixture.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import structlog

from app.agents.patcher.exceptions import EditApplyError
from app.agents.patcher.schemas import AppliedEdit, CodeEdit

log = structlog.get_logger(__name__)


def apply_edits(repo_path: Path, edits: list[CodeEdit]) -> list[AppliedEdit]:
    """Apply each edit in order. Raise EditApplyError on the first failure.

    If a later edit in the batch fails, any earlier edits in the same call
    are rolled back so the workspace isn't left half-patched.

    Failure modes intentionally surfaced:
      - search text not found
      - search text appears more than once (ambiguous — refuse rather than
        guess wrong)
      - search non-empty but file doesn't exist
      - search empty but file already exists (would clobber)
      - escape attempts (paths containing `..`, `.git`, or absolute paths)
    """
    applied: list[AppliedEdit] = []
    # Map each touched path to its original bytes before the first edit in
    # this batch (or None if the file did not exist before this batch).
    snapshots: dict[Path, bytes | None] = {}
    repo_root = repo_path.resolve()

    try:
        for edit in edits:
            rel = _validate_rel_path(edit.path)
            full = (repo_path / rel).resolve()
            # Defense in depth: re-check after resolve that we're still inside
            # the workspace. Symlinks + ../ traversal could otherwise escape.
            if not _is_inside(full, repo_root):
                raise EditApplyError(
                    f"path escapes workspace: {edit.path}", path=edit.path,
                )

            if full not in snapshots:
                snapshots[full] = full.read_bytes() if full.is_file() else None

            if not edit.search:
                applied.append(_apply_new_file(full, edit))
            else:
                applied.append(_apply_replace(full, edit))
    except EditApplyError:
        for path, original_bytes in reversed(list(snapshots.items())):
            try:
                if original_bytes is None:
                    if path.is_file():
                        path.unlink()
                else:
                    path.write_bytes(original_bytes)
            except OSError as rollback_err:
                log.warning(
                    "patcher_rollback_failed",
                    path=str(path),
                    error=str(rollback_err),
                )
        raise

    return applied


# ---------------------------------------------------------------------------
# Per-edit handlers
# ---------------------------------------------------------------------------

def _apply_new_file(full: Path, edit: CodeEdit) -> AppliedEdit:
    if full.exists():
        raise EditApplyError(
            f"file already exists; refusing to clobber via empty-search edit: "
            f"{edit.path}",
            path=edit.path,
        )
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(edit.replace, encoding="utf-8", newline="\n")
    return AppliedEdit(
        path=edit.path,
        explanation=edit.explanation,
        new_file=True,
        bytes_added=len(edit.replace.encode("utf-8")),
        bytes_removed=0,
    )


def _apply_replace(full: Path, edit: CodeEdit) -> AppliedEdit:
    if not full.exists():
        raise EditApplyError(
            f"file not found: {edit.path}", path=edit.path,
        )
    try:
        # Universal-newline read, so the LLM's `\n`-only search text still
        # matches CRLF files (the prompt shows it the same normalized view).
        with full.open("r", encoding="utf-8") as f:
            original = f.read()
            # "\n" / "\r\n" / "\r" for a consistent file, a tuple if mixed,
            # None if the file has no line breaks at all.
            eol = f.newlines
    except UnicodeDecodeError as e:
        raise EditApplyError(
            f"file not utf-8: {edit.path} ({e})", path=edit.path,
        ) from e

    count = original.count(edit.search)
    if count == 0:
        raise EditApplyError(
            f"search text not found in {edit.path}. "
            f"(LLM hallucinated the snippet — add more surrounding context "
            f"and retry.)",
            path=edit.path,
        )
    if count > 1:
        raise EditApplyError(
            f"search text matches {count} places in {edit.path}; "
            f"refusing to pick one. Add more surrounding context.",
            path=edit.path,
        )

    new_text = original.replace(edit.search, edit.replace, 1)
    # Write back with the file's own line ending. Always writing "\n" turned
    # every line of a CRLF file into a change, so a one-line fix became a
    # whole-file rewrite in the captured diff (and in the pushed PR).
    newline = eol if isinstance(eol, str) else "\n"
    full.write_text(new_text, encoding="utf-8", newline=newline)
    return AppliedEdit(
        path=edit.path,
        explanation=edit.explanation,
        new_file=False,
        bytes_added=max(0, len(new_text) - len(original)),
        bytes_removed=max(0, len(original) - len(new_text)),
    )


# ---------------------------------------------------------------------------
# Diff capture
# ---------------------------------------------------------------------------

def capture_diff(repo_path: Path, *, max_bytes: int = 200_000) -> str:
    """Run `git diff` (including new files) in repo_path. Returns the
    unified-diff text, truncated at `max_bytes` with a footer if oversized.

    Requires the directory to be a git repo. Workspaces created via the
    sandbox `Workspace.clone()` always satisfy this.
    """
    try:
        proc = subprocess.run(
            ["git", "diff", "--no-color", "--unified=4", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except FileNotFoundError as e:
        raise EditApplyError("`git` not on PATH; can't capture diff") from e
    except subprocess.TimeoutExpired as e:
        raise EditApplyError("`git diff` timed out") from e

    if proc.returncode != 0:
        # `git diff` exits 0 even with changes — non-zero is unexpected
        # (corrupt repo, not a git dir, etc.).
        raise EditApplyError(
            f"git diff failed ({proc.returncode}): {proc.stderr[:300]}",
        )

    # `git diff HEAD` doesn't include brand-new files until they're staged.
    # Stage everything to a temporary index just for the diff, then capture
    # again. We use `git add -N` (intent-to-add) so new files show up in
    # diff but aren't fully staged — keeps the workspace in a sensible state.
    try:
        proc_add = subprocess.run(
            ["git", "add", "-N", "."],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if proc_add.returncode != 0:
            log.warning("patcher_git_add_n_failed", stderr=proc_add.stderr[:200])
            # Continue with what we had — partial diff is better than no diff.
            diff_text = proc.stdout
        else:
            proc2 = subprocess.run(
                ["git", "diff", "--no-color", "--unified=4", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            diff_text = proc2.stdout if proc2.returncode == 0 else proc.stdout
    except FileNotFoundError as e:
        raise EditApplyError("`git` not on PATH; can't capture diff") from e
    except subprocess.TimeoutExpired as e:
        raise EditApplyError("`git diff` timed out") from e

    if len(diff_text.encode("utf-8")) > max_bytes:
        return (
            diff_text[: max_bytes // 2]
            + f"\n\n... (diff truncated at {max_bytes} bytes)\n"
        )
    return diff_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_rel_path(path: str) -> str:
    """Reject paths that look like escape attempts. Returns the cleaned path."""
    p = path.replace("\\", "/").strip()
    if not p:
        raise EditApplyError("empty path", path=path)
    if p.startswith("/") or (len(p) >= 2 and p[1] == ":"):
        raise EditApplyError(f"absolute paths refused: {path}", path=path)
    parts = p.split("/")
    if ".." in parts:
        raise EditApplyError(f"`..` not allowed in path: {path}", path=path)
    if any(part.lower() == ".git" for part in parts):
        raise EditApplyError(f"`.git` not allowed in path: {path}", path=path)
    return p


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
