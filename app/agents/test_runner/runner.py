"""Test Runner orchestrator.

Today supports Python only. Two phases run sequentially inside the sandbox:

    Phase 1 — syntax_check
        `python -m py_compile <changed-files>` (or all .py files in the
        project sub-dir, capped, when no changed list is given). Pure
        syntax validation, no imports executed. Fast and deterministic;
        if this fails, the patch is broken and the Reviewer should retry.

    Phase 2 — collect_tests
        `pytest --collect-only -q`. Proves tests are at least discoverable.
        Often fails because the project's runtime imports need
        `pip install -e .`, which the sandbox doesn't do today — those
        failures classify as `needs_env` (not the patch's fault).

The runner returns a structured TestRunResult either way; it never raises
for sandbox-side problems (image missing, timeout) — those become
classification="error".
"""
from __future__ import annotations

import time
from pathlib import Path

import structlog

from app.agents.test_runner.detector import detect_project
from app.agents.test_runner.schemas import (
    Classification,
    Language,
    PhaseResult,
    TestRunResult,
)
from app.sandbox import (
    DockerNotAvailableError,
    ImageMissingError,
    SandboxRunError,
    SandboxRunner,
    SandboxTimeoutError,
    Workspace,
)
from app.sandbox.container import SandboxResult

log = structlog.get_logger(__name__)

# Output truncation. Keep the TAIL because test failure summaries are at
# the end. Reviewer prompts also benefit from "last error" framing.
_PHASE_OUTPUT_TAIL_BYTES = 8_000
_FAILURE_EXCERPT_BYTES = 2_000

# Don't scan more than this many .py files when no changed-list is given.
# A repo with 10k .py files would otherwise overflow the argv list.
_MAX_FILES_FOR_BULK_SYNTAX = 50


async def run_tests(
    workspace: Workspace,
    repo_path: Path,
    *,
    changed_files: list[str] | None = None,
    runner: SandboxRunner | None = None,
    timeout_s: int = 120,
) -> TestRunResult:
    """Detect + execute the test phases for this workspace.

    Args:
        workspace: per-investigation workspace (used for bind-mount path).
        repo_path: the cloned repo dir on the host. Must be inside
            `workspace.host_path`.
        changed_files: repo-relative paths the patch touched. When given,
            phase 1 only syntax-checks these. When None, falls back to a
            capped bulk scan of the project sub-dir.
        runner: SandboxRunner to use; constructed with defaults if None.
        timeout_s: cap each phase.
    """
    started = time.monotonic()
    runner = runner or SandboxRunner()

    info = detect_project(repo_path)

    if info.language == "unknown":
        return TestRunResult(
            language="unknown",
            classification="no_project",
            summary=(
                "Couldn't detect a known project type "
                "(no pyproject.toml / package.json / go.mod / Cargo.toml)."
            ),
            phases=[],
            duration_s=time.monotonic() - started,
        )

    if info.language != "python":
        # We can't run this language's test suite yet (the sandbox runner is
        # Python-only), but the patch already applied cleanly to a real,
        # recognized project. Classify as needs_env — the same "syntax was
        # fine, tests just couldn't run" bucket Python uses — so the Reviewer
        # ACCEPTS with a caveat instead of giving up. A human still reviews the
        # diff before it becomes a PR, so we're not merging anything blind.
        return TestRunResult(
            language=info.language,
            classification="needs_env",
            summary=(
                f"Patch applied to a {info.language} project "
                f"(detected {info.marker or 'project marker'}). Automated "
                "tests weren't run — the sandbox runner supports Python only "
                "for now — so review the diff before merging."
            ),
            phases=[],
            duration_s=time.monotonic() - started,
        )

    workdir_in_sandbox = _workdir_for(workspace, repo_path, info.subdir)

    # ---- Phase 1: syntax check ----
    syntax_files = _resolve_syntax_targets(
        repo_path=repo_path,
        subdir=info.subdir,
        changed_files=changed_files,
    )
    if not syntax_files and changed_files:
        # The patch touched only non-.py files (e.g. pyproject.toml, README.md,
        # .yaml, .pyi) or files outside the detected subdir. Fall back to the
        # bulk .py scan so we still verify the Python project's syntax and
        # test collection rather than misclassifying the repo as no_project.
        syntax_files = _resolve_syntax_targets(
            repo_path=repo_path,
            subdir=info.subdir,
            changed_files=None,
        )
    if not syntax_files:
        # Edge case: detected Python but found no .py files anywhere in the
        # project (header-only or all-stubs).
        return TestRunResult(
            language="python",
            classification="no_project",
            summary="Python project detected but no .py files to syntax-check.",
            phases=[],
            duration_s=time.monotonic() - started,
        )

    syntax_phase = await _run_phase(
        runner=runner,
        workspace=workspace,
        name="syntax_check",
        argv=["python", "-m", "py_compile", *syntax_files],
        workdir=workdir_in_sandbox,
        timeout_s=timeout_s,
    )

    if syntax_phase is None:
        # Infra error already logged inside _run_phase.
        return _make_error_result(
            language="python",
            phases=[],
            summary="Sandbox infra problem during syntax_check (see logs).",
            elapsed=time.monotonic() - started,
        )

    if not syntax_phase.ok:
        # Skip collection entirely — running pytest on broken syntax would
        # just produce a less-useful error.
        collect_skipped = PhaseResult(
            name="collect_tests",
            argv=["pytest", "--collect-only", "-q"],
            exit_code=-1,
            duration_s=0.0,
            skipped=True,
        )
        return TestRunResult(
            language="python",
            classification="fail",
            summary=(
                f"Syntax check FAILED on "
                f"{len(syntax_files)} file(s) — patch is broken."
            ),
            phases=[syntax_phase, collect_skipped],
            duration_s=time.monotonic() - started,
            failure_excerpt=_excerpt(syntax_phase),
        )

    # ---- Phase 2: pytest collect ----
    collect_phase = await _run_phase(
        runner=runner,
        workspace=workspace,
        name="collect_tests",
        argv=["pytest", "--collect-only", "-q"],
        workdir=workdir_in_sandbox,
        timeout_s=timeout_s,
    )

    if collect_phase is None:
        return _make_error_result(
            language="python",
            phases=[syntax_phase],
            summary="Syntax OK; sandbox infra problem during collect_tests.",
            elapsed=time.monotonic() - started,
        )

    if collect_phase.ok:
        return TestRunResult(
            language="python",
            classification="pass",
            summary="Syntax OK + pytest collected the test suite cleanly.",
            phases=[syntax_phase, collect_phase],
            duration_s=time.monotonic() - started,
        )

    if collect_phase.timed_out:
        return TestRunResult(
            language="python",
            classification="fail",
            summary="pytest collection timed out — patch may have introduced an import-time hang.",
            phases=[syntax_phase, collect_phase],
            duration_s=time.monotonic() - started,
            failure_excerpt=_excerpt(collect_phase),
        )

    # Collect failed but syntax was fine — almost always means the project's
    # imports need `pip install -e .`, which the sandbox doesn't do yet.
    # Surface it as 'needs_env' so the Reviewer doesn't retry the patch
    # over a missing dep.
    return TestRunResult(
        language="python",
        classification="needs_env",
        summary=(
            "Syntax OK but pytest collection failed — usually means the "
            "project's deps weren't installed (sandbox runs read-only without "
            "network)."
        ),
        phases=[syntax_phase, collect_phase],
        duration_s=time.monotonic() - started,
        failure_excerpt=_excerpt(collect_phase),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

async def _run_phase(
    *,
    runner: SandboxRunner,
    workspace: Workspace,
    name: str,
    argv: list[str],
    workdir: str,
    timeout_s: int,
) -> PhaseResult | None:
    """Run a single phase. Returns None when infra was broken (caller
    converts that to classification="error")."""
    log.info("test_runner_phase_start", name=name, argv=argv[:3], workdir=workdir)
    started = time.monotonic()
    try:
        sb: SandboxResult = runner.run(
            workspace,
            argv,
            workdir=workdir,
            timeout_s=timeout_s,
        )
    except SandboxTimeoutError:
        return PhaseResult(
            name=name,
            argv=argv,
            exit_code=124,  # convention: timeout
            duration_s=time.monotonic() - started,
            stdout="",
            stderr=f"(timed out after {timeout_s}s)",
            timed_out=True,
        )
    except (DockerNotAvailableError, ImageMissingError, SandboxRunError) as e:
        log.warning("test_runner_phase_infra_error", name=name, error=str(e))
        return None

    return PhaseResult(
        name=name,
        argv=argv,
        exit_code=sb.exit_code,
        duration_s=sb.duration_s,
        stdout=_tail(sb.stdout, _PHASE_OUTPUT_TAIL_BYTES),
        stderr=_tail(sb.stderr, _PHASE_OUTPUT_TAIL_BYTES),
        timed_out=False,
    )


def _resolve_syntax_targets(
    *,
    repo_path: Path,
    subdir: str,
    changed_files: list[str] | None,
) -> list[str]:
    """Pick the file list py_compile should look at.

    Returned paths are repo-relative POSIX (suitable to pass on the
    sandbox CLI from inside the workdir).
    """
    project_root = repo_path / subdir if subdir else repo_path

    if changed_files:
        # Filter to .py files that live inside the project sub-dir AND
        # actually exist on disk after the patch.
        out: list[str] = []
        for path in changed_files:
            if not path.endswith(".py"):
                continue
            normalized = path.replace("\\", "/")
            on_disk = repo_path / normalized
            if not on_disk.is_file():
                continue
            if subdir and not normalized.startswith(subdir + "/"):
                continue
            rel_to_project = (
                normalized[len(subdir) + 1 :] if subdir else normalized
            )
            out.append(rel_to_project)
        return out

    # No changed_files list — bulk scan, capped. Sorted for determinism so
    # repeated runs produce the same argv (useful for test snapshots later).
    py_files = sorted(
        p for p in project_root.rglob("*.py")
        if "__pycache__" not in p.parts and ".venv" not in p.parts
    )
    py_files = py_files[:_MAX_FILES_FOR_BULK_SYNTAX]
    return [str(p.relative_to(project_root)).replace("\\", "/") for p in py_files]


def _workdir_for(
    workspace: Workspace, repo_path: Path, subdir: str,
) -> str:
    """Build the in-sandbox workdir, e.g. /workspace/Hello-World/backend."""
    # The sandbox mounts workspace.host_path at /workspace, so repo_path
    # is expected to be `<workspace.host_path>/<repo_name>/...`.
    rel = repo_path.relative_to(workspace.host_path).as_posix()
    if subdir:
        rel = f"{rel}/{subdir}"
    return rel


def _tail(s: str, n_bytes: int) -> str:
    if not s:
        return ""
    encoded = s.encode("utf-8", errors="replace")
    if len(encoded) <= n_bytes:
        return s
    return (
        "...(truncated)...\n"
        + encoded[-n_bytes:].decode("utf-8", errors="replace")
    )


def _excerpt(phase: PhaseResult) -> str:
    """Combine stdout + stderr (in that order) and tail-truncate.

    stderr is concatenated AFTER stdout so the tail-bias preserves it —
    that's where the actionable error message lives for `py_compile`,
    pytest collection failures, and most CLI test runners.
    """
    combined = (phase.stdout + "\n" + phase.stderr).strip()
    return _tail(combined, _FAILURE_EXCERPT_BYTES)


def _make_error_result(
    *, language: Language, phases: list[PhaseResult], summary: str, elapsed: float,
) -> TestRunResult:
    return TestRunResult(
        language=language,
        classification="error",
        summary=summary,
        phases=phases,
        duration_s=elapsed,
    )


# Imported just so call sites can `from app.agents.test_runner.runner import
# Classification` for type annotations. Keeps the public surface tidy.
__all__ = ["Classification", "run_tests"]
