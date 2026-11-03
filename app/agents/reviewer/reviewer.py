"""Reviewer — the loop controller that turns the autonomous pilot into an
actual autonomous system.

Per iteration:
    1. Call Patch Writer (with prior failures fed in as context after
       attempt 1).
    2. If the patch produced edits → run Test Runner against the workspace.
    3. Inspect the outcome and decide: retry / accept / give_up.
    4. If retrying, reset the workspace to its post-clone state so the
       next attempt starts from clean.

Termination conditions:
    - Decision is `accept`  → return success.
    - Decision is `give_up` → return failure with reason.
    - Hit `max_attempts`    → return failure.
    - Two attempts in a row produce IDENTICAL diffs → return failure
      (LLM isn't learning; stop burning tokens).

Decision logic (deterministic — no extra LLM call here, intentional):

    patch.success = False
        → give_up (Patch Writer already gave up; e.g. zero edits returned)

    patch.success = True AND test.classification = "pass"
        → accept

    patch.success = True AND test.classification = "fail"
        → retry (clear signal: the patch broke syntax)

    patch.success = True AND test.classification = "needs_env"
        → accept-with-caveat: still success, since the patch itself didn't
          break anything we can detect. Mark with summary so downstream knows.

    patch.success = True AND test.classification in ("error", "no_project")
        → give_up (infra problem or unsupported language)
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import structlog
from sqlalchemy.orm import Session

from app.agents.explorer.schemas import FileCandidate
from app.agents.patcher import PriorAttempt, write_patch
from app.agents.reviewer.schemas import Attempt, Decision, ReviewerResult
from app.agents.test_runner import TestRunResult, run_tests
from app.sandbox import SandboxRunner, Workspace

log = structlog.get_logger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TEST_TIMEOUT_S = 120


async def review_and_iterate(
    *,
    repo: str,
    repo_path: Path,
    workspace: Workspace,
    issue_title: str,
    issue_body: str | None,
    issue_labels: list[str] | None,
    candidates: list[FileCandidate],
    router,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    test_timeout_s: int = DEFAULT_TEST_TIMEOUT_S,
    sandbox_runner: SandboxRunner | None = None,
    investigation_id: str | None = None,
    user_id: int | None = None,
    session: Session | None = None,
) -> ReviewerResult:
    """Run the Patch → Test → Review loop. See module docstring for rules.

    The workspace is mutated in place and reset between attempts via
    `git restore . && git clean -fd`. On exit the workspace is left in
    the state of the LAST attempt (accepted or otherwise) — so the caller
    can capture the final diff if `success=True`, or inspect the failed
    state if debugging.
    """
    started = time.monotonic()
    sandbox_runner = sandbox_runner or SandboxRunner()

    attempts: list[Attempt] = []
    prior: list[PriorAttempt] = []

    for n in range(1, max_attempts + 1):
        log.info(
            "reviewer_attempt_start",
            attempt=n, max=max_attempts, repo=repo, priors=len(prior),
        )

        # ---- Phase 1: write patch ----
        patch = await write_patch(
            repo=repo,
            repo_path=repo_path,
            issue_title=issue_title,
            issue_body=issue_body,
            issue_labels=issue_labels,
            candidates=candidates,
            router=router,
            prior_attempts=prior,
            investigation_id=investigation_id,
            user_id=user_id,
            session=session,
        )

        # Patch didn't apply cleanly — decide whether to give up or retry.
        if not patch.success:
            # Rate-limit is a capacity wall, not a real failure — retrying
            # immediately just hits the same wall. Stop and propagate the flag
            # so the Pilot reports 'rate_limited' (retry) not 'rejected'.
            if patch.rate_limited:
                attempt = Attempt(
                    attempt_number=n,
                    patch_result=patch,
                    test_result=None,
                    decision="give_up",
                    decision_reason=patch.error or "patch generation failed",
                )
                attempts.append(attempt)
                return _make_result(
                    success=False,
                    summary=(
                        "Stopped: all LLM providers were rate-limited or "
                        "unavailable — retry in a minute."
                    ),
                    attempts=attempts,
                    rate_limited=True,
                    elapsed=time.monotonic() - started,
                )
            # The LLM proposed edits, but `apply_edits` rejected them (e.g.
            # search snippet not found or ambiguous). Feed the applier error
            # back into the next attempt so the model can add context.
            if patch.edits_attempted > 0:
                reason = patch.error or "edit failed to apply"
                attempt = Attempt(
                    attempt_number=n,
                    patch_result=patch,
                    test_result=None,
                    decision="retry",
                    decision_reason=reason,
                )
                attempts.append(attempt)
                prior.append(PriorAttempt(
                    attempt_number=n,
                    summary=patch.summary or "(no summary)",
                    failure_excerpt=reason,
                    edits_applied_paths=[e.path for e in patch.edits_applied],
                ))
                _reset_workspace(repo_path)
                continue
            attempt = Attempt(
                attempt_number=n,
                patch_result=patch,
                test_result=None,
                decision="give_up",
                decision_reason=patch.error or "patch generation failed",
            )
            attempts.append(attempt)
            return _make_result(
                success=False,
                summary=f"Patch generation failed on attempt {n}: {patch.error}",
                attempts=attempts,
                elapsed=time.monotonic() - started,
            )

        # Bail if this attempt's diff is identical to a previous one's.
        # LLM is stuck — burning more tokens won't break the cycle.
        if any(
            a.patch_result.success
            and patch.unified_diff == a.patch_result.unified_diff
            for a in attempts
        ):
            log.warning("reviewer_diff_loop_detected", repo=repo, attempt=n)
            attempt = Attempt(
                attempt_number=n,
                patch_result=patch,
                test_result=None,
                decision="give_up",
                decision_reason="LLM produced the same diff as last attempt "
                                "— giving up to avoid an infinite loop",
            )
            attempts.append(attempt)
            return _make_result(
                success=False,
                summary="Stopped: LLM stuck on the same (failing) approach.",
                attempts=attempts,
                elapsed=time.monotonic() - started,
            )

        # ---- Phase 2: run tests ----
        changed = [e.path for e in patch.edits_applied]
        test_result = await run_tests(
            workspace,
            repo_path,
            changed_files=changed,
            runner=sandbox_runner,
            timeout_s=test_timeout_s,
        )

        # ---- Phase 3: decide ----
        decision, reason = _decide(test_result)
        attempt = Attempt(
            attempt_number=n,
            patch_result=patch,
            test_result=test_result,
            decision=decision,
            decision_reason=reason,
        )
        attempts.append(attempt)

        log.info(
            "reviewer_attempt_done",
            attempt=n,
            decision=decision,
            classification=test_result.classification,
        )

        if decision == "accept":
            return _make_result(
                success=True,
                summary=(
                    f"Accepted on attempt {n}: "
                    f"{test_result.classification}. {reason}"
                ),
                attempts=attempts,
                accepted_attempt_number=n,
                final_diff=patch.unified_diff,
                elapsed=time.monotonic() - started,
            )

        if decision == "give_up":
            return _make_result(
                success=False,
                summary=f"Gave up on attempt {n}: {reason}",
                attempts=attempts,
                elapsed=time.monotonic() - started,
            )

        # ---- Retry: build PriorAttempt + reset workspace ----
        prior.append(_to_prior(patch, test_result, attempt_number=n))
        _reset_workspace(repo_path)

    # Ran out of attempts.
    last_cls = (
        attempts[-1].test_result.classification
        if attempts and attempts[-1].test_result
        else "n/a"
    )
    return _make_result(
        success=False,
        summary=(
            f"Out of attempts ({max_attempts}). Last classification: "
            f"{last_cls}."
        ),
        attempts=attempts,
        elapsed=time.monotonic() - started,
    )


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _decide(test: TestRunResult) -> tuple[Decision, str]:
    """Map test classification to a Decision + one-line reason."""
    c = test.classification
    if c == "pass":
        return "accept", "tests passed cleanly"
    if c == "fail":
        return "retry", "patch broke syntax — feeding failure back to the LLM"
    if c == "needs_env":
        # The patch itself didn't break anything we can detect — pytest
        # couldn't even collect because of missing deps. Surface as
        # acceptance with a caveat in the summary so a downstream PR
        # writer knows tests weren't fully run.
        return "accept", "syntax OK; tests couldn't run (missing env deps, not patch's fault)"
    if c == "error":
        return "give_up", "sandbox infra error — can't continue"
    if c == "no_project":
        return "give_up", "couldn't detect a supported project type"
    # Defensive default — shouldn't reach.
    return "give_up", f"unknown classification: {c}"


def _to_prior(
    patch, test: TestRunResult | None, *, attempt_number: int,
) -> PriorAttempt:
    """Build the context blob the next Patch Writer call will see."""
    excerpt = (test.failure_excerpt if test else "") or ""
    return PriorAttempt(
        attempt_number=attempt_number,
        summary=patch.summary or "(no summary)",
        failure_excerpt=excerpt,
        edits_applied_paths=[e.path for e in patch.edits_applied],
    )


# ---------------------------------------------------------------------------
# Workspace reset
# ---------------------------------------------------------------------------

def _reset_workspace(repo_path: Path) -> None:
    """Revert all uncommitted changes + drop new files.

    `git reset -q` clears intent-to-add entries (`git add -N .` from
    `capture_diff`) so new files become untracked again.
    `git restore .` undoes modifications to tracked files.
    `git clean -fd` removes untracked files + dirs (where the LLM created
    new files like `src/helper.py`).

    Best-effort: if any command fails we log + continue. The next
    patch attempt may produce a confusing result (stacked edits), but
    that's still surfaced via the diff so the loop can detect it.
    """
    for argv in (
        ["git", "reset", "-q"],
        ["git", "restore", "."],
        ["git", "clean", "-fd"],
    ):
        try:
            proc = subprocess.run(
                argv, cwd=repo_path, capture_output=True, text=True, timeout=15,
            )
            if proc.returncode != 0:
                log.warning(
                    "reviewer_workspace_reset_step_failed",
                    argv=argv, stderr=proc.stderr[:200],
                )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.warning("reviewer_workspace_reset_error", argv=argv, error=str(e))


def _make_result(
    *,
    success: bool,
    summary: str,
    attempts: list[Attempt],
    elapsed: float,
    accepted_attempt_number: int | None = None,
    final_diff: str = "",
    rate_limited: bool = False,
) -> ReviewerResult:
    return ReviewerResult(
        success=success,
        summary=summary,
        attempts=attempts,
        accepted_attempt_number=accepted_attempt_number,
        final_diff=final_diff,
        rate_limited=rate_limited,
        elapsed_s=elapsed,
    )
