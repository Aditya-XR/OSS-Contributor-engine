"""Runner tests with a fake SandboxRunner — no Docker required."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.test_runner import run_tests
from app.agents.test_runner.schemas import PhaseResult
from app.sandbox import (
    ImageMissingError,
    SandboxResult,
    SandboxTimeoutError,
    Workspace,
)


class _FakeRunner:
    """Returns canned SandboxResults in order. Records call args for asserts."""

    def __init__(self, results: list[SandboxResult | Exception]):
        self._results = list(results)
        self.calls: list[dict] = []

    def run(self, workspace, command, *, workdir=None, timeout_s=None, **_):
        self.calls.append({
            "workspace": workspace,
            "command": command,
            "workdir": workdir,
            "timeout_s": timeout_s,
        })
        if not self._results:
            raise RuntimeError("FakeRunner: no more canned results")
        nxt = self._results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _ok(stdout: str = "", stderr: str = "") -> SandboxResult:
    return SandboxResult(
        exit_code=0, stdout=stdout, stderr=stderr, duration_s=0.05,
    )


def _fail(code: int = 1, *, stdout: str = "", stderr: str = "") -> SandboxResult:
    return SandboxResult(
        exit_code=code, stdout=stdout, stderr=stderr, duration_s=0.05,
    )


def _make_python_repo(tmp_path: Path) -> tuple[Workspace, Path]:
    """tmp_path becomes the workspace; a `proj/` subdir becomes the repo."""
    ws = Workspace(investigation_id="inv-test", host_path=tmp_path)
    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (repo / "src.py").write_text("x = 1\n", encoding="utf-8")
    return ws, repo


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_run_tests_pass(tmp_path):
    ws, repo = _make_python_repo(tmp_path)
    runner = _FakeRunner([_ok(stdout="Compiling...\n"), _ok(stdout="collected 5 items\n")])
    result = await run_tests(
        ws, repo, changed_files=["src.py"], runner=runner,
    )
    assert result.classification == "pass"
    assert result.language == "python"
    assert len(result.phases) == 2
    assert result.phases[0].name == "syntax_check"
    assert result.phases[1].name == "collect_tests"
    assert all(p.ok for p in result.phases)


@pytest.mark.unit
async def test_phase_workdir_includes_subdir(tmp_path):
    ws, repo = _make_python_repo(tmp_path)
    runner = _FakeRunner([_ok(), _ok()])
    await run_tests(ws, repo, changed_files=["src.py"], runner=runner)
    # Workdir should be `proj` (the repo dir name) — relative to the
    # workspace root which is the bind-mount target /workspace.
    for call in runner.calls:
        assert call["workdir"] == "proj"


@pytest.mark.unit
async def test_syntax_check_passes_changed_files_only(tmp_path):
    ws, repo = _make_python_repo(tmp_path)
    (repo / "other.py").write_text("y = 2\n", encoding="utf-8")
    runner = _FakeRunner([_ok(), _ok()])
    await run_tests(
        ws, repo, changed_files=["src.py"], runner=runner,
    )
    syntax_call = runner.calls[0]
    assert syntax_call["command"][:3] == ["python", "-m", "py_compile"]
    # Only src.py present, not other.py — that's the whole point of
    # passing changed_files.
    assert "src.py" in syntax_call["command"]
    assert "other.py" not in syntax_call["command"]


@pytest.mark.unit
async def test_bulk_syntax_when_no_changed_files(tmp_path):
    ws, repo = _make_python_repo(tmp_path)
    (repo / "a.py").write_text("", encoding="utf-8")
    (repo / "b.py").write_text("", encoding="utf-8")
    runner = _FakeRunner([_ok(), _ok()])
    await run_tests(ws, repo, runner=runner)
    syntax_call = runner.calls[0]
    py_args = [a for a in syntax_call["command"] if a.endswith(".py")]
    # Three .py files in the project: src.py + a.py + b.py
    assert sorted(py_args) == ["a.py", "b.py", "src.py"]


# ---------------------------------------------------------------------------
# Failure classifications
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_syntax_fail_classifies_as_fail(tmp_path):
    ws, repo = _make_python_repo(tmp_path)
    runner = _FakeRunner([
        _fail(1, stderr="SyntaxError: invalid syntax\n"),
    ])
    result = await run_tests(ws, repo, changed_files=["src.py"], runner=runner)
    assert result.classification == "fail"
    # Phase 2 must be present and marked skipped.
    assert len(result.phases) == 2
    assert result.phases[1].skipped is True
    assert "SyntaxError" in result.failure_excerpt
    # Pytest must NOT have been invoked.
    assert len(runner.calls) == 1


@pytest.mark.unit
async def test_syntax_ok_collect_fail_classifies_needs_env(tmp_path):
    ws, repo = _make_python_repo(tmp_path)
    runner = _FakeRunner([
        _ok(),
        _fail(2, stderr="ImportError: No module named 'numpy'\n"),
    ])
    result = await run_tests(ws, repo, changed_files=["src.py"], runner=runner)
    assert result.classification == "needs_env"
    assert result.phases[0].ok
    assert not result.phases[1].ok
    assert "ImportError" in result.failure_excerpt


@pytest.mark.unit
async def test_timeout_in_syntax_classifies_as_fail(tmp_path):
    """A timeout during syntax check is still a failure — exit_code 124."""
    ws, repo = _make_python_repo(tmp_path)
    runner = _FakeRunner([SandboxTimeoutError("ran out of time")])
    result = await run_tests(
        ws, repo, changed_files=["src.py"], runner=runner, timeout_s=10,
    )
    assert result.classification == "fail"
    assert result.phases[0].timed_out is True
    assert result.phases[0].exit_code == 124


@pytest.mark.unit
async def test_timeout_in_collect_tests_classifies_as_fail(tmp_path):
    """A timeout during pytest collection (e.g. import-time infinite loop)
    must classify as 'fail', not 'needs_env' (which Reviewer would accept)."""
    ws, repo = _make_python_repo(tmp_path)
    runner = _FakeRunner([
        _ok(),
        SandboxTimeoutError("collection hung"),
    ])
    result = await run_tests(
        ws, repo, changed_files=["src.py"], runner=runner, timeout_s=10,
    )
    assert result.classification == "fail"
    assert result.phases[1].timed_out is True
    assert result.phases[1].exit_code == 124


@pytest.mark.unit
async def test_non_py_changed_file_in_python_repo_falls_back_to_bulk_syntax(tmp_path):
    """Editing pyproject.toml or README.md in a Python project shouldn't
    classify the repo as no_project (which makes Reviewer give up)."""
    ws, repo = _make_python_repo(tmp_path)
    runner = _FakeRunner([_ok(), _ok()])
    result = await run_tests(
        ws, repo, changed_files=["pyproject.toml"], runner=runner,
    )
    assert result.classification == "pass"
    assert "src.py" in runner.calls[0]["command"]


@pytest.mark.unit
async def test_image_missing_classifies_as_error(tmp_path):
    ws, repo = _make_python_repo(tmp_path)
    runner = _FakeRunner([ImageMissingError("sandbox image not built")])
    result = await run_tests(ws, repo, changed_files=["src.py"], runner=runner)
    assert result.classification == "error"
    assert "infra problem" in result.summary
    assert result.phases == []


# ---------------------------------------------------------------------------
# No-project / unsupported branches
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_no_project_when_no_markers(tmp_path):
    ws = Workspace(investigation_id="inv-empty", host_path=tmp_path)
    repo = tmp_path / "empty"
    repo.mkdir()
    runner = _FakeRunner([])
    result = await run_tests(ws, repo, runner=runner)
    assert result.classification == "no_project"
    assert result.language == "unknown"
    assert runner.calls == []  # never ran anything


@pytest.mark.unit
async def test_non_python_language_accepts_with_caveat(tmp_path):
    """A detected-but-unsupported language (JS/TS/Go/Rust) shouldn't make the
    pilot give up. The patch already applied to a real project, so we classify
    needs_env (→ Reviewer accepts with a 'tests not run' caveat) rather than
    no_project (→ give up)."""
    ws = Workspace(investigation_id="inv-js", host_path=tmp_path)
    repo = tmp_path / "node-thing"
    repo.mkdir()
    (repo / "package.json").write_text('{"name":"x"}', encoding="utf-8")
    runner = _FakeRunner([])
    result = await run_tests(ws, repo, runner=runner)
    assert result.classification == "needs_env"
    assert result.language == "javascript"
    assert "javascript" in result.summary
    # Honest "tests weren't run" caveat so the PR body / UI can surface it.
    assert "review the diff" in result.summary or "weren't run" in result.summary
    assert runner.calls == []  # never tried to run anything


@pytest.mark.unit
async def test_python_project_with_no_py_files_returns_no_project(tmp_path):
    ws = Workspace(investigation_id="inv-noslip", host_path=tmp_path)
    repo = tmp_path / "weird"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    # No .py files. Bulk scan finds nothing, changed_files is None.
    runner = _FakeRunner([])
    result = await run_tests(ws, repo, runner=runner)
    assert result.classification == "no_project"
    assert result.language == "python"
    assert "no .py files" in result.summary


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_failure_excerpt_is_tail_truncated(tmp_path):
    """Verify the failure excerpt is bounded for prompt-friendly feedback."""
    ws, repo = _make_python_repo(tmp_path)
    # Build a long-ish stdout/stderr.
    big = "noise line\n" * 1000
    runner = _FakeRunner([
        _fail(1, stdout=big, stderr="LAST: SyntaxError\n"),
    ])
    result = await run_tests(ws, repo, changed_files=["src.py"], runner=runner)
    assert result.classification == "fail"
    # Excerpt should fit in the 2KB tail budget, and the final marker
    # ("LAST:") should survive — that's the whole reason we tail-bias.
    assert len(result.failure_excerpt) <= 2_500
    assert "LAST" in result.failure_excerpt


@pytest.mark.unit
async def test_phase_result_serializes_cleanly():
    """PhaseResult is part of the public TestRunResult contract — pydantic
    round-trip should work for any downstream JSON consumer (the Reviewer
    will dump these into prompts)."""
    pr = PhaseResult(
        name="syntax_check", argv=["python", "-m", "py_compile", "x.py"],
        exit_code=0, duration_s=0.4,
        stdout="ok", stderr="",
    )
    blob = pr.model_dump_json()
    rt = PhaseResult.model_validate_json(blob)
    assert rt == pr
    assert rt.ok is True
