"""Orchestrator tests with a fake LLM router."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agents.explorer.schemas import FileCandidate
from app.agents.patcher import PatchResult, write_patch


def _seed(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        full = root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8", newline="\n")


def _git_init(root: Path) -> None:
    for argv in (
        ["git", "init", "-q"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "add", "."],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-q", "-m", "init"],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True)


class _FakeRouter:
    """Returns whatever JSON the test queues up."""
    def __init__(self, response_json: str):
        self._json = response_json
        self.call_count = 0
        self.last_user_msg = None
        self.model_list = [{"litellm_params": {"model": "gemini/gemini-2.5-flash"}}]

    def completion(self, **kwargs):
        self.call_count += 1
        for m in kwargs.get("messages", []):
            if m.get("role") == "user":
                self.last_user_msg = m["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._json))],
            model="gemini/gemini-2.5-flash",
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=80),
            _hidden_params={"response_cost": 0.0002},
        )


def _cand(path: str, confidence: float = 0.8) -> FileCandidate:
    return FileCandidate(path=path, confidence=confidence)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_write_patch_applies_simple_edit(tmp_path):
    _seed(tmp_path, {"foo.py": "x = 1\ny = 2\nz = 3\n"})
    _git_init(tmp_path)

    fake = _FakeRouter(json.dumps({
        "summary": "bump y to 42",
        "confidence": 0.9,
        "edits": [
            {"path": "foo.py", "search": "y = 2", "replace": "y = 42",
             "explanation": "fix per issue"},
        ],
    }))

    result = await write_patch(
        repo="acme/widget",
        repo_path=tmp_path,
        issue_title="y should be 42",
        issue_body=None,
        issue_labels=[],
        candidates=[_cand("foo.py")],
        router=fake,
    )
    assert isinstance(result, PatchResult)
    assert result.success is True
    assert result.confidence == 0.9
    assert result.edits_attempted == 1
    assert len(result.edits_applied) == 1
    assert result.edits_applied[0].path == "foo.py"
    assert result.edits_applied[0].new_file is False
    assert "+y = 42" in result.unified_diff
    assert (tmp_path / "foo.py").read_text() == "x = 1\ny = 42\nz = 3\n"


@pytest.mark.integration
async def test_write_patch_creates_new_file(tmp_path):
    _seed(tmp_path, {"existing.py": "old\n"})
    _git_init(tmp_path)
    fake = _FakeRouter(json.dumps({
        "summary": "add helper module",
        "confidence": 0.7,
        "edits": [
            {"path": "src/helper.py", "search": "",
             "replace": "def f(): return 1\n",
             "explanation": "add new helper"},
        ],
    }))
    result = await write_patch(
        repo="acme/widget", repo_path=tmp_path,
        issue_title="need a helper", issue_body="",
        issue_labels=[], candidates=[_cand("existing.py")],
        router=fake,
    )
    assert result.success is True
    assert result.edits_applied[0].new_file is True
    assert (tmp_path / "src/helper.py").read_text() == "def f(): return 1\n"
    assert "src/helper.py" in result.unified_diff


# ---------------------------------------------------------------------------
# LLM-side failure modes
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_no_router_returns_failure(tmp_path):
    result = await write_patch(
        repo="acme/x", repo_path=tmp_path,
        issue_title="x", issue_body=None, issue_labels=[],
        candidates=[_cand("foo.py")], router=None,
    )
    assert result.success is False
    assert "router" in (result.error or "")


@pytest.mark.unit
async def test_no_candidates_returns_failure(tmp_path):
    fake = _FakeRouter("{}")  # never called
    result = await write_patch(
        repo="acme/x", repo_path=tmp_path,
        issue_title="x", issue_body=None, issue_labels=[],
        candidates=[], router=fake,
    )
    assert result.success is False
    assert fake.call_count == 0


@pytest.mark.unit
async def test_invalid_json_returns_failure(tmp_path):
    _seed(tmp_path, {"x.py": "x\n"})
    fake = _FakeRouter("definitely not json at all")
    result = await write_patch(
        repo="acme/x", repo_path=tmp_path,
        issue_title="x", issue_body=None, issue_labels=[],
        candidates=[_cand("x.py")], router=fake,
    )
    assert result.success is False
    assert "unparseable" in (result.error or "")


@pytest.mark.unit
async def test_zero_edits_returns_failure(tmp_path):
    _seed(tmp_path, {"x.py": "x\n"})
    fake = _FakeRouter(json.dumps({
        "summary": "I can't fix this from these files",
        "confidence": 0.2,
        "edits": [],
    }))
    result = await write_patch(
        repo="acme/x", repo_path=tmp_path,
        issue_title="x", issue_body=None, issue_labels=[],
        candidates=[_cand("x.py")], router=fake,
    )
    assert result.success is False
    assert result.confidence == 0.2
    assert result.summary == "I can't fix this from these files"
    assert "zero edits" in (result.error or "")


# ---------------------------------------------------------------------------
# Apply-time failure modes
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_failed_apply_returns_failure(tmp_path):
    _seed(tmp_path, {"x.py": "hello\n"})
    _git_init(tmp_path)
    fake = _FakeRouter(json.dumps({
        "summary": "fix it",
        "confidence": 0.9,
        "edits": [
            {"path": "x.py", "search": "MISSING_TEXT", "replace": "anything"},
        ],
    }))
    result = await write_patch(
        repo="acme/x", repo_path=tmp_path,
        issue_title="x", issue_body=None, issue_labels=[],
        candidates=[_cand("x.py")], router=fake,
    )
    assert result.success is False
    assert "x.py" in (result.error or "")
    assert (tmp_path / "x.py").read_text() == "hello\n"  # unchanged


@pytest.mark.integration
async def test_noop_patch_marked_as_failure(tmp_path):
    """An edit that replaces text with itself produces no real diff."""
    _seed(tmp_path, {"x.py": "hello\n"})
    _git_init(tmp_path)
    fake = _FakeRouter(json.dumps({
        "summary": "no real change",
        "confidence": 0.5,
        "edits": [
            {"path": "x.py", "search": "hello", "replace": "hello"},
        ],
    }))
    result = await write_patch(
        repo="acme/x", repo_path=tmp_path,
        issue_title="x", issue_body=None, issue_labels=[],
        candidates=[_cand("x.py")], router=fake,
    )
    assert result.success is False
    assert "no visible change" in (result.error or "")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_prompt_includes_file_content(tmp_path):
    _seed(tmp_path, {"src/auth.py": "def login(): pass\n"})
    fake = _FakeRouter(json.dumps({
        "summary": "x", "confidence": 0.0, "edits": [],
    }))
    await write_patch(
        repo="acme/x", repo_path=tmp_path,
        issue_title="login bug",
        issue_body="See src/auth.py",
        issue_labels=["bug"],
        candidates=[_cand("src/auth.py", 0.95)],
        router=fake,
    )
    assert fake.last_user_msg is not None
    assert "src/auth.py" in fake.last_user_msg
    assert "def login(): pass" in fake.last_user_msg
    assert "login bug" in fake.last_user_msg
    # Labels surfaced explicitly.
    assert "bug" in fake.last_user_msg


@pytest.mark.unit
async def test_prompt_includes_prior_attempts_when_provided(tmp_path):
    from app.agents.patcher import PriorAttempt
    _seed(tmp_path, {"src/auth.py": "def login(): pass\n"})
    fake = _FakeRouter(json.dumps({
        "summary": "x", "confidence": 0.0, "edits": [],
    }))
    prior = [
        PriorAttempt(
            attempt_number=1,
            summary="renamed login to authenticate",
            failure_excerpt="SyntaxError: invalid name on line 7",
            edits_applied_paths=["src/auth.py"],
        ),
    ]
    await write_patch(
        repo="acme/x", repo_path=tmp_path,
        issue_title="login bug", issue_body=None, issue_labels=[],
        candidates=[_cand("src/auth.py")],
        router=fake,
        prior_attempts=prior,
    )
    msg = fake.last_user_msg
    assert msg is not None
    assert "PREVIOUS ATTEMPTS" in msg
    assert "renamed login to authenticate" in msg
    assert "SyntaxError" in msg
    # Anti-loop instruction is present.
    assert "DIFFERENT" in msg


@pytest.mark.unit
async def test_prompt_omits_prior_section_when_empty(tmp_path):
    _seed(tmp_path, {"x.py": "y = 1\n"})
    fake = _FakeRouter(json.dumps({
        "summary": "x", "confidence": 0.0, "edits": [],
    }))
    await write_patch(
        repo="acme/x", repo_path=tmp_path,
        issue_title="t", issue_body=None, issue_labels=[],
        candidates=[_cand("x.py")], router=fake,
        prior_attempts=None,
    )
    assert "PREVIOUS ATTEMPTS" not in fake.last_user_msg


# ---------------------------------------------------------------------------
# Large files: excerpts + budget sharing
# ---------------------------------------------------------------------------

def _large_source(n: int, overrides: dict[int, str]) -> str:
    """`n` lines (~35 chars each, so 1,000 lines ≈ 35k chars); `overrides`
    maps 1-based line numbers to text."""
    return "\n".join(
        overrides.get(i, f"value_{i} = compute({i})  # filler line")
        for i in range(1, n + 1)
    ) + "\n"


@pytest.mark.integration
async def test_fix_deep_in_large_file_reaches_prompt_and_applies(tmp_path):
    """The bug lives at line 901 of a ~35k-char file. Showing only the head
    of the file (the old behaviour) kept line 901 out of the prompt, so the
    LLM could never write a `search` string that matches it."""
    buggy = "    return price * rate  # BUG: should be price * (1 - rate)"
    _seed(tmp_path, {"pricing.py": _large_source(1000, {
        900: "def apply_discount(price, rate):",
        901: buggy,
    })})
    _git_init(tmp_path)
    fake = _FakeRouter(json.dumps({
        "summary": "return the discounted price",
        "confidence": 0.9,
        "edits": [{
            "path": "pricing.py",
            "search": buggy,
            "replace": "    return price * (1 - rate)",
            "explanation": "apply the discount instead of returning it",
        }],
    }))
    result = await write_patch(
        repo="acme/shop", repo_path=tmp_path,
        issue_title="apply_discount returns the discount, not the discounted price",
        issue_body=None, issue_labels=["bug"],
        candidates=[_cand("pricing.py", 0.9)], router=fake,
    )
    msg = fake.last_user_msg
    assert "def apply_discount(price, rate):" in msg
    assert buggy in msg
    assert "of 1000 lines shown" in msg
    assert "not shown] ..." in msg
    assert result.success is True
    assert "+    return price * (1 - rate)" in result.unified_diff


@pytest.mark.unit
async def test_small_file_is_not_starved_by_large_files_ahead_of_it(tmp_path):
    """Two large high-confidence files used to eat the whole budget (12k +
    4k chars), leaving a small third file as "omitted"."""
    _seed(tmp_path, {
        "big_one.py": _large_source(1000, {}),
        "big_two.py": _large_source(1000, {}),
        "tiny.py": "def tiny_helper():\n    return 1\n",
    })
    fake = _FakeRouter(json.dumps({"summary": "x", "confidence": 0.0, "edits": []}))
    await write_patch(
        repo="acme/x", repo_path=tmp_path,
        issue_title="tiny_helper should return 2", issue_body=None,
        issue_labels=[],
        candidates=[
            _cand("big_one.py", 0.9), _cand("big_two.py", 0.8),
            _cand("tiny.py", 0.3),
        ],
        router=fake,
    )
    msg = fake.last_user_msg
    assert "def tiny_helper():\n    return 1" in msg
    assert "omitted" not in msg
    # The shared budget still holds: two capped excerpts + the small file.
    file_section = msg.split("CANDIDATE FILES", 1)[1]
    assert len(file_section) < 16_000 + 1_000


@pytest.mark.unit
async def test_retry_prompt_shows_the_line_the_error_points_at(tmp_path):
    from app.agents.patcher import PriorAttempt
    target = "    total = combine_parts(first_part, second_part"
    _seed(tmp_path, {"big.py": _large_source(1200, {1100: target})})
    fake = _FakeRouter(json.dumps({"summary": "x", "confidence": 0.0, "edits": []}))
    await write_patch(
        repo="acme/x", repo_path=tmp_path,
        issue_title="crash on startup", issue_body=None, issue_labels=[],
        candidates=[_cand("big.py", 0.9)], router=fake,
        prior_attempts=[PriorAttempt(
            attempt_number=1,
            summary="closed a bracket",
            failure_excerpt='  File "big.py", line 1100\nSyntaxError: unexpected EOF',
            edits_applied_paths=["big.py"],
        )],
    )
    assert target in fake.last_user_msg


@pytest.mark.unit
def test_allocate_budgets_shows_small_files_whole():
    from app.agents.patcher.patcher import _allocate_budgets
    alloc = _allocate_budgets(
        [100, 50_000, 200], [0.3, 0.9, 0.5], total=16_000, per_file_cap=12_000,
    )
    assert alloc == [100, 12_000, 200]


@pytest.mark.unit
def test_allocate_budgets_splits_large_files_by_confidence():
    from app.agents.patcher.patcher import _allocate_budgets
    alloc = _allocate_budgets(
        [50_000, 50_000], [0.9, 0.3], total=16_000, per_file_cap=12_000,
    )
    assert alloc == [12_000, 4_000]


@pytest.mark.unit
@pytest.mark.parametrize("weights", [[0.0, 0.0, 0.0], [1.0, 0.5, 0.1], [0.2, 0.9, 0.9]])
def test_allocate_budgets_never_exceeds_total(weights):
    from app.agents.patcher.patcher import _allocate_budgets
    alloc = _allocate_budgets(
        [30_000, 900, 70_000], weights, total=16_000, per_file_cap=12_000,
    )
    assert sum(alloc) <= 16_000
    assert alloc[1] == 900
