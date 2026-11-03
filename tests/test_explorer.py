"""End-to-end-ish tests for the Code Explorer orchestrator.

LLM router is faked. Filesystem is real (tmp_path).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agents.explorer import explore


def _mkrepo(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        full = root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    return root


class _FakeRouter:
    """Returns whatever JSON the test queues up."""
    def __init__(self, response_json: str):
        self._json = response_json
        self.call_count = 0
        self.last_messages = None
        self.model_list = [{"litellm_params": {"model": "gemini/gemini-2.5-flash"}}]

    def completion(self, **kwargs):
        self.call_count += 1
        self.last_messages = kwargs.get("messages")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._json))],
            model="gemini/gemini-2.5-flash",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40),
            _hidden_params={"response_cost": 0.0001},
        )


# ---------------------------------------------------------------------------
# Deterministic-only path (router=None)
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_explore_without_llm_returns_deterministic_top_n(tmp_path):
    _mkrepo(tmp_path, {
        "src/auth/handler.py": "def login(): ...",
        "src/db/m.py": "schema = ...",
        "docs/x.md": "# x",
    })
    result = await explore(
        repo="acme/widget",
        repo_path=tmp_path,
        issue_title="login bug",
        issue_body="Login crashes in auth/handler.py",
        router=None,
        max_candidates=3,
    )
    assert result.repo == "acme/widget"
    assert result.used_llm_rerank is False
    assert len(result.candidates) > 0
    paths = [c.path for c in result.candidates]
    assert "src/auth/handler.py" in paths
    for c in result.candidates:
        # No LLM rerank → no reason populated; deterministic signals present.
        assert c.reason == ""
        assert c.signals
    # Sorted desc by confidence.
    confs = [c.confidence for c in result.candidates]
    assert confs == sorted(confs, reverse=True)


@pytest.mark.unit
async def test_explore_returns_empty_on_unrelated_issue(tmp_path):
    _mkrepo(tmp_path, {
        "src/foo.py": "x",
        "src/bar.py": "y",
    })
    result = await explore(
        repo="acme/widget",
        repo_path=tmp_path,
        issue_title="totally unrelated topic",
        issue_body="",
        router=None,
    )
    # Generic stopwordy issue + filenames that don't match → no candidates
    # past the score-zero floor.
    assert result.candidates == []
    assert result.files_scanned > 0


# ---------------------------------------------------------------------------
# LLM rerank path
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_explore_with_llm_rerank_uses_llm_confidence_and_reason(tmp_path):
    _mkrepo(tmp_path, {
        "src/auth/handler.py": "def login(): ...",
        "src/auth/util.py": "def helper(): ...",
    })
    # LLM returns a different ordering than the deterministic step would.
    fake_router = _FakeRouter(json.dumps({
        "candidates": [
            {
                "path": "src/auth/util.py",
                "confidence": 0.95,
                "reason": "helper is the actual login entry point per snippet",
            },
            {
                "path": "src/auth/handler.py",
                "confidence": 0.50,
                "reason": "looks related but only invokes helper",
            },
        ],
    }))
    result = await explore(
        repo="acme/widget",
        repo_path=tmp_path,
        issue_title="login bug",
        issue_body="something about auth",
        router=fake_router,
    )
    assert result.used_llm_rerank is True
    assert fake_router.call_count == 1
    paths = [c.path for c in result.candidates]
    assert paths == ["src/auth/util.py", "src/auth/handler.py"]
    assert result.candidates[0].confidence == 0.95
    assert "helper" in result.candidates[0].reason
    # Deterministic signals carried through from the scan, not the LLM.
    assert result.candidates[0].signals


@pytest.mark.unit
async def test_explore_drops_hallucinated_paths(tmp_path):
    _mkrepo(tmp_path, {"src/auth.py": "x"})
    fake_router = _FakeRouter(json.dumps({
        "candidates": [
            {"path": "src/auth.py", "confidence": 0.9, "reason": "real"},
            {"path": "made/up/path.py", "confidence": 0.8, "reason": "fake"},
        ],
    }))
    result = await explore(
        repo="acme/widget",
        repo_path=tmp_path,
        issue_title="auth bug",
        issue_body=None,
        router=fake_router,
    )
    paths = [c.path for c in result.candidates]
    assert "made/up/path.py" not in paths
    assert "src/auth.py" in paths


@pytest.mark.unit
async def test_explore_falls_back_to_deterministic_on_llm_parse_failure(tmp_path):
    _mkrepo(tmp_path, {"src/auth.py": "x"})
    # Invalid JSON → call_llm's pydantic validation returns parsed=None.
    fake_router = _FakeRouter("this is not valid json at all")
    result = await explore(
        repo="acme/widget",
        repo_path=tmp_path,
        issue_title="auth bug",
        issue_body=None,
        router=fake_router,
    )
    # Used the LLM (flag stays True), but parsed nothing → fell back.
    assert result.used_llm_rerank is True
    assert len(result.candidates) > 0
    # No LLM reason since the fallback path uses empty strings.
    assert all(c.reason == "" for c in result.candidates)


@pytest.mark.unit
async def test_explore_caps_at_max_candidates(tmp_path):
    files = {f"src/{name}.py": "auth login" for name in "abcdefghij"}
    _mkrepo(tmp_path, files)
    # 10 deterministic candidates exist; we ask for 3.
    result = await explore(
        repo="acme/widget",
        repo_path=tmp_path,
        issue_title="bug in src/a.py auth login crash",
        issue_body=None,
        router=None,
        max_candidates=3,
    )
    assert len(result.candidates) <= 3


@pytest.mark.unit
async def test_explore_records_telemetry(tmp_path):
    _mkrepo(tmp_path, {"src/foo.py": "x"})
    result = await explore(
        repo="acme/widget",
        repo_path=tmp_path,
        issue_title="foo bug",
        issue_body=None,
        router=None,
    )
    assert result.files_scanned >= 1
    assert result.elapsed_s >= 0.0


@pytest.mark.unit
async def test_explore_extracts_referenced_path_from_title_and_ignores_cargo_lock(tmp_path):
    _mkrepo(tmp_path, {
        "Cargo.lock": "# lockfile",
        "src/auth.controller.ts": "export function login() {}",
        "src/other.ts": "export const x = 1;",
    })
    result = await explore(
        repo="acme/widget",
        repo_path=tmp_path,
        issue_title="Fix crash in src/auth.controller.ts",
        issue_body="",
        router=None,
    )
    paths = [c.path for c in result.candidates]
    assert "Cargo.lock" not in paths
    assert paths[0] == "src/auth.controller.ts"
    assert "ref:src/auth.controller.ts" in result.candidates[0].signals
