"""Scanner tests — fake repos on disk, no Docker, no LLM."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.explorer.scanner import read_snippet, scan_workspace


def _mkrepo(root: Path, files: dict[str, str]) -> Path:
    """Materialize a fake repo. `files` maps relative path -> file content."""
    for rel, content in files.items():
        full = root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    return root


@pytest.mark.unit
def test_scan_returns_top_k_by_score(tmp_path):
    _mkrepo(tmp_path, {
        "src/auth/handlers.py": "def login(): ...",
        "src/db/migrations.py": "schema = []",
        "tests/test_auth.py": "def test_login(): ...",
        "docs/README.md": "# README",
    })
    top, examined = scan_workspace(
        tmp_path,
        keywords=["auth", "login"],
        references=[],
        issue_text="bug in login crashes auth",
        top_k=10,
    )
    paths = [c.path for c in top]
    assert "src/auth/handlers.py" in paths
    assert "tests/test_auth.py" in paths
    # The migrations + README scored 0, so they shouldn't appear.
    assert "src/db/migrations.py" not in paths
    assert "docs/README.md" not in paths
    assert examined >= 4


@pytest.mark.unit
def test_scan_prunes_ignored_dirs(tmp_path):
    _mkrepo(tmp_path, {
        "node_modules/auth/index.js": "x",   # should be ignored
        "src/auth/index.js": "x",            # should be considered
    })
    top, _ = scan_workspace(
        tmp_path,
        keywords=["auth"],
        references=[],
        issue_text="auth bug",
        top_k=10,
    )
    paths = [c.path for c in top]
    assert "src/auth/index.js" in paths
    assert not any(p.startswith("node_modules/") for p in paths)


@pytest.mark.unit
def test_scan_skips_zero_byte_files(tmp_path):
    _mkrepo(tmp_path, {"src/auth.py": ""})
    top, _ = scan_workspace(
        tmp_path, keywords=["auth"], references=[],
        issue_text="auth bug", top_k=10,
    )
    assert top == []


@pytest.mark.unit
def test_scan_sorted_descending_by_score(tmp_path):
    _mkrepo(tmp_path, {
        "src/auth/handler.py": "x",
        "src/something/auth_util.py": "x",
        "src/random/totally_other.py": "x",
    })
    top, _ = scan_workspace(
        tmp_path,
        keywords=["auth", "handler"],
        references=[],
        issue_text="bug with auth handler",
        top_k=10,
    )
    assert len(top) >= 2
    for i in range(len(top) - 1):
        assert top[i].score >= top[i + 1].score


@pytest.mark.unit
def test_scan_referenced_paths_outrank_keyword_matches(tmp_path):
    _mkrepo(tmp_path, {
        "src/auth/explicit.py": "x",
        "src/auth/something_auth_related.py": "x",
    })
    top, _ = scan_workspace(
        tmp_path,
        keywords=["auth"],
        references=["src/auth/explicit.py"],
        issue_text="bug in src/auth/explicit.py",
        top_k=10,
    )
    assert top[0].path == "src/auth/explicit.py"


@pytest.mark.unit
def test_scan_raises_on_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_workspace(
            tmp_path / "nope", keywords=[], references=[], issue_text="",
        )


@pytest.mark.unit
def test_read_snippet_returns_head(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("\n".join(f"line {i}" for i in range(200)), encoding="utf-8")
    snippet = read_snippet(tmp_path, "x.py", max_lines=5)
    assert snippet.splitlines() == ["line 0", "line 1", "line 2", "line 3", "line 4"]


@pytest.mark.unit
def test_read_snippet_handles_missing_file(tmp_path):
    snippet = read_snippet(tmp_path, "ghost.py", max_lines=10)
    assert "snippet read failed" in snippet


@pytest.mark.unit
def test_read_snippet_with_keywords_shows_the_matching_code(tmp_path):
    lines = [f"line {i}" for i in range(300)]
    lines[250] = "def refresh_session_token(user):"
    (tmp_path / "x.py").write_text("\n".join(lines), encoding="utf-8")
    snippet = read_snippet(
        tmp_path, "x.py", max_lines=30, keywords=["refresh_session_token"],
    )
    out = snippet.split("\n")
    assert len(out) <= 30
    assert out[0] == "line 0"
    assert "def refresh_session_token(user):" in out
    assert any("not shown] ..." in line for line in out)


@pytest.mark.unit
def test_read_snippet_with_keywords_keeps_short_files_whole(tmp_path):
    (tmp_path / "x.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    snippet = read_snippet(tmp_path, "x.py", max_lines=30, keywords=["zzz"])
    assert snippet == "a = 1\nb = 2"


@pytest.mark.unit
def test_read_snippet_with_keywords_handles_missing_file(tmp_path):
    snippet = read_snippet(tmp_path, "ghost.py", max_lines=10, keywords=["x"])
    assert "snippet read failed" in snippet
