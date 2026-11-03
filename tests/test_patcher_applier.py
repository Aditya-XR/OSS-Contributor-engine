"""Pure file-ops tests for the patcher applier."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.agents.patcher.applier import apply_edits, capture_diff
from app.agents.patcher.exceptions import EditApplyError
from app.agents.patcher.schemas import CodeEdit


def _seed(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        full = root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8", newline="\n")


def _git_init(root: Path) -> None:
    """Bare-bones git repo so capture_diff has something to diff against."""
    for argv in (
        ["git", "init", "-q"],
        # Store bytes verbatim, like a Linux host. A global autocrlf=true
        # (Windows default) would normalize CRLF and hide line-ending diffs.
        ["git", "config", "core.autocrlf", "false"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "add", "."],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-q", "-m", "init"],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# apply_edits — happy paths
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_apply_simple_replace(tmp_path):
    _seed(tmp_path, {"foo.py": "x = 1\ny = 2\nz = 3\n"})
    [edit] = apply_edits(tmp_path, [
        CodeEdit(path="foo.py", search="y = 2", replace="y = 42",
                 explanation="bump y"),
    ])
    assert edit.path == "foo.py"
    assert edit.new_file is False
    assert (tmp_path / "foo.py").read_text(encoding="utf-8") == "x = 1\ny = 42\nz = 3\n"


@pytest.mark.unit
def test_apply_multiple_edits_in_order(tmp_path):
    _seed(tmp_path, {"a.py": "alpha\n", "b.py": "beta\n"})
    applied = apply_edits(tmp_path, [
        CodeEdit(path="a.py", search="alpha", replace="ALPHA"),
        CodeEdit(path="b.py", search="beta", replace="BETA"),
    ])
    assert len(applied) == 2
    assert (tmp_path / "a.py").read_text() == "ALPHA\n"
    assert (tmp_path / "b.py").read_text() == "BETA\n"


@pytest.mark.unit
def test_create_new_file_via_empty_search(tmp_path):
    [edit] = apply_edits(tmp_path, [
        CodeEdit(
            path="src/new_module.py",
            search="",
            replace="def hello():\n    return 'world'\n",
            explanation="add helper",
        ),
    ])
    assert edit.new_file is True
    assert (tmp_path / "src/new_module.py").read_text() == \
        "def hello():\n    return 'world'\n"


@pytest.mark.unit
def test_byte_counts_recorded(tmp_path):
    _seed(tmp_path, {"x.py": "abc\n"})
    [edit] = apply_edits(tmp_path, [
        CodeEdit(path="x.py", search="abc", replace="abcdef"),
    ])
    assert edit.bytes_added == 3  # added 3 chars (def)
    assert edit.bytes_removed == 0


# ---------------------------------------------------------------------------
# apply_edits — failure modes
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_search_not_found_raises(tmp_path):
    _seed(tmp_path, {"x.py": "hello\n"})
    with pytest.raises(EditApplyError, match="not found"):
        apply_edits(tmp_path, [
            CodeEdit(path="x.py", search="goodbye", replace="hi"),
        ])


@pytest.mark.unit
def test_search_ambiguous_raises(tmp_path):
    _seed(tmp_path, {"x.py": "z = 1\nz = 2\n"})
    with pytest.raises(EditApplyError, match="matches 2 places"):
        apply_edits(tmp_path, [
            CodeEdit(path="x.py", search="z = ", replace="zed = "),
        ])


@pytest.mark.unit
def test_missing_file_raises(tmp_path):
    with pytest.raises(EditApplyError, match="file not found"):
        apply_edits(tmp_path, [
            CodeEdit(path="ghost.py", search="x", replace="y"),
        ])


@pytest.mark.unit
def test_new_file_refuses_to_clobber(tmp_path):
    _seed(tmp_path, {"x.py": "existing\n"})
    with pytest.raises(EditApplyError, match="already exists"):
        apply_edits(tmp_path, [
            CodeEdit(path="x.py", search="", replace="anything"),
        ])


@pytest.mark.unit
@pytest.mark.parametrize("bad_path", [
    "../escape.py", "/etc/passwd", "C:/Windows/System32/foo.txt",
    "ok/../../escape.py", "",
    ".git/config", ".git/hooks/pre-commit", "sub/.git/config", ".GIT/config",
])
def test_path_validation_rejects_escapes(tmp_path, bad_path):
    with pytest.raises(EditApplyError):
        apply_edits(tmp_path, [
            CodeEdit(path=bad_path, search="", replace="x"),
        ])


@pytest.mark.unit
def test_stops_at_first_failure(tmp_path):
    _seed(tmp_path, {"a.py": "hello\n", "b.py": "world\n"})
    with pytest.raises(EditApplyError):
        apply_edits(tmp_path, [
            CodeEdit(path="a.py", search="MISSING", replace="x"),
            CodeEdit(path="b.py", search="world", replace="WORLD"),
        ])
    # b.py should be UNMODIFIED because we never got to it.
    assert (tmp_path / "b.py").read_text() == "world\n"


@pytest.mark.unit
def test_failed_multi_edit_rolls_back_earlier_edits(tmp_path):
    _seed(tmp_path, {"a.py": "alpha\n", "b.py": "beta\n"})
    with pytest.raises(EditApplyError, match="not found"):
        apply_edits(tmp_path, [
            CodeEdit(path="a.py", search="alpha", replace="MODIFIED"),
            CodeEdit(path="created.py", search="", replace="new file\n"),
            CodeEdit(path="b.py", search="MISSING", replace="BETA"),
        ])
    assert (tmp_path / "a.py").read_text() == "alpha\n"
    assert not (tmp_path / "created.py").exists()
    assert (tmp_path / "b.py").read_text() == "beta\n"


@pytest.mark.unit
def test_windows_backslashes_in_path_accepted(tmp_path):
    _seed(tmp_path, {"src/foo.py": "x\n"})
    apply_edits(tmp_path, [
        CodeEdit(path="src\\foo.py", search="x", replace="y"),
    ])
    assert (tmp_path / "src/foo.py").read_text() == "y\n"


@pytest.mark.unit
def test_replace_preserves_crlf_line_endings(tmp_path):
    (tmp_path / "win.py").write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    # The LLM sees (and searches with) "\n"-normalized text.
    apply_edits(tmp_path, [
        CodeEdit(path="win.py", search="a = 1\nb = 2", replace="a = 1\nb = 42"),
    ])
    assert (tmp_path / "win.py").read_bytes() == b"a = 1\r\nb = 42\r\nc = 3\r\n"


@pytest.mark.unit
def test_replace_keeps_lf_line_endings(tmp_path):
    (tmp_path / "unix.py").write_bytes(b"a = 1\nb = 2\n")
    apply_edits(tmp_path, [
        CodeEdit(path="unix.py", search="b = 2", replace="b = 42"),
    ])
    assert (tmp_path / "unix.py").read_bytes() == b"a = 1\nb = 42\n"


# ---------------------------------------------------------------------------
# capture_diff
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_capture_diff_reflects_applied_changes(tmp_path):
    _seed(tmp_path, {"x.py": "hello\n"})
    _git_init(tmp_path)
    apply_edits(tmp_path, [
        CodeEdit(path="x.py", search="hello", replace="HELLO"),
    ])
    diff = capture_diff(tmp_path)
    assert "x.py" in diff
    assert "-hello" in diff
    assert "+HELLO" in diff


@pytest.mark.integration
def test_capture_diff_includes_new_files(tmp_path):
    _seed(tmp_path, {"existing.py": "z\n"})
    _git_init(tmp_path)
    apply_edits(tmp_path, [
        CodeEdit(path="brand_new.py", search="", replace="from new\n"),
    ])
    diff = capture_diff(tmp_path)
    assert "brand_new.py" in diff
    assert "+from new" in diff


@pytest.mark.integration
def test_capture_diff_crlf_edit_touches_only_changed_line(tmp_path):
    lines = [f"v{i} = {i}" for i in range(20)]
    (tmp_path / "win.py").write_bytes(("\r\n".join(lines) + "\r\n").encode())
    _git_init(tmp_path)
    apply_edits(tmp_path, [
        CodeEdit(path="win.py", search="v10 = 10", replace="v10 = 100"),
    ])
    diff = capture_diff(tmp_path)
    changed = [
        ln for ln in diff.splitlines()
        if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
    ]
    assert len(changed) == 2, diff
    assert changed[0].startswith("-v10 = 10")
    assert changed[1].startswith("+v10 = 100")


@pytest.mark.integration
def test_capture_diff_empty_when_no_changes(tmp_path):
    _seed(tmp_path, {"x.py": "x\n"})
    _git_init(tmp_path)
    diff = capture_diff(tmp_path)
    assert diff.strip() == ""
