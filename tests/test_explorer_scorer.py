"""Pure unit tests for the explorer scorer — no I/O, no LLM."""
from __future__ import annotations

import pytest

from app.agents.explorer.scorer import (
    combine_scores,
    dir_score,
    extract_keywords,
    extract_referenced_paths,
    filename_score,
    referenced_score,
)

# Aliased so pytest doesn't auto-collect this as a test function in the
# module namespace just because its name starts with `test_`.
from app.agents.explorer.scorer import test_affinity as _test_affinity

# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_keywords_pulls_identifiers():
    kws = extract_keywords(
        "AuthHandler raises ValueError on empty token in JsonResponse",
    )
    assert "authhandler" in kws
    assert "valueerror" in kws
    assert "auth" in kws  # camelCase split
    assert "handler" in kws
    assert "token" in kws


@pytest.mark.unit
def test_extract_keywords_filters_stopwords():
    kws = extract_keywords("the bug is that the new feature should not crash")
    assert "the" not in kws
    assert "should" not in kws
    assert "feature" not in kws  # also in stopwords (issue/bug/feature)


@pytest.mark.unit
def test_extract_keywords_dedupes():
    kws = extract_keywords("auth auth Auth AUTH authentication")
    auths = [k for k in kws if "auth" in k]
    # "auth" once, "authentication" once — no duplicate "auth".
    assert auths.count("auth") == 1


@pytest.mark.unit
def test_extract_keywords_handles_empty():
    assert extract_keywords(None) == []
    assert extract_keywords("") == []


# ---------------------------------------------------------------------------
# extract_referenced_paths
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_referenced_paths_picks_up_paths():
    refs = extract_referenced_paths(
        "See `src/auth/handlers.py` and tests/test_auth.py for context.",
    )
    assert "src/auth/handlers.py" in refs
    assert "tests/test_auth.py" in refs


@pytest.mark.unit
def test_extract_referenced_paths_handles_windows_slashes():
    refs = extract_referenced_paths("Look at src\\auth\\foo.py")
    assert "src/auth/foo.py" in refs


@pytest.mark.unit
def test_extract_referenced_paths_picks_up_bare_filenames():
    refs = extract_referenced_paths("The bug is in handlers.py around line 42.")
    assert "handlers.py" in refs


@pytest.mark.unit
def test_extract_referenced_paths_handles_multi_dot_and_dotted_dirs():
    refs = extract_referenced_paths(
        "Check `src/auth.controller.ts`, `.github/workflows/ci.yml`, "
        "and `Button.stories.tsx`.",
    )
    assert "src/auth.controller.ts" in refs
    assert ".github/workflows/ci.yml" in refs
    assert "Button.stories.tsx" in refs
    score, _ = referenced_score("src/auth.controller.ts", refs)
    assert score == 1.0
    score_base, _ = referenced_score("web/components/Button.stories.tsx", refs)
    assert score_base == 0.7


# ---------------------------------------------------------------------------
# referenced_score
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_referenced_score_full_path_match_is_1():
    score, sigs = referenced_score(
        "src/auth/handlers.py", ["src/auth/handlers.py"],
    )
    assert score == 1.0
    assert any(s.startswith("ref:") for s in sigs)


@pytest.mark.unit
def test_referenced_score_basename_match_is_partial():
    score, sigs = referenced_score(
        "src/auth/handlers.py", ["handlers.py"],
    )
    assert 0.5 < score < 1.0
    assert any(s.startswith("refname:") for s in sigs)


@pytest.mark.unit
def test_referenced_score_no_match_is_zero():
    score, sigs = referenced_score("foo/bar.py", ["baz.py"])
    assert score == 0.0
    assert sigs == []


# ---------------------------------------------------------------------------
# filename_score
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_filename_score_matches_keyword():
    score, sigs = filename_score("src/auth_handler.py", ["auth", "handler"])
    assert score > 0
    assert any("keyword:auth" in s for s in sigs)
    assert any("keyword:handler" in s for s in sigs)


@pytest.mark.unit
def test_filename_score_zero_without_keywords():
    score, sigs = filename_score("anything.py", [])
    assert score == 0.0
    assert sigs == []


@pytest.mark.unit
def test_filename_score_caps_at_one():
    # Lots of matching keywords against a single short name.
    score, _ = filename_score("auth.py", ["auth"] * 20)
    assert score <= 1.0


# ---------------------------------------------------------------------------
# dir_score
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_dir_score_matches_dir_components():
    score, sigs = dir_score("src/auth/handlers.py", ["auth"])
    assert score > 0
    assert "dir:auth" in sigs


@pytest.mark.unit
def test_dir_score_zero_when_no_directory():
    score, sigs = dir_score("toplevel.py", ["auth"])
    assert score == 0.0
    assert sigs == []


# ---------------------------------------------------------------------------
# test_affinity
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_test_affinity_boosts_tests_when_bug_flavored():
    score, sigs = _test_affinity(
        "tests/test_auth.py", "There is a crash when login fails",
    )
    assert score > 0
    assert "test:bug-related" in sigs


@pytest.mark.unit
def test_test_affinity_demerits_tests_when_feature_request():
    score, _ = _test_affinity(
        "tests/test_auth.py", "Add support for SSO via SAML",
    )
    assert score < 0


@pytest.mark.unit
def test_test_affinity_neutral_for_non_tests():
    score, _ = _test_affinity("src/handler.py", "anything")
    assert score == 0.0


# ---------------------------------------------------------------------------
# combine_scores — end-to-end of the pure layer
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_combine_scores_referenced_file_dominates():
    score_ref, _ = combine_scores(
        path="src/auth/handlers.py",
        keywords=["other", "stuff"],
        references=["src/auth/handlers.py"],
        issue_text="See src/auth/handlers.py",
    )
    score_kw, _ = combine_scores(
        path="src/auth/handlers.py",
        keywords=["auth", "handlers"],
        references=[],
        issue_text="auth handlers bug",
    )
    # A direct path reference should outweigh keyword matches alone.
    assert score_ref > score_kw


@pytest.mark.unit
def test_combine_scores_clamps_to_unit_interval():
    score, _ = combine_scores(
        path="auth.py",
        keywords=["auth"],
        references=[],
        issue_text="bug crash error",  # bug-flavored
    )
    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_combine_scores_unrelated_file_scores_zero():
    score, sigs = combine_scores(
        path="docs/license.md",
        keywords=["websocket", "transport"],
        references=[],
        issue_text="Connection drops on websocket transport",
    )
    assert score == 0.0
    assert sigs == []
