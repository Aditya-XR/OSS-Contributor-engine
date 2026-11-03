"""HTTP route tests for /investigations/{id}/pilot.

We exercise auth, ownership checks, the "must be completed" precondition,
the concurrent-pilot 409, and the basic GET/list shapes. The background
task spawn is patched out — we test that it's *invoked correctly*, not
the loop itself (Reviewer tests already cover that).
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from app.db.models import Investigation, Issue, PilotRun, Repo


def _seed_inv(
    session, user, *, status: str = "completed",
) -> Investigation:
    repo = Repo(
        id=991, full_name="acme/foo", name="foo",
        html_url="https://github.com/acme/foo",
    )
    session.add(repo)
    session.flush()
    issue = Issue(
        id=10001, repo_id=repo.id, number=7, title="t",
        html_url="https://github.com/acme/foo/issues/7",
        issue_created_at=datetime.now(UTC).replace(tzinfo=None),
        issue_updated_at=datetime.now(UTC).replace(tzinfo=None),
    )
    session.add(issue)
    session.flush()
    inv = Investigation(
        user_id=user.id, issue_id=issue.id, status=status,
        report_md="# r",
    )
    session.add(inv)
    session.commit()
    return inv


@pytest.fixture(autouse=True)
def _pilot_route_env(api_app, monkeypatch):
    """Keep these tests independent of the developer's `.env`.

    `create_pilot` depends on `get_llm_router`, which lazily builds a real
    router from GEMINI_API_KEY / GROQ_API_KEY and answers 503 when neither
    is set (a fresh clone, CI). And a local `PILOT_ENABLED=false` would 503
    every request too. The spawn is patched out, so the router is never
    called: any non-None object will do.
    """
    monkeypatch.setattr("app.api.routes.pilot.settings.pilot_enabled", True)
    api_app.state.llm_router = object()


@pytest.fixture
def patched_spawn(monkeypatch):
    """Stub out the pilot's background spawn so the test doesn't fire
    a real LLM/sandbox run. Returns a MagicMock recording all calls.

    Closes the inbound coroutine so we don't get RuntimeWarning about
    coroutines never being awaited.
    """
    mock = MagicMock()

    def _capture(coro, *args, **kwargs):
        # Close the awaitable so Python doesn't whine about it.
        if hasattr(coro, "close"):
            coro.close()
        return mock(coro, *args, **kwargs)

    monkeypatch.setattr("app.api.routes.pilot.spawn", _capture)
    return mock


# ---------------------------------------------------------------------------
# Auth + ownership
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_create_pilot_requires_auth(client):
    r = client.post("/investigations/some-id/pilot")
    assert r.status_code == 401


@pytest.mark.unit
def test_get_pilot_requires_auth(client):
    r = client.get("/investigations/some-id/pilot")
    assert r.status_code == 401


@pytest.mark.unit
def test_create_pilot_returns_404_for_other_users_investigation(
    client, session, make_logged_in_user, patched_spawn,
):
    """A user who didn't create the investigation gets 404, not 403 —
    we don't leak existence."""
    make_logged_in_user(github_login="alice", github_id=11)
    # Different user owns the investigation.
    from app.db.models import User
    bob = User(github_login="bob", github_id=22, name=None)
    session.add(bob)
    session.flush()
    inv = _seed_inv(session, bob)

    r = client.post(f"/investigations/{inv.id}/pilot")
    assert r.status_code == 404
    assert patched_spawn.called is False
    # Also: GET should 404 for the same reason
    assert client.get(f"/investigations/{inv.id}/pilot").status_code == 404


@pytest.mark.unit
def test_create_pilot_404_for_nonexistent_investigation(
    client, make_logged_in_user, patched_spawn,
):
    make_logged_in_user()
    r = client.post("/investigations/does-not-exist/pilot")
    assert r.status_code == 404
    assert patched_spawn.called is False


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pilot_refuses_non_completed_investigation(
    client, session, make_logged_in_user, patched_spawn,
):
    user = make_logged_in_user()
    inv = _seed_inv(session, user, status="running")
    r = client.post(f"/investigations/{inv.id}/pilot")
    assert r.status_code == 409
    assert "completed" in r.json()["detail"]
    assert patched_spawn.called is False


@pytest.mark.unit
def test_pilot_refuses_concurrent_run(
    client, session, make_logged_in_user, patched_spawn,
):
    user = make_logged_in_user()
    inv = _seed_inv(session, user)

    # Seed a queued pilot for this investigation directly.
    existing = PilotRun(
        investigation_id=inv.id, user_id=user.id, status="queued",
    )
    session.add(existing)
    session.commit()

    r = client.post(f"/investigations/{inv.id}/pilot")
    assert r.status_code == 409
    assert "already running" in r.json()["detail"]
    assert patched_spawn.called is False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_create_pilot_returns_202_and_spawns_background(
    client, session, make_logged_in_user, patched_spawn,
):
    user = make_logged_in_user()
    inv = _seed_inv(session, user)

    r = client.post(f"/investigations/{inv.id}/pilot")
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert "pilot_id" in body

    # Spawn must have been called exactly once with our pilot id.
    assert patched_spawn.call_count == 1

    # PilotRun row exists with status='queued'.
    row = session.get(PilotRun, body["pilot_id"])
    assert row is not None
    assert row.investigation_id == inv.id
    assert row.user_id == user.id
    assert row.status == "queued"


@pytest.mark.unit
def test_get_latest_pilot_returns_most_recent(
    client, session, make_logged_in_user,
):
    user = make_logged_in_user()
    inv = _seed_inv(session, user)
    # Two pilots, explicit timestamps because SQLite's CURRENT_TIMESTAMP
    # has 1-second resolution and we'd otherwise race.
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = datetime(2026, 1, 1, 12, 0, 5)
    p1 = PilotRun(
        investigation_id=inv.id, user_id=user.id, status="rejected",
        summary="first try", attempts_made=3,
        created_at=t0, updated_at=t0,
    )
    session.add(p1)
    session.commit()
    p2 = PilotRun(
        investigation_id=inv.id, user_id=user.id, status="accepted",
        summary="second worked", attempts_made=1, accepted_attempt_number=1,
        accepted_diff="diff --git a/x b/x\n",
        created_at=t1, updated_at=t1,
    )
    session.add(p2)
    session.commit()

    r = client.get(f"/investigations/{inv.id}/pilot")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == p2.id
    assert body["status"] == "accepted"
    assert body["accepted_attempt_number"] == 1
    assert body["accepted_diff"].startswith("diff --git")


@pytest.mark.unit
def test_get_latest_pilot_404_when_none_exist(
    client, session, make_logged_in_user,
):
    user = make_logged_in_user()
    inv = _seed_inv(session, user)
    r = client.get(f"/investigations/{inv.id}/pilot")
    assert r.status_code == 404


@pytest.mark.unit
def test_list_pilots_returns_all_newest_first(
    client, session, make_logged_in_user,
):
    user = make_logged_in_user()
    inv = _seed_inv(session, user)
    base = datetime(2026, 1, 1, 12, 0, 0)
    for i in range(3):
        ts = base.replace(second=i * 5)
        session.add(PilotRun(
            investigation_id=inv.id, user_id=user.id,
            status="rejected", summary=f"try {i}",
            created_at=ts, updated_at=ts,
        ))
        session.commit()

    r = client.get(f"/investigations/{inv.id}/pilot/all")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 3
    summaries = [r["summary"] for r in rows]
    # Newest first => "try 2", "try 1", "try 0"
    assert summaries == ["try 2", "try 1", "try 0"]


@pytest.mark.unit
def test_transcript_json_parses_to_dict(
    client, session, make_logged_in_user,
):
    """Round-trip: store a transcript JSON, fetch via API, verify it's a dict."""
    user = make_logged_in_user()
    inv = _seed_inv(session, user)
    session.add(PilotRun(
        investigation_id=inv.id, user_id=user.id, status="accepted",
        transcript_json='{"success": true, "attempts": [{"attempt_number": 1}]}',
    ))
    session.commit()

    r = client.get(f"/investigations/{inv.id}/pilot")
    body = r.json()
    assert body["transcript"] == {"success": True, "attempts": [{"attempt_number": 1}]}


@pytest.mark.unit
def test_transcript_corrupt_json_becomes_none(
    client, session, make_logged_in_user,
):
    """Defensive — corrupt JSON in DB shouldn't 500 the API."""
    user = make_logged_in_user()
    inv = _seed_inv(session, user)
    session.add(PilotRun(
        investigation_id=inv.id, user_id=user.id, status="rejected",
        transcript_json="this is not json at all",
    ))
    session.commit()

    r = client.get(f"/investigations/{inv.id}/pilot")
    assert r.status_code == 200
    assert r.json()["transcript"] is None


# ---------------------------------------------------------------------------
# Safety rails (Batch 37)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_create_pilot_refused_when_cost_cap_exceeded(
    client, session, make_logged_in_user, patched_spawn, monkeypatch,
):
    from app.api.routes import pilot as pilot_routes
    from app.db.models import AgentRun

    user = make_logged_in_user()
    inv = _seed_inv(session, user)
    session.add(AgentRun(
        user_id=user.id, agent_name="x", provider="gemini",
        model="m", cost_usd=99.0,
    ))
    session.commit()
    monkeypatch.setattr(pilot_routes, "cost_cap_exceeded",
                        lambda s, uid: (True, 99.0, 5.0))

    r = client.post(f"/investigations/{inv.id}/pilot")
    assert r.status_code == 402
    assert "cost cap" in r.json()["detail"].lower()
    assert patched_spawn.called is False


@pytest.mark.unit
def test_create_pilot_refused_when_repo_on_refuse_list(
    client, session, make_logged_in_user, patched_spawn, monkeypatch,
):
    from app.api.routes import pilot as pilot_routes

    user = make_logged_in_user()
    inv = _seed_inv(session, user)  # repo is acme/foo
    monkeypatch.setattr(pilot_routes, "is_repo_refused",
                        lambda full: full == "acme/foo")

    r = client.post(f"/investigations/{inv.id}/pilot")
    assert r.status_code == 403
    assert "refuse-list" in r.json()["detail"]
    assert "acme/foo" in r.json()["detail"]
    assert patched_spawn.called is False


@pytest.mark.unit
def test_create_pilot_allowed_when_cost_under_and_repo_allowed(
    client, session, make_logged_in_user, patched_spawn, monkeypatch,
):
    """Sanity: with cap not exceeded and repo not refused, pilot starts."""
    from app.api.routes import pilot as pilot_routes

    user = make_logged_in_user()
    inv = _seed_inv(session, user)
    monkeypatch.setattr(pilot_routes, "cost_cap_exceeded",
                        lambda s, uid: (False, 0.0, 5.0))
    monkeypatch.setattr(pilot_routes, "is_repo_refused", lambda full: False)

    r = client.post(f"/investigations/{inv.id}/pilot")
    assert r.status_code == 202
    assert patched_spawn.call_count == 1
