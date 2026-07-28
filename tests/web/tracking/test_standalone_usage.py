"""The usage sink for work that has no TaskTracker.

Recording usage is only half of metering: something has to bind a TokenUsage
and hand it to the quota hook. Four production entry points (KB ingest over
HTTP and Celery, /speech/transcribe, Telegram voice) have no TaskTracker, so
without this their recorded usage lands in a throwaway object.
"""

import concurrent.futures
from typing import Any, Optional

import pytest

from xagent.core.model.chat.token_context import add_media_usage, get_token_usage
from xagent.web.tracking.standalone_usage import bind_usage_to_thread, usage_scope


class _FakeSession:
    def __init__(self) -> None:
        self.rolled_back = False
        self.closed = False

    def in_transaction(self) -> bool:
        return False

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch):
    """Capture what reaches quota_hooks.record_usage."""
    calls: list[dict[str, Any]] = []
    session = _FakeSession()

    def _record(db: Any, user_id: Any, details: list, actions: int) -> None:
        calls.append(
            {"db": db, "user_id": user_id, "details": details, "actions": actions}
        )

    from xagent.web.models import database
    from xagent.web.services import quota_hooks

    monkeypatch.setattr(quota_hooks, "record_usage", _record)
    monkeypatch.setattr(database, "get_session_local", lambda: lambda: session)
    return calls, session


def test_usage_recorded_in_scope_reaches_the_quota_hook(captured) -> None:
    calls, _ = captured
    with usage_scope(42):
        add_media_usage(unit="seconds", quantity=3, model="asr-1", call_type="asr")

    assert len(calls) == 1
    assert calls[0]["user_id"] == 42
    assert calls[0]["details"][0]["call_type"] == "asr"
    # These paths make provider calls, not agent tool calls.
    assert calls[0]["actions"] == 0


def test_scope_reports_even_when_the_body_raises(captured) -> None:
    """A provider call that already happened is billable regardless of what
    fails afterwards."""
    calls, _ = captured
    with pytest.raises(RuntimeError):
        with usage_scope(7):
            add_media_usage(unit="images", quantity=1, model="i", call_type="edit_image")
            raise RuntimeError("downstream failure")

    assert len(calls) == 1
    assert calls[0]["details"][0]["call_type"] == "edit_image"


def test_no_usage_means_no_hook_call(captured) -> None:
    calls, _ = captured
    with usage_scope(1):
        pass
    assert calls == []


def test_compatibility_session_is_disposed(captured) -> None:
    """The hook manages its own durability; the session handed to it must be
    closed by us and never left holding a transaction."""
    calls, session = captured
    with usage_scope(5):
        add_media_usage(unit="texts", quantity=2, model="e", call_type="embedding")

    assert session.closed is True
    assert calls[0]["db"] is session


def test_bind_usage_to_thread_carries_usage_across_the_hop(captured) -> None:
    """run_in_executor / ThreadPoolExecutor drop contextvars; the wrapper is
    what keeps the worker's records attached to the caller."""
    calls, _ = captured

    with usage_scope(9):

        def work() -> None:
            add_media_usage(unit="texts", quantity=4, model="e", call_type="embedding")

        bound = bind_usage_to_thread(work)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pool.submit(bound).result()

    assert len(calls) == 1
    assert calls[0]["details"][0]["quantity"] == 4.0


def test_unbound_thread_hop_loses_usage(captured) -> None:
    """Documents why the wrapper exists: the same code without it records
    nothing, which is the defect this module fixes."""
    calls, _ = captured

    with usage_scope(9):

        def work() -> None:
            add_media_usage(unit="texts", quantity=4, model="e", call_type="embedding")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pool.submit(work).result()  # not bound

    assert calls == []


def test_scope_restores_the_previous_context(captured) -> None:
    """A nested scope must not leak its usage object into the outer one."""
    _calls, _ = captured
    with usage_scope(1) as outer:
        add_media_usage(unit="images", quantity=1, model="a", call_type="generate_image")
        with usage_scope(2):
            add_media_usage(unit="images", quantity=1, model="b", call_type="edit_image")
        # Back on the outer usage, and the inner call did not land here.
        assert get_token_usage() is outer
        assert len(outer.details) == 1


def test_none_user_id_is_tolerated(captured) -> None:
    """record_usage no-ops on a missing user; the scope must not raise."""
    calls, _ = captured
    user_id: Optional[int] = None
    with usage_scope(user_id):
        add_media_usage(unit="seconds", quantity=1, model="m", call_type="asr")
    assert calls[0]["user_id"] is None


def test_hook_failure_does_not_break_the_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metering must never break the operation it measures."""
    from xagent.web.models import database
    from xagent.web.services import quota_hooks

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("quota backend down")

    monkeypatch.setattr(quota_hooks, "record_usage", _boom)
    monkeypatch.setattr(database, "get_session_local", lambda: _FakeSession)

    with usage_scope(1):
        add_media_usage(unit="seconds", quantity=1, model="m", call_type="asr")
    # No exception escaped.


def test_patched_entry_points_bind_a_scope() -> None:
    """Guard the wiring itself: if a future refactor drops the scope from an
    entry point, its usage silently stops being billed again."""
    import inspect

    from xagent.web.api import kb, model
    from xagent.web.channels.telegram import bot
    from xagent.web.jobs import kb_tasks

    assert "usage_scope(" in inspect.getsource(kb.ingest)
    assert "usage_scope(" in inspect.getsource(kb.ingest_cloud)
    assert "usage_scope(" in inspect.getsource(kb_tasks.handle_kb_ingest_document)
    assert "usage_scope(" in inspect.getsource(model.transcribe_speech_input)
    assert "usage_scope(" in inspect.getsource(
        bot.TelegramBotInstance._transcribe_uploaded_voice_files
    )
