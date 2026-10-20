"""``usage_scope`` delivers non-run usage to the standalone hook exactly once."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from xagent.core.model.chat.token_context import (
    MediaCallType,
    TokenContextManager,
    add_media_usage,
    get_token_usage,
    token_context,
)
from xagent.web.services import quota_hooks
from xagent.web.tracking.standalone_usage import usage_scope


class _Hook:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, list[dict], str]] = []
        self.threads: list[int] = []

    def __call__(self, user_id: Any, details: list[dict], source: str) -> None:
        self.calls.append((user_id, details, source))
        self.threads.append(threading.get_ident())


@pytest.fixture
def hook():
    recorder = _Hook()
    quota_hooks.set_standalone_usage_hook(recorder)
    yield recorder
    quota_hooks.set_standalone_usage_hook(None)


def _embed(texts: int = 1) -> None:
    add_media_usage(call_type=MediaCallType.EMBEDDING, quantity=texts, model_id="e")


def test_sync_scope_delivers_its_rows_once(hook):
    with usage_scope(7, usage_source="kb_ingest_job") as usage:
        _embed(2)
        _embed(3)
        assert get_token_usage() is usage

    [(user_id, details, source)] = hook.calls
    assert (user_id, source) == (7, "kb_ingest_job")
    assert [row["quantity"] for row in details] == [2.0, 3.0]


def test_rows_recorded_before_the_body_raises_are_delivered_once(hook):
    with pytest.raises(RuntimeError, match="after billing"):
        with usage_scope(7, usage_source="kb_search"):
            _embed()
            raise RuntimeError("after billing")

    assert len(hook.calls) == 1
    assert [row["quantity"] for row in hook.calls[0][1]] == [1.0]


async def test_async_scope_delivers_off_the_event_loop(hook):
    loop_thread = threading.get_ident()
    async with usage_scope(3, usage_source="speech_transcribe"):
        _embed()
        # asyncio.to_thread carries the scope into the worker.
        await asyncio.to_thread(_embed)

    [(user_id, details, _source)] = hook.calls
    assert user_id == 3 and len(details) == 2
    assert hook.threads[0] != loop_thread


async def test_async_scope_raising_body_delivers_once(hook):
    with pytest.raises(ValueError):
        async with usage_scope(3, usage_source="telegram_voice"):
            _embed()
            raise ValueError("transcript empty")

    assert len(hook.calls) == 1


async def test_cancelled_delivery_still_delivers_once(hook):
    entered = threading.Event()
    release = threading.Event()

    def slow_hook(user_id, details, source):
        entered.set()
        release.wait(5)
        hook(user_id, details, source)

    quota_hooks.set_standalone_usage_hook(slow_hook)

    async def scoped():
        async with usage_scope(3, usage_source="kb_search"):
            _embed()

    task = asyncio.create_task(scoped())
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    for _ in range(100):
        if hook.calls:
            break
        await asyncio.sleep(0.01)
    assert len(hook.calls) == 1


def test_run_in_executor_needs_the_context_handed_over(hook):
    async def ingest(handed_over: bool):
        loop = asyncio.get_running_loop()
        async with usage_scope(5, usage_source="kb_ingest"):
            if handed_over:
                await loop.run_in_executor(None, contextvars.copy_context().run, _embed)
            else:
                await loop.run_in_executor(None, _embed)

    asyncio.run(ingest(handed_over=False))
    assert hook.calls == []
    asyncio.run(ingest(handed_over=True))
    assert len(hook.calls) == 1


def test_nothing_recorded_means_no_call(hook):
    with usage_scope(1, usage_source="memory_api"):
        pass
    assert hook.calls == []


def test_without_a_hook_the_scope_is_inert():
    quota_hooks.set_standalone_usage_hook(None)
    with usage_scope(1, usage_source="memory_api") as usage:
        _embed()
    assert len(usage.details) == 1


def test_unattributed_usage_is_delivered_and_logged(hook, caplog):
    caplog.set_level(logging.INFO, logger="xagent.web.tracking.standalone_usage")
    with usage_scope(None, usage_source="kb_ingest_job"):
        _embed()

    assert hook.calls[0][0] is None
    assert "no attributable user" in caplog.text


def test_a_failing_hook_never_fails_the_work(caplog):
    def broken(*_args):
        raise RuntimeError("db down")

    quota_hooks.set_standalone_usage_hook(broken)
    try:
        with usage_scope(1, usage_source="kb_search"):
            _embed()
    finally:
        quota_hooks.set_standalone_usage_hook(None)
    assert "Standalone usage hook failed" in caplog.text


def test_scope_restores_the_previous_binding_and_nests(hook):
    with TokenContextManager() as outer:
        with usage_scope(1, usage_source="outer"):
            _embed(1)
            with usage_scope(2, usage_source="inner"):
                _embed(2)
        assert token_context.get() is outer.get_usage()

    assert [(c[0], [r["quantity"] for r in c[1]]) for c in hook.calls] == [
        (2, [2.0]),
        (1, [1.0]),
    ]
    assert outer.get_usage().details == []


def test_delivered_rows_are_copies(hook):
    with usage_scope(1, usage_source="kb_search") as usage:
        _embed()
    hook.calls[0][1][0]["quantity"] = 99
    assert usage.details[0]["quantity"] == 1.0


def test_pool_threads_do_not_inherit_the_scope(hook):
    with usage_scope(1, usage_source="kb_search"):
        with ThreadPoolExecutor(1) as pool:
            pool.submit(_embed).result()
    assert hook.calls == []
