"""Every channel turn binds the task row's ``source`` into its agent context.

``task_source`` is the key the MCP approval gate selects a registration by
(``mcp_approval_gate``'s module docstring: an unregistered or absent source is
not gated, so a host that forgets to bind it simply gets no approval prompt).
The three chat bots build their own ``context`` dict and call
``AgentService.execute_task`` directly, bypassing the WebSocket turn path that
binds it, so each one has to bind it itself -- from the task row it already
loaded, never from the inbound chat event.

The shared-turn path does not carry these dicts at all: it re-builds one in
``shared_channel_execution.execute_channel_background`` and is covered there.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from xagent.web.channels.feishu.bot import FeishuBotInstance
from xagent.web.channels.slack.bot import SlackBotInstance
from xagent.web.channels.telegram.bot import TelegramBotInstance
from xagent.web.services.task_execution_context_service import (
    TaskExecutionRecoverySnapshot,
)
from xagent.web.services.task_lease_service import TaskLease


def _snapshot(source: str | None) -> Any:
    return SimpleNamespace(
        runtime_user=None,
        conversation_history=(),
        conversation_watermark=None,
        execution_recovery=TaskExecutionRecoverySnapshot(),
        task=SimpleNamespace(source=source),
    )


class _FakeTracer:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def add_handler(self, handler: Any) -> None:
        self.handlers.append(handler)

    def remove_handler(self, handler: Any) -> None:
        if handler in self.handlers:
            self.handlers.remove(handler)


def _agent_service() -> Any:
    return SimpleNamespace(
        workspace=None,
        tracer=_FakeTracer(),
        set_conversation_history=lambda _messages, *, watermark=None: None,
        set_execution_context_messages=lambda _messages: None,
        set_recovered_skill_context=lambda _context: None,
    )


def _agent_manager(contexts: list[Any], service: Any) -> Any:
    class FakeAgentManager:
        async def get_agent_for_task(self, *_args: Any, **_kwargs: Any) -> Any:
            return service

        async def execute_task(self, **kwargs: Any) -> dict[str, Any]:
            contexts.append(kwargs["context"])
            return {"success": True, "output": "ok"}

    return FakeAgentManager()


class _FakeManagedLease:
    heartbeat_task = None

    def __init__(self, lease: TaskLease) -> None:
        self.lease = lease

    async def close(self) -> bool:
        return True

    async def finalize_result(self, **_kwargs: Any) -> bool:
        return True


@pytest.mark.parametrize("source", ["slack", None])
@pytest.mark.asyncio
async def test_slack_turn_binds_the_task_rows_source(
    monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    bot = object.__new__(SlackBotInstance)
    bot.channel_id = 7
    bot.channel_name = "Support Slack"
    bot.bot_user_id = "U_BOT"
    bot.web_client = object()
    bot.active_tasks = {}
    bot.event_queues = {}
    bot.event_tasks = {}
    bot._recent_event_ids = []
    bot._recent_event_id_set = set()
    bot._accepting = True
    bot._save_active_tasks = lambda: None

    managed = _FakeManagedLease(
        TaskLease(task_id=45, runner_id="runner-a", run_id="run-a")
    )
    contexts: list[Any] = []
    service = _agent_service()

    async def prepare(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            user_id=5, task_id=45, is_new_task=True, managed_lease=managed
        )

    async def persist(**_kwargs: Any) -> None:
        return None

    async def send_text(_channel_id: str, _text: str, *, thread_ts: Any) -> str:
        return "loading-ts"

    async def send_final_text(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("xagent.web.channels.slack.bot.prepare_channel_task", prepare)
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.load_task_setup_snapshot_sync",
        lambda *_args: _snapshot(source),
    )
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.get_agent_manager",
        lambda: _agent_manager(contexts, service),
    )
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.persist_channel_user_message", persist
    )
    bot._send_text = send_text
    bot._send_final_text = send_final_text

    await bot._process_event(
        "T1:D1:U1:direct",
        {},
        {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "user": "U1",
            "ts": "1.0",
            "text": "hello",
        },
    )

    assert len(contexts) == 1
    assert contexts[0]["task_source"] == source


@pytest.mark.parametrize("source", ["telegram", None])
@pytest.mark.asyncio
async def test_telegram_turn_binds_the_task_rows_source(
    monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    bot = object.__new__(TelegramBotInstance)
    bot.channel_id = 1
    bot.channel_name = "Telegram source binding"
    bot.active_tasks = {}
    bot.bot = object()
    bot._accepting = True
    bot._ingress_stopped = False
    bot._stop_lock = None
    bot._stop_loop = None
    bot.user_message_queues = {}
    bot.user_message_tasks = {}
    bot.user_active_executions = {}
    bot.user_active_trace_handlers = {}
    bot.user_preparing_executions = set()
    bot.user_stop_events = {}
    bot.user_conversation_generations = {}
    bot.user_switch_locks = {}
    bot.selected_agents = {}
    bot._save_selected_agents = lambda: True
    bot._save_active_tasks = lambda: True
    bot._consume_user_stop_request = lambda _user_id: False
    bot._clear_user_stop_request = lambda _user_id: None

    managed = _FakeManagedLease(
        TaskLease(task_id=48, runner_id="runner-a", run_id="run-a")
    )
    contexts: list[Any] = []
    service = _agent_service()

    async def prepare(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            user_id=5,
            task_id=48,
            is_new_task=True,
            managed_lease=managed,
            requested_agent_missing=False,
        )

    async def persist(**_kwargs: Any) -> None:
        return None

    async def extract_text(_message: Any) -> tuple[str, list[Any]]:
        return "hello", []

    async def await_execution(_user_id: Any, execution: Any, *, reason: Any) -> Any:
        return await execution

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.prepare_channel_task", prepare
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_task_setup_snapshot_sync",
        lambda *_args: _snapshot(source),
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.get_agent_manager",
        lambda: _agent_manager(contexts, service),
    )
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.persist_channel_user_message", persist
    )
    bot._extract_message_content = extract_text
    bot._await_execution_with_stop_monitor = await_execution

    class LoadingMessage:
        message_id = 77

        async def edit_text(self, _text: str, **_kwargs: Any) -> None:
            return None

        async def delete(self) -> None:
            return None

    class Message:
        from_user = SimpleNamespace(id=123)
        chat = SimpleNamespace(id=456)

        async def answer(self, _text: str, **_kwargs: Any) -> LoadingMessage:
            return LoadingMessage()

    await bot._process_user_messages_batch(123, [Message()])

    assert len(contexts) == 1
    assert contexts[0]["task_source"] == source


@pytest.mark.parametrize("source", ["feishu", None])
@pytest.mark.asyncio
async def test_feishu_turn_binds_the_task_rows_source(
    monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    bot = object.__new__(FeishuBotInstance)
    bot.channel_id = 1
    bot.channel_name = "Feishu source binding"
    bot.active_tasks = {"open-id": "45"}
    bot.api_client = object()
    bot._save_active_tasks = lambda: None

    managed = _FakeManagedLease(
        TaskLease(task_id=45, runner_id="runner-a", run_id="run-a")
    )
    contexts: list[Any] = []
    service = _agent_service()

    async def prepare(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            user_id=5, task_id=45, is_new_task=False, managed_lease=managed
        )

    async def persist(**_kwargs: Any) -> None:
        return None

    async def send_text(_chat_id: str, _text: str) -> str:
        return "loading-message-id"

    async def update_text(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("xagent.web.channels.feishu.bot.prepare_channel_task", prepare)
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.load_task_setup_snapshot_sync",
        lambda *_args: _snapshot(source),
    )
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.get_agent_manager",
        lambda: _agent_manager(contexts, service),
    )
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.persist_channel_user_message", persist
    )
    bot._send_text = send_text
    bot._update_text = update_text

    message = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                chat_id="chat-id",
                message_id="message-id",
                message_type="text",
                content='{"text": "hello"}',
            )
        )
    )
    await bot._process_messages_batch("open-id", [message])

    assert len(contexts) == 1
    assert contexts[0]["task_source"] == source


@pytest.mark.asyncio
async def test_shared_channel_turn_binds_the_snapshot_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared-turn executor is the fourth channel entry point.

    When shared task execution is enabled all three bots hand their turn to
    ``execute_channel_background``, which builds its own context dict instead
    of forwarding theirs. The binding has to exist there too, or the gate
    silently stops firing for every tenant on the shared path.
    """

    from xagent.web.services import shared_channel_execution

    contexts: list[Any] = []
    manager = _agent_manager(contexts, _agent_service())
    snapshot = SimpleNamespace(
        runtime_user=None,
        conversation_history=(),
        conversation_watermark=None,
        execution_recovery=TaskExecutionRecoverySnapshot(),
        task=SimpleNamespace(source="slack", user_id=5),
    )

    async def _materialize(_recovery: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(
        "xagent.web.services.agent_service_manager.get_agent_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        "xagent.web.services.task_execution_context_service."
        "materialize_task_execution_recovery_state",
        _materialize,
    )

    class _Forwarder:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def close(self, *, drain: bool = False) -> None:
            return None

    monkeypatch.setattr(
        shared_channel_execution, "ChannelProgressForwarder", _Forwarder
    )
    monkeypatch.setattr(
        shared_channel_execution, "get_task_event_bridge", lambda: MagicMock()
    )

    # Stop right after execute_task: everything past it (result projection,
    # lease finalization, command settlement) needs a live database and is
    # covered by test_shared_channel_execution.py. A sentinel is used rather
    # than a bare ``except Exception`` so an unrelated failure inside
    # execute_task is not swallowed.
    class _StopAfterExecute(Exception):
        pass

    def _explode(_result: Any) -> Any:
        raise _StopAfterExecute

    monkeypatch.setattr(
        "xagent.web.services.execution_result_projection."
        "project_execution_result_for_channel",
        _explode,
    )

    with pytest.raises(_StopAfterExecute):
        await shared_channel_execution.execute_channel_background(
            command=SimpleNamespace(task_id=45, command_id="cmd-1"),
            lease=TaskLease(task_id=45, runner_id="runner-a", run_id="run-a"),
            heartbeat_task=None,
            snapshot=snapshot,
            payload=SimpleNamespace(
                for_agent="hello",
                transcript_message="hello",
                attachments=[],
                file_ids=(),
            ),
        )

    assert len(contexts) == 1
    assert contexts[0]["task_source"] == "slack"
