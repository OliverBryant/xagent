import errno
import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.channels.telegram.bot import TelegramBotInstance
from xagent.web.channels.telegram.handler import TelegramTraceHandler
from xagent.web.models import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import channel_runtime
from xagent.web.services.channel_runtime import TelegramChannelTaskSnapshot


@pytest.fixture
def telegram_db(tmp_path: Path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'telegram.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _bot(channel_id: int) -> TelegramBotInstance:
    bot = object.__new__(TelegramBotInstance)
    bot.channel_id = channel_id
    bot.active_tasks = {}
    bot.user_message_queues = {}
    bot.user_active_executions = {}
    bot.user_active_trace_handlers = {}
    bot.user_preparing_executions = set()
    bot.user_stop_events = {}
    bot._accepting = True
    bot.saved = False

    def _save() -> bool:
        bot.saved = True
        return True

    bot._save_active_tasks = _save
    return bot


def _owner_and_channel(
    db: Session,
    *,
    allowed_users: list[str] | None,
) -> tuple[User, UserChannel]:
    owner = User(username="owner", password_hash="hash")
    db.add(owner)
    db.flush()
    channel = UserChannel(
        user_id=owner.id,
        channel_type="telegram",
        channel_name="Test Telegram",
        config={"allowed_users": allowed_users},
        is_active=True,
    )
    db.add(channel)
    db.commit()
    return owner, channel


def _task(
    db: Session,
    *,
    owner: User,
    channel: UserChannel,
    sender: str | None,
    title: str,
    updated_at: datetime | None = None,
) -> Task:
    task = Task(
        user_id=owner.id,
        channel_id=channel.id,
        channel_name=channel.channel_name,
        telegram_user_id=sender,
        title=title,
        status=TaskStatus.COMPLETED,
        updated_at=updated_at,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _use_telegram_db(
    monkeypatch: pytest.MonkeyPatch,
    db: Session,
) -> None:
    factory = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: factory)


def test_telegram_tasks_are_sender_scoped_and_newest_first(
    telegram_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, channel = _owner_and_channel(
        telegram_db,
        allowed_users=["101", "202"],
    )
    now = datetime.now(UTC)
    older = _task(
        telegram_db,
        owner=owner,
        channel=channel,
        sender="101",
        title="older",
        updated_at=now - timedelta(days=1),
    )
    newer = _task(
        telegram_db,
        owner=owner,
        channel=channel,
        sender="101",
        title="newer",
        updated_at=now,
    )
    _task(
        telegram_db,
        owner=owner,
        channel=channel,
        sender="202",
        title="someone else's task",
    )
    other_channel = UserChannel(
        user_id=owner.id,
        channel_type="telegram",
        channel_name="Other bot",
        config={"allowed_users": ["101"]},
        is_active=True,
    )
    telegram_db.add(other_channel)
    telegram_db.commit()
    _task(
        telegram_db,
        owner=owner,
        channel=other_channel,
        sender="101",
        title="other bot task",
    )

    _use_telegram_db(monkeypatch, telegram_db)
    tasks = channel_runtime._load_telegram_channel_tasks_sync(
        channel_id=int(channel.id),
        external_user_id="101",
        active_task_id=None,
    )

    assert [task.task_id for task in tasks] == [newer.id, older.id]


def test_legacy_history_is_not_bulk_claimed_from_current_single_user_allowlist(
    telegram_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, channel = _owner_and_channel(telegram_db, allowed_users=["101"])
    legacy = _task(
        telegram_db,
        owner=owner,
        channel=channel,
        sender=None,
        title="legacy",
    )

    _use_telegram_db(monkeypatch, telegram_db)
    tasks = channel_runtime._load_telegram_channel_tasks_sync(
        channel_id=int(channel.id),
        external_user_id="101",
        active_task_id=None,
    )
    telegram_db.refresh(legacy)

    assert tasks == ()
    assert legacy.telegram_user_id is None


@pytest.mark.parametrize("allowed_users", [None, ["101", "202"]])
def test_legacy_history_is_not_exposed_when_sender_is_ambiguous(
    telegram_db: Session,
    allowed_users: list[str] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, channel = _owner_and_channel(
        telegram_db,
        allowed_users=allowed_users,
    )
    legacy = _task(
        telegram_db,
        owner=owner,
        channel=channel,
        sender=None,
        title="ambiguous legacy task",
    )

    _use_telegram_db(monkeypatch, telegram_db)
    tasks = channel_runtime._load_telegram_channel_tasks_sync(
        channel_id=int(channel.id),
        external_user_id="101",
        active_task_id=None,
    )
    telegram_db.refresh(legacy)

    assert tasks == ()
    assert legacy.telegram_user_id is None


def test_legacy_active_mapping_can_safely_claim_one_task(
    telegram_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, channel = _owner_and_channel(
        telegram_db,
        allowed_users=["101", "202"],
    )
    legacy = _task(
        telegram_db,
        owner=owner,
        channel=channel,
        sender=None,
        title="active legacy task",
    )
    _use_telegram_db(monkeypatch, telegram_db)
    tasks = channel_runtime._load_telegram_channel_tasks_sync(
        channel_id=int(channel.id),
        external_user_id="101",
        active_task_id=int(legacy.id),
    )
    telegram_db.refresh(legacy)

    assert [task.task_id for task in tasks] == [legacy.id]
    assert legacy.telegram_user_id == "101"


def test_list_limits_results_to_most_recent_tasks(
    telegram_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, channel = _owner_and_channel(telegram_db, allowed_users=["101"])
    for index in range(channel_runtime.TELEGRAM_TASK_LIST_LIMIT + 5):
        _task(
            telegram_db,
            owner=owner,
            channel=channel,
            sender="101",
            title=f"task-{index}",
            updated_at=datetime.now(UTC) + timedelta(seconds=index),
        )

    _use_telegram_db(monkeypatch, telegram_db)
    tasks = channel_runtime._load_telegram_channel_tasks_sync(
        channel_id=int(channel.id),
        external_user_id="101",
        active_task_id=None,
    )

    assert len(tasks) == channel_runtime.TELEGRAM_TASK_LIST_LIMIT
    assert tasks[0].title == (f"task-{channel_runtime.TELEGRAM_TASK_LIST_LIMIT + 4}")


def test_deactivated_channel_rejects_task_history(
    telegram_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _owner, channel = _owner_and_channel(telegram_db, allowed_users=["101"])
    channel.is_active = False
    telegram_db.commit()
    _use_telegram_db(monkeypatch, telegram_db)

    with pytest.raises(channel_runtime.ChannelConfigurationError):
        channel_runtime._load_telegram_channel_tasks_sync(
            channel_id=int(channel.id),
            external_user_id="101",
            active_task_id=None,
        )


@pytest.mark.parametrize("mismatch", ["sender", "channel", "owner"])
def test_active_task_lookup_rejects_mismatched_scope(
    telegram_db: Session,
    mismatch: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, channel = _owner_and_channel(
        telegram_db,
        allowed_users=["101", "202"],
    )
    task_owner = owner
    task_channel = channel
    sender = "101"
    if mismatch == "sender":
        sender = "202"
    elif mismatch == "channel":
        task_channel = UserChannel(
            user_id=owner.id,
            channel_type="telegram",
            channel_name="Other Telegram",
            config={"allowed_users": ["101"]},
            is_active=True,
        )
        telegram_db.add(task_channel)
        telegram_db.commit()
    else:
        task_owner = User(username="other-owner", password_hash="hash")
        telegram_db.add(task_owner)
        telegram_db.commit()
        task_channel = UserChannel(
            user_id=task_owner.id,
            channel_type="telegram",
            channel_name="Other owner's Telegram",
            config={"allowed_users": ["101"]},
            is_active=True,
        )
        telegram_db.add(task_channel)
        telegram_db.commit()

    inaccessible = _task(
        telegram_db,
        owner=task_owner,
        channel=task_channel,
        sender=sender,
        title="must not resume",
    )
    _use_telegram_db(monkeypatch, telegram_db)
    assert (
        channel_runtime._load_telegram_channel_task_sync(
            channel_id=int(channel.id),
            external_user_id="101",
            task_id=int(inaccessible.id),
            active_task_id=None,
        )
        is None
    )


class _Message:
    def __init__(self, sender: int, text: str) -> None:
        self.from_user = SimpleNamespace(id=sender)
        self.text = text
        self.answers: list[str] = []

    async def answer(self, text: str, **_kwargs: object) -> None:
        self.answers.append(text)


@pytest.mark.asyncio
async def test_switch_rejects_another_telegram_senders_task(
    telegram_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, channel = _owner_and_channel(
        telegram_db,
        allowed_users=["101", "202"],
    )
    private_task = _task(
        telegram_db,
        owner=owner,
        channel=channel,
        sender="202",
        title="private",
    )
    bot = _bot(int(channel.id))

    async def load_task(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_telegram_channel_task",
        load_task,
    )
    message = _Message(101, f"/switch {private_task.id}")

    await bot._handle_switch_command(message)  # type: ignore[arg-type]

    assert bot.active_tasks == {}
    assert message.answers == [
        "Task not found or not accessible. Use /list to see your tasks."
    ]


@pytest.mark.asyncio
async def test_switch_stops_current_run_clears_queue_and_persists_selection(
    telegram_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, channel = _owner_and_channel(telegram_db, allowed_users=["101"])
    current = _task(
        telegram_db,
        owner=owner,
        channel=channel,
        sender="101",
        title="current",
    )
    target = _task(
        telegram_db,
        owner=owner,
        channel=channel,
        sender="101",
        title="target",
    )
    bot = _bot(int(channel.id))
    events: list[str] = []

    class _ActiveTasks(dict[int, int]):
        def __setitem__(self, key: int, value: int) -> None:
            events.append(f"select:{value}")
            super().__setitem__(key, value)

    class _Queues(dict[int, list[str]]):
        def pop(self, key: int, default=None):  # type: ignore[no-untyped-def]
            events.append("clear-queue")
            return super().pop(key, default)

    bot.active_tasks = _ActiveTasks({101: int(current.id)})
    bot.user_message_queues = _Queues({101: ["pending"]})

    class _AgentService:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def pause_execution_by_id(
            self,
            execution_id: str,
            reason: str | None = None,
        ) -> bool:
            events.append("pause")
            self.calls.append((execution_id, reason))
            return True

    service = _AgentService()
    bot.user_active_executions[101] = (int(current.id), service)

    class _TraceHandler(TelegramTraceHandler):
        def cancel(self) -> None:
            events.append("cancel-stream")
            super().cancel()

    trace_handler = _TraceHandler(
        task_id=int(current.id),
        bot=object(),  # type: ignore[arg-type]
        chat_id=101,
    )
    bot.user_active_trace_handlers[101] = trace_handler

    snapshot = TelegramChannelTaskSnapshot(
        task_id=int(target.id),
        title=str(target.title),
        status="completed",
        created_at=target.created_at,
        updated_at=target.updated_at,
    )

    async def load_task(**_kwargs: object) -> TelegramChannelTaskSnapshot:
        return snapshot

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_telegram_channel_task",
        load_task,
    )
    message = _Message(101, f"/switch {target.id}")

    await bot._handle_switch_command(message)  # type: ignore[arg-type]

    assert bot.active_tasks[101] == target.id
    assert bot.user_message_queues == {}
    assert bot.saved is True
    assert service.calls == [
        (str(current.id), "Telegram task switch requested"),
    ]
    assert trace_handler.cancelled is True
    # The selection is persisted first: tearing down the old conversation is
    # irreversible, so it must only happen once the switch is durable.
    assert events == [
        f"select:{target.id}",
        "clear-queue",
        "cancel-stream",
        "pause",
    ]
    assert message.answers[-1].startswith(f"Switched to task <code>{target.id}</code>")


def test_task_list_messages_escape_titles_and_stay_below_telegram_limit() -> None:
    bot = _bot(1)
    bot.task_list_message_limit = 260
    tasks = [
        TelegramChannelTaskSnapshot(
            task_id=index,
            title=f"<unsafe & task {index}> " + ("😀" * 100),
            status="completed",
            updated_at=datetime(2026, 7, index, tzinfo=UTC),
            created_at=None,
        )
        for index in range(1, 9)
    ]

    messages = bot._format_task_list_messages(tasks, active_task_id=2)  # type: ignore[arg-type]

    assert len(messages) > 1
    assert all(
        bot._telegram_text_units(message) <= bot.task_list_message_limit
        for message in messages
    )
    assert all("<unsafe" not in message for message in messages)
    assert any("● <code>2</code>" in message for message in messages)
    assert any("50 most recent" in message for message in messages)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/switch 42", 42),
        ("/switch@xagent_bot 42", 42),
        ("/switch", None),
        ("/switch abc", None),
        ("/switch 0", None),
        ("/switch 42 extra", None),
    ],
)
def test_switch_task_id_parsing(text: str, expected: int | None) -> None:
    assert TelegramBotInstance._switch_task_id(text) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("busy_state", ["executing", "preparing", "idle"])
async def test_switch_to_active_task_reports_whether_work_is_underway(
    busy_state: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dequeued batch is active work even before its execution registers."""

    bot = _bot(1)
    bot.active_tasks[101] = 42
    if busy_state == "executing":
        bot.user_active_executions[101] = (42, object())
    elif busy_state == "preparing":
        bot.user_preparing_executions.add(101)
    snapshot = TelegramChannelTaskSnapshot(
        task_id=42,
        title="running",
        status="running",
        created_at=None,
        updated_at=None,
    )

    async def load_task(**_kwargs: object) -> TelegramChannelTaskSnapshot:
        return snapshot

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_telegram_channel_task",
        load_task,
    )
    message = _Message(101, "/switch 42")

    await bot._handle_switch_command(message)  # type: ignore[arg-type]

    if busy_state == "idle":
        assert "still working" not in message.answers[-1]
    else:
        assert "is still working" in message.answers[-1]


@pytest.mark.asyncio
async def test_switch_is_not_confirmed_when_selection_cannot_be_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = _bot(1)
    bot.active_tasks[101] = 7
    bot._save_active_tasks = lambda: False

    snapshot = TelegramChannelTaskSnapshot(
        task_id=9,
        title="target",
        status="completed",
        created_at=None,
        updated_at=None,
    )

    async def load_task(**_kwargs: object) -> TelegramChannelTaskSnapshot:
        return snapshot

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_telegram_channel_task",
        load_task,
    )
    message = _Message(101, "/switch 9")

    await bot._handle_switch_command(message)  # type: ignore[arg-type]

    assert bot.active_tasks[101] == 7
    assert "couldn't save the switch" in message.answers[-1]
    assert not any("Switched to task" in answer for answer in message.answers)


def test_save_active_tasks_is_atomic_and_reports_failure(tmp_path: Path) -> None:
    bot = _bot(1)
    bot.instance_id = "inst"
    bot.active_tasks_file = tmp_path / "active.json"
    del bot._save_active_tasks

    bot.active_tasks = {101: 7}
    assert TelegramBotInstance._save_active_tasks(bot) is True
    assert bot.active_tasks_file.read_text() == '{"101": 7}'
    assert list(tmp_path.glob("*.tmp")) == []

    # A serialization failure must not truncate the previously saved mapping.
    bot.active_tasks = {101: object()}
    assert TelegramBotInstance._save_active_tasks(bot) is False
    assert bot.active_tasks_file.read_text() == '{"101": 7}'
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_output_attachments_stop_when_cancelled_mid_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.channels.telegram.utils import TelegramImageRef

    bot = _bot(1)
    handler = TelegramTraceHandler(7, bot=None, chat_id=1, message_id=1)  # type: ignore[arg-type]
    refs = [TelegramImageRef(file_id=f"f{i}", alt_text="a") for i in range(3)]

    async def load_files(**_kwargs: object) -> list:
        return []

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_channel_output_files",
        load_files,
    )

    # Cancelled before the first send: nothing is delivered and no failed refs
    # are returned, so the caller cannot emit a stale fallback message either.
    handler.cancel()
    failed = await bot._send_output_images(
        image_refs=refs,
        user_id=1,
        task_id=7,
        reply_to=_Message(101, ""),  # type: ignore[arg-type]
        is_cancelled=lambda: handler.cancelled,
    )
    assert failed == []


@pytest.mark.asyncio
async def test_failed_switch_save_leaves_old_conversation_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistence is the commit point: a failed save must not tear down the
    previous conversation's queue, trace handler, or running execution."""

    bot = _bot(1)
    bot.active_tasks[101] = 7
    bot.user_message_queues[101] = ["pending"]
    handler = TelegramTraceHandler(7, bot=None, chat_id=1, message_id=1)  # type: ignore[arg-type]
    bot.user_active_trace_handlers[101] = handler

    paused: list[str] = []

    class _AgentService:
        def pause_execution_by_id(
            self,
            execution_id: str,
            reason: str | None = None,
        ) -> bool:
            paused.append(execution_id)
            return True

    bot.user_active_executions[101] = (7, _AgentService())
    bot._save_active_tasks = lambda: False

    snapshot = TelegramChannelTaskSnapshot(
        task_id=9,
        title="target",
        status="completed",
        created_at=None,
        updated_at=None,
    )

    async def load_task(**_kwargs: object) -> TelegramChannelTaskSnapshot:
        return snapshot

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_telegram_channel_task",
        load_task,
    )
    message = _Message(101, "/switch 9")

    await bot._handle_switch_command(message)  # type: ignore[arg-type]

    assert bot.active_tasks[101] == 7
    assert bot.user_message_queues[101] == ["pending"]
    assert handler.cancelled is False
    assert paused == []
    assert "couldn't save the switch" in message.answers[-1]


def test_failed_new_conversation_save_keeps_previous_selection() -> None:
    bot = _bot(1)
    bot.active_tasks[101] = 7
    bot.user_message_queues[101] = ["pending"]
    handler = TelegramTraceHandler(7, bot=None, chat_id=1, message_id=1)  # type: ignore[arg-type]
    bot.user_active_trace_handlers[101] = handler
    bot._save_active_tasks = lambda: False

    stopped, persisted = bot._start_new_conversation(101)

    assert (stopped, persisted) == (False, False)
    assert bot.active_tasks[101] == 7
    assert bot.user_message_queues[101] == ["pending"]
    assert handler.cancelled is False


def test_save_active_tasks_tracks_unsaved_state_for_retry(tmp_path: Path) -> None:
    bot = _bot(1)
    bot.instance_id = "inst"
    bot.active_tasks_file = tmp_path / "active.json"
    bot._active_tasks_unsaved = False
    del bot._save_active_tasks

    bot.active_tasks = {101: object()}
    assert TelegramBotInstance._save_active_tasks(bot) is False
    assert bot._active_tasks_unsaved is True

    # A later successful save clears the retry flag.
    bot.active_tasks = {101: 7}
    assert TelegramBotInstance._save_active_tasks(bot) is True
    assert bot._active_tasks_unsaved is False
    assert bot.active_tasks_file.read_text() == '{"101": 7}'


def test_save_active_tasks_survives_unsupported_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = _bot(1)
    bot.instance_id = "inst"
    bot.active_tasks_file = tmp_path / "active.json"
    bot._active_tasks_unsaved = False
    del bot._save_active_tasks
    bot.active_tasks = {101: 7}

    real_open = os.open

    def fake_open(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(path) == str(tmp_path):
            # Only an errno that means "unsupported" may be tolerated; a bare
            # OSError is indistinguishable from a real durability failure.
            raise OSError(errno.ENOTSUP, "directory fsync unsupported")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", fake_open)

    assert TelegramBotInstance._save_active_tasks(bot) is True
    assert bot.active_tasks_file.read_text() == '{"101": 7}'


@pytest.mark.asyncio
async def test_trace_handler_skips_text_fallback_after_cancellation() -> None:
    """A cancellation landing while the HTML edit is in flight must stop the
    plain-text fallback too."""

    sent: list[str] = []

    class _Bot:
        def __init__(self, handler_box: list) -> None:
            self.handler_box = handler_box

        async def edit_message_text(self, **kwargs: object) -> None:
            if "parse_mode" in kwargs:
                # Cancel mid-flight, then fail so the fallback path is taken.
                self.handler_box[0].cancel()
                raise RuntimeError("bad html")
            sent.append(str(kwargs.get("text")))

    box: list = [None]
    handler = TelegramTraceHandler(7, _Bot(box), chat_id=1, message_id=1)  # type: ignore[arg-type]
    box[0] = handler

    await handler._update_message("hello")

    assert sent == []


def test_agent_selection_is_applied_only_after_a_durable_reset() -> None:
    """A failed reset must not leave the new agent selected: the next message
    would otherwise start a fresh task with an agent the user was told failed."""

    import asyncio

    bot = _bot(1)
    bot.active_tasks[101] = 7
    bot.selected_agents = {101: 3}
    bot.user_conversation_generations = {}
    saved_agents: list[dict] = []
    bot._save_selected_agents = lambda: saved_agents.append(dict(bot.selected_agents))
    bot._save_active_tasks = lambda: False

    class _Callback:
        def __init__(self) -> None:
            self.from_user = SimpleNamespace(id=101)
            self.data = "agsel:default"
            self.message = None
            self.answers: list[tuple[tuple, dict]] = []

        async def answer(self, *args: object, **kwargs: object) -> None:
            self.answers.append((args, kwargs))

    callback = _Callback()
    asyncio.run(bot._handle_agent_selection_callback(callback))  # type: ignore[arg-type]

    assert bot.selected_agents == {101: 3}
    assert saved_agents == []
    assert bot.active_tasks[101] == 7
    assert "couldn't save" in callback.answers[-1][0][0]


def test_format_task_timestamp_normalizes_aware_values_to_utc() -> None:
    from datetime import timedelta

    naive = datetime(2026, 7, 28, 10, 30)
    aware_utc = datetime(2026, 7, 28, 10, 30, tzinfo=UTC)
    aware_offset = datetime(2026, 7, 28, 18, 30, tzinfo=timezone(timedelta(hours=8)))

    fmt = TelegramBotInstance._format_task_timestamp
    # Naive values are the project's SQLite UTC convention; aware values from
    # PostgreSQL may arrive in a non-UTC session timezone.
    assert fmt(naive) == "2026-07-28 10:30"
    assert fmt(aware_utc) == "2026-07-28 10:30"
    assert fmt(aware_offset) == "2026-07-28 10:30"
    assert fmt(None) == "unknown"


def test_help_text_lists_every_registered_command() -> None:
    help_text = TelegramBotInstance._help_text()
    registered = {command.command for command in TelegramBotInstance.bot_commands}

    for name in registered:
        assert f"/{name}" in help_text, name


def test_save_active_tasks_reports_failure_for_real_directory_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a genuinely unsupported fsync is tolerated. A permission or I/O
    error must leave the save dirty rather than claiming durability."""

    import errno
    import os as os_module

    bot = _bot(1)
    bot.instance_id = "inst"
    bot.active_tasks_file = tmp_path / "active.json"
    bot._active_tasks_unsaved = False
    del bot._save_active_tasks
    bot.active_tasks = {101: 7}

    real_open = os_module.open

    def fake_open(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(path) == str(tmp_path):
            raise OSError(errno.EACCES, "permission denied")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os_module, "open", fake_open)

    assert TelegramBotInstance._save_active_tasks(bot) is False
    assert bot._active_tasks_unsaved is True

    # An unsupported operation still counts as a successful save.
    def unsupported_open(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(path) == str(tmp_path):
            raise OSError(errno.EINVAL, "not supported")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os_module, "open", unsupported_open)

    assert TelegramBotInstance._save_active_tasks(bot) is True
    assert bot._active_tasks_unsaved is False
