from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.channels.telegram.bot import TelegramBotInstance
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
    bot.user_preparing_executions = set()
    bot.user_stop_events = {}
    bot._accepting = True
    bot.saved = False
    bot._save_active_tasks = lambda: setattr(bot, "saved", True)
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


def test_legacy_history_backfills_only_for_exclusive_allowed_sender(
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

    assert [task.task_id for task in tasks] == [legacy.id]
    assert legacy.telegram_user_id == "101"


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
    assert events == ["clear-queue", "pause", f"select:{target.id}"]
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


def test_stale_task_output_is_not_delivered_after_switch() -> None:
    bot = _bot(1)
    bot.active_tasks[101] = 42

    assert bot._is_active_telegram_task(101, 42) is True
    assert bot._is_active_telegram_task(101, 41) is False
