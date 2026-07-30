"""Concurrent channel-manager syncs must not act on a stale bot view.

Channel CRUD endpoints schedule ``_sync_bots_async`` as a background task, so
two syncs can overlap. Each one reads ``self.bots`` and then awaits while
starting or stopping bots; without serialization the second sync observes the
pre-start view and starts a duplicate bot for the same credential.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from xagent.web.channels.feishu.bot import FeishuChannelManager
from xagent.web.channels.telegram.bot import TelegramChannelManager
from xagent.web.services.channel_runtime import ChannelConfigSnapshot


@pytest.mark.asyncio
async def test_telegram_concurrent_sync_starts_each_bot_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TelegramChannelManager()
    snapshot = ChannelConfigSnapshot(
        channel_id=1,
        channel_name="Telegram Bot",
        config_items=(("bot_token", "tg-token"),),
    )

    async def load_configs(**_kwargs: Any) -> tuple[ChannelConfigSnapshot, ...]:
        # Yield so a second sync can interleave right after the DB read.
        await asyncio.sleep(0)
        return (snapshot,)

    started: list[str] = []

    async def start_bot(token: str, *_args: Any, **_kwargs: Any) -> None:
        started.append(token)
        await asyncio.sleep(0)
        manager.bots[token] = object()  # type: ignore[assignment]

    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_active_channel_configs",
        load_configs,
    )
    manager._start_bot_for_token = start_bot  # type: ignore[method-assign]

    await asyncio.gather(manager._sync_bots_async(), manager._sync_bots_async())

    assert started == ["tg-token"]


@pytest.mark.asyncio
async def test_feishu_concurrent_sync_starts_each_bot_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = FeishuChannelManager()
    snapshot = ChannelConfigSnapshot(
        channel_id=2,
        channel_name="Feishu Bot",
        config_items=(("app_id", "cli_123"), ("app_secret", "secret")),
    )

    async def load_configs(**_kwargs: Any) -> tuple[ChannelConfigSnapshot, ...]:
        await asyncio.sleep(0)
        return (snapshot,)

    started: list[str] = []

    async def start_bot(app_id: str, *_args: Any, **_kwargs: Any) -> None:
        started.append(app_id)
        await asyncio.sleep(0)
        manager.bots[app_id] = object()  # type: ignore[assignment]

    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.load_active_channel_configs",
        load_configs,
    )
    manager._start_bot_for_appid = start_bot  # type: ignore[method-assign]

    await asyncio.gather(manager._sync_bots_async(), manager._sync_bots_async())

    assert started == ["cli_123"]
