"""Tests for Auto failing a call over to its fallback model."""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx
import openai
import pytest

from xagent.core.model.chat.basic import router as router_module
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.call_boundary import ProviderCallError
from xagent.core.model.chat.basic.router import RouterLLM
from xagent.core.model.chat.exceptions import LLMToolProtocolError
from xagent.core.model.chat.types import ChunkType, StreamChunk

PRIMARY = "deepseek/deepseek-v4-flash"
FALLBACK = "openai/gpt-5.5"
MESSAGES = [{"role": "user", "content": "summarize the report"}]

_REQUEST = httpx.Request("POST", "https://provider.example/v1/chat/completions")


def _status_error(cls: type[openai.APIStatusError], status: int) -> Exception:
    return cls(
        "provider said no",
        response=httpx.Response(status, request=_REQUEST),
        body=None,
    )


class _FakeLLM(BaseLLM):
    def __init__(
        self,
        model_id: str,
        *,
        error: BaseException | None = None,
        chunks: list[StreamChunk] | None = None,
        error_after_chunks: bool = False,
    ) -> None:
        self._model_id = model_id
        self.error = error
        self.chunks = chunks or [
            StreamChunk(type=ChunkType.TOKEN, content=model_id, delta=model_id)
        ]
        self.error_after_chunks = error_after_chunks
        self.calls = 0

    @property
    def abilities(self) -> list[str]:
        return ["chat", "tool_calling"]

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        del messages, kwargs
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self._model_id

    async def stream_chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[StreamChunk]:
        del messages, kwargs
        self.calls += 1
        if self.error is not None and not self.error_after_chunks:
            raise self.error
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


@pytest.fixture(autouse=True)
def _fresh_cooldowns():
    router_module._CANDIDATE_COOLDOWNS.clear()
    yield
    router_module._CANDIDATE_COOLDOWNS.clear()


def _router(
    monkeypatch: pytest.MonkeyPatch,
    models: dict[str, _FakeLLM],
    *,
    selected: str = PRIMARY,
    fallback: str | None = FALLBACK,
    configured: bool = True,
) -> RouterLLM:
    router = RouterLLM(
        downstream_resolver=lambda profile: models[profile],
        candidate_models=list(models) if configured else None,
        fallback_model=fallback,
        use_environment_fallback=False,
    )

    async def select_model(_prompt: str, **_kwargs: Any) -> str:
        return selected

    monkeypatch.setattr(router, "_select_model", select_model)
    monkeypatch.setattr(router, "_profile_context_window", lambda _model: None)
    return router


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=_REQUEST),
        openai.APITimeoutError(request=_REQUEST),
        _status_error(openai.InternalServerError, 503),
        _status_error(openai.RateLimitError, 429),
        _status_error(openai.AuthenticationError, 401),
        _status_error(openai.NotFoundError, 404),
        RuntimeError("DeepSeek request failed"),
        ProviderCallError("provider_unavailable"),
    ],
    ids=[
        "connection",
        "timeout",
        "503",
        "429",
        "401",
        "404",
        "wrapped-connection",
        "guarded-unavailable",
    ],
)
async def test_availability_failure_fails_over_to_fallback(monkeypatch, error):
    if type(error) is RuntimeError:
        # Adapters re-raise provider errors wrapped in a RuntimeError.
        error.__cause__ = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM("saved-deepseek", error=error)
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    assert await prepared.chat(MESSAGES) == "saved-gpt"
    assert (primary.calls, fallback.calls) == (1, 1)
    # Usage and later calls on this selection belong to the fallback now.
    assert prepared.model_name == FALLBACK
    assert prepared.model_id == "saved-gpt"
    assert await prepared.chat(MESSAGES) == "saved-gpt"
    assert (primary.calls, fallback.calls) == (1, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("This model's maximum context length is 128000 tokens"),
        _status_error(openai.BadRequestError, 400),
        LLMToolProtocolError(
            provider="deepseek",
            code="malformed_tool_arguments",
            message="bad tool arguments",
        ),
        RuntimeError("unexpected response shape"),
    ],
    ids=["context-length", "400", "tool-protocol", "unrecognized"],
)
async def test_request_failure_does_not_fail_over(monkeypatch, error):
    primary = _FakeLLM("saved-deepseek", error=error)
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    with pytest.raises(type(error)) as raised:
        await prepared.chat(MESSAGES)
    assert raised.value is error
    assert fallback.calls == 0
    assert not router_module._CANDIDATE_COOLDOWNS.active("saved-deepseek")


@pytest.mark.asyncio
async def test_failed_candidate_is_skipped_until_its_cooldown_ends(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(router_module.time, "monotonic", lambda: now[0])
    monkeypatch.setenv("XAGENT_AUTO_CANDIDATE_COOLDOWN_SECONDS", "60")
    primary = _FakeLLM(
        "saved-deepseek", error=openai.APIConnectionError(request=_REQUEST)
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    first = await router.prepare_for_call(MESSAGES)
    assert await first.chat(MESSAGES) == "saved-gpt"
    assert primary.calls == 1

    now[0] += 59
    during = await router.prepare_for_call(MESSAGES)
    assert during.model_name == FALLBACK
    assert await during.chat(MESSAGES) == "saved-gpt"
    assert primary.calls == 1

    now[0] += 2
    primary.error = None
    after = await router.prepare_for_call(MESSAGES)
    assert after.model_name == PRIMARY
    assert await after.chat(MESSAGES) == "saved-deepseek"
    assert primary.calls == 2


@pytest.mark.asyncio
async def test_legacy_auto_fails_over_without_a_shared_cooldown(monkeypatch):
    primary = _FakeLLM(
        "router:deepseek", error=openai.APIConnectionError(request=_REQUEST)
    )
    fallback = _FakeLLM("router:gpt")
    router = _router(
        monkeypatch, {PRIMARY: primary, FALLBACK: fallback}, configured=False
    )

    assert await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES) == (
        "router:gpt"
    )
    # Legacy children share ids across callers with different keys, so one
    # caller's failure must not reroute anyone else.
    second = await router.prepare_for_call(MESSAGES)
    assert second.model_name == PRIMARY
    assert primary.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected", "fallback"),
    [(PRIMARY, None), (FALLBACK, FALLBACK)],
    ids=["no-fallback", "selected-is-fallback"],
)
async def test_nothing_to_fail_over_to_raises(monkeypatch, selected, fallback):
    error = openai.APIConnectionError(request=_REQUEST)
    models = {
        PRIMARY: _FakeLLM("saved-deepseek", error=error),
        FALLBACK: _FakeLLM("saved-gpt", error=error),
    }
    router = _router(monkeypatch, models, selected=selected, fallback=fallback)

    with pytest.raises(openai.APIConnectionError):
        await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES)
    assert sum(model.calls for model in models.values()) == 1


@pytest.mark.asyncio
async def test_fallback_without_a_required_modality_is_not_used(monkeypatch):
    error = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM("saved-deepseek", error=error)
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})
    monkeypatch.setattr(
        router,
        "_profile_input_modalities",
        lambda model: ("text", "image") if model == PRIMARY else ("text",),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            ],
        }
    ]

    with pytest.raises(openai.APIConnectionError):
        await (await router.prepare_for_call(messages)).chat(messages)
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_fallback_failure_is_reported_without_the_first_error(monkeypatch):
    primary = _FakeLLM(
        "saved-deepseek", error=openai.APIConnectionError(request=_REQUEST)
    )
    fallback_error = _status_error(openai.InternalServerError, 503)
    fallback = _FakeLLM("saved-gpt", error=fallback_error)
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    with pytest.raises(openai.InternalServerError) as raised:
        await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES)
    assert raised.value is fallback_error
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_fallback_that_cannot_be_built_reports_the_first_error(monkeypatch):
    error = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM("saved-deepseek", error=error)

    def resolve(profile: str) -> BaseLLM:
        if profile == FALLBACK:
            raise RuntimeError("fallback endpoint is misconfigured")
        return primary

    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: primary})
    router._downstream_resolver = resolve

    with pytest.raises(openai.APIConnectionError) as raised:
        await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES)
    assert raised.value is error


async def _collect(stream: AsyncIterator[StreamChunk]) -> list[str]:
    return [chunk.content async for chunk in stream]


@pytest.mark.asyncio
async def test_stream_fails_over_before_its_first_chunk(monkeypatch):
    primary = _FakeLLM(
        "saved-deepseek", error=_status_error(openai.InternalServerError, 502)
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    assert await _collect(prepared.stream_chat(MESSAGES)) == ["saved-gpt"]
    assert prepared.model_id == "saved-gpt"


@pytest.mark.asyncio
async def test_stream_fails_over_on_a_leading_error_chunk(monkeypatch):
    error = openai.APIConnectionError(request=_REQUEST)
    primary = _FakeLLM(
        "saved-claude",
        chunks=[StreamChunk(type=ChunkType.ERROR, content="failed", raw=error)],
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    prepared = await router.prepare_for_call(MESSAGES)

    assert await _collect(prepared.stream_chat(MESSAGES)) == ["saved-gpt"]


@pytest.mark.asyncio
async def test_stream_does_not_fail_over_after_its_first_chunk(monkeypatch):
    primary = _FakeLLM(
        "saved-deepseek",
        error=openai.APIConnectionError(request=_REQUEST),
        error_after_chunks=True,
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    received: list[str] = []
    with pytest.raises(openai.APIConnectionError):
        async for chunk in (await router.prepare_for_call(MESSAGES)).stream_chat(
            MESSAGES
        ):
            received.append(chunk.content)
    assert received == ["saved-deepseek"]
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_rate_limit_fails_over_without_a_cooldown(monkeypatch):
    primary = _FakeLLM(
        "saved-deepseek", error=_status_error(openai.RateLimitError, 429)
    )
    fallback = _FakeLLM("saved-gpt")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: fallback})

    assert await (await router.prepare_for_call(MESSAGES)).chat(MESSAGES) == (
        "saved-gpt"
    )
    assert (await router.prepare_for_call(MESSAGES)).model_name == PRIMARY


@pytest.mark.asyncio
async def test_cooldown_keeps_the_candidate_when_the_fallback_cannot_be_built(
    monkeypatch,
):
    primary = _FakeLLM("saved-deepseek")
    router = _router(monkeypatch, {PRIMARY: primary, FALLBACK: primary})

    def resolve(profile: str) -> BaseLLM:
        if profile == FALLBACK:
            raise RuntimeError("fallback endpoint is misconfigured")
        return primary

    router._downstream_resolver = resolve
    router_module._CANDIDATE_COOLDOWNS.start("saved-deepseek", 60)

    prepared = await router.prepare_for_call(MESSAGES)

    assert prepared.model_name == PRIMARY
    assert await prepared.chat(MESSAGES) == "saved-deepseek"
