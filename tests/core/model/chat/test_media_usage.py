"""Media (non-LLM) usage tracking: image/video/tts/asr/embedding/rerank.

These modalities record usage via ``add_media_usage`` into the same
``TokenUsage.details`` list that LLM tokens use, so they flow through DB
persistence and the quota ``delta_details`` contract without special-casing.
"""

import threading

import pytest

from xagent.core.model.chat.token_context import (
    MediaCallType,
    MediaUnit,
    TokenContextManager,
    TokenUsage,
    add_media_usage,
    add_token_usage,
    aggregate_media_usage_by_model,
    aggregate_token_usage_by_model,
)


def test_add_media_usage_appends_media_entry_and_counts_call() -> None:
    with TokenContextManager() as manager:
        add_media_usage(
            unit="images",
            quantity=2,
            model="sd-xl",
            model_id="m1",
            call_type="generate_image",
            resolution="1K",
        )
        usage = manager.get_usage()

    assert usage.media_calls == 1
    # Media does not count as an LLM call or add tokens.
    assert usage.llm_calls == 0
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert len(usage.details) == 1
    entry = usage.details[0]
    assert entry["type"] == "media"
    assert entry["unit"] == "images"
    assert entry["quantity"] == 2.0
    assert entry["model"] == "sd-xl"
    assert entry["model_id"] == "m1"
    assert entry["call_type"] == "generate_image"
    assert entry["resolution"] == "1K"


def test_add_media_usage_carries_accompanying_tokens() -> None:
    with TokenContextManager() as manager:
        add_media_usage(
            unit="images",
            quantity=1,
            model="gemini-image",
            call_type="generate_image",
            input_tokens=5,
            output_tokens=3,
        )
        usage = manager.get_usage()

    entry = usage.details[0]
    # Stored under provider_tokens, never "tokens": a consumer that sums the
    # "tokens" key across all entries must not pick up media counts.
    assert "tokens" not in entry
    assert entry["provider_tokens"] == 8
    assert entry["provider_input_tokens"] == 5
    assert entry["provider_output_tokens"] == 3
    assert entry["tokens_estimated"] is False
    # Media token passthrough must NOT inflate the LLM token totals.
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


def test_estimated_tokens_are_flagged() -> None:
    with TokenContextManager() as manager:
        add_media_usage(
            unit="texts",
            quantity=2,
            model="embed",
            call_type="embedding",
            input_tokens=12,
            tokens_estimated=True,
        )
        details = manager.get_usage().details

    assert details[0]["tokens_estimated"] is True
    # The flag survives aggregation so billing can refuse to price an estimate.
    assert aggregate_media_usage_by_model(details)[0]["tokens_estimated"] is True


def test_dirty_quantity_is_coerced_and_does_not_raise() -> None:
    with TokenContextManager() as manager:
        add_media_usage(unit="requests", quantity=None, model="x")  # type: ignore[arg-type]
        add_media_usage(unit="requests", quantity="oops", model="x")  # type: ignore[arg-type]
        usage = manager.get_usage()

    assert usage.media_calls == 2
    assert all(entry["quantity"] == 0.0 for entry in usage.details)


def test_to_dict_from_dict_roundtrip_preserves_media() -> None:
    with TokenContextManager() as manager:
        add_token_usage(input_tokens=10, output_tokens=4, model="gpt", model_id="g1")
        add_media_usage(unit="seconds", quantity=3.5, model="tts", call_type="tts")
        usage = manager.get_usage()

    data = usage.to_dict()
    assert data["media_calls"] == 1
    assert data["llm_calls"] == 1

    restored = TokenUsage.from_dict(data)
    assert restored.media_calls == 1
    assert restored.llm_calls == 1
    assert restored.input_tokens == 10
    # 2 token entries (one input, one output) + 1 media entry.
    assert len(restored.details) == 3
    assert sum(1 for d in restored.details if d["type"] == "media") == 1


def test_merge_combines_media_calls_and_details() -> None:
    a = TokenUsage()
    a.add_media_usage(unit="images", quantity=1, model="x")
    a.increment_media_calls()

    b = TokenUsage()
    b.add_media_usage(unit="seconds", quantity=2, model="y")
    b.increment_media_calls()

    a.merge(b)
    assert a.media_calls == 2
    assert len(a.details) == 2


def test_token_aggregation_ignores_media_entries() -> None:
    with TokenContextManager() as manager:
        add_token_usage(input_tokens=10, output_tokens=5, model="gpt", model_id="g1")
        add_media_usage(
            unit="images", quantity=2, model="sd", call_type="generate_image"
        )
        details = manager.get_usage().details

    token_groups = aggregate_token_usage_by_model(details)
    assert len(token_groups) == 1
    assert token_groups[0]["model_name"] == "gpt"
    assert token_groups[0]["input_tokens"] == 10
    assert token_groups[0]["output_tokens"] == 5


def test_media_aggregation_groups_by_model_unit_and_call_type() -> None:
    with TokenContextManager() as manager:
        add_media_usage(
            unit="images", quantity=2, model="sd", call_type="generate_image"
        )
        add_media_usage(
            unit="images", quantity=3, model="sd", call_type="generate_image"
        )
        add_media_usage(unit="seconds", quantity=4, model="tts", call_type="tts")
        # LLM tokens must never appear in the media aggregation.
        add_token_usage(input_tokens=7, output_tokens=2, model="gpt", model_id="g1")
        details = manager.get_usage().details

    media_groups = aggregate_media_usage_by_model(details)
    assert len(media_groups) == 2

    by_unit = {group["unit"]: group for group in media_groups}
    assert by_unit["images"]["quantity"] == 5.0
    assert by_unit["images"]["calls"] == 2
    assert by_unit["images"]["call_type"] == "generate_image"
    assert by_unit["seconds"]["quantity"] == 4.0
    assert by_unit["seconds"]["calls"] == 1


def test_media_aggregation_splits_by_resolution() -> None:
    # Same model+call_type at different resolutions must bill as separate line
    # items, since an image model's price varies by resolution.
    with TokenContextManager() as manager:
        add_media_usage(
            unit="images",
            quantity=1,
            model="gemini-image",
            call_type="generate_image",
            resolution="1K",
        )
        add_media_usage(
            unit="images",
            quantity=1,
            model="gemini-image",
            call_type="generate_image",
            resolution="4K",
        )
        details = manager.get_usage().details

    groups = aggregate_media_usage_by_model(details)
    assert len(groups) == 2
    by_res = {group["resolution"]: group for group in groups}
    assert set(by_res) == {"1K", "4K"}
    assert by_res["1K"]["calls"] == 1
    assert by_res["4K"]["calls"] == 1


def test_aggregations_tolerate_non_list_and_dirty_entries() -> None:
    assert aggregate_media_usage_by_model(None) == []
    assert aggregate_media_usage_by_model("nope") == []
    # Non-dict junk is ignored; a bare media entry still counts as a call, since
    # a media row's existence is itself the billing signal.
    groups = aggregate_media_usage_by_model([{"type": "media"}, 42, "junk"])
    assert len(groups) == 1
    assert groups[0]["calls"] == 1
    assert groups[0]["quantity"] == 0.0


def test_zero_quantity_media_entries_stay_visible() -> None:
    # A duration-billed call the provider never measured records 0 seconds.
    # That entry must survive aggregation: it is the only evidence the task
    # made a billable provider call, and dropping it would report
    # media_calls=0 (and hide the whole popover) for a task that did.
    with TokenContextManager() as manager:
        add_media_usage(unit="seconds", quantity=0, model="tts", call_type="tts")
        add_media_usage(unit="seconds", quantity=5, model="tts", call_type="tts")
        details = manager.get_usage().details

    groups = aggregate_media_usage_by_model(details)
    assert len(groups) == 1
    assert groups[0]["quantity"] == 5.0
    assert groups[0]["calls"] == 2  # both calls counted, including the unmeasured one


def test_only_unmeasured_calls_still_surface() -> None:
    # The async-video case: no duration is available yet for any call, so the
    # whole group is zero-quantity. It must still be reported.
    with TokenContextManager() as manager:
        for _ in range(3):
            add_media_usage(unit="seconds", quantity=0, model="veo", call_type="video")
        groups = aggregate_media_usage_by_model(manager.get_usage().details)

    assert len(groups) == 1
    assert groups[0]["calls"] == 3
    assert groups[0]["quantity"] == 0.0


@pytest.mark.parametrize("bad_unit", ["image", "second", "tokens", "IMAGES", ""])
def test_unknown_unit_is_rejected(bad_unit: str) -> None:
    # A typo'd unit mints a new billing dimension that the aggregator will
    # happily key off, and a written usage record cannot be repaired
    # retroactively. The write boundary is the last point it is still fixable.
    with TokenContextManager() as manager:
        with pytest.raises(ValueError, match="Unknown media unit"):
            add_media_usage(unit=bad_unit, quantity=1, model="m", call_type="tts")
        usage = manager.get_usage()

    # A rejected call must leave no partial state behind: media_calls is
    # incremented before the detail entry is appended, so validating late would
    # record a call with no matching entry.
    assert usage.media_calls == 0
    assert usage.details == []


def test_unknown_call_type_is_rejected() -> None:
    with TokenContextManager() as manager:
        with pytest.raises(ValueError, match="Unknown media call type"):
            add_media_usage(unit="seconds", quantity=1, model="m", call_type="speech")
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_empty_call_type_is_allowed() -> None:
    # call_type is optional metadata rather than a billing dimension of its
    # own, so omitting it stays legal while a typo does not.
    with TokenContextManager() as manager:
        add_media_usage(unit="requests", quantity=1, model="m")
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["call_type"] == ""


def test_none_call_type_is_normalised_to_empty() -> None:
    with TokenContextManager() as manager:
        add_media_usage(unit="requests", quantity=1, model="m", call_type=None)
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["call_type"] == ""


def test_none_unit_is_rejected_with_clear_error() -> None:
    with TokenContextManager() as manager:
        with pytest.raises(ValueError, match="Media unit cannot be None"):
            add_media_usage(unit=None, quantity=1, model="m")
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_enum_members_are_accepted() -> None:
    with TokenContextManager() as manager:
        add_media_usage(
            unit=MediaUnit.SECONDS,
            quantity=3,
            model="whisper",
            call_type=MediaCallType.ASR,
        )
        usage = manager.get_usage()

    entry = usage.details[0]
    # Stored as plain strings so details stays JSON-serialisable.
    assert entry["unit"] == "seconds"
    assert entry["call_type"] == "asr"
    assert isinstance(entry["unit"], str)


def test_concurrent_media_records_lose_no_counts() -> None:
    # One TokenUsage is shared across worker threads (RAG ingestion pools,
    # bind_usage_to_thread callers), where ``+=`` is a read-modify-write.
    usage = TokenUsage()
    workers, per_worker = 8, 200

    def record() -> None:
        for _ in range(per_worker):
            usage.increment_media_calls()
            usage.add_media_usage(unit="images", quantity=1, call_type="generate_image")

    threads = [threading.Thread(target=record) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    expected = workers * per_worker
    assert usage.media_calls == expected
    assert len(usage.details) == expected


def test_concurrent_merge_loses_no_counts() -> None:
    # merge() snapshots the source under its own lock before taking the
    # target's, so concurrent merges neither deadlock nor drop entries.
    target = TokenUsage()
    sources = []
    for _ in range(8):
        source = TokenUsage()
        source.increment_media_calls()
        source.add_media_usage(unit="seconds", quantity=2, call_type="asr")
        sources.append(source)

    threads = [threading.Thread(target=target.merge, args=(s,)) for s in sources]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert target.media_calls == len(sources)
    assert len(target.details) == len(sources)


@pytest.mark.parametrize(
    "bad_quantity",
    [True, False, -1, -0.5, float("nan"), float("inf"), float("-inf"), "abc", None],
)
def test_invalid_quantities_record_zero_not_garbage(bad_quantity) -> None:
    # This is the last boundary before the value is persisted into task and
    # quota details, and a written record cannot be repaired. NaN/inf are not
    # JSON-serialisable (they emit literal NaN/Infinity that strict parsers
    # reject); negatives would subtract from a bill; booleans are a caller bug.
    with TokenContextManager() as manager:
        add_media_usage(
            unit="images", quantity=bad_quantity, model="m", call_type="generate_image"
        )
        entry = manager.get_usage().details[0]

    assert entry["quantity"] == 0.0
    # Recorded, not dropped: the call still happened, it is just unmeasured.
    assert manager.get_usage().media_calls == 1


def test_valid_fractional_quantity_survives() -> None:
    # The guard must not clamp legitimate fractional durations.
    with TokenContextManager() as manager:
        add_media_usage(unit="seconds", quantity=2.5, model="m", call_type="asr")
        assert manager.get_usage().details[0]["quantity"] == 2.5


def test_direct_record_media_call_also_validates() -> None:
    # TokenUsage.record_media_call is the real write boundary; a caller holding
    # a TokenUsage directly must get the same guarantees as add_media_usage.
    usage = TokenUsage()
    usage.record_media_call(unit="images", quantity=float("inf"), call_type="video")

    assert usage.details[0]["quantity"] == 0.0


def test_media_count_and_detail_are_written_atomically() -> None:
    # A snapshot taken between the counter bump and the detail append would
    # report mutually inconsistent billing. Assert every observed snapshot has
    # media_calls equal to the number of media detail rows.
    usage = TokenUsage()
    torn: list[tuple[int, int]] = []
    stop = threading.Event()

    def writer() -> None:
        for _ in range(400):
            usage.record_media_call(
                unit="images", quantity=1, call_type="generate_image"
            )
        stop.set()

    def reader() -> None:
        while not stop.is_set():
            snap = usage.to_dict()
            media = sum(1 for d in snap["details"] if d.get("type") == "media")
            if snap["media_calls"] != media:
                torn.append((snap["media_calls"], media))

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not torn, f"observed torn snapshots: {torn[:5]}"
    assert usage.media_calls == 400


def test_usage_survives_asdict_deepcopy_and_pickle() -> None:
    # The mutation lock is not picklable, so as a dataclass field it would break
    # every generic consumer of this object (dataclasses.asdict walks
    # __dataclass_fields__ and ignores __getstate__).
    import copy
    import dataclasses
    import pickle

    usage = TokenUsage()
    usage.record_media_call(unit="images", quantity=2, call_type="generate_image")

    assert dataclasses.asdict(usage)["media_calls"] == 1
    assert copy.deepcopy(usage).media_calls == 1
    revived = pickle.loads(pickle.dumps(usage))
    assert revived.media_calls == 1
    # The revived object gets a working lock of its own, not a shared one.
    revived.record_media_call(unit="images", quantity=1, call_type="generate_image")
    assert revived.media_calls == 2


def test_media_calls_is_not_positionally_before_legacy_fields() -> None:
    # Guards the public constructor shape: media_calls sits after llm_calls and
    # before tool_calls, and the private lock must not occupy a positional slot.
    usage = TokenUsage(1, 2, 3, 4, 5, [])

    assert usage.input_tokens == 1
    assert usage.output_tokens == 2
    assert usage.llm_calls == 3
    assert usage.media_calls == 4
    assert usage.tool_calls == 5


def test_concurrent_merges_in_both_directions_do_not_deadlock() -> None:
    # merge() snapshots the source under its own lock before taking the
    # target's; holding both would deadlock on opposing merges.
    a = TokenUsage()
    b = TokenUsage()
    for _ in range(50):
        a.record_media_call(unit="images", quantity=1, call_type="generate_image")
        b.record_media_call(unit="seconds", quantity=1, call_type="asr")

    threads = [
        threading.Thread(target=a.merge, args=(b,)),
        threading.Thread(target=b.merge, args=(a,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    for t in threads:
        assert not t.is_alive(), "merge deadlocked in opposing directions"
    # Neither side lost rows: each gained at least the other's original 50.
    assert len(a.details) >= 100
    assert len(b.details) >= 100


def test_estimate_tokens_accepts_any_iterable_of_strings() -> None:
    # The docstring promises an iterable; a generator previously estimated 0,
    # so an embedding producer passing one would have billed nothing.
    from xagent.core.model.chat.token_context import estimate_tokens

    assert estimate_tokens(x for x in ["abcd", "efgh"]) == 2
    assert estimate_tokens(["abcd", "efgh"]) == 2
    # Non-iterables and non-string members are ignored, never raised on.
    assert estimate_tokens(42) == 0
    assert estimate_tokens(["abcd", 42, None]) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 0), ("a", 1), ("ab", 1), ("abc", 1), ("abcd", 1), ("abcde", 2)],
)
def test_estimate_tokens_never_undercounts_short_text(text, expected) -> None:
    # Truncating division estimated 0 for any 1-3 character Latin string, so
    # short non-empty text billed nothing. Empty still estimates 0.
    from xagent.core.model.chat.token_context import estimate_tokens

    assert estimate_tokens(text) == expected


def test_estimate_tokens_counts_cjk_per_character() -> None:
    from xagent.core.model.chat.token_context import estimate_tokens

    # CJK is roughly one token per character; a flat chars/4 heuristic would
    # undercount Chinese by close to 4x.
    assert estimate_tokens("中文") == 2
    assert estimate_tokens("中文abcd") == 3
