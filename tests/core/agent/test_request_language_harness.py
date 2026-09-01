from __future__ import annotations

import json

import pytest

from xagent.core.agent.context import ExecutionContext
from xagent.core.agent.language import (
    OUTPUT_LANGUAGE_METADATA_KEY,
    request_only_language_harness,
)
from xagent.core.agent.pattern.auto.auto import AutoPattern
from xagent.core.agent.pattern.dag.dag import DAGPattern
from xagent.core.agent.pattern.dag.plan_generator import (
    LLMPlanGenerator,
    PlanGenerationRequest,
)
from xagent.core.agent.pattern.react.react import ReActPattern

ENGLISH_REQUEST = "Summarize the latest customer email and draft a concise reply."
POLLUTED_EXECUTION_REQUEST = (
    f"{ENGLISH_REQUEST}\n\n"
    "[From: Gerard Santos <gerard.santos@example.es>]\n"
    "Connector context: bandeja de entrada, correo electrónico, responder."
)


def _polluted_context() -> ExecutionContext:
    context = ExecutionContext(execution_id="request-language-harness")
    context.add_user_message(
        POLLUTED_EXECUTION_REQUEST,
        metadata={"display_message": ENGLISH_REQUEST},
    )
    return context


@pytest.mark.parametrize(
    "user_request",
    [
        "Translate the following note to Spanish: The launch is tomorrow.",
        "请把最新的客户邮件整理成简短摘要。",
        "請把最新的客戶郵件整理成簡短摘要。",
        "OK?",
        "Review este draft and keep the product names unchanged.",
    ],
)
def test_request_language_harness_preserves_the_whole_request_without_detection(
    user_request: str,
) -> None:
    harness = request_only_language_harness(user_request)

    assert json.dumps(user_request, ensure_ascii=False) in harness
    assert "Request-only response language harness" in harness
    assert "explicit and implicit requests" in harness
    assert "too short, mixed-language, or depends on conversation context" in harness
    assert "Output language:" not in harness


def test_root_language_harness_uses_only_the_user_authored_request() -> None:
    system_context = _polluted_context()._system_context()
    harness = system_context.split("Request-only response language harness:\n", 1)[1]

    assert json.dumps(ENGLISH_REQUEST) in harness
    assert "Gerard Santos" not in harness
    assert "example.es" not in harness
    assert "bandeja de entrada" not in harness


def test_dag_language_consumers_receive_the_same_user_authored_request() -> None:
    context = _polluted_context()
    plan_payload = json.loads(
        LLMPlanGenerator()._build_prompt(
            PlanGenerationRequest(
                context=context,
                execution_id=context.execution_id,
                available_tool_names=[],
            )
        )
    )
    completion_payload = json.loads(
        DAGPattern(lambda **_: None)._completion_assessment_messages(context)[1][
            "content"
        ]
    )

    assert plan_payload["latest_user_request"] == ENGLISH_REQUEST
    assert json.dumps(ENGLISH_REQUEST) in plan_payload["output_language_policy"]
    assert completion_payload["user_authored_language_request"] == ENGLISH_REQUEST
    assert json.dumps(ENGLISH_REQUEST) in completion_payload["output_language_policy"]
    assert "Gerard Santos" not in completion_payload["output_language_policy"]


def test_caller_pinned_language_remains_the_only_hard_authority() -> None:
    context = _polluted_context()
    context.metadata["request_context"] = {OUTPUT_LANGUAGE_METADATA_KEY: "French"}
    context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "French"

    system_context = context._system_context()
    completion_payload = json.loads(
        DAGPattern(lambda **_: None)._completion_assessment_messages(context)[1][
            "content"
        ]
    )

    assert "Output language: French" in system_context
    assert "Request-only response language harness" not in system_context
    assert "Output language: French" in completion_payload["output_language_policy"]
    assert completion_payload["user_authored_language_request"] == ""


def test_final_answer_schemas_follow_the_shared_language_guidance() -> None:
    react_schema = ReActPattern()._final_answer_tool_schema()
    react_function = react_schema["function"]
    auto_function = AutoPattern()._decision_tool_schema()["function"]

    assert "authoritative output language guidance" in react_function["description"]
    assert "connector metadata" in react_function["description"]
    assert "authoritative output language guidance" in auto_function["description"]
    assert "connector metadata" in auto_function["description"]
