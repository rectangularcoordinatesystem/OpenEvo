from __future__ import annotations

from dataclasses import dataclass

from openevo_asta_bench.solver import (
    ASTABENCH_OPENEVO_SYSTEM,
    _tool_message_details,
    _tool_results_have_errors,
)


@dataclass
class Message:
    role: str
    content: str = ""
    function: str | None = None
    error: object | None = None


def test_python_session_instruction_requires_code() -> None:
    assert '"code"' in ASTABENCH_OPENEVO_SYSTEM


def test_tool_error_is_preserved_for_next_model_turn() -> None:
    details = _tool_message_details(
        Message(
            role="tool",
            function="python_session",
            error={"type": "parsing", "message": "'code' is a required property"},
        )
    )

    assert "ERROR:" in details
    assert "'code' is a required property" in details


def test_tool_error_does_not_count_as_successful_step() -> None:
    assert _tool_results_have_errors([Message(role="tool", error={"type": "parsing"})])
    assert not _tool_results_have_errors([Message(role="tool", content="42")])
