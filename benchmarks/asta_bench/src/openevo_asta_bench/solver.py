from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from openevo_asta_bench import openai_compat as _openai_compat
from openevo_asta_bench.constants import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_REFLECTOR_MODEL,
    GPT55_INSPECT_MODEL_ALIASES,
)
from openevo_asta_bench.submission import finalize_state_output

try:
    from inspect_ai.solver import Generate, Solver, TaskState, solver
except Exception:
    Generate = Any  # type: ignore[assignment]
    Solver = Any  # type: ignore[assignment]
    TaskState = Any  # type: ignore[assignment]

    def solver(func: Any) -> Any:
        return func


ASTABENCH_OPENEVO_SYSTEM = """You are solving one AstaBench sample.
Use only the tools provided by this Inspect TaskState. Preserve task-specific
date, corpus, sandbox, and submission constraints. Do not use hidden scorer
feedback or prior test answers. Submit the final answer in the format requested
by the task. When calling python_session, always provide a non-empty code
argument, for example {"code": "print('hello')"}. If a tool reports an input
validation error, correct the arguments before retrying."""

PAPER_FINDER_SYSTEM = """For Paper Finder tasks, return only valid JSON with
output.results. Each result must use the Semantic Scholar numeric corpus_id as
paper_id. Do not submit SHA hashes, arXiv IDs, DOIs, titles, or URLs as paper_id.
If a tool result does not expose a numeric corpus_id, search or inspect another
result until you have numeric corpus IDs. Submit at most 3 highly relevant
results, ordered by relevance."""


@solver
def astabench_openevo_solver(
    *,
    model: str = DEFAULT_AGENT_MODEL,
    backend: str = "inspect_model",
    codex_model: str = DEFAULT_CODEX_MODEL,
    codex_bin: str = "codex",
    codex_timeout_seconds: float = 180.0,
    reflector_model: str = DEFAULT_REFLECTOR_MODEL,
    max_steps: int = 24,
    model_client_timeout: float | None = None,
    openai_tool_schema: str = "standard",
    revision_manifest: str | None = None,
    task_name: str | None = None,
    **tool_options: Any,
) -> Solver:
    """Return an Inspect solver for AstaBench.

    The function imports Inspect/AstaBench lazily so this package remains
    smoke-testable before the official benchmark environment is installed.
    """

    if backend not in {"inspect_model", "codex_cli"}:
        raise ValueError("backend must be inspect_model or codex_cli")
    if openai_tool_schema not in {"standard", "flat"}:
        raise ValueError("openai_tool_schema must be standard or flat")
    selected_codex_model = codex_model if backend == "codex_cli" else _canonical_gpt55_model(model)
    if selected_codex_model != DEFAULT_CODEX_MODEL or reflector_model != DEFAULT_REFLECTOR_MODEL:
        raise ValueError("AstaBench OpenEvo runs are pinned to gpt-5.5 for agent and reflector")
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    try:
        from astabench.tools import ToolsetConfig
        from astabench.util.state import merge_tools_with_state
    except Exception as exc:
        raise RuntimeError("astabench must be installed to create the AstaBench solver") from exc

    config = ToolsetConfig.model_validate(tool_options)
    optional_tools = config.create_tools()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        merged = merge_tools_with_state(optional_tools)
        state = await merged(state, generate)
        _inject_revision_context(state, revision_manifest=revision_manifest)
        if not _has_system_message(state):
            state.messages.insert(0, ChatMessageSystem(content=ASTABENCH_OPENEVO_SYSTEM))
        if _is_paper_finder_task(task_name or _state_task_name(state)):
            state.messages.insert(0, ChatMessageSystem(content=PAPER_FINDER_SYSTEM))

        if backend == "codex_cli":
            state.output.completion = await _run_codex_cli(
                state,
                codex_bin=codex_bin,
                codex_model=codex_model,
                timeout_seconds=codex_timeout_seconds,
            )
        else:
            from inspect_ai.model import call_tools, get_model

            model_kwargs = {}
            if model_client_timeout is not None:
                model_kwargs["client_timeout"] = model_client_timeout
            agent_model = get_model(
                _inspect_model_name(
                    model,
                    openai_tool_schema=openai_tool_schema,
                ),
                **model_kwargs,
            )
            needs_final_answer = False
            successful_tool_steps = 0
            total_turns = 0
            while successful_tool_steps < max_steps and total_turns < max_steps * 2:
                total_turns += 1
                output = await agent_model.generate(
                    input=_messages_for_model(state.messages),
                    tools=state.tools,
                )
                state.messages.append(output.message)
                if getattr(output.message, "tool_calls", None):
                    needs_final_answer = True
                    results = await call_tools(output.message, state.tools, max_output=None)
                    state.messages.extend(results)
                    if _tool_results_have_errors(results):
                        state.messages.append(
                            ChatMessageUser(
                                content=(
                                    "The tool call failed. Read the error above, correct the tool "
                                    "arguments, and retry. For python_session you must provide "
                                    'a non-empty {"code": "..."} argument.'
                                )
                            )
                        )
                    else:
                        successful_tool_steps += 1
                        state.messages.append(
                            ChatMessageUser(content="Continue, and submit when complete.")
                        )
                    continue
                state.output = output
                needs_final_answer = False
                break
            if needs_final_answer:
                state.messages.append(
                    ChatMessageUser(
                        content=(
                            "Tool budget reached. Based only on the task, answer choices, "
                            "and tool results above, provide the final answer now. If the "
                            "task asks for JSON, output only valid JSON."
                        )
                    )
                )
                state.output = await agent_model.generate(
                    input=_messages_for_model(state.messages),
                    tools=[],
                )
                state.messages.append(state.output.message)

        return await finalize_state_output(state, task_name=task_name or _state_task_name(state))

    return solve


def _inject_revision_context(state: Any, *, revision_manifest: str | None) -> None:
    if not revision_manifest:
        return
    text = Path(revision_manifest).read_text(encoding="utf-8")
    try:
        from inspect_ai.model import ChatMessageSystem
    except Exception:
        return
    state.messages.insert(
        0,
        ChatMessageSystem(
            content=(
                "Frozen OpenEvo revision context follows. Treat it as general "
                "method guidance, not task answers.\n\n" + text
            )
        ),
    )


def _has_system_message(state: Any) -> bool:
    messages = getattr(state, "messages", [])
    return any(getattr(message, "role", None) == "system" for message in messages)


def _state_task_name(state: Any) -> str | None:
    metadata = getattr(state, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("task_name", "benchmark", "name"):
            value = metadata.get(key)
            if isinstance(value, str):
                return value
    return None


def _is_paper_finder_task(task_name: str | None) -> bool:
    normalized = (task_name or "").lower()
    return "paper_finder" in normalized or "paperfinding" in normalized


async def _run_codex_cli(
    state: Any,
    *,
    codex_bin: str,
    codex_model: str,
    timeout_seconds: float,
) -> str:
    if getattr(state, "tools", None):
        raise ValueError(
            "codex_cli backend cannot yet consume Inspect state.tools; use inspect_model "
            "for Standard AstaBench tool tasks"
        )
    prompt = _codex_prompt_from_state(state)
    with tempfile.TemporaryDirectory(prefix="openevo-astabench-codex-") as tmpdir:
        output_path = Path(tmpdir) / "last-message.txt"
        process = await asyncio.create_subprocess_exec(
            codex_bin,
            "exec",
            "--model",
            codex_model,
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--output-last-message",
            str(output_path),
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_codex_env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise TimeoutError("codex_cli backend timed out") from exc
        if process.returncode != 0:
            detail = (stderr or stdout).decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"codex_cli backend failed with exit {process.returncode}: {detail}")
        completion = output_path.read_text(encoding="utf-8", errors="replace").strip()
        if not completion:
            completion = stdout.decode("utf-8", errors="replace").strip()
        return completion


def _codex_prompt_from_state(state: Any) -> str:
    parts = [
        "Solve this AstaBench sample. Return only the final answer requested by the task.",
        "Do not mention hidden data or scoring. If JSON is requested, output valid JSON.",
        "",
    ]
    for message in getattr(state, "messages", []):
        role = getattr(message, "role", "message")
        content = _message_content_text(getattr(message, "content", ""))
        if content:
            parts.append(f"{role.upper()}:\n{content}\n")
    return "\n".join(parts)


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_message_content_text(item) for item in content)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content)
    return str(content)


def _messages_for_model(messages: list[Any]) -> list[Any]:
    try:
        from inspect_ai.model import ChatMessageUser
    except Exception:
        return messages
    converted: list[Any] = []
    for message in messages:
        if getattr(message, "role", None) == "tool":
            tool_name = getattr(message, "function", None) or getattr(message, "name", None) or "tool"
            converted.append(
                ChatMessageUser(
                    content=f"Tool result from {tool_name}:\n{_tool_message_details(message)}",
                )
            )
        else:
            converted.append(message)
    return converted


def _tool_results_have_errors(results: list[Any]) -> bool:
    return any(getattr(result, "error", None) for result in results)


def _tool_message_details(message: Any) -> str:
    content = _message_content_text(getattr(message, "content", ""))
    error_text = _structured_text(getattr(message, "error", None))
    if error_text:
        return f"ERROR: {error_text}" + (f"\n{content}" if content else "")
    return content


def _structured_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _codex_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.endswith("_API_KEY") or key in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"}:
            env.pop(key, None)
    return env


def _codex_model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _canonical_gpt55_model(model: str) -> str:
    name = _codex_model_name(model)
    if name in GPT55_INSPECT_MODEL_ALIASES:
        return DEFAULT_CODEX_MODEL
    return name


def _inspect_model_name(
    model: str,
    *,
    openai_tool_schema: str = "standard",
) -> str:
    if openai_tool_schema == "standard":
        return model if "/" in model else f"openai/{model}"
    if openai_tool_schema != "flat":
        raise ValueError("openai_tool_schema must be standard or flat")

    if "/" in model:
        provider, model_name = model.split("/", 1)
        if provider not in {"openai", _openai_compat.FLAT_OPENAI_PROVIDER}:
            raise ValueError(
                "flat OpenAI tool schema only supports OpenAI-compatible models"
            )
    else:
        model_name = model
    return f"{_openai_compat.FLAT_OPENAI_PROVIDER}/{model_name}"
