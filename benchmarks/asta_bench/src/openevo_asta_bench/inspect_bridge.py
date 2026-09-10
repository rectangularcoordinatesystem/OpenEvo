from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openevo.evolution.models import EventIngestRequest

from openevo_asta_bench.constants import DEFAULT_AGENT_MODEL, EVENT_TYPE, SOURCE
from openevo_asta_bench.leakage import leakage_basis_from_sample
from openevo_asta_bench.metrics import primary_score, task_metric_for


DEFAULT_MAX_MESSAGE_CHARS = 60000
DEFAULT_MAX_TOOL_EVENTS = 96


class AstaBenchBridgeError(ValueError):
    """Raised when an Inspect/AstaBench log cannot be converted safely."""


def build_astabench_events(
    path: str | Path,
    *,
    policy_version: str | None = None,
    rollout_step: int | None = None,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    max_tool_events: int = DEFAULT_MAX_TOOL_EVENTS,
) -> list[EventIngestRequest]:
    root = Path(path)
    files = _eval_files(root)
    if not files:
        raise AstaBenchBridgeError(f"no .eval or JSON Inspect logs found under {root}")
    events: list[EventIngestRequest] = []
    for file_path in files:
        events.extend(
            _events_from_log(
                file_path,
                policy_version=policy_version,
                rollout_step=rollout_step,
                max_message_chars=max_message_chars,
                max_tool_events=max_tool_events,
            )
        )
    return events


def _events_from_log(
    path: Path,
    *,
    policy_version: str | None,
    rollout_step: int | None,
    max_message_chars: int,
    max_tool_events: int,
) -> list[EventIngestRequest]:
    payload = _read_json(path)
    samples = _samples(payload)
    if not samples:
        raise AstaBenchBridgeError(f"Inspect log has no samples: {path}")
    eval_id = _text(payload.get("eval_id")) or _text(payload.get("run_id")) or path.stem
    task_name = _task_name(payload)
    model_name = _model_name(payload)
    return [
        _event_from_sample(
            sample,
            path=path,
            eval_id=eval_id,
            task_name=task_name,
            model_name=model_name,
            policy_version=policy_version,
            rollout_step=rollout_step,
            max_message_chars=max_message_chars,
            max_tool_events=max_tool_events,
        )
        for sample in samples
    ]


def _event_from_sample(
    sample: dict[str, Any],
    *,
    path: Path,
    eval_id: str,
    task_name: str,
    model_name: str,
    policy_version: str | None,
    rollout_step: int | None,
    max_message_chars: int,
    max_tool_events: int,
) -> EventIngestRequest:
    sample_id = _sample_id(sample)
    metrics = _sample_metrics(sample)
    score = primary_score(metrics, task_name)
    status = _sample_status(sample, score)
    messages = _messages(sample, max_chars=max_message_chars)
    tools = _tool_events(sample, max_events=max_tool_events)
    task_metric = task_metric_for(task_name)
    metadata = {
        "builder": "astabench_inspect_bridge",
        "capture_mode": "inspect_eval_log",
        "token_level_metrics_available": True,
        "inspect_eval_log": str(path),
        "task_name": task_name,
        "task_tags": list(task_metric.tags) if task_metric else [],
        "official_primary_metric": task_metric.primary_metric if task_metric else None,
        "metrics": metrics,
        "usage": _usage(sample),
        "tool_events": tools,
        "leakage_basis": leakage_basis_from_sample(sample),
    }
    trajectory = {
        "status": status,
        "metadata": metadata,
        "traces": [
            {
                "prompt_ids": [],
                "response_ids": [],
                "loss_mask": [],
                "prompt_messages": messages["prompt"],
                "response_messages": messages["response"],
                "finish_reason": _finish_reason(sample),
                "response_logprobs": None,
                "reward": score,
                "metadata": metadata,
            }
        ],
    }
    return EventIngestRequest(
        source=SOURCE,
        event_type=EVENT_TYPE,
        source_event_id=f"astabench:{eval_id}:{sample_id}",
        created_at=_created_at(sample),
        task_id=f"{task_name}:{sample_id}",
        session_id=f"{eval_id}:{sample_id}",
        policy_version=policy_version,
        rollout_step=rollout_step,
        agent={"harness": "inspect", "model_name": model_name, "benchmark": "astabench"},
        base_model=model_name,
        reward=score,
        status=status,
        payload={
            "session_result": {
                "session_id": f"{eval_id}:{sample_id}",
                "task_id": f"{task_name}:{sample_id}",
                "status": status,
                "trajectory": trajectory,
                "metadata": {"astabench": metadata},
            }
        },
    )


def _eval_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        raise AstaBenchBridgeError(f"path does not exist: {root}")
    return sorted([*root.rglob("*.eval"), *root.rglob("*.json")])


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AstaBenchBridgeError(f"{path} is not a JSON Inspect log; use --log-format json") from exc


def _samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("samples", "sample_summaries"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _task_name(payload: dict[str, Any]) -> str:
    for path in (("eval", "task"), ("eval", "task_name"), ("task",), ("task_name",), ("eval", "task_id")):
        value = _nested(payload, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _model_name(payload: dict[str, Any]) -> str:
    for path in (("eval", "model"), ("plan", "model"), ("model",), ("metadata", "model")):
        value = _nested(payload, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return DEFAULT_AGENT_MODEL


def _sample_id(sample: dict[str, Any]) -> str:
    for key in ("id", "sample_id", "epoch", "uuid"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "sample"


def _sample_metrics(sample: dict[str, Any]) -> dict[str, Any]:
    for key in ("scores", "metrics", "score"):
        value = sample.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _sample_status(sample: dict[str, Any], score: float | None) -> str:
    raw = _text(sample.get("status")) or _text(sample.get("error"))
    if raw:
        upper = raw.upper()
        if "ERROR" in upper or "FAIL" in upper or "TIMEOUT" in upper:
            return "ERROR"
    if score is None:
        return "ERROR"
    return "COMPLETED"


def _messages(sample: dict[str, Any], *, max_chars: int) -> dict[str, list[dict[str, str]]]:
    raw_messages = sample.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = _nested(sample, ("transcript", "messages"))
    prompt: list[dict[str, str]] = []
    response: list[dict[str, str]] = []
    if isinstance(raw_messages, list):
        for message in raw_messages:
            if not isinstance(message, dict):
                continue
            role = _text(message.get("role")) or "unknown"
            content = _truncate(_text(message.get("content")) or "", max_chars)
            if role == "assistant":
                response.append({"role": role, "content": content})
            elif role in {"user", "system", "developer"}:
                prompt.append({"role": role, "content": content})
    if not prompt:
        text = _text(sample.get("input")) or _text(sample.get("prompt")) or ""
        if text:
            prompt.append({"role": "user", "content": _truncate(text, max_chars)})
    if not response:
        text = _text(sample.get("completion")) or _text(_nested(sample, ("output", "completion"))) or ""
        if text:
            response.append({"role": "assistant", "content": _truncate(text, max_chars)})
    return {"prompt": prompt, "response": response}


def _tool_events(sample: dict[str, Any], *, max_events: int) -> list[dict[str, Any]]:
    events = sample.get("events")
    if not isinstance(events, list):
        return []
    out: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = _text(event.get("event")) or _text(event.get("type")) or ""
        if "tool" not in event_type.lower():
            continue
        out.append(_sanitize_tool_event(event))
        if len(out) >= max_events:
            break
    return out


def _sanitize_tool_event(event: dict[str, Any]) -> dict[str, Any]:
    allowed = {}
    for key in ("event", "type", "name", "tool", "arguments", "result", "error"):
        if key in event:
            allowed[key] = event[key]
    return allowed


def _usage(sample: dict[str, Any]) -> dict[str, Any]:
    value = sample.get("usage") or sample.get("model_usage")
    return value if isinstance(value, dict) else {}


def _finish_reason(sample: dict[str, Any]) -> str:
    return _text(sample.get("finish_reason")) or "inspect_eval"


def _created_at(sample: dict[str, Any]) -> str | None:
    return _text(sample.get("completed_at")) or _text(sample.get("created_at"))


def _nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 14)] + "\n[truncated]\n"
