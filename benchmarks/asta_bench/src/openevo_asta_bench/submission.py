from __future__ import annotations

import json
import re
from typing import Any

MAX_PAPER_FINDER_RESULTS = 3


def completion_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(part for part in parts if part)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_completion_for_task(task_name: str | None, completion: Any) -> str:
    text = completion_text(completion).strip()
    normalized_task = (task_name or "").lower()
    if "litqa" in normalized_task and "search" not in normalized_task:
        answer = _extract_multiple_choice_answer(text)
        return answer or text
    if "paper_finder" in normalized_task or "paperfinding" in normalized_task:
        parsed = _extract_json_object(text)
        if parsed is not None:
            return json.dumps(_normalize_paper_finder_json(parsed), ensure_ascii=False)
        papers = _extract_ordered_ids(text)
        if papers:
            return json.dumps(
                {
                    "output": {
                        "results": [
                            {"paper_id": paper_id, "markdown_evidence": ""}
                            for paper_id in _dedupe_digit_ids(papers)[:MAX_PAPER_FINDER_RESULTS]
                        ]
                    }
                },
                ensure_ascii=False,
            )
        return text
    if "ds1000" in normalized_task:
        return _strip_markdown_fence(text)
    return text


def submission_manager_completion(default_completion: Any = None) -> str:
    """Return an AstaBench submission-manager value when the package is available."""

    try:
        from astabench.tools.submission import get_submission_manager
    except Exception:
        return completion_text(default_completion)
    manager = get_submission_manager()
    if manager.has_submission():
        return completion_text(manager.get_submission())
    return completion_text(default_completion)


async def finalize_state_output(state: Any, *, task_name: str | None = None) -> Any:
    raw = submission_manager_completion(getattr(getattr(state, "output", None), "completion", None))
    output = getattr(state, "output", None)
    if output is None:
        raise ValueError("Inspect TaskState output object is missing")
    output.completion = normalize_completion_for_task(task_name, raw)
    return state


def _extract_multiple_choice_answer(text: str) -> str | None:
    patterns = [
        r"(?i)\banswer\s*[:：]\s*['\"]?([A-Z])\b",
        r"(?i)\bfinal\s*[:：]\s*['\"]?([A-Z])\b",
        r"(?i)\boption\s+([A-Z])\b",
        r"\b([A-Z])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def _extract_ordered_ids(text: str) -> list[str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    ids = re.findall(r"\b(?:CorpusId|S2|paper|arXiv)[:/_-]?([A-Za-z0-9.:-]{4,})\b", text)
    return [item for item in ids if item]


def _normalize_paper_finder_json(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    output = value.get("output")
    if not isinstance(output, dict):
        return value
    results = output.get("results")
    if not isinstance(results, list):
        return value
    normalized_results: list[dict[str, str]] = []
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        paper_id = _normalize_digit_corpus_id(result.get("paper_id"))
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        evidence = result.get("markdown_evidence")
        normalized_results.append(
            {
                "paper_id": paper_id,
                "markdown_evidence": evidence if isinstance(evidence, str) else "",
            }
        )
        if len(normalized_results) >= MAX_PAPER_FINDER_RESULTS:
            break
    value = dict(value)
    value["output"] = dict(output)
    value["output"]["results"] = normalized_results
    return value


def _normalize_digit_corpus_id(value: Any) -> str | None:
    text = str(value or "").strip()
    match = re.fullmatch(r"(?i)(?:corpusid[:\s]*)?(\d+)", text)
    return match.group(1) if match else None


def _dedupe_digit_ids(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        paper_id = _normalize_digit_corpus_id(value)
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            output.append(paper_id)
    return output


def _extract_json_object(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _strip_markdown_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text
