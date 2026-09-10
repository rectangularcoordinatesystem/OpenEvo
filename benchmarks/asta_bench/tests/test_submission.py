from __future__ import annotations

import json

import pytest

from openevo_asta_bench.submission import finalize_state_output, normalize_completion_for_task


class Output:
    completion = "Reasoning... Answer: c"


class State:
    output = Output()


def test_litqa_completion_extracts_choice() -> None:
    assert normalize_completion_for_task("LitQA2_FullText_validation", "Reasoning\nAnswer: c") == "C"


def test_ds1000_completion_strips_python_fence() -> None:
    assert normalize_completion_for_task("DS1000_validation", "```python\nprint(1)\n```") == "print(1)"


def test_paper_finder_keeps_only_digit_corpus_ids_and_truncates() -> None:
    results = [
        {"paper_id": "1f53e69f94a1020b48ea7d282221c576555c38a3", "markdown_evidence": "sha"},
        {"paper_id": "CorpusId:12345", "markdown_evidence": "a"},
    ] + [
        {"paper_id": str(20000 + index), "markdown_evidence": "b"}
        for index in range(35)
    ]
    normalized = normalize_completion_for_task(
        "astabench/paper_finder_validation",
        json.dumps({"output": {"results": results}}),
    )
    data = json.loads(normalized)
    paper_ids = [item["paper_id"] for item in data["output"]["results"]]

    assert len(paper_ids) == 3
    assert paper_ids[0] == "12345"
    assert "1f53e69f94a1020b48ea7d282221c576555c38a3" not in paper_ids


@pytest.mark.asyncio
async def test_finalize_state_output_without_astabench_submission_manager() -> None:
    state = await finalize_state_output(State(), task_name="LitQA2_FullText_validation")
    assert state.output.completion == "C"
