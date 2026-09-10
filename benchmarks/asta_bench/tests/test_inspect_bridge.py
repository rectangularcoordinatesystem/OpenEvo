from __future__ import annotations

import json
import os

from openevo.evolution.models import DatasetCreateRequest
from openevo.evolution.store import EvolutionStore
from openevo_asta_bench.inspect_bridge import build_astabench_events


def test_build_astabench_event_from_json_eval_log(tmp_path) -> None:
    log_path = tmp_path / "sample.eval"
    log_path.write_text(
        json.dumps(
            {
                "eval_id": "eval-1",
                "eval": {"task": "astabench/paper_finder_validation", "model": "gpt-5.5"},
                "samples": [
                    {
                        "id": "42",
                        "input": "Find papers about calibration.",
                        "completion": '["CorpusId:12345"]',
                        "scores": {
                            "score_paper_finder": {"adjusted_f1_micro_avg": 0.5}
                        },
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "metadata": {
                            "paper_ids": ["secret-paper-id"],
                            "answer": "hidden reference",
                        },
                        "events": [
                            {
                                "event": "tool_call",
                                "name": "search",
                                "arguments": {"query": "calibration"},
                                "result": "ok",
                                "api_key": "must-not-copy",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    [event] = build_astabench_events(log_path, policy_version="rev-a")

    assert event.source == "asta_bench.inspect"
    assert event.source_event_id == "astabench:eval-1:42"
    assert event.task_id == "astabench/paper_finder_validation:42"
    assert event.policy_version == "rev-a"
    assert event.reward == 0.5
    assert event.status == "COMPLETED"
    trace = event.payload["session_result"]["trajectory"]["traces"][0]
    assert trace["prompt_messages"] == [
        {"role": "user", "content": "Find papers about calibration."}
    ]
    metadata = trace["metadata"]
    assert metadata["official_primary_metric"] == "score_paper_finder/adjusted_f1_micro_avg"
    assert metadata["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert metadata["tool_events"][0]["name"] == "search"
    assert "api_key" not in metadata["tool_events"][0]
    assert metadata["leakage_basis"]["paper_ids"] == ["secret-paper-id"]

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    (artifact_root / "datasets").mkdir(mode=0o700)
    os.chmod(artifact_root, 0o700)
    os.chmod(artifact_root / "datasets", 0o700)
    store = EvolutionStore(db_path=tmp_path / "evolution.db", artifact_root=artifact_root)
    store.initialize()
    assert store.ingest_event(event).ingested is True
    dataset = store.create_dataset(
        DatasetCreateRequest(
            name="astabench_smoke",
            purpose="astabench_validation_reflection",
            query={"source": "asta_bench.inspect", "event_types": ["openevo.session_completed"]},
        )
    )
    assert dataset.event_count == 1
    assert dataset.trace_count == 1
