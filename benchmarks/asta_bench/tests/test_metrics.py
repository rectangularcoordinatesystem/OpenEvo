from __future__ import annotations

from openevo_asta_bench.metrics import compute_official_macro, primary_score


def test_primary_score_uses_official_metric_path() -> None:
    metrics = {"score_paper_finder": {"recall_at_30": 0.75}, "score": 0.1}
    assert primary_score(metrics, "LitQA2_FullText_Search_validation") == 0.75


def test_primary_score_accepts_inspect_correct_incorrect_values() -> None:
    assert primary_score({"check_arithmetic": {"value": "C"}}) == 1.0
    assert primary_score({"check_arithmetic": {"value": "I"}}) == 0.0


def test_macro_uses_litqa_half_weights_and_category_macro() -> None:
    result = compute_official_macro(
        [
            {"task_name": "LitQA2_FullText_validation", "score": 0.0},
            {"task_name": "LitQA2_FullText_Search_validation", "score": 1.0},
            {"task_name": "PaperFindingBench_validation", "score": 1.0},
            {"task_name": "DiscoveryBench_validation", "score": 0.5},
            {"task_name": "DS1000_validation", "score": 0.25},
            {"task_name": "E2E_Bench_validation", "score": 0.75},
        ]
    )

    assert result["categories"]["lit"] == 0.75
    assert result["categories"]["data"] == 0.5
    assert result["categories"]["code"] == 0.25
    assert result["categories"]["discovery"] == 0.75
    assert result["overall"] == 0.5625
