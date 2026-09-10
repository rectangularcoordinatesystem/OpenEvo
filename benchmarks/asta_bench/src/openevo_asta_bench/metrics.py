from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class AstaBenchTaskMetric:
    name: str
    path: str
    primary_metric: str
    tags: tuple[str, ...]
    weight: float = 1.0


ASTABENCH_CONFIG_1_0_0: tuple[AstaBenchTaskMetric, ...] = (
    AstaBenchTaskMetric("ArxivDIGESTables_Clean_validation", "astabench/arxivdigestables_validation", "score_tables/mean", ("lit",)),
    AstaBenchTaskMetric("ScholarQA_CS2_validation", "astabench/sqa_dev", "global_avg/mean", ("lit",)),
    AstaBenchTaskMetric("LitQA2_FullText_validation", "astabench/litqa2_validation", "is_correct/accuracy", ("lit",), 0.5),
    AstaBenchTaskMetric("PaperFindingBench_validation", "astabench/paper_finder_validation", "score_paper_finder/adjusted_f1_micro_avg", ("lit",)),
    AstaBenchTaskMetric("LitQA2_FullText_Search_validation", "astabench/paper_finder_litqa2_validation", "score_paper_finder/recall_at_30", ("lit",), 0.5),
    AstaBenchTaskMetric("DiscoveryBench_validation", "astabench/discoverybench_validation", "score_discoverybench/mean", ("data",)),
    AstaBenchTaskMetric("SUPER_Expert_validation", "astabench/super_validation", "entrypoint/mean", ("code",)),
    AstaBenchTaskMetric("CORE_Bench_Hard_validation", "astabench/core_bench_validation", "score_with_stderr/accuracy", ("code",)),
    AstaBenchTaskMetric("DS1000_validation", "astabench/ds1000_validation", "ds1000_scorer/accuracy", ("code",)),
    AstaBenchTaskMetric("E2E_Bench_validation", "astabench/e2e_discovery_validation", "score_rubric/accuracy", ("discovery",)),
    AstaBenchTaskMetric("E2E_Bench_Hard_validation", "astabench/e2e_discovery_hard_validation", "score_rubric/accuracy", ("discovery",)),
    AstaBenchTaskMetric("ArxivDIGESTables_Clean_test", "astabench/arxivdigestables_test", "score_tables/mean", ("lit",)),
    AstaBenchTaskMetric("ScholarQA_CS2_test", "astabench/sqa_test", "global_avg/mean", ("lit",)),
    AstaBenchTaskMetric("LitQA2_FullText_test", "astabench/litqa2_test", "is_correct/accuracy", ("lit",), 0.5),
    AstaBenchTaskMetric("PaperFindingBench_test", "astabench/paper_finder_test", "score_paper_finder/adjusted_f1_micro_avg", ("lit",)),
    AstaBenchTaskMetric("LitQA2_FullText_Search_test", "astabench/paper_finder_litqa2_test", "score_paper_finder/recall_at_30", ("lit",), 0.5),
    AstaBenchTaskMetric("DiscoveryBench_test", "astabench/discoverybench_test", "score_discoverybench/mean", ("data",)),
    AstaBenchTaskMetric("SUPER_Expert_test", "astabench/super_test", "output_match/mean", ("code",)),
    AstaBenchTaskMetric("CORE_Bench_Hard_test", "astabench/core_bench_test", "score_with_stderr/accuracy", ("code",)),
    AstaBenchTaskMetric("DS1000_test", "astabench/ds1000_test", "ds1000_scorer/accuracy", ("code",)),
    AstaBenchTaskMetric("E2E_Bench_test", "astabench/e2e_discovery_test", "score_rubric/accuracy", ("discovery",)),
    AstaBenchTaskMetric("E2E_Bench_Hard_test", "astabench/e2e_discovery_hard_test", "score_rubric/accuracy", ("discovery",)),
)

_BY_NAME = {task.name: task for task in ASTABENCH_CONFIG_1_0_0}
_BY_PATH = {task.path: task for task in ASTABENCH_CONFIG_1_0_0}


def task_metric_for(identifier: str) -> AstaBenchTaskMetric | None:
    return _BY_NAME.get(identifier) or _BY_PATH.get(identifier)


def primary_score(metrics: dict[str, Any], task_name: str | None = None) -> float | None:
    candidates: list[str] = []
    if task_name:
        task_metric = task_metric_for(task_name)
        if task_metric is not None:
            candidates.append(task_metric.primary_metric)
    candidates.extend(
        [
            "score/value",
            "accuracy",
            "mean",
            "score",
            "value",
        ]
    )
    for key in candidates:
        value = _number_at(metrics, key)
        if value is not None:
            return value
    for value in metrics.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        score_value = _inspect_score_value(value)
        if score_value is not None:
            return score_value
    return None


def compute_official_macro(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute AstaBench-style category macro scores from per-task records.

    Each record should include `task_name` or `task_path` and either `score` or `metrics`.
    Missing tasks reduce coverage and are ignored in category means.
    """

    by_tag: dict[str, list[tuple[float, float]]] = {"lit": [], "data": [], "code": [], "discovery": []}
    seen_tasks: set[str] = set()
    for record in records:
        task = _record_task_metric(record)
        if task is None:
            continue
        score = _record_score(record, task)
        if score is None:
            continue
        seen_tasks.add(task.name)
        for tag in task.tags:
            by_tag.setdefault(tag, []).append((float(score), task.weight))

    category_scores = {
        tag: _weighted_mean(values)
        for tag, values in by_tag.items()
        if values
    }
    overall_values = [category_scores[tag] for tag in ("lit", "data", "code", "discovery") if tag in category_scores]
    return {
        "overall": sum(overall_values) / len(overall_values) if overall_values else None,
        "categories": category_scores,
        "coverage": {
            "tasks_scored": len(seen_tasks),
            "configured_tasks": len(ASTABENCH_CONFIG_1_0_0),
            "task_fraction": len(seen_tasks) / len(ASTABENCH_CONFIG_1_0_0),
        },
    }


def _record_task_metric(record: dict[str, Any]) -> AstaBenchTaskMetric | None:
    for key in ("task_name", "task", "task_path", "path"):
        value = record.get(key)
        if isinstance(value, str):
            metric = task_metric_for(value)
            if metric is not None:
                return metric
    return None


def _record_score(record: dict[str, Any], task: AstaBenchTaskMetric) -> float | None:
    value = record.get("score")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        return primary_score(metrics, task.name)
    return None


def _number_at(metrics: dict[str, Any], key: str) -> float | None:
    if key in metrics and isinstance(metrics[key], (int, float)) and not isinstance(metrics[key], bool):
        return float(metrics[key])
    current: Any = metrics
    for part in key.split("/"):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return float(current)
    return None


def _inspect_score_value(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("value")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        if raw == "C":
            return 1.0
        if raw == "I":
            return 0.0
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    weight = sum(item[1] for item in values)
    if weight <= 0:
        return 0.0
    return sum(score * item_weight for score, item_weight in values) / weight
