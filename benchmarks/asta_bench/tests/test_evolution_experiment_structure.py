from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from openevo_asta_bench.evolution_experiment import (
    AttemptRecord,
    _default_sample_ids,
    _evolve_revision_artifacts,
    _parse_log,
    _run_attempt,
    _write_core_evolution_artifacts,
    _write_freeze_manifest,
    _write_revision_context,
    _write_summary,
)


def test_run_attempt_can_disable_scoring_and_retries(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(
        "openevo_asta_bench.evolution_experiment.subprocess.run", fake_run
    )

    _run_attempt(
        asta_root=tmp_path,
        env={},
        uv="uv",
        task="astabench/discoverybench_validation",
        sample_id="sample-001",
        sample_index=1,
        sample_selector="limit",
        model="openai/cc-gpt-5.5",
        solver_spec="package/solver",
        max_steps=24,
        time_limit=1800,
        task_args=[],
        log_dir=tmp_path / "logs" / "sample-01-evo-0",
        revision_manifest_path=None,
        no_score=True,
        retry_on_error=0,
        openai_tool_schema="flat",
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert "--no-score" in command
    assert "--retry-on-error" not in command
    assert "openai_tool_schema=flat" in command


def test_revision_context_combines_global_policy_and_task_local_memory(tmp_path) -> None:
    global_policy = tmp_path / "global_policy.md"
    task_memory = tmp_path / "task_memory.md"
    global_policy.write_text("Use verified evidence before final answers.\n", encoding="utf-8")
    task_memory.write_text("Retry with narrower search terms.\n", encoding="utf-8")

    context = _write_revision_context(
        context_dir=tmp_path / "contexts",
        sample_index=3,
        evolution_count=1,
        global_policy_path=global_policy,
        task_memory_path=task_memory,
        task_skill_path=None,
        task_agent_system_path=None,
        mode="validation_evolve",
    )

    assert context is not None
    text = context.read_text(encoding="utf-8")
    assert "# Frozen Global Policy" in text
    assert "Use verified evidence" in text
    assert "# Task-Local Retry Memory" in text
    assert "narrower search terms" in text


def test_revision_context_combines_skill_and_agent_system(tmp_path) -> None:
    memory = tmp_path / "memory.md"
    skill = tmp_path / "skill" / "SKILL.md"
    agent_system = tmp_path / "agent_system.md"
    memory.write_text("Use narrower search terms.\n", encoding="utf-8")
    skill.parent.mkdir()
    skill.write_text("# AstaBench Retry Skill\n\n- Verify corpus IDs.\n", encoding="utf-8")
    agent_system.write_text("# AstaBench Agent-System Guidance\n\n- Preserve schema.\n", encoding="utf-8")

    context = _write_revision_context(
        context_dir=tmp_path / "contexts",
        sample_index=2,
        evolution_count=1,
        global_policy_path=None,
        task_memory_path=memory,
        task_skill_path=skill,
        task_agent_system_path=agent_system,
        mode="validation_evolve",
    )

    assert context is not None
    text = context.read_text(encoding="utf-8")
    assert "# Task-Local Retry Memory" in text
    assert "# Task-Local Skill Bundle" in text
    assert "Verify corpus IDs" in text
    assert "# Task-Local Agent System" in text
    assert "Preserve schema" in text


def test_frozen_revision_context_excludes_task_local_memory(tmp_path) -> None:
    global_policy = tmp_path / "global_policy.md"
    task_memory = tmp_path / "task_memory.md"
    global_policy.write_text("Global only.\n", encoding="utf-8")
    task_memory.write_text("Validation retry memory.\n", encoding="utf-8")

    context = _write_revision_context(
        context_dir=tmp_path / "contexts",
        sample_index=1,
        evolution_count=0,
        global_policy_path=global_policy,
        task_memory_path=None,
        task_skill_path=None,
        task_agent_system_path=None,
        mode="frozen_eval",
    )

    assert context is not None
    text = context.read_text(encoding="utf-8")
    assert "Global only" in text
    assert "Validation retry memory" not in text


def test_summary_records_actual_task_and_mode(tmp_path) -> None:
    attempt = AttemptRecord(
        sample_id="litqa2_2",
        attempt_index=0,
        evolution_count=0,
        log_dir="logs/sample-01-evo-0",
        log_file=None,
        status="success",
        correct=True,
        score=1.0,
    )

    _write_summary(
        tmp_path,
        [attempt],
        [
            {
                "sample_id": "litqa2_2",
                "final_score": 1.0,
                "final_correct": True,
                "evolution_count": 0,
                "attempts": 1,
                "first_score": 1.0,
                "first_correct": True,
            }
        ],
        task="astabench/paper_finder_litqa2_validation",
        mode="frozen_eval",
        global_policy_path=None,
        task_local_memory_dir=tmp_path / "task_local_memory",
        max_evolutions=0,
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["task"] == "astabench/paper_finder_litqa2_validation"
    assert summary["mode"] == "frozen_eval"
    assert summary["max_evolutions"] == 0
    assert summary["test_feedback_to_evolution"] is False


def test_evolve_revision_artifacts_writes_memory_skill_and_agent_system(tmp_path) -> None:
    attempt = AttemptRecord(
        sample_id="litqa2_2",
        attempt_index=0,
        evolution_count=0,
        log_dir="logs/sample-01-evo-0",
        log_file=None,
        status="success",
        correct=False,
        score=0.0,
        completion="Answered with the wrong schema.",
    )

    _evolve_revision_artifacts(
        env={},
        codex_bin="missing-codex",
        codex_model="gpt-5.5",
        reflector_backend="openai_api",
        reflector_model="cc-gpt-5.5",
        task="astabench/litqa2_validation",
        sample_id="litqa2_2",
        attempts=[attempt],
        memory_output_path=tmp_path / "task_local_memory" / "sample-01-after-evo-1.md",
        skill_output_path=tmp_path / "task_local_skills" / "sample-01-after-evo-1" / "SKILL.md",
        agent_system_output_path=tmp_path
        / "task_local_agent_system"
        / "sample-01-after-evo-1.md",
    )

    memory = (tmp_path / "task_local_memory" / "sample-01-after-evo-1.md").read_text(
        encoding="utf-8"
    )
    skill = (
        tmp_path / "task_local_skills" / "sample-01-after-evo-1" / "SKILL.md"
    ).read_text(encoding="utf-8")
    agent_system = (
        tmp_path / "task_local_agent_system" / "sample-01-after-evo-1.md"
    ).read_text(encoding="utf-8")
    assert "Retry Memory" in memory
    assert "# AstaBench Retry Skill" in skill
    assert "Do not copy validation labels" in skill
    assert "# AstaBench Agent-System Guidance" in agent_system
    assert "Treat validation-derived guidance as method advice" in agent_system


def test_core_evolution_artifacts_write_formal_openevo_manifest(tmp_path) -> None:
    attempt = AttemptRecord(
        sample_id="sample-001",
        attempt_index=0,
        evolution_count=0,
        log_dir="logs/sample-01-evo-0",
        log_file=None,
        status="success",
        correct=False,
        score=0.25,
        completion='{"hypothesis": "h", "workflow": "w"}',
    )
    memory = tmp_path / "task_local_memory" / "sample-01-after-evo-1.md"
    skill = tmp_path / "task_local_skills" / "sample-01-after-evo-1" / "SKILL.md"
    agent_system = tmp_path / "task_local_agent_system" / "sample-01-after-evo-1.md"
    memory.parent.mkdir(parents=True)
    skill.parent.mkdir(parents=True)
    agent_system.parent.mkdir(parents=True)
    memory.write_text("# Memory\n", encoding="utf-8")
    skill.write_text("# Skill\n", encoding="utf-8")
    agent_system.write_text("# Agent\n", encoding="utf-8")

    manifest_path = _write_core_evolution_artifacts(
        output_dir=tmp_path,
        core_artifacts_dir=tmp_path / "core_artifacts",
        task="astabench/discoverybench_validation",
        sample_index=1,
        sample_id="sample-001",
        attempts=[attempt],
        methods=("text_memory", "skill_bundle", "agent_system"),
        reflector_backend="openai_api",
        reflector_model="cc-gpt-5.5",
        env={},
        memory_path=memory,
        skill_path=skill,
        agent_system_path=agent_system,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "openevo_core_evolution_artifacts"
    assert manifest["dataset_artifact"]["type"] == "dataset"
    assert manifest["errors"] == []
    assert [artifact["type"] for artifact in manifest["artifacts"]] == [
        "text_memory",
        "skill_bundle",
        "agent_system",
    ]
    assert (tmp_path / "core_artifacts" / "index.json").exists()


def test_experiment_freeze_manifest_marks_no_upload_or_test_feedback(tmp_path) -> None:
    policy = tmp_path / "global_policy.md"
    policy.write_text("Frozen policy.\n", encoding="utf-8")

    manifest_path = _write_freeze_manifest(
        path=tmp_path / "freeze_manifest.json",
        task="astabench/litqa2_validation",
        mode="frozen_eval",
        model="openai/cc-gpt-5.5",
        reflector_backend="openai_api",
        reflector_model="cc-gpt-5.5",
        global_policy_path=policy,
        task_local_memory_dir=tmp_path / "task_local_memory",
        max_steps=2,
        max_evolutions=0,
        sample_count=10,
        output_dir=tmp_path / "run",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["benchmark"] == "astabench"
    assert manifest["test_feedback_to_evolution"] is False
    assert manifest["leaderboard_upload_performed"] is False
    assert manifest["global_policy"]["exists"] is True


def test_sample_count_generates_limit_placeholders() -> None:
    samples = _default_sample_ids("astabench/paper_finder_validation", sample_count=66)

    assert len(samples) == 66
    assert samples[0] == "sample-001"
    assert samples[-1] == "sample-066"


def test_discoverybench_uses_official_split_sizes() -> None:
    validation = _default_sample_ids("astabench/discoverybench_validation")
    test = _default_sample_ids("astabench/discoverybench_test")

    assert len(validation) == 25
    assert len(test) == 239


def test_discoverybench_rejects_sample_count_beyond_split() -> None:
    with pytest.raises(ValueError, match="exceeds the 25 samples"):
        _default_sample_ids("astabench/discoverybench_validation", sample_count=239)


def test_parse_log_marks_empty_inspect_run_as_no_samples(tmp_path) -> None:
    log = tmp_path / "empty.json"
    log.write_text(
        json.dumps(
            {
                "status": "success",
                "eval": {"dataset": {"samples": 25, "sample_ids": []}},
                "samples": [],
                "stats": {"model_usage": {}},
            }
        ),
        encoding="utf-8",
    )

    parsed = _parse_log(log, task_name="astabench/discoverybench_validation")

    assert parsed["status"] == "no_samples"
    assert parsed["score"] == 0.0
    assert parsed["error"] == "Inspect selected no samples from a dataset of 25"
