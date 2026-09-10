from __future__ import annotations

import json

import pytest

from openevo_asta_bench.cli import main
from openevo_asta_bench.evolution_runner import build_astabench_eval_command, build_freeze_manifest, write_freeze_manifest


def test_eval_command_is_local_non_uploading_and_pins_gpt55() -> None:
    command = build_astabench_eval_command(log_dir="logs/smoke", limit=1)

    assert command[:3] == ["uv", "run", "astabench"]
    assert "--solver" in command
    assert "openevo_asta_bench/astabench_openevo_solver" in command
    assert command[command.index("--model") + 1] == "openai/cc-gpt-5.5"
    assert "model=openai/cc-gpt-5.5" in command
    assert "--log-format" in command
    assert "upload" not in " ".join(command)


def test_freeze_manifest_disables_test_feedback_and_upload(tmp_path) -> None:
    manifest = build_freeze_manifest(revision="rev-123", split="test")
    output = write_freeze_manifest(tmp_path / "freeze.json", manifest)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["agent_model"] == "openai/cc-gpt-5.5"
    assert data["codex_model"] == "gpt-5.5"
    assert data["reflector_provider"] == "codex_cli"
    assert data["reflector_model"] == "gpt-5.5"
    assert data["test_feedback_to_evolution"] is False
    assert data["upload_enabled"] is False


def test_freeze_rejects_non_codex_reflector() -> None:
    with pytest.raises(ValueError, match="codex_cli"):
        build_freeze_manifest(
            revision="rev-123",
            split="validation",
            reflector_provider="openai_chat",
        )


def test_cli_prints_eval_command(capsys) -> None:
    assert main(["eval-command", "--log-dir", "logs/smoke", "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert "astabench eval" in out
    assert "--model openai/cc-gpt-5.5" in out


def test_eval_command_for_single_task_uses_inspect_eval() -> None:
    command = build_astabench_eval_command(
        log_dir="logs/sqa-smoke",
        split="validation",
        task="astabench/sqa_dev",
        limit=1,
    )

    assert command[:5] == ["uv", "run", "inspect", "eval", "astabench/sqa_dev"]
    assert "--split" not in command
    assert "--ignore-git" not in command
    assert "--log-dir" in command
    assert command[command.index("--log-dir") + 1] == "logs/sqa-smoke"


def test_cli_run_invokes_runner(monkeypatch) -> None:
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr("openevo_asta_bench.cli.run_astabench_eval", fake_run)

    assert (
        main(
            [
                "run",
                "--log-dir",
                "logs/sqa-smoke",
                "--task",
                "astabench/sqa_dev",
                "--limit",
                "1",
                "--no-score",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "log_dir": "logs/sqa-smoke",
            "split": "validation",
            "task": "astabench/sqa_dev",
            "model": "openai/cc-gpt-5.5",
            "limit": 1,
            "max_samples": 1,
            "revision_manifest": None,
            "no_score": True,
        }
    ]
