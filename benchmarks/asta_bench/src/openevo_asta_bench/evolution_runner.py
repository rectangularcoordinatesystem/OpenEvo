from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess

from openevo_asta_bench.constants import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_REFLECTOR_MODEL,
    DEFAULT_REFLECTOR_PROVIDER,
    DEFAULT_TOOLSET,
    GPT55_INSPECT_MODEL_ALIASES,
)


@dataclass(frozen=True)
class FreezeManifest:
    benchmark: str
    astabench_version: str
    config_version: str
    split: str
    revision: str
    agent_model: str
    codex_model: str
    reflector_provider: str
    reflector_model: str
    toolset: str
    test_feedback_to_evolution: bool = False
    upload_enabled: bool = False


def build_freeze_manifest(
    *,
    revision: str,
    split: str,
    astabench_version: str = "v0.3.1",
    config_version: str = "1.0.0",
    agent_model: str = DEFAULT_AGENT_MODEL,
    reflector_provider: str = DEFAULT_REFLECTOR_PROVIDER,
    reflector_model: str = DEFAULT_REFLECTOR_MODEL,
    toolset: str = DEFAULT_TOOLSET,
) -> FreezeManifest:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if _canonical_gpt55_model(agent_model) != DEFAULT_CODEX_MODEL or reflector_model != DEFAULT_REFLECTOR_MODEL:
        raise ValueError("AstaBench OpenEvo runs are pinned to gpt-5.5")
    if reflector_provider != DEFAULT_REFLECTOR_PROVIDER:
        raise ValueError("OpenEvo background reflector must use codex_cli")
    return FreezeManifest(
        benchmark="astabench",
        astabench_version=astabench_version,
        config_version=config_version,
        split=split,
        revision=revision,
        agent_model=agent_model,
        codex_model=_canonical_gpt55_model(agent_model),
        reflector_provider=reflector_provider,
        reflector_model=reflector_model,
        toolset=toolset,
    )


def write_freeze_manifest(path: str | Path, manifest: FreezeManifest) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_astabench_eval_command(
    *,
    log_dir: str,
    split: str = "validation",
    task: str | None = None,
    model: str = DEFAULT_AGENT_MODEL,
    limit: int | None = None,
    max_samples: int | None = 1,
    revision_manifest: str | None = None,
    no_score: bool = False,
) -> list[str]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if _canonical_gpt55_model(model) != DEFAULT_CODEX_MODEL:
        raise ValueError("AstaBench OpenEvo solver model is pinned to gpt-5.5")
    command = ["uv", "run"]
    is_single_task = bool(task)
    if is_single_task:
        command.extend(["inspect", "eval", task])
    else:
        command.extend(["astabench", "eval", "--split", split])
    command.extend(
        [
            "--solver",
            "openevo_asta_bench/astabench_openevo_solver",
            "--model",
            _inspect_model_name(model),
            "--log-dir",
            log_dir,
            "--log-format",
            "json",
            "-S",
            f"model={_inspect_model_name(model)}",
            "-S",
            "with_asta_tools=1",
            "-S",
            "with_stateful_python=1",
            "-S",
            "with_report_editor=1",
            "-S",
            "with_table_editor=1",
            "-S",
            "with_editor_submit=1",
        ]
    )
    if not is_single_task:
        command.append("--ignore-git")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples), "--max-sandboxes", str(max_samples)])
    if no_score:
        command.append("--no-score")
    if revision_manifest:
        command.extend(["-S", f"revision_manifest={revision_manifest}"])
    return command


def build_astabench_config_command(*, log_dir: str, split: str = "validation") -> list[str]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    return [
        "uv",
        "run",
        "astabench",
        "eval",
        "--config-only",
        "--log-dir",
        log_dir,
        "--split",
        split,
        "--ignore-git",
    ]


def run_astabench_eval(
    *,
    log_dir: str,
    split: str = "validation",
    task: str | None = None,
    model: str = DEFAULT_AGENT_MODEL,
    limit: int | None = None,
    max_samples: int | None = 1,
    revision_manifest: str | None = None,
    no_score: bool = False,
) -> int:
    command = build_astabench_eval_command(
        log_dir=log_dir,
        split=split,
        task=task,
        model=model,
        limit=limit,
        max_samples=max_samples,
        revision_manifest=revision_manifest,
        no_score=no_score,
    )
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    if task:
        config_completed = subprocess.run(
            build_astabench_config_command(log_dir=log_dir, split=split),
            check=False,
        )
        if config_completed.returncode != 0:
            return config_completed.returncode
    return 0


def _codex_model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _canonical_gpt55_model(model: str) -> str:
    name = _codex_model_name(model)
    if name in GPT55_INSPECT_MODEL_ALIASES:
        return DEFAULT_CODEX_MODEL
    return name


def _inspect_model_name(model: str) -> str:
    return model if "/" in model else f"openai/{model}"
