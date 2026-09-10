from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from openevo.evolution.methods import run_method
from openevo.evolution.models import ArtifactRegisterRequest, ArtifactType, WorkerClaimInputArtifact, WorkerClaimedJob
from openevo_asta_bench.inspect_bridge import AstaBenchBridgeError, build_astabench_events
from openevo_asta_bench.metrics import primary_score


DEFAULT_UV = "/home/mark/.local/bin/uv"
DEFAULT_MODEL = "openai/cc-gpt-5.5"
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_TASK = "astabench/litqa2_validation"
RUN_MODES = ("validation_evolve", "frozen_eval")
DEFAULT_CORE_METHODS = (
    "text_memory",
    "text_memory_reflector",
    "skill_bundle",
    "skill_bundle_reflector",
    "agent_system",
    "agent_system_reflector",
)
SUPPORTED_CORE_METHODS_NOT_RUN_BY_DEFAULT = (
    "text_memory_expel_reflector",
    "agent_system_history_reflector",
    "agent_system_pareto_reflector",
    "agent_system_gepa_reflector",
    "parametric_memory_register",
    "parametric_memory_lora_sft",
)

KNOWN_TASK_SAMPLE_COUNTS = {
    "astabench/discoverybench_validation": 25,
    "astabench/discoverybench_test": 239,
}
DEFAULT_SAMPLE_IDS = [
    "76184ccf-4bf0-469e-a442-11d04b4ff8b0",
    "247eeb85-a552-4b87-b83e-327538fcb8a9",
    "8d7fa642-ee46-4a13-8ea9-61cc2d4f4ddd",
    "5a2128ad-3127-4595-b810-db128d1a2335",
    "aa1835b2-2b1c-4986-b7af-e174da0124b0",
    "b105af85-833e-48bc-ac78-48f73c9673fd",
    "634f6745-f3b3-4cb1-9859-96ffb954b98a",
    "9088251a-99fe-4b91-b6a9-375154ec4f58",
    "b2a0249b-2850-4576-8b78-8408c1e47324",
    "afb36e40-0836-4810-909e-eabfe17dcd8b",
]
PAPER_FINDER_LITQA2_VALIDATION_IDS = [
    "litqa2_2",
    "litqa2_11",
    "litqa2_17",
    "litqa2_26",
    "litqa2_41",
    "litqa2_53",
    "litqa2_156",
    "litqa2_162",
    "litqa2_176",
    "litqa2_188",
]


@dataclass
class AttemptRecord:
    sample_id: str
    attempt_index: int
    evolution_count: int
    log_dir: str
    log_file: str | None
    status: str
    correct: bool
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    completion: str = ""
    error: str = ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openevo-asta-evolve")
    parser.add_argument("--asta-root", default="/home/mark/asta_openevo/third_party/asta-bench")
    parser.add_argument("--env-file", default="/home/mark/asta_openevo/third_party/asta-bench/.env")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--openai-tool-schema",
        choices=["standard", "flat"],
        default="standard",
        help=(
            "Outbound Chat Completions tool declaration format. Use flat only "
            "for a relay that explicitly requires tools[].name."
        ),
    )
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--reflector-backend", choices=["openai_api", "codex_cli"], default="openai_api")
    parser.add_argument("--reflector-model", default="cc-gpt-5.5")
    parser.add_argument("--mode", choices=RUN_MODES, default="validation_evolve")
    parser.add_argument("--global-policy", default="")
    parser.add_argument("--task-local-memory-dir", default="")
    parser.add_argument("--task-local-skill-dir", default="")
    parser.add_argument("--task-local-agent-system-dir", default="")
    parser.add_argument("--core-artifacts-dir", default="")
    parser.add_argument("--core-method", action="append", default=[])
    parser.add_argument("--skip-core-artifacts", action="store_true")
    parser.add_argument("--context-dir", default="")
    parser.add_argument("--promote-global-policy", default="")
    parser.add_argument("--write-freeze-manifest", default="")
    parser.add_argument("--reset-task-state", action="store_true")
    parser.add_argument("--max-evolutions", type=int, default=5)
    parser.add_argument("--target-score", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--time-limit", type=int, default=240)
    parser.add_argument("--task-arg", action="append", default=[])
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--sample-start-index", type=int, default=1)
    parser.add_argument("--sample-end-index", type=int, default=0)
    parser.add_argument("--sample-selector", choices=["limit", "id"], default="limit")
    parser.add_argument(
        "--no-score",
        action="store_true",
        help=(
            "Skip benchmark scoring. Use this for infrastructure/API smoke tests "
            "so a failed solver cannot trigger paid scorer calls."
        ),
    )
    parser.add_argument(
        "--retry-on-error",
        type=int,
        default=None,
        help=(
            "Number of Inspect retries after a sample error. Defaults to 0 with "
            "--no-score and 2 otherwise."
        ),
    )
    parser.add_argument("--uv", default=DEFAULT_UV)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--solver-spec", default="")
    args = parser.parse_args(argv)

    if args.no_score and args.mode != "frozen_eval":
        parser.error("--no-score is only supported with --mode frozen_eval")
    if args.retry_on_error is not None and args.retry_on_error < 0:
        parser.error("--retry-on-error must be non-negative")
    retry_on_error = (
        args.retry_on_error
        if args.retry_on_error is not None
        else (0 if args.no_score else 2)
    )

    asta_root = Path(args.asta_root)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(asta_root)
    if args.reset_task_state:
        _reset_task_state(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = Path(args.task_local_memory_dir) if args.task_local_memory_dir else output_dir / "task_local_memory"
    memory_dir.mkdir(exist_ok=True)
    skill_dir = Path(args.task_local_skill_dir) if args.task_local_skill_dir else output_dir / "task_local_skills"
    skill_dir.mkdir(exist_ok=True)
    agent_system_dir = (
        Path(args.task_local_agent_system_dir)
        if args.task_local_agent_system_dir
        else output_dir / "task_local_agent_system"
    )
    agent_system_dir.mkdir(exist_ok=True)
    core_artifacts_dir = (
        Path(args.core_artifacts_dir) if args.core_artifacts_dir else output_dir / "core_artifacts"
    )
    if not args.skip_core_artifacts:
        core_artifacts_dir.mkdir(exist_ok=True)
    context_dir = Path(args.context_dir) if args.context_dir else output_dir / "revision_context"
    context_dir.mkdir(exist_ok=True)
    logs_root = output_dir / "logs"
    logs_root.mkdir(exist_ok=True)

    env = os.environ.copy()
    env.update(_read_env_file(Path(args.env_file)))
    samples = (
        list(enumerate(args.sample_id, start=1))
        if args.sample_id
        else _sample_specs(
            args.task,
            sample_count=args.sample_count,
            sample_start_index=args.sample_start_index,
            sample_end_index=args.sample_end_index,
        )
    )
    global_policy_path = Path(args.global_policy) if args.global_policy else None
    core_methods = tuple(args.core_method) if args.core_method else DEFAULT_CORE_METHODS
    solver_spec = args.solver_spec or _default_solver_spec()
    max_evolutions = 0 if args.mode == "frozen_eval" else args.max_evolutions
    attempts = _load_attempts(output_dir / "attempts.json")
    final_records = _load_final_records(output_dir / "summary.json")

    for sample_index, sample_id in samples:
        if any(record.get("sample_id") == sample_id for record in final_records):
            print(f"[sample {sample_index}/{len(samples)}] {sample_id} already complete, skipping", flush=True)
            continue
        sample_attempts = [attempt for attempt in attempts if attempt.sample_id == sample_id]
        next_evolution = (
            max((attempt.evolution_count for attempt in sample_attempts), default=-1) + 1
        )
        task_memory_path: Path | None = (
            memory_dir / f"sample-{sample_index:02d}-after-evo-{next_evolution}.md"
            if next_evolution > 0
            else None
        )
        task_skill_path: Path | None = (
            skill_dir / f"sample-{sample_index:02d}-after-evo-{next_evolution}" / "SKILL.md"
            if next_evolution > 0
            else None
        )
        task_agent_system_path: Path | None = (
            agent_system_dir / f"sample-{sample_index:02d}-after-evo-{next_evolution}.md"
            if next_evolution > 0
            else None
        )
        print(f"[sample {sample_index}/{len(samples)}] {sample_id}", flush=True)
        for evolution_count in range(next_evolution, max_evolutions + 1):
            log_dir = logs_root / f"sample-{sample_index:02d}-evo-{evolution_count}"
            revision_context = _write_revision_context(
                context_dir=context_dir,
                sample_index=sample_index,
                evolution_count=evolution_count,
                global_policy_path=global_policy_path,
                task_memory_path=task_memory_path if args.mode == "validation_evolve" else None,
                task_skill_path=task_skill_path if args.mode == "validation_evolve" else None,
                task_agent_system_path=task_agent_system_path if args.mode == "validation_evolve" else None,
                mode=args.mode,
            )
            record = _run_attempt(
                asta_root=asta_root,
                env=env,
                uv=args.uv,
                task=args.task,
                sample_id=sample_id,
                sample_index=sample_index,
                sample_selector=args.sample_selector,
                model=args.model,
                openai_tool_schema=args.openai_tool_schema,
                solver_spec=solver_spec,
                max_steps=args.max_steps,
                time_limit=args.time_limit,
                task_args=args.task_arg,
                log_dir=log_dir,
                revision_manifest_path=revision_context,
                no_score=args.no_score,
                retry_on_error=retry_on_error,
            )
            attempts.append(record)
            sample_attempts.append(record)
            print(
                f"  evo={evolution_count} status={record.status} "
                f"score={record.score:.3f} correct={record.correct}",
                flush=True,
            )
            _write_json(output_dir / "attempts.json", [r.__dict__ for r in attempts])
            if record.status == "no_samples":
                print(
                    "no Inspect sample was selected; stopping because the requested "
                    "sample index is outside the task dataset",
                    flush=True,
                )
                _write_summary(
                    output_dir,
                    attempts,
                    final_records,
                    task=args.task,
                    mode=args.mode,
                    global_policy_path=global_policy_path,
                    task_local_memory_dir=memory_dir,
                    task_local_skill_dir=skill_dir,
                    task_local_agent_system_dir=agent_system_dir,
                    core_artifacts_dir=core_artifacts_dir,
                    core_methods=core_methods,
                    max_evolutions=max_evolutions,
                )
                return 4
            if _is_fatal_api_error(record.error):
                print(
                    "fatal API error detected; stopping this shard before "
                    "polluting remaining samples with invalid zero scores",
                    flush=True,
                )
                _write_summary(
                    output_dir,
                    attempts,
                    final_records,
                    task=args.task,
                    mode=args.mode,
                    global_policy_path=global_policy_path,
                    task_local_memory_dir=memory_dir,
                    task_local_skill_dir=skill_dir,
                    task_local_agent_system_dir=agent_system_dir,
                    core_artifacts_dir=core_artifacts_dir,
                    core_methods=core_methods,
                    max_evolutions=max_evolutions,
                )
                return 2
            if _is_transient_infra_error(record.error):
                print(
                    "transient infrastructure error persisted after retries; "
                    "stopping this shard so it can be resumed cleanly",
                    flush=True,
                )
                _write_summary(
                    output_dir,
                    attempts,
                    final_records,
                    task=args.task,
                    mode=args.mode,
                    global_policy_path=global_policy_path,
                    task_local_memory_dir=memory_dir,
                    task_local_skill_dir=skill_dir,
                    task_local_agent_system_dir=agent_system_dir,
                    core_artifacts_dir=core_artifacts_dir,
                    core_methods=core_methods,
                    max_evolutions=max_evolutions,
                )
                return 3
            if record.correct:
                break
            stop_reason = _adaptive_stop_reason(
                sample_attempts,
                target_score=args.target_score,
                patience=args.patience,
            )
            if stop_reason:
                print(f"  stopping sample early: {stop_reason}", flush=True)
                break
            if args.mode == "frozen_eval":
                break
            if evolution_count >= max_evolutions:
                break
            task_memory_path = memory_dir / f"sample-{sample_index:02d}-after-evo-{evolution_count + 1}.md"
            task_skill_path = skill_dir / f"sample-{sample_index:02d}-after-evo-{evolution_count + 1}" / "SKILL.md"
            task_agent_system_path = agent_system_dir / f"sample-{sample_index:02d}-after-evo-{evolution_count + 1}.md"
            _evolve_revision_artifacts(
                env=env,
                codex_bin=args.codex_bin,
                codex_model=args.codex_model,
                reflector_backend=args.reflector_backend,
                reflector_model=args.reflector_model,
                task=args.task,
                sample_id=sample_id,
                attempts=sample_attempts,
                memory_output_path=task_memory_path,
                skill_output_path=task_skill_path,
                agent_system_output_path=task_agent_system_path,
            )
        if sample_attempts:
            final = sample_attempts[-1]
            best = max(sample_attempts, key=lambda attempt: attempt.score)
            final_records.append(
                {
                    "sample_id": sample_id,
                    "final_score": final.score,
                    "final_correct": final.correct,
                    "best_score": best.score,
                    "best_correct": best.correct,
                    "best_evolution_count": best.evolution_count,
                    "evolution_count": final.evolution_count,
                    "attempts": len(sample_attempts),
                    "first_score": sample_attempts[0].score,
                    "first_correct": sample_attempts[0].correct,
                }
            )
            core_manifest_path = None
            if not args.skip_core_artifacts:
                core_manifest_path = _write_core_evolution_artifacts(
                    output_dir=output_dir,
                    core_artifacts_dir=core_artifacts_dir,
                    task=args.task,
                    sample_index=sample_index,
                    sample_id=sample_id,
                    attempts=sample_attempts,
                    methods=core_methods,
                    reflector_backend=args.reflector_backend,
                    reflector_model=args.reflector_model,
                    env=env,
                    memory_path=task_memory_path,
                    skill_path=task_skill_path,
                    agent_system_path=task_agent_system_path,
                )
                final_records[-1]["core_artifacts_manifest"] = str(core_manifest_path)
            _write_summary(
                output_dir,
                attempts,
                final_records,
                task=args.task,
                mode=args.mode,
                global_policy_path=global_policy_path,
                task_local_memory_dir=memory_dir,
                task_local_skill_dir=skill_dir,
                task_local_agent_system_dir=agent_system_dir,
                core_artifacts_dir=core_artifacts_dir,
                core_methods=core_methods,
                max_evolutions=max_evolutions,
            )

    policy_update_path = (
        _write_task_policy_update(
            env=env,
            reflector_model=args.reflector_model,
            task=args.task,
            attempts=attempts,
            final_records=final_records,
            output_path=Path(args.promote_global_policy),
        )
        if args.promote_global_policy and args.mode == "validation_evolve"
        else None
    )
    _write_summary(
        output_dir,
        attempts,
        final_records,
        task=args.task,
        mode=args.mode,
        global_policy_path=global_policy_path,
        task_local_memory_dir=memory_dir,
        task_local_skill_dir=skill_dir,
        task_local_agent_system_dir=agent_system_dir,
        core_artifacts_dir=core_artifacts_dir,
        core_methods=core_methods,
        max_evolutions=max_evolutions,
        policy_update_path=policy_update_path,
    )
    if args.write_freeze_manifest:
        _write_freeze_manifest(
            path=Path(args.write_freeze_manifest),
            task=args.task,
            mode=args.mode,
            model=args.model,
            reflector_backend=args.reflector_backend,
            reflector_model=args.reflector_model,
            global_policy_path=global_policy_path,
            task_local_memory_dir=memory_dir,
            task_local_skill_dir=skill_dir,
            task_local_agent_system_dir=agent_system_dir,
            max_steps=args.max_steps,
            max_evolutions=max_evolutions,
            sample_count=len(samples),
            output_dir=output_dir,
        )
    _write_svg_charts(output_dir, attempts, final_records)
    print(f"summary: {output_dir / 'summary.json'}", flush=True)
    print(f"charts: {output_dir / 'charts.svg'}", flush=True)
    return 0


def _run_attempt(
    *,
    asta_root: Path,
    env: dict[str, str],
    uv: str,
    task: str,
    sample_id: str,
    sample_index: int,
    sample_selector: str,
    model: str,
    openai_tool_schema: str,
    solver_spec: str,
    max_steps: int,
    time_limit: int,
    task_args: list[str],
    log_dir: Path,
    revision_manifest_path: Path | None,
    no_score: bool,
    retry_on_error: int,
) -> AttemptRecord:
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        uv,
        "run",
        "inspect",
        "eval",
        "--solver",
        solver_spec,
        "--model",
        model,
        "-S",
        f"model={model}",
        "-S",
        f"openai_tool_schema={openai_tool_schema}",
        "-S",
        f"max_steps={max_steps}",
        "-S",
        f"task_name={task}",
        task,
        "--max-sandboxes",
        "1",
        "--time-limit",
        str(time_limit),
        "--no-fail-on-error",
        "--log-dir",
        str(log_dir),
        "--log-format",
        "json",
        "--display",
        "plain",
    ]
    if no_score:
        command.append("--no-score")
    if retry_on_error > 0:
        command.extend(["--retry-on-error", str(retry_on_error)])
    for task_arg in task_args:
        command.extend(["-T", task_arg])
    if sample_selector == "id":
        command.extend(["--sample-id", sample_id, "--max-samples", "1"])
    else:
        command.extend(["--limit", f"{sample_index}-{sample_index}"])
    if revision_manifest_path is not None:
        command.extend(["-S", f"revision_manifest={revision_manifest_path}"])
    timed_out = False
    try:
        result = subprocess.run(
            command,
            cwd=asta_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=time_limit + max(90, int(time_limit / 2) + 60),
        )
        stdout = result.stdout
        returncode = result.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _timeout_output_text(exc)
        returncode = 124
    log_file = _latest_json(log_dir)
    parsed = _parse_log(log_file, task_name=task) if log_file is not None else {}
    return AttemptRecord(
        sample_id=sample_id,
        attempt_index=len(list(log_dir.parent.glob(f"{log_dir.name.split('-evo-')[0]}-evo-*"))) - 1,
        evolution_count=int(log_dir.name.rsplit("-evo-", 1)[1]),
        log_dir=str(log_dir),
        log_file=str(log_file) if log_file else None,
        status=parsed.get("status") or ("timeout" if timed_out else ("error" if returncode else "unknown")),
        correct=bool(parsed.get("correct")),
        score=float(parsed.get("score") or 0.0),
        metrics=parsed.get("metrics") or {},
        usage=parsed.get("usage") or {},
        completion=parsed.get("completion") or "",
        error=parsed.get("error") or (_truncate_tail(stdout, 4000) if returncode else ""),
    )


def _parse_log(path: Path, *, task_name: str | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples") or []
    if not samples:
        dataset_size = ((data.get("eval") or {}).get("dataset") or {}).get("samples")
        return {
            "status": "no_samples",
            "score": 0.0,
            "correct": False,
            "metrics": {},
            "usage": (data.get("stats") or {}).get("model_usage") or {},
            "completion": "",
            "error": (
                "Inspect selected no samples"
                + (f" from a dataset of {dataset_size}" if dataset_size is not None else "")
            ),
        }
    sample = samples[0] if samples else {}
    sample_error = sample.get("error") if isinstance(sample.get("error"), dict) else None
    metrics: dict[str, float] = {}
    score = 0.0
    correct = False
    for scorer_payload in (sample.get("scores") or {}).values():
        if not isinstance(scorer_payload, dict):
            continue
        value = scorer_payload.get("value")
        if value == "C":
            score = max(score, 1.0)
            correct = True
        elif isinstance(value, dict):
            is_correct = value.get("is_correct")
            if is_correct is True or is_correct == "C" or is_correct == 1:
                score = max(score, 1.0)
                correct = True
        elif isinstance(value, (int, float)):
            score = max(score, float(value))
            correct = correct or float(value) >= 1.0
    for result_score in (data.get("results") or {}).get("scores") or []:
        for name, metric in (result_score.get("metrics") or {}).items():
            if isinstance(metric, dict) and isinstance(metric.get("value"), (int, float)):
                key = f"{result_score.get('name')}/{name}"
                metrics[key] = float(metric["value"])
                if key.endswith("is_correct/accuracy") or key.endswith("accuracy"):
                    score = max(score, float(metric["value"]))
                    correct = correct or float(metric["value"]) >= 1.0
    official_primary_score = primary_score(metrics, task_name)
    if official_primary_score is not None:
        score = official_primary_score
        correct = correct or official_primary_score >= 1.0
    completion = ""
    try:
        completion = _message_content_text(sample["output"]["choices"][0]["message"]["content"])
    except Exception:
        completion = ""
    return {
        "status": "error" if sample_error else (data.get("status") or "unknown"),
        "score": score,
        "correct": correct,
        "metrics": metrics,
        "usage": (data.get("stats") or {}).get("model_usage") or {},
        "completion": completion,
        "error": json.dumps(sample_error, ensure_ascii=False) if sample_error else "",
    }


def _evolve_memory(
    *,
    env: dict[str, str],
    codex_bin: str,
    codex_model: str,
    reflector_backend: str,
    reflector_model: str,
    task: str,
    sample_id: str,
    attempts: list[AttemptRecord],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous = "\n\n".join(
        f"Attempt {a.evolution_count}: status={a.status}, score={a.score}, "
        f"correct={a.correct}, completion={a.completion[:500]!r}"
        for a in attempts
    )
    prompt = f"""
You are OpenEvo's background reflector for an AstaBench validation task.
Model: {codex_model}. Produce a concise Markdown memory update for the next
attempt on the same task.

Constraints:
- Do not invent the final answer.
- Do not copy hidden target labels, hidden references, or scorer internals.
- Improve the method: search query formation, evidence extraction, tool use,
  ranking, and the task's required final-answer format.
- For LitQA2 multiple-choice tasks, choose one of the option letters shown in
  the prompt (A, B, ... as available) and preserve the task prompt's answer
  schema.
- Keep it under 350 words.

Task: {task}
Sample id: {sample_id}
Previous attempt summaries:
{previous}
"""
    if reflector_backend == "openai_api":
        memory = _evolve_memory_openai_api(prompt, env=env, model=reflector_model)
        if memory.strip():
            output_path.write_text(memory, encoding="utf-8")
            return
    with tempfile.TemporaryDirectory(prefix="openevo-asta-evolve-") as tmpdir:
        out = Path(tmpdir) / "memory.md"
        process_env = dict(os.environ)
        for key in list(process_env):
            if key.endswith("_API_KEY"):
                process_env.pop(key, None)
        try:
            result = subprocess.run(
                [
                    codex_bin,
                    "exec",
                    "--model",
                    codex_model,
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--output-last-message",
                    str(out),
                    "-",
                ],
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=process_env,
                timeout=240,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            result = None
        if (
            result is not None
            and result.returncode == 0
            and out.exists()
            and out.read_text(encoding="utf-8").strip()
        ):
            output_path.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            output_path.write_text(
                "# LitQA2 Retry Memory\n\n"
                "- Re-read the question and choices before searching.\n"
                "- Use targeted snippet_search queries with exact scientific phrases.\n"
                "- Ground the answer in retrieved evidence and answer with one provided option letter using the requested schema.\n",
                encoding="utf-8",
            )


def _evolve_revision_artifacts(
    *,
    env: dict[str, str],
    codex_bin: str,
    codex_model: str,
    reflector_backend: str,
    reflector_model: str,
    task: str,
    sample_id: str,
    attempts: list[AttemptRecord],
    memory_output_path: Path,
    skill_output_path: Path,
    agent_system_output_path: Path,
) -> None:
    _evolve_memory(
        env=env,
        codex_bin=codex_bin,
        codex_model=codex_model,
        reflector_backend=reflector_backend,
        reflector_model=reflector_model,
        task=task,
        sample_id=sample_id,
        attempts=attempts,
        output_path=memory_output_path,
    )
    memory = _read_optional_text(memory_output_path).strip()
    _write_task_skill_bundle(
        output_path=skill_output_path,
        task=task,
        sample_id=sample_id,
        attempts=attempts,
        memory=memory,
    )
    _write_task_agent_system(
        output_path=agent_system_output_path,
        task=task,
        sample_id=sample_id,
        attempts=attempts,
        memory=memory,
    )


def _write_core_evolution_artifacts(
    *,
    output_dir: Path,
    core_artifacts_dir: Path,
    task: str,
    sample_index: int,
    sample_id: str,
    attempts: list[AttemptRecord],
    methods: tuple[str, ...],
    reflector_backend: str,
    reflector_model: str,
    env: dict[str, str],
    memory_path: Path | None,
    skill_path: Path | None,
    agent_system_path: Path | None,
) -> Path:
    sample_dir = core_artifacts_dir / f"sample-{sample_index:02d}"
    dataset_dir = sample_dir / "dataset"
    artifact_root = sample_dir / "artifacts"
    sample_dir.mkdir(parents=True, exist_ok=True)
    dataset_artifact = _write_core_dataset_artifact(
        dataset_dir=dataset_dir,
        task=task,
        sample_id=sample_id,
        attempts=attempts,
    )
    input_artifacts = [dataset_artifact]
    prior_artifacts = _configured_prior_artifacts(
        sample_dir=sample_dir,
        memory_path=memory_path,
        skill_path=skill_path,
        agent_system_path=agent_system_path,
    )
    produced: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for method in methods:
        job = WorkerClaimedJob(
            job_id=f"asta-{_safe_id(sample_id)}-{method}",
            lease_id=f"lease-{_safe_id(sample_id)}-{method}",
            job_type="astabench_core_evolution",
            method=method,
            input_artifacts=input_artifacts + prior_artifacts,
            config=_core_method_config(
                method=method,
                task=task,
                sample_id=sample_id,
                reflector_backend=reflector_backend,
                reflector_model=reflector_model,
                env=env,
            ),
        )
        try:
            artifacts = run_method(job, artifact_root=artifact_root)
        except Exception as exc:
            errors.append({"method": method, "error": str(exc)})
            continue
        for artifact in artifacts:
            produced.append(_artifact_request_payload(artifact))
            input_artifacts.append(_artifact_as_worker_input(artifact, artifact_id=f"{method}-{len(produced)}"))
    manifest = {
        "kind": "openevo_core_evolution_artifacts",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "sample_id": sample_id,
        "dataset_artifact": dataset_artifact.model_dump(mode="json"),
        "methods_requested": list(methods),
        "methods_succeeded": [item["manifest"].get("method") or item["type"] for item in produced],
        "supported_but_not_run_by_default": list(SUPPORTED_CORE_METHODS_NOT_RUN_BY_DEFAULT),
        "artifacts": produced,
        "errors": errors,
        "notes": (
            "These are OpenEvo core ArtifactRegisterRequest payloads produced by "
            "openevo.evolution.methods.run_method from the AstaBench trajectory dataset."
        ),
    }
    manifest_path = sample_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_core_index(core_artifacts_dir)
    return manifest_path


def _write_core_dataset_artifact(
    *,
    dataset_dir: Path,
    task: str,
    sample_id: str,
    attempts: list[AttemptRecord],
) -> WorkerClaimInputArtifact:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    records = _core_dataset_records(task=task, sample_id=sample_id, attempts=attempts)
    records_path = dataset_dir / "records.jsonl"
    records_text = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    records_path.write_text(records_text, encoding="utf-8")
    records_bytes = records_path.read_bytes()
    manifest = {
        "dataset_id": f"astabench-{_safe_id(task)}-{_safe_id(sample_id)}",
        "name": f"AstaBench {task} {sample_id} attempts",
        "purpose": "astabench_core_evolution_reflection",
        "records_path": "records.jsonl",
        "records_uri": records_path.resolve().as_uri(),
        "event_count": len(records),
        "records_byte_size": len(records_bytes),
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "source": "openevo_asta_bench.evolution_experiment",
        "task": task,
        "sample_id": sample_id,
    }
    manifest_path = dataset_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return WorkerClaimInputArtifact(
        artifact_id=f"dataset-{_safe_id(sample_id)}",
        type=ArtifactType.DATASET,
        uri=manifest_path.resolve().as_uri(),
        name=str(manifest["name"]),
    )


def _core_dataset_records(
    *,
    task: str,
    sample_id: str,
    attempts: list[AttemptRecord],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for attempt in attempts:
        if attempt.log_file:
            try:
                events = build_astabench_events(
                    attempt.log_file,
                    policy_version=f"evo-{attempt.evolution_count}",
                    rollout_step=attempt.evolution_count,
                )
            except (AstaBenchBridgeError, OSError, ValueError):
                events = []
            for event in events:
                record = event.model_dump(mode="json")
                record["payload"]["attempt"] = attempt.__dict__
                records.append(record)
        if not any(record.get("session_id", "").endswith(f":{sample_id}") for record in records):
            records.append(_fallback_core_record(task=task, sample_id=sample_id, attempt=attempt))
    return records


def _fallback_core_record(*, task: str, sample_id: str, attempt: AttemptRecord) -> dict[str, Any]:
    status = "COMPLETED" if attempt.status == "success" else "ERROR"
    prompt = f"Run AstaBench task {task} sample {sample_id}."
    response = attempt.completion or attempt.error or "No completion captured."
    return {
        "event_id": f"fallback-{_safe_id(sample_id)}-{attempt.evolution_count}",
        "source": "asta_bench.inspect",
        "event_type": "openevo.session_completed",
        "source_event_id": f"fallback:{task}:{sample_id}:{attempt.evolution_count}",
        "task_id": f"{task}:{sample_id}",
        "session_id": f"{task}:{sample_id}:evo-{attempt.evolution_count}",
        "policy_version": f"evo-{attempt.evolution_count}",
        "rollout_step": attempt.evolution_count,
        "agent": {"harness": "inspect", "benchmark": "astabench"},
        "base_model": DEFAULT_MODEL,
        "reward": attempt.score,
        "status": status,
        "payload": {
            "session_result": {
                "session_id": f"{task}:{sample_id}:evo-{attempt.evolution_count}",
                "task_id": f"{task}:{sample_id}",
                "status": status,
                "trajectory": {
                    "status": status,
                    "metadata": {"attempt": attempt.__dict__},
                    "traces": [
                        {
                            "prompt_messages": [{"role": "user", "content": prompt}],
                            "response_messages": [{"role": "assistant", "content": response}],
                            "reward": attempt.score,
                            "metadata": {"attempt": attempt.__dict__},
                        }
                    ],
                },
                "metadata": {"attempt": attempt.__dict__},
            }
        },
    }


def _configured_prior_artifacts(
    *,
    sample_dir: Path,
    memory_path: Path | None,
    skill_path: Path | None,
    agent_system_path: Path | None,
) -> list[WorkerClaimInputArtifact]:
    artifacts: list[WorkerClaimInputArtifact] = []
    if memory_path and memory_path.exists():
        artifacts.append(
            WorkerClaimInputArtifact(
                artifact_id="adapter-text-memory",
                type=ArtifactType.TEXT_MEMORY,
                uri=memory_path.resolve().as_uri(),
                name="Adapter text memory",
            )
        )
    if skill_path and skill_path.exists():
        artifacts.append(
            WorkerClaimInputArtifact(
                artifact_id="adapter-skill-bundle",
                type=ArtifactType.SKILL_BUNDLE,
                uri=skill_path.parent.resolve().as_uri(),
                name="Adapter skill bundle",
            )
        )
    if agent_system_path and agent_system_path.exists():
        target_path = sample_dir / "adapter-agent-system" / "AGENTS.md"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(agent_system_path, target_path)
        artifacts.append(
            WorkerClaimInputArtifact(
                artifact_id="adapter-agent-system",
                type=ArtifactType.AGENT_SYSTEM,
                uri=target_path.resolve().as_uri(),
                name="Adapter agent system",
            )
        )
    return artifacts


def _core_method_config(
    *,
    method: str,
    task: str,
    sample_id: str,
    reflector_backend: str,
    reflector_model: str,
    env: dict[str, str],
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": f"{task} {sample_id} {method}",
        "tags": ["astabench", _safe_id(task)],
        "compatibility": {"task_tags": ["astabench", _safe_id(task)], "agent_harness": ["inspect"]},
        "promoted": True,
        "reflector_model": reflector_model,
        "reflector_provider": "codex_cli" if reflector_backend == "codex_cli" else "openai_chat",
        "reflector_base_url": env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE") or "",
        "reflector_api_key": env.get("OPENAI_API_KEY") or "",
        "target_path": "AGENTS.md",
        "agent_system_audit": {"enabled": False},
    }
    if method == "skill_bundle":
        config["skill_markdown"] = (
            f"# {task} Skill\n\n"
            f"Use OpenEvo trajectory evidence from sample `{sample_id}` to preserve "
            "the task schema, verify data provenance, and avoid hidden-answer leakage.\n"
        )
    if method == "agent_system":
        config["agent_system_markdown"] = (
            f"# {task} Agent System\n\n"
            "- Use only task-visible data, tools, and retrieved evidence.\n"
            "- Preserve the exact DiscoveryBench hypothesis/workflow JSON schema.\n"
            "- Treat validation trajectories as method guidance, never as target answers.\n"
        )
    return config


def _artifact_request_payload(artifact: ArtifactRegisterRequest) -> dict[str, Any]:
    payload = artifact.model_dump(mode="json")
    path = _artifact_content_path(artifact)
    if path is not None:
        payload["local_path"] = str(path)
    return payload


def _artifact_as_worker_input(artifact: ArtifactRegisterRequest, *, artifact_id: str) -> WorkerClaimInputArtifact:
    return WorkerClaimInputArtifact(
        artifact_id=artifact_id,
        type=artifact.type,
        uri=artifact.uri,
        name=artifact.name,
    )


def _artifact_content_path(artifact: ArtifactRegisterRequest) -> Path | None:
    if not artifact.uri.startswith("file://"):
        return None
    return Path(artifact.uri.removeprefix("file://"))


def _write_core_index(core_artifacts_dir: Path) -> None:
    manifests = sorted(
        path for path in core_artifacts_dir.glob("sample-*/manifest.json") if path.is_file()
    )
    index = {
        "kind": "openevo_core_evolution_artifacts_index",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_manifests": [str(path) for path in manifests],
        "sample_count": len(manifests),
    }
    _write_json(core_artifacts_dir / "index.json", index)


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return normalized[:96] or "item"


def _write_task_skill_bundle(
    *,
    output_path: Path,
    task: str,
    sample_id: str,
    attempts: list[AttemptRecord],
    memory: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_attempts = [attempt for attempt in attempts if not attempt.correct]
    latest = attempts[-1] if attempts else None
    content = f"""# AstaBench Retry Skill

Use this skill when solving `{task}` samples with OpenEvo validation-derived
method guidance.

## Trigger

Apply these habits before the final answer for task `{task}`.

## Procedure

- Re-read the task prompt, answer schema, and available choices before using tools.
- Convert the question into narrow search queries with exact entities, mechanisms, and quoted phrases.
- Prefer evidence that directly resolves the requested relation over broad review snippets.
- Track why each candidate answer is supported or contradicted before submitting.
- Preserve the exact final-answer format requested by the benchmark.
- Do not copy validation labels, hidden references, scorer internals, or sample-specific answers.

## Latest Reflection

{memory or "- No memory reflection was available; follow the procedure above."}

## Provenance

- task: `{task}`
- sample_id: `{sample_id}`
- latest_status: `{latest.status if latest else "unknown"}`
- latest_score: `{latest.score if latest else 0.0:.3f}`
- failed_attempts: `{len(failed_attempts)}`
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _write_task_agent_system(
    *,
    output_path: Path,
    task: str,
    sample_id: str,
    attempts: list[AttemptRecord],
    memory: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latest = attempts[-1] if attempts else None
    content = f"""# AstaBench Agent-System Guidance

- Solve `{task}` using only task-visible prompts, tools, and retrieved evidence.
- Treat validation-derived guidance as method advice, not as an answer source.
- Before searching, identify the exact output schema and the minimum evidence needed.
- After tool use, compare candidate answers against the retrieved evidence and eliminate weak matches.
- For Paper Finder tasks, submit numeric Semantic Scholar corpus IDs only when the tool result exposes them.
- For LitQA2 multiple-choice tasks, answer with one of the shown option letters in the requested schema.
- Do not include hidden labels, scorer feedback, private references, or memorized validation answers.

## Current Reflection

{memory or "- No memory reflection was available; use the rules above."}

## Provenance

- task: `{task}`
- sample_id: `{sample_id}`
- latest_status: `{latest.status if latest else "unknown"}`
- latest_score: `{latest.score if latest else 0.0:.3f}`
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _evolve_memory_openai_api(prompt: str, *, env: dict[str, str], model: str) -> str:
    try:
        from openai import OpenAI
    except Exception:
        return ""
    api_key = env.get("OPENAI_API_KEY")
    if not api_key:
        return ""
    base_url = env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE")
    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    try:
        response = client.chat.completions.create(
            model=model.split("/", 1)[1] if model.startswith("openai/") else model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are OpenEvo's background reflector. Produce concise "
                        "method memory only; do not solve the hidden benchmark item."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
    except Exception:
        return ""
    content = response.choices[0].message.content if response.choices else ""
    return content or ""


def _write_summary(
    output_dir: Path,
    attempts: list[AttemptRecord],
    final_records: list[dict[str, Any]],
    *,
    task: str,
    mode: str,
    global_policy_path: Path | None,
    task_local_memory_dir: Path,
    max_evolutions: int,
    task_local_skill_dir: Path | None = None,
    task_local_agent_system_dir: Path | None = None,
    core_artifacts_dir: Path | None = None,
    core_methods: tuple[str, ...] = (),
    policy_update_path: Path | None = None,
) -> None:
    final_count = len(final_records)
    accuracy = (
        sum(1 for record in final_records if record["final_correct"]) / final_count
        if final_count
        else 0.0
    )
    first_accuracy = (
        sum(1 for record in final_records if record["first_correct"]) / final_count
        if final_count
        else 0.0
    )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "mode": mode,
        "global_policy": _path_metadata(global_policy_path),
        "task_local_memory_dir": str(task_local_memory_dir),
        "task_local_skill_dir": str(task_local_skill_dir) if task_local_skill_dir else "",
        "task_local_agent_system_dir": (
            str(task_local_agent_system_dir) if task_local_agent_system_dir else ""
        ),
        "core_artifacts_dir": str(core_artifacts_dir) if core_artifacts_dir else "",
        "core_methods": list(core_methods),
        "core_supported_but_not_run_by_default": list(SUPPORTED_CORE_METHODS_NOT_RUN_BY_DEFAULT),
        "max_evolutions": max_evolutions,
        "mean_initial_score": (
            sum(float(record.get("first_score") or 0.0) for record in final_records) / final_count
            if final_count
            else 0.0
        ),
        "mean_final_score": (
            sum(float(record.get("final_score") or 0.0) for record in final_records) / final_count
            if final_count
            else 0.0
        ),
        "mean_best_score": (
            sum(float(record.get("best_score", record.get("final_score") or 0.0)) for record in final_records) / final_count
            if final_count
            else 0.0
        ),
        "test_feedback_to_evolution": False if mode == "frozen_eval" else None,
        "policy_update_path": str(policy_update_path) if policy_update_path else "",
        "samples_completed": final_count,
        "accuracy_initial": first_accuracy,
        "accuracy_final": accuracy,
        "mean_evolutions": (
            sum(record["evolution_count"] for record in final_records) / final_count
            if final_count
            else 0.0
        ),
        "final_records": final_records,
        "attempts": [attempt.__dict__ for attempt in attempts],
    }
    _write_json(output_dir / "summary.json", summary)


def _write_revision_context(
    *,
    context_dir: Path,
    sample_index: int,
    evolution_count: int,
    global_policy_path: Path | None,
    task_memory_path: Path | None,
    task_skill_path: Path | None = None,
    task_agent_system_path: Path | None = None,
    mode: str,
) -> Path | None:
    sections: list[str] = []
    global_policy = _read_optional_text(global_policy_path)
    task_memory = _read_optional_text(task_memory_path)
    task_skill = _read_optional_text(task_skill_path)
    task_agent_system = _read_optional_text(task_agent_system_path)
    if global_policy:
        sections.append("# Frozen Global Policy\n\n" + global_policy.strip())
    if task_memory:
        sections.append("# Task-Local Retry Memory\n\n" + task_memory.strip())
    if task_skill:
        sections.append("# Task-Local Skill Bundle\n\n" + task_skill.strip())
    if task_agent_system:
        sections.append("# Task-Local Agent System\n\n" + task_agent_system.strip())
    if not sections:
        return None
    context_dir.mkdir(parents=True, exist_ok=True)
    path = context_dir / f"sample-{sample_index:02d}-evo-{evolution_count}-{mode}.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path


def _write_task_policy_update(
    *,
    env: dict[str, str],
    reflector_model: str,
    task: str,
    attempts: list[AttemptRecord],
    final_records: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempts_by_sample: dict[str, list[AttemptRecord]] = {}
    for attempt in attempts:
        attempts_by_sample.setdefault(attempt.sample_id, []).append(attempt)
    best_trajectories = []
    for sample_id, sample_attempts in attempts_by_sample.items():
        best = max(sample_attempts, key=lambda item: (item.score, item.evolution_count))
        best_trajectories.append(
            {
                "sample_id": sample_id,
                "evolution_count": best.evolution_count,
                "score": best.score,
                "completion_excerpt": best.completion[:500],
            }
        )
    compact = {
        "task": task,
        "samples_completed": len(final_records),
        "initial_correct": sum(1 for item in final_records if item.get("first_correct")),
        "final_correct": sum(1 for item in final_records if item.get("final_correct")),
        "mean_evolutions": (
            sum(float(item.get("evolution_count") or 0) for item in final_records) / len(final_records)
            if final_records
            else 0.0
        ),
        "best_trajectories": best_trajectories,
        "regressions": [
            {
                "sample_id": item.get("sample_id"),
                "initial_score": item.get("first_score"),
                "final_score": item.get("final_score"),
                "best_score": item.get("best_score"),
            }
            for item in final_records
            if float(item.get("final_score") or 0.0) < float(item.get("best_score") or 0.0)
        ],
    }
    prompt = f"""
You are OpenEvo's background reflector after an AstaBench validation task.
Write a task-level policy update that can be reviewed and optionally promoted
to the frozen global policy for future tasks.

Constraints:
- Generalize tactics; do not preserve sample-specific answers.
- Do not include hidden target labels, hidden references, or scorer internals.
- Learn from every sample's best trajectory, including partially successful ones.
- Do not turn tool or infrastructure failures into a policy of refusing to analyze data.
- Preserve useful analysis tactics while avoiding patterns associated with regressions.
- Keep the update under 500 words.
- Use Markdown bullets.

Validation aggregate:
{json.dumps(compact, ensure_ascii=False, indent=2)}
"""
    update = _evolve_memory_openai_api(prompt, env=env, model=reflector_model).strip()
    if not update:
        update = (
            "# Task Policy Update\n\n"
            "- Prefer concise, evidence-grounded tool use before final answers.\n"
            "- Preserve the exact output schema required by the task.\n"
            "- When retrieval is weak, reformulate queries around entities, mechanisms, and quoted phrases.\n"
        )
    output_path.write_text(update + "\n", encoding="utf-8")
    return output_path


def _write_freeze_manifest(
    *,
    path: Path,
    task: str,
    mode: str,
    model: str,
    reflector_backend: str,
    reflector_model: str,
    global_policy_path: Path | None,
    task_local_memory_dir: Path,
    max_steps: int,
    max_evolutions: int,
    sample_count: int,
    output_dir: Path,
    task_local_skill_dir: Path | None = None,
    task_local_agent_system_dir: Path | None = None,
) -> Path:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": "astabench",
        "task": task,
        "mode": mode,
        "agent_model": model,
        "reflector_backend": reflector_backend,
        "reflector_model": reflector_model,
        "max_steps": max_steps,
        "max_evolutions": max_evolutions,
        "sample_count": sample_count,
        "global_policy": _path_metadata(global_policy_path),
        "task_local_memory_dir": str(task_local_memory_dir),
        "task_local_skill_dir": str(task_local_skill_dir) if task_local_skill_dir else "",
        "task_local_agent_system_dir": (
            str(task_local_agent_system_dir) if task_local_agent_system_dir else ""
        ),
        "output_dir": str(output_dir),
        "test_feedback_to_evolution": False,
        "leaderboard_upload_performed": False,
        "notes": "Use frozen_eval for test/leaderboard runs; do not evolve from test scorer feedback.",
    }
    _write_json(path, manifest)
    return path


def _path_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _read_optional_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _reset_task_state(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for name in ["attempts.json", "summary.json", "charts.svg"]:
        path = output_dir / name
        if path.exists():
            path.unlink()
    for name in ["logs", "task_local_memory", "task_local_skills", "task_local_agent_system", "revision_context"]:
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)


def _write_svg_charts(
    output_dir: Path,
    attempts: list[AttemptRecord],
    final_records: list[dict[str, Any]],
) -> None:
    width = 1100
    height = 720
    margin = 70
    rows = []
    # Chart 1: cumulative best accuracy after each sample.
    cumulative = []
    correct = 0
    for index, record in enumerate(final_records, start=1):
        correct += 1 if record["final_correct"] else 0
        cumulative.append(correct / index)
    # Chart 2: per-sample evolution count and final score.
    bars = [record["evolution_count"] for record in final_records]
    def polyline(values: list[float], x0: int, y0: int, w: int, h: int) -> str:
        if not values:
            return ""
        points = []
        for i, value in enumerate(values):
            x = x0 + (w * i / max(1, len(values) - 1))
            y = y0 + h - h * max(0.0, min(1.0, value))
            points.append(f"{x:.1f},{y:.1f}")
        return " ".join(points)

    rows.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    rows.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    rows.append('<text x="70" y="45" font-size="24" font-family="Arial" font-weight="700">OpenEvo AstaBench LitQA2 Validation Evolution Smoke</text>')
    rows.append('<text x="70" y="72" font-size="13" font-family="Arial" fill="#555">Task: LitQA2-FullText validation, single-sample retry with max 5 evolutions, no leaderboard upload</text>')
    # Line chart.
    x0, y0, w, h = margin, 120, 450, 230
    rows.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="white" stroke="#ccc"/>')
    rows.append(f'<text x="{x0}" y="{y0-18}" font-size="16" font-family="Arial" font-weight="700">Cumulative Final Accuracy</text>')
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y0 + h - h * t
        rows.append(f'<line x1="{x0}" x2="{x0+w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eee"/>')
        rows.append(f'<text x="{x0-38}" y="{y+4:.1f}" font-size="11" font-family="Arial">{t:.2f}</text>')
    rows.append(f'<polyline points="{polyline(cumulative, x0, y0, w, h)}" fill="none" stroke="#2166ac" stroke-width="3"/>')
    # Bars.
    x1, y1, w1, h1 = 610, 120, 420, 230
    rows.append(f'<rect x="{x1}" y="{y1}" width="{w1}" height="{h1}" fill="white" stroke="#ccc"/>')
    rows.append(f'<text x="{x1}" y="{y1-18}" font-size="16" font-family="Arial" font-weight="700">Evolutions Per Sample</text>')
    max_bar = max([5, *bars]) if bars else 5
    bar_w = w1 / max(1, len(bars))
    for i, value in enumerate(bars):
        bh = h1 * value / max_bar
        x = x1 + i * bar_w + 4
        y = y1 + h1 - bh
        rows.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(2, bar_w-8):.1f}" height="{bh:.1f}" fill="#b2182b"/>')
        rows.append(f'<text x="{x+bar_w/2-8:.1f}" y="{y1+h1+16}" font-size="10" font-family="Arial">{i+1}</text>')
    # Table.
    table_y = 420
    rows.append(f'<text x="70" y="{table_y-25}" font-size="16" font-family="Arial" font-weight="700">Per-Sample Summary</text>')
    headers = ["#", "Initial", "Final", "Evolutions", "Attempts"]
    xs = [70, 140, 230, 320, 430]
    for x, header in zip(xs, headers):
        rows.append(f'<text x="{x}" y="{table_y}" font-size="12" font-family="Arial" font-weight="700">{header}</text>')
    for i, record in enumerate(final_records, start=1):
        y = table_y + i * 24
        vals = [
            str(i),
            "1" if record["first_correct"] else "0",
            "1" if record["final_correct"] else "0",
            str(record["evolution_count"]),
            str(record["attempts"]),
        ]
        for x, val in zip(xs, vals):
            rows.append(f'<text x="{x}" y="{y}" font-size="12" font-family="Arial">{val}</text>')
    rows.append("</svg>")
    (output_dir / "charts.svg").write_text("\n".join(rows), encoding="utf-8")


def _read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key] = value
    return env


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _load_attempts(path: Path) -> list[AttemptRecord]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [AttemptRecord(**item) for item in data]


def _load_final_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("final_records")
    return list(records) if isinstance(records, list) else []


def _timeout_output_text(exc: subprocess.TimeoutExpired) -> str:
    output = exc.output or exc.stdout or ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return str(output) + f"\nTimed out after {exc.timeout} seconds.\n"


def _sample_specs(
    task: str,
    *,
    sample_count: int = 0,
    sample_start_index: int = 1,
    sample_end_index: int = 0,
) -> list[tuple[int, str]]:
    if sample_start_index < 1:
        raise ValueError("--sample-start-index must be >= 1")
    samples = _default_sample_ids(task, sample_count=sample_count)
    end = sample_end_index if sample_end_index > 0 else len(samples)
    end = min(end, len(samples))
    if end < sample_start_index:
        return []
    return [(index, samples[index - 1]) for index in range(sample_start_index, end + 1)]


def _default_sample_ids(task: str, *, sample_count: int = 0) -> list[str]:
    known_count = KNOWN_TASK_SAMPLE_COUNTS.get(task)
    if known_count is not None:
        if sample_count > known_count:
            raise ValueError(
                f"--sample-count {sample_count} exceeds the {known_count} samples in {task}"
            )
        count = sample_count or known_count
        return [f"sample-{index:03d}" for index in range(1, count + 1)]
    if sample_count > 0:
        return [f"sample-{index:03d}" for index in range(1, sample_count + 1)]
    if "paper_finder_litqa2" in task:
        return PAPER_FINDER_LITQA2_VALIDATION_IDS
    return DEFAULT_SAMPLE_IDS


def _latest_json(path: Path) -> Path | None:
    files = sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 14] + "\n[truncated]\n"


def _truncate_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "\n[truncated]\n" + text[-(limit - 14) :]


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_message_content_text(item) for item in content)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content") or ""
        return str(text)
    return str(content)


def _is_fatal_api_error(error: str) -> bool:
    normalized = (error or "").lower()
    fatal_markers = [
        "authenticationerror",
        "error code: 401",
        "no api key",
        "quota",
        "remainquota",
        "额度已用尽",
    ]
    return any(marker in normalized for marker in fatal_markers)


def _is_transient_infra_error(error: str) -> bool:
    normalized = (error or "").lower()
    transient_markers = [
        "connecterror",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
        "remoteprotocolerror",
    ]
    return any(marker in normalized for marker in transient_markers)


def _adaptive_stop_reason(
    attempts: list[AttemptRecord],
    *,
    target_score: float,
    patience: int,
) -> str:
    if not attempts:
        return ""
    best_score = max(attempt.score for attempt in attempts)
    if attempts[-1].status == "success" and best_score >= target_score:
        return f"target score reached: best={best_score:.3f} >= {target_score:.3f}"
    if patience <= 0 or len(attempts) <= patience:
        return ""
    previous_best = max(attempt.score for attempt in attempts[: -patience])
    recent_best = max(attempt.score for attempt in attempts[-patience:])
    if recent_best <= previous_best:
        return f"no best-score improvement in last {patience} attempts"
    return ""


def _default_output_dir(asta_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return asta_root / "logs" / f"openevo-litqa2-evolution-{stamp}"


def _default_solver_spec() -> str:
    return f"{Path(__file__).with_name('solver.py')}@astabench_openevo_solver"


if __name__ == "__main__":
    raise SystemExit(main())
