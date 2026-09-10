from __future__ import annotations

import argparse
import json
import shlex
import sys

from openevo_asta_bench.constants import DEFAULT_AGENT_MODEL
from openevo_asta_bench.evolution_runner import (
    build_astabench_eval_command,
    build_freeze_manifest,
    run_astabench_eval,
    write_freeze_manifest,
)
from openevo_asta_bench.metrics import compute_official_macro


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openevo-asta-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    events = sub.add_parser("events", help="Convert JSON Inspect .eval logs to OpenEvo events.")
    events.add_argument("--input", required=True)
    events.add_argument("--output")
    events.add_argument("--policy-version")
    events.add_argument("--rollout-step", type=int)

    dataset = sub.add_parser("dataset", help="Ingest AstaBench events into an OpenEvo dataset.")
    dataset.add_argument("--input", required=True)
    dataset.add_argument("--db", default=".openevo/evolution/evolution.db")
    dataset.add_argument("--artifact-root", default=".openevo/evolution")
    dataset.add_argument("--name", required=True)
    dataset.add_argument("--purpose", default="astabench_validation_reflection")
    dataset.add_argument("--policy-version")
    dataset.add_argument("--rollout-step", type=int)
    dataset.add_argument("--status", action="append", default=["COMPLETED", "ERROR"])

    score = sub.add_parser("macro", help="Compute local official-style macro score from records JSON.")
    score.add_argument("--input", required=True)

    command = sub.add_parser("eval-command", help="Print a non-uploading AstaBench eval command.")
    command.add_argument("--log-dir", required=True)
    command.add_argument("--split", choices=["validation", "test"], default="validation")
    command.add_argument("--task", help="Run one Inspect task path, e.g. astabench/sqa_dev.")
    command.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    command.add_argument("--limit", type=int)
    command.add_argument("--max-samples", type=int, default=1)
    command.add_argument("--revision-manifest")
    command.add_argument("--no-score", action="store_true")

    run = sub.add_parser("run", help="Run a local non-uploading AstaBench eval.")
    run.add_argument("--log-dir", required=True)
    run.add_argument("--split", choices=["validation", "test"], default="validation")
    run.add_argument("--task", help="Run one Inspect task path, e.g. astabench/sqa_dev.")
    run.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    run.add_argument("--limit", type=int)
    run.add_argument("--max-samples", type=int, default=1)
    run.add_argument("--revision-manifest")
    run.add_argument("--no-score", action="store_true")

    freeze = sub.add_parser("freeze", help="Write a frozen OpenEvo/AstaBench run manifest.")
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--revision", required=True)
    freeze.add_argument("--split", choices=["validation", "test"], default="test")
    freeze.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    freeze.add_argument("--reflector-model", default="gpt-5.5")
    freeze.add_argument("--reflector-provider", default="codex_cli")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "events":
        from openevo_asta_bench.inspect_bridge import build_astabench_events

        events = build_astabench_events(
            args.input,
            policy_version=args.policy_version,
            rollout_step=args.rollout_step,
        )
        _write_jsonl(args.output, [event.model_dump(mode="json") for event in events])
        return 0
    if args.command == "dataset":
        from openevo.evolution.models import DatasetCreateRequest
        from openevo.evolution.store import EvolutionStore
        from openevo_asta_bench.inspect_bridge import build_astabench_events

        events = build_astabench_events(
            args.input,
            policy_version=args.policy_version,
            rollout_step=args.rollout_step,
        )
        store = EvolutionStore(db_path=args.db, artifact_root=args.artifact_root)
        store.initialize()
        for event in events:
            store.ingest_event(event)
        created = store.create_dataset(
            DatasetCreateRequest(
                name=args.name,
                purpose=args.purpose,
                query={
                    "source": "asta_bench.inspect",
                    "event_types": ["openevo.session_completed"],
                    "status": args.status,
                },
            )
        )
        print(created.model_dump_json(indent=2))
        return 0
    if args.command == "macro":
        with open(args.input, encoding="utf-8") as handle:
            records = json.load(handle)
        print(json.dumps(compute_official_macro(records), indent=2, sort_keys=True))
        return 0
    if args.command == "eval-command":
        command = build_astabench_eval_command(
            log_dir=args.log_dir,
            split=args.split,
            task=args.task,
            model=args.model,
            limit=args.limit,
            max_samples=args.max_samples,
            revision_manifest=args.revision_manifest,
            no_score=args.no_score,
        )
        print(shlex.join(command))
        return 0
    if args.command == "run":
        return run_astabench_eval(
            log_dir=args.log_dir,
            split=args.split,
            task=args.task,
            model=args.model,
            limit=args.limit,
            max_samples=args.max_samples,
            revision_manifest=args.revision_manifest,
            no_score=args.no_score,
        )
    if args.command == "freeze":
        manifest = build_freeze_manifest(
            revision=args.revision,
            split=args.split,
            agent_model=args.model,
            reflector_provider=args.reflector_provider,
            reflector_model=args.reflector_model,
        )
        output = write_freeze_manifest(args.output, manifest)
        print(str(output))
        return 0
    raise AssertionError(args.command)


def _write_jsonl(path: str | None, rows: list[dict[str, object]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
