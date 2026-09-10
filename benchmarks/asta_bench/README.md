# OpenEvo AstaBench Adapter

This package keeps AstaBench/Inspect in control of samples, tools, sandboxes,
scoring, and `.eval` logs while exposing OpenEvo evolution assets to a custom
Inspect solver.

The adapter intentionally does not upload leaderboard results. It only creates
local eval commands, local freeze manifests, and OpenEvo event records derived
from Inspect logs.

## Smoke Test

```bash
python3 -m pytest benchmarks/asta_bench/tests
```

The smoke tests use mocks and do not call AstaBench, Asta tools, or a model.

For a live infrastructure/API smoke test, run the evolution experiment in
frozen mode with scoring disabled:

```bash
python -m openevo_asta_bench.evolution_experiment \
  --mode frozen_eval \
  --no-score \
  --skip-core-artifacts \
  --sample-count 1 \
  ...
```

`--no-score` prevents a failed or empty solver response from triggering paid
DiscoveryBench scorer calls. It also defaults Inspect retries to zero; use
`--retry-on-error N` to opt into retries explicitly. Scored runs keep the
historical default of two retries. DiscoveryBench itself also returns a local
zero score without calling its LLM evaluator when the model output contains an
error or an empty completion.

Relays that return Chat Completions responses but require flat function-tool
declarations can be selected explicitly:

```bash
python -m openevo_asta_bench.evolution_experiment \
  --model openai/cc-gpt-5.5 \
  --openai-tool-schema flat \
  ...
```

The default, `--openai-tool-schema standard`, retains the official OpenAI Chat
Completions request shape. Flat mode changes only outbound `tools` and named
`tool_choice` declarations; messages and standard Chat response parsing remain
unchanged. It forces Chat Completions and does not attempt to bypass a Responses
API rejection.

## Official Validation Run Shape

After installing AstaBench `v0.3.1`, accepting the Hugging Face dataset license,
and exporting `ASTA_TOOL_KEY`, use the generated command as the run skeleton:

```bash
openevo-asta-bench eval-command \
  --split validation \
  --limit 1 \
  --log-dir logs/openevo-astabench-smoke \
  --model openai/gpt-5.5
```

For full test runs, freeze first and run test only once:

```bash
openevo-asta-bench freeze \
  --output .openevo/asta_bench/freeze-test.json \
  --model openai/gpt-5.5 \
  --revision REVISION_ID
```

Do not use test `.eval` logs as reflector input.
