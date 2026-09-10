# OpenEvo AstaBench CLI Runbook

Run commands from:

```bash
cd /home/mark/asta_openevo/OpenEvo/benchmarks/asta_bench
```

Required environment:

```bash
export ASTA_TOOL_KEY=...
```

## Print A Command

```bash
uv run openevo-asta-bench eval-command \
  --split validation \
  --task astabench/sqa_dev \
  --limit 1 \
  --max-samples 1 \
  --log-dir /home/mark/asta_openevo/third_party/asta-bench/logs/openevo-sqa-dev-smoke
```

## Run One Validation Task

```bash
uv run openevo-asta-bench run \
  --split validation \
  --task astabench/sqa_dev \
  --limit 1 \
  --max-samples 1 \
  --log-dir /home/mark/asta_openevo/third_party/asta-bench/logs/openevo-sqa-dev-smoke
```

The adapter uses `inspect eval` for `--task` runs, then writes the AstaBench
`eval_config.json` with `astabench eval --config-only`.

## Score A Finished Log Directory

```bash
cd /home/mark/asta_openevo/third_party/asta-bench
uv run astabench score logs/openevo-sqa-dev-smoke
```

## Full Validation Split

```bash
uv run openevo-asta-bench run \
  --split validation \
  --max-samples 1 \
  --log-dir /home/mark/asta_openevo/third_party/asta-bench/logs/openevo-validation-full
```

## Full Test Split

Freeze the exact OpenEvo revision first:

```bash
uv run openevo-asta-bench freeze \
  --output /home/mark/asta_openevo/third_party/asta-bench/logs/openevo-test-freeze.json \
  --split test \
  --revision REVISION_ID
```

Then run:

```bash
uv run openevo-asta-bench run \
  --split test \
  --max-samples 1 \
  --revision-manifest /home/mark/asta_openevo/third_party/asta-bench/logs/openevo-test-freeze.json \
  --log-dir /home/mark/asta_openevo/third_party/asta-bench/logs/openevo-test-full
```

## Task Paths

Validation order in `astabench/config/v1.0.0.yml`:

1. `astabench/arxivdigestables_validation`
2. `astabench/sqa_dev`
3. `astabench/litqa2_validation`
4. `astabench/paper_finder_validation`
5. `astabench/paper_finder_litqa2_validation`

Test order in `astabench/config/v1.0.0.yml`:

1. `astabench/paper_finder_test`
2. `astabench/paper_finder_litqa2_test`
3. `astabench/sqa_test`
4. `astabench/arxivdigestables_test`
5. `astabench/litqa2_test`

Use `--task` with any of these paths to run a single task.
