# AstaBench Validation Log Analysis

Analysis date: 2026-08-10.

Source logs are under `/home/mark/asta_openevo/third_party/asta-bench/logs`.

## Task Number Ambiguity

AstaBench `v1.0.0` uses different task orders for validation and test.

Validation task 3/4 are:

- task3: `astabench/litqa2_validation`
- task4: `astabench/paper_finder_validation`

Test task 3/4 are:

- task3: `astabench/sqa_test`
- task4: `astabench/arxivdigestables_test`

The existing logs I found cover `litqa2_validation`, `paper_finder_litqa2_validation`,
`paper_finder_validation` failed probes, and `sqa_dev`. I did not find an
ArxivDIGESTables run log, so conclusions about test task4 cannot be made from
local evidence.

## Why Task 3/4 Runs Were Long

For the validation-style logs, the long-running tasks are the literature-search
tasks with heavy tool and scorer usage.

Observed per-attempt timings from Inspect JSON `samples[0].total_time`:

- `astabench/litqa2_validation`: 63 parsed attempts, mean 99-125 seconds by run,
  max 237.5 seconds.
- `astabench/paper_finder_litqa2_validation`: 49 parsed attempts, mean 103-128
  seconds by run, max 237.9 seconds.
- `astabench/sqa_dev`: 53 parsed attempts, mean 208-270 seconds by run, max
  521.6 seconds.

Concrete evidence:

- Slow SQA example:
  `/home/mark/asta_openevo/third_party/asta-bench/logs/openevo-scholarqa-cs2-validation-adaptive-2shard-a/logs/sample-17-evo-2/2026-08-03T20-54-39+08-00_sqa-dev_7EFhSDoKp6Pv2MxPN5GCxS.json`
  took 521.627 seconds and used 92,603 total tokens. Its score metadata contains
  LLM scorer prompt logs, and `aggregate_scorer` combines ingredient recall,
  answer precision, citation recall, and citation precision.
- Slow PaperFinder-LitQA2 example:
  `/home/mark/asta_openevo/third_party/asta-bench/logs/openevo-paperfinder-litqa2-validation-structured/logs/sample-08-evo-3/2026-07-31T17-13-16+08-00_paper-finder-litqa2-validation_g3sKUJi62Mz5TxHxt25yRC.json`
  took 237.921 seconds and used 43,074 total tokens.
- Slow LitQA2 example:
  `/home/mark/asta_openevo/third_party/asta-bench/logs/openevo-litqa2-validation-full-rerun-20260803_153714/logs/sample-04-evo-3/2026-08-03T15-49-02+08-00_litqa2-validation_iFBtwJx2Mn7gC33FWySsU4.json`
  took 237.542 seconds and used 38,986 total tokens.

The specific cause is not local CPU. The logs show high wall time paired with
large model usage, many Asta literature tool messages, and for SQA additional
LLM-based scoring. SQA is especially slow because it requires generating a
cited JSON report and then running multiple grader components over the answer
and citations. PaperFinder is also slow because it repeatedly searches and
inspects papers, often consuming 40k-140k tokens per attempt.

Several attempts hit exact time-limit ceilings such as 210, 237-240, or timeout
errors in invalid probe directories. Those are task/eval limits being exhausted,
not normal short completions.

## Sample7 Regression

The sample7 case is in:

`/home/mark/asta_openevo/third_party/asta-bench/logs/openevo-scholarqa-cs2-validation-adaptive-2shard-a/attempts.json`

It is `sample-007`, an SQA question:

`During pre-training, why is the transformer embedding layer initialized randomly, rather than with pre-trained embeddings from existing models?`

Score sequence:

- evo0: 0.7847222222
- evo1: 0.8013888889
- evo2: 0.5763888889
- evo3: 0.5833333333

The reason is content drift, not an execution failure.

Rubric-critical ingredients include:

- pre-trained embeddings have incompatible parameter distributions;
- high-variance embeddings can interfere with positional encodings;
- random initialization improves gradient flow and training stability;
- empirical evidence that random initialization can outperform imported
  embeddings;
- standardization of pre-trained embeddings as a mitigation.

evo1 cites and centers the key paper `On Initializing Transformers with
Pre-trained Embeddings`, and scores:

- ingredient recall: 0.8055555556
- answer precision: 1.0
- citation recall: 0.4
- citation precision: 1.0

evo2/evo3 drift to generic BERT/Transformer/static-embedding explanations. They
remain plausible enough for answer precision to stay 1.0, but they lose the
rubric-specific evidence:

- evo2 ingredient recall: 0.3055555556, citation recall: 0.3333333333
- evo3 ingredient recall: 0.3333333333, citation recall: 0.1666666667

So the concrete cause is that later evolution attempts replaced a rubric-aligned
answer with a more generic architecture/contextual-embedding answer.

## Is It Accidental?

I did not re-run the external model/scorer, because that would spend live model
and Asta tool quota. I did reproduce the phenomenon from local logs by parsing
the raw attempts and Inspect score records.

It is not unique to sample7. Similar improve-then-regress SQA sequences exist:

- adaptive sample-004: 0.7543 -> 0.8381 -> 0.7677 -> 0.7876
- full shard sample-001: 0.7981 -> 0.8661 -> 0.8006 -> 0.7232 ...
- full shard sample-051: 0.7865 -> 0.8125 -> 0.7240 -> 0.8043 ...

The pattern is therefore plausible and recurring for SQA. It is not a law that
more evolution always hurts: other samples improve later. The safer conclusion
is that OpenEvo should select the best scored attempt per sample, not assume the
latest evolved attempt is best.
