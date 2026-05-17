# Evaluation suite

## Run

```bash
uv run python -m eval.runner \
    --dataset eval/dataset/questions.jsonl \
    --adversarial eval/dataset/adversarial.jsonl \
    --output eval/results/run_$(date +%Y%m%dT%H%M%SZ).jsonl
```

Each Q gets a fresh session and Chroma collection — sessions don't leak between Qs.

## Metrics

| Metric | What it measures | Output |
|---|---|---|
| `task_completion` | Composite: judge_pass AND tools_match AND has_citation | binary per Q; rate aggregated |
| `faithfulness` | Every claim supported by retrieved chunks | 0 / 0.5 / 1 |
| `citation_precision` | Cited claims that actually appear in the cited chunk | 0..1 |
| `citation_recall` | Did the answer cite when it should have? (proxy: any cite present) | 0..1 |
| `tool_selection_match` | actual tools == expected tools (multi-hop allows ≥1 retrieve) | bool |
| `doc_filter_correct` | When a company is named, was doc_filter passed? | bool |
| `iterative_retrieval_used` | FM-1 instrumentation: ≥2 retrieve_from_docs calls? | bool |
| `recall_at_k` | Was an expected page in top-k retrieved? (k = 1, 3, 5) | 0..1 |
| `refusal` | (adversarial/should_refuse) Did the agent refuse? | bool |
| `clarification` | (adversarial/should_clarify) Did the agent ask to clarify? | bool |

## SLO targets (the bar v1 commits to clear)

| Metric | Target |
|---|---|
| task_completion_rate (overall) | ≥ 0.70 |
| task_completion_rate (multi_hop, numerical) | ≥ 0.50 |
| faithfulness | ≥ 0.80 |
| citation_recall | ≥ 0.85 |
| retrieval_recall@5 | ≥ 0.80 |
| refusal_rate (should_refuse) | ≥ 0.90 |
| clarification_rate (should_clarify) | ≥ 0.80 |

## Baseline

`eval/results/v0_baseline.jsonl` is the frozen pre-tuning baseline. Do not regenerate after Task 41 (FM-1 fix). All subsequent runs compare against it.
