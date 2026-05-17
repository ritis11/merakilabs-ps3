# Eval dataset schema

## `questions.jsonl`

One JSON object per line:

| Field | Required | Description |
|---|---|---|
| `id` | yes | Stable identifier, e.g. `q_001` |
| `question` | yes | The user-facing prompt |
| `category` | yes | One of: `single_hop_factual`, `multi_hop`, `numerical`, `comparative`, `meta` |
| `expected_docs` | yes | List of doc_names the agent should use |
| `expected_pages` | yes (except `meta`) | List of 1-indexed pages where the answer is grounded |
| `expected_tools` | yes | Tools the agent should call (e.g. `["retrieve_from_docs", "calculate"]`) |
| `expected_doc_filter` | no | If the question names a company, the doc_names the agent should pass to `retrieve_from_docs` |
| `ground_truth` | yes | Reference answer for the LLM judge |
| `expected_answer_contains` | no | Optional substring sanity check (not part of task_completion score) |

## `adversarial.jsonl`

Same shape with one extra field:

| Field | Required | Description |
|---|---|---|
| `subset` | yes | `should_refuse` or `should_clarify` |

For `should_refuse`: `expected_tools` typically empty or `["retrieve_from_docs"]` only (the agent looks, finds nothing, refuses).
For `should_clarify`: `ground_truth` describes the clarifying question the agent should ask.
