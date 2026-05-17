# Meraki Multi-Turn Agent : Minimal Agent, Maximum Reliability

Conversational agent for question-answering over Indian listed-company FY25
annual reports (fiscal year ended March 31, 2025).

The system has 4 typed tools (corpus retrieval, calculator, document listing,
web search via Tavily), session memory, runtime-enforced reliability gates
(`UsageLimits(request_limit=5)` on the agent loop, citation requirement
enforced by the `Answer` Pydantic validator), and a quantifiable eval suite
with published SLO targets.

## Quick start (5 minutes from a clean clone)

```bash
# 1. Install dependencies
uv sync

# 2. Configure secrets
cp .env.example .env
# edit .env with your GEMINI_API_KEY and TAVILY_API_KEY

# 3. Save any extra corpus PDFs into data/pdfs/
# There are 3 docs existing in folder

# 4. Run backend
uv run uvicorn backend.main:app --reload --port 8000

# 5. In a second terminal, run frontend
uv run streamlit run frontend/streamlit_app.py
# open http://localhost:8501
```

Or via Docker:
```bash
cp .env.example .env
# edit .env
docker compose up
# open http://localhost:8000/docs for the OpenAPI playground
# open http://localhost:8501 for the Streamlit chat UI (run separately)
```

## Corpus

Three Indian listed-company FY25 annual reports (April 2024 – March 2025).
are saved into `data/pdfs/` with the EXACT filenames below — the eval dataset
references these names:

- **Eternal Ltd (formerly Zomato Ltd) FY25** — save as `zomato_fy25.pdf`
  — source: eternal.com investor-relations
- **FSN E-Commerce (Nykaa) FY25** — save as `nykaa_fy25.pdf`
  — source: nykaa.com investor-relations
- **PB Fintech (Policybazaar / Paisabazaar) FY25** — save as `pbfintech_fy25.pdf`
  — source: pbfintech.com investor-relations

Why this corpus: three different sectors enable cross-document comparative
eval; new-age tech businesses are domain-relevant; FY25 data is post-cutoff
for most current LLMs, so the retrieval layer genuinely earns its keep.
## Run the eval suite

```bash
uv run python -m eval.runner \
    --dataset eval/dataset/questions.jsonl \
    --adversarial eval/dataset/adversarial.jsonl \
    --output eval/results/run_$(date +%Y%m%dT%H%M%SZ).jsonl
```

Aggregate report and per-metric SLO red/green print at the end. The
v0 baseline (`eval/results/v0_baseline.jsonl`) and the post-FM1-fix run
(`eval/results/v1_after_fm1_fix.jsonl`) are committed once generated.


## Run tests

```bash
uv run pytest -xvs
```

Smoke test (`tests/test_agent_smoke.py`) self-skips without a real
`GEMINI_API_KEY`. The rest run fully offline.

## Project layout

```
backend/         # FastAPI app, agent, ingestion, retrieval, session
frontend/        # Streamlit UI (chat + tool-trace expander)
eval/            # dataset, metrics, runner, results
data/pdfs/       # corpus (.gitignore'd; download per "Corpus" above)
tests/           # pytest unit tests
```
