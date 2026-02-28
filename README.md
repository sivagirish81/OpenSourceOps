# OpenSourceOps

OpenSourceOps is an intent-driven open-source intelligence platform for two primary workflows: **Company Adopter** and **Open-Source Contributor**. It discovers and evaluates GitHub repositories, computes deterministic maturity/risk signals, and generates persona-specific strategy outputs through a 4-agent orchestration pipeline.

The system supports iterative, feedback-driven refinement with auditability. Users can refine specific sections (recommendations, risk analysis, plans, opportunities) and regenerate either with full re-discovery or AI-only updates, while preserving a run history and justification trail.

For Adopter mode, the Adoption Playbook tab now includes a **Repo Deep-Dive Chat** bound to a selected repository and commit SHA. The chat is evidence-grounded over repo docs/code/config/releases and enforces a strict answer contract (direct answer, citations, confidence, and next checks when evidence is weak).

## Personas

### 1) Company / Adopter
- Input: domain + intent + constraints
- Output: ranked repositories, maturity/risk analysis, alternatives, and tailored 30/60/90 adoption playbook
- Optional action: export playbook to Notion via Composio (preview + confirmation)

### 2) Developer / Contributor
- Input: domain + intent + constraints
- Output: ranked repositories and **Top Contribution Opportunities** from GitHub issues + discussions (with fallback when discussions unavailable), plus weekly contribution plan

## Architecture

```text
[Streamlit UI]
  - Persona toggle, intent input, constraints, feedback refine controls
  - Adopter tabs: Repositories, Adoption Playbook, Run Log
  - Contributor tabs: Repositories, Opportunities, Contribution Plan, Run Log

[CoordinatorAgent]
  - Decides full re-search vs AI-only regeneration
  - Maintains changes & justification
  - Handles Notion export action

[ScoutAgent]
  - Query optimizer via Snowflake Cortex AI_COMPLETE
  - GitHub repo discovery (rate-limit capped)
  - Contributor mode: issues + discussions retrieval

[AnalystAgent]
  - Deterministic repo scoring + intent-match scoring
  - Opportunity scoring (engagement, recency, relevance, scope, ease)

[StrategistAgent]
  - Per-repo AI summaries
  - Adopter playbook OR contributor plan generation

[Snowflake]
  - Persistence, run context, feedback, action logs, cache retrieval
  - Evidence index + chat traces + eval logs for grounded repo Q&A

[Repo Deep-Dive Chat]
  - Ingest selected GitHub repo URL (+ optional tag/branch)
  - Build semantic evidence index with metadata (path/lines/symbol/type)
  - Hybrid retrieval (BM25 lexical + embedding similarity)
  - RetrieverAgent -> VerifierAgent -> WriterAgent contract enforcement
```

## Repository Layout

- `/app/main.py`: Streamlit app (persona-adaptive UX + feedback regeneration)
- `/src/github_client.py`: GitHub REST/GraphQL discovery
- `/src/query_pack.py`: domain/intent/constraints query generation
- `/src/snowflake_client.py`: Snowflake persistence, typed inserts, cache retrieval
- `/src/scoring.py`: deterministic repository/opportunity scoring
- `/src/ai.py`: Cortex AI wrappers (optimizer + persona outputs)
- `/src/agents.py`: Scout/Analyst/Strategist/Coordinator orchestration
- `/src/composio_integration.py`: Notion export (preview/dry-run/live)
- `/src/run_logger.py`: run trace helper
- `/src/repo_chat.py`: repo ingestion, hybrid retrieval, grounded chat pipeline
- `/src/store.py`: SnowflakeStore + LocalStore adapters for playbooks/evidence/chat traces
- `/src/chat_api.py`: FastAPI endpoints for repo indexing and `/chat`
- `/sql/setup.sql`: schema and table setup
- `/tests/`: dry-run sanity tests

## Setup

Prerequisite: Python `3.10` to `3.12` (recommended: `3.11` or `3.12`).

1. Copy `.env.example` to `.env` and set credentials.
2. Install dependencies:

```bash
make setup
```

3. Initialize Snowflake schema:

```bash
make init_snowflake
```

4. Run app:

```bash
make run
```

5. (Optional) Run chat API endpoint:

```bash
make run_api
```

## Environment Variables

### Required for full live mode
- `GITHUB_TOKEN`
- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_ROLE` (recommended for dedicated app RBAC)
- `SNOWFLAKE_WAREHOUSE`
- One of `SNOWFLAKE_PASSWORD` or `SNOWFLAKE_PRIVATE_KEY`
- `SNOWFLAKE_DATABASE` (default `OPENSOURCEOPS`)
- `SNOWFLAKE_SCHEMA` (default `PUBLIC`)

### Optional
- `COMPOSIO_API_KEY`
- `NOTION_DATABASE_ID` or `NOTION_PARENT_PAGE_ID`
- `COMPOSIO_NOTION_INTEGRATION_ID`
- `CACHE_TTL_HOURS` (default `6`)
- `CORTEX_MODEL_PRIMARY` (default `snowflake-arctic`)
- `CORTEX_MODEL_FALLBACK` (default `mistral-7b`)

### GitHub call throttling (rate-limit safety)
- `GITHUB_MAX_SEARCH_QUERIES` (max `6`, default `3`)
- `GITHUB_TOP_K_REPOS` (default `25`)
- `GITHUB_PER_QUERY_PAGE_SIZE` (default `15`)
- `GITHUB_ISSUE_REPO_LIMIT` (default `5`)
- `GITHUB_ISSUES_PER_REPO` (default `10`)
- `GITHUB_DISCUSSION_REPO_LIMIT` (default `4`)
- `GITHUB_DISCUSSIONS_PER_REPO` (default `8`)
- `GITHUB_FETCH_CONTRIBUTORS` (default `false`)

When credentials are missing, the app runs in dry-run/fallback mode and records intended actions in run logs.

## Caching Behavior

Cache key is computed from:
- persona
- domain
- intent hash
- constraints hash

Artifacts are persisted and reused for up to `CACHE_TTL_HOURS` (default `6`):
- optimizer output/query pack
- repositories and scores
- opportunities
- AI outputs
- repo evidence index chunks
- chat traces and evaluation logs

`Force refresh` bypasses cache. Feedback regeneration uses intelligent scope:
- Repo/constraints/opportunity feedback -> full rediscovery
- Plan/detail feedback -> reuse candidates and regenerate AI outputs only

## Feedback Loop

Each major section exposes **Refine** controls. Feedback is stored in run history and used during regeneration. A **Changes & Justification** section explains whether recommendations changed and why.

## Sample Runs

### Adopter sample
- Domain: `RAG evaluation`
- Intent: `We need to adopt a reliable RAG evaluation stack for internal model governance with low operational risk.`
- Constraints: `language=Python`, `maturity=Stable only`

### Contributor sample
- Domain: `RAG evaluation`
- Intent: `I want to contribute to high-impact RAG evaluation tooling to build maintainer trust and portfolio evidence.`
- Constraints: `language=Python`, `maturity=OK emerging`

## Repo Deep-Dive Chat Contract

For every answer:
1. Direct answer
2. Citations (`repo URL + file path + line range`)
3. `Confidence: High|Medium|Low`
4. If weak evidence: exact phrase `Not confirmed from repo evidence` + `next_checks`
5. No capability claims without evidence

### API endpoints
- `POST /index` with `{"repo_url":"https://github.com/owner/repo","ref":"optional-tag-or-branch"}`
- `POST /chat` with `{"repoId":"<repo_id>","question":"...","mode":"grounded_qa|scenario_brainstorm|risk_review"}`

### Example questions (A-E)
1. A: `Where in the code is retry orchestration implemented?`
2. B: `How do I configure this repo for production deployment?`
3. C: `Does it support Kafka and Postgres?`
4. D: `What are the biggest integration risks for adopting this repo?`
5. E: `What are breaking changes between recent releases?`

Expected response shape:
```json
{
  "answer": "string",
  "citations": [
    {
      "file_path": "string",
      "start_line": 1,
      "end_line": 30,
      "url": "https://github.com/owner/repo/blob/<sha>/path#L1",
      "snippet": "string"
    }
  ],
  "confidence": "High|Medium|Low",
  "next_checks": ["string"],
  "question_type": "A|B|C|D|E"
}
```

## Tests

```bash
make test
```

## GitHub Actions

Free-tier compatible workflows:
- `.github/workflows/ci.yml` (PR + push checks)
- `.github/workflows/deploy-check.yml` (deployment readiness gate)

## Known Constraints

- GitHub discussions may be unavailable for some repositories/tokens; app falls back to issues-only opportunities and labels this in UI.
- Snowflake Cortex function signatures vary by account; wrapper handles fallback paths.
