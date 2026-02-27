# OpenSourceOps

OpenSourceOps is an autonomous Open Source Intelligence web application and multi-agent pipeline for evaluating and operationalizing open-source ecosystems. It supports the **Developer / Contributor** persona by identifying high-impact repositories and contribution opportunities in a chosen domain, scoring opportunities, and generating a weekly contribution plan focused on measurable reputation impact.

It also supports the **Company / Adopter** persona by discovering candidate projects, computing maturity and risk signals, generating a structured 30/60/90 adoption playbook with Snowflake Cortex `AI_COMPLETE`, and optionally publishing the playbook to Notion through Composio with explicit user confirmation.

## Architecture

```text
[Streamlit UI]
   |
   | Run Analysis (Domain, Language, Mode)
   v
[CoordinatorAgent] ---> [RUN_LOG]
   |
   +--> [ScoutAgent] ----> GitHub Search API ----> REPOS / ISSUES
   |
   +--> [AnalystAgent] --> scoring.py -----------> REPO_SCORES / ISSUE_SCORES
   |
   +--> [StrategistAgent] -> Snowflake AI_COMPLETE -> REPO_AI / DOMAIN_AI
   |
   +--> [CoordinatorAgent] -> Composio (optional) -> Notion page URL
```

## Repository Layout

- `/app/main.py` Streamlit web app
- `/src/github_client.py` GitHub repo + issues discovery
- `/src/query_pack.py` domain-to-query generation
- `/src/snowflake_client.py` Snowflake connection + insert/select helpers + setup CLI
- `/src/scoring.py` deterministic scoring formulas
- `/src/ai.py` Snowflake `AI_COMPLETE` wrappers + JSON parsing + markdown renderer
- `/src/agents.py` four agents (Scout, Analyst, Strategist, Coordinator)
- `/src/composio_integration.py` Notion creation with dry-run mode
- `/src/run_logger.py` RUN_LOG helper
- `/sql/setup.sql` database/schema/tables
- `/tests/` dry-run sanity tests

## Setup

Prerequisite: Python `3.10` to `3.12` (recommended: `3.11` or `3.12`). Python `3.13+` is currently not supported by some upstream dependencies.

1. Copy `.env.example` to `.env` and fill values.
2. Run:

```bash
make setup
```

3. Initialize Snowflake tables (optional but recommended when credentials exist):

```bash
make init_snowflake
```

4. Start app:

```bash
make run
```

## Environment Variables

Required for full live mode:

- `GITHUB_TOKEN`
- Optional GitHub rate-limit knobs:
  - `GITHUB_MAX_SEARCH_QUERIES` (default `2`)
  - `GITHUB_TOP_K_REPOS` (default `25`)
  - `GITHUB_PER_QUERY_PAGE_SIZE` (default `15`)
  - `GITHUB_ISSUE_REPO_LIMIT` (default `5`)
  - `GITHUB_ISSUES_PER_REPO` (default `10`)
  - `GITHUB_FETCH_CONTRIBUTORS` (default `false`; set `true` to add contributor count API calls)
- `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_WAREHOUSE`, and one of:
  - `SNOWFLAKE_PASSWORD`, or
  - `SNOWFLAKE_PRIVATE_KEY`
- `SNOWFLAKE_DATABASE` (default `OPENSOURCEOPS`), `SNOWFLAKE_SCHEMA` (default `PUBLIC`)
- `COMPOSIO_API_KEY`
- `NOTION_DATABASE_ID` or `NOTION_PARENT_PAGE_ID`
- `COMPOSIO_NOTION_INTEGRATION_ID` (if required by your Composio workspace)

When credentials are missing, OpenSourceOps uses dry-run behavior and logs intended actions to `RUN_LOG`.

## Quickstart Flow

1. Launch app with `make run`.
2. In sidebar set:
   - Domain: `RAG evaluation`
   - Language: `Python` (optional)
   - Mode: `Enterprise` or `Contributor`
3. Click **Run Analysis**.
4. Open **Repositories** tab to inspect ranked repos with health/risk/maturity and AI summaries.
5. Open **Adoption Playbook** tab to view generated markdown playbook.
6. Click **Create Notion Playbook**:
   - Review preview modal payload.
   - Check confirmation box.
   - Execute dry-run (if no Composio key) or live Notion page creation.
7. In **Run Log** tab, verify full Scout -> Analyst -> Strategist -> Coordinator trace.

## Tests

```bash
make test
```

- `test_github_dryrun.py`: validates query pack and token-aware GitHub client setup.
- `test_snowflake_dryrun.py`: validates Snowflake dry-run or connection ability.

## GitHub Actions

The repository includes two lightweight workflows that run on GitHub-hosted `ubuntu-latest` runners and are compatible with GitHub's free-tier Actions usage:

- `.github/workflows/ci.yml`
  - Triggers on pull requests and pushes to `main`/`master`
  - Installs dependencies, runs Python syntax checks, and executes `pytest`
- `.github/workflows/deploy-check.yml`
  - Triggers on pushes to `main` and manual dispatch
  - Re-runs syntax and test checks as a deployment readiness gate

## Limitations

- GitHub issue difficulty/reputation scoring is heuristic and intentionally simple.
- `AI_COMPLETE` function signatures can vary by Snowflake account version; fallback path handles this.
- Composio action payloads may require adjustment based on SDK version and workspace tool naming.

## Roadmap Ideas

1. Add richer contributor matching by skill tags and issue embeddings.
2. Add incremental Snowflake upserts and dedupe keys.
3. Add automatic weekly digest export (email/Slack/Notion).
4. Add model evaluation checks for JSON schema conformance.
