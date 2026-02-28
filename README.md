# OpenSourceOps

OpenSourceOps is a Snowflake-backed OSS Due Diligence platform for companies evaluating open source software before adoption. The product captures company context, scouts the best candidate repositories, runs a multi-agent due diligence analysis, and produces a grounded decision report with prioritized owner-assigned next steps.

The application is built as a Streamlit-first system where Streamlit handles both UI and backend orchestration. Snowflake is the mandatory system of record for company profiles, scouting runs, ingestion metadata, evidence chunks, findings, final reports, and agent transcripts.

The repository indexing subsystem uses a **Signal-Based Evidence Index**. It does not clone and embed full source trees. Instead, it collects high-signal GitHub metadata and key governance/ops files (README/docs/SECURITY/LICENSE/CI/deploy configs), then applies deterministic parsers to extract features with citations.

## Core Workflow

1. Company onboarding with industry-standard due diligence inputs.
2. Context-aware scouting returns Top 3 repositories with explainable scores.
3. User selects one repository.
4. Repo ingestion captures commit SHA, file manifest, evidence chunks, and budget usage.
5. Multi-agent due diligence run produces GO / CONDITIONAL_GO / NO_GO report.

## Product Scope

- Company-only (no contributor mode).
- No adoption playbook output. Focus is due diligence decision quality.
- Grounded Answer Contract enforced for findings and report.

## Architecture

```text
Streamlit App (UI + Backend)
  |
  +-- SnowflakeStore (mandatory persistence)
  |     +-- COMPANY_PROFILES
  |     +-- SCOUTING_RUNS
  |     +-- REPO_CANDIDATES
  |     +-- REPO_INGESTIONS
  |     +-- REPO_EVIDENCE
  |     +-- REPO_SIGNALS
  |     +-- DUE_DILIGENCE_RUNS
  |     +-- FINDINGS
  |     +-- REPORTS
  |     +-- AGENT_TRANSCRIPTS
  |
  +-- ScoutingAgent
  +-- RepoLibrarianAgent
  +-- SecuritySupplyChainAgent
  +-- LicenseComplianceAgent
  +-- ReliabilityOpsAgent
  +-- ArchitectureIntegrationAgent
  +-- CommunityMaintenanceAgent
  +-- JudgeVerifierAgent
  |
  +-- Composio (optional task creation)
  +-- CrewAI (orchestration integration with deterministic fallback)
```

## Grounded Answer Contract

For every claim:

- Include citations (`repo URL + file path + line range` or document section).
- Include confidence (`High | Medium | Low`).
- If evidence is weak/missing, output exactly:
  - `Not confirmed from repo evidence`
  - plus `next_checks` list.
- Do not assert unsupported capabilities without evidence.

## Context-Aware Severity Model

Each finding includes:

- `base_severity`
- `context_multiplier`
- `adjusted_severity`
- `multiplier_rationale`

Severity is adjusted based on company profile requirements (for example, SOC2 + required audit logs + low risk tolerance increases security severity).

## Repository Structure

- [app/main.py](/Users/sivagirish/Documents/Work/Project/OpenSourceOps/app/main.py): Streamlit pages and orchestration entrypoint.
- [storage/snowflake_store.py](/Users/sivagirish/Documents/Work/Project/OpenSourceOps/storage/snowflake_store.py): mandatory typed Snowflake persistence layer.
- [storage/schema.sql](/Users/sivagirish/Documents/Work/Project/OpenSourceOps/storage/schema.sql): Snowflake schema DDL.
- [src/due_diligence_agents.py](/Users/sivagirish/Documents/Work/Project/OpenSourceOps/src/due_diligence_agents.py): context-aware agents, grounding checks, judge/verifier, report schema validation.
- [src/due_diligence_pipeline.py](/Users/sivagirish/Documents/Work/Project/OpenSourceOps/src/due_diligence_pipeline.py): scouting, ingestion, due diligence run functions.
- [src/evidence_collector.py](/Users/sivagirish/Documents/Work/Project/OpenSourceOps/src/evidence_collector.py): lightweight GitHub evidence collection with strict caps.
- [src/evidence_parsers.py](/Users/sivagirish/Documents/Work/Project/OpenSourceOps/src/evidence_parsers.py): deterministic signal extractors and citation-backed features.
- [src/github_client.py](/Users/sivagirish/Documents/Work/Project/OpenSourceOps/src/github_client.py): GitHub repository discovery.
- [tests/](/Users/sivagirish/Documents/Work/Project/OpenSourceOps/tests): pytest test suite.

## Setup

Prerequisite: Python `3.10` - `3.12`.

1. Copy `.env.example` to `.env`.
2. Set Snowflake and GitHub credentials.
3. Install dependencies:

```bash
make setup
```

4. Initialize Snowflake schema:

```bash
make init_snowflake
```

5. Start app:

```bash
make run
```

## Environment Variables

Required:

- `GITHUB_TOKEN`
- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD` or `SNOWFLAKE_PRIVATE_KEY`
- `SNOWFLAKE_ROLE`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE` (default `OPENSOURCEOPS`)
- `SNOWFLAKE_SCHEMA` (default `PUBLIC`)

Optional:

- `COMPOSIO_API_KEY` (task creation integration)
- `SKYFIRE_MAX_FILES` (default `50`)
- `SKYFIRE_MAX_TOKENS` (default `120000`)
- `INDEX_MINIMAL_MAX_FILES` (default `20`)
- `INDEX_STANDARD_MAX_FILES` (default `50`)
- `MAX_FILES` (default `80`)
- `MAX_BYTES` (default `8388608`)
- `MAX_DOC_FILES` (default `35`)
- `MAX_WORKFLOW_FILES` (default `20`)

## Streamlit Pages

1. Company Onboarding
2. Scouting Results (Top 3)
3. Repo Selection + Ingestion Progress
4. Due Diligence Dashboard
5. War Room Transcript

## Due Diligence Output

Validated report JSON contains:

- `decision` (`GO` / `CONDITIONAL_GO` / `NO_GO`)
- `risk_register` with adjusted severities and context rationales
- `prioritized_next_steps` with owners/effort
- `maintainer_questions`

Exports:

- JSON
- Markdown

## Demo Script

1. Open **Company Onboarding** and create profile:
   - Industry: FinTech
   - Frameworks: SOC2, GDPR
   - Risk tolerance: Low
   - Require commercial support: Yes
2. Open **Scouting Results**:
   - Requirements: "Need durable workflow orchestration with auditability and production reliability."
   - Run scouting and inspect Top 3 score breakdown.
3. Select one repo and move to **Repo Selection + Ingestion**:
   - Choose indexing depth:
     - Minimal (fast): ~10-30s, indexes ~10-20 most important files
     - Standard: ~30-90s, caps at ~50 key files
   - Indexing also includes small high-signal context from latest releases, recent issues, and recent discussions.
   - Start background ingestion and monitor progress/chunk availability.
   - You can navigate away; ingestion continues in background and persists to Snowflake.
4. Open **Due Diligence Dashboard**:
   - Run due diligence with available chunks (partial allowed), then rerun after indexing completes for fuller coverage.
   - Review decision, risk register, owner-assigned next steps, and maintainer questions.
   - Download JSON/Markdown exports.
5. Open **War Room Transcript**:
   - Review agent-by-agent outputs and judge resolution trace.

## Testing

Run tests:

```bash
make test
```

Coverage includes:

- Snowflake schema creation and store write/read mock happy path.
- Citation enforcement and missing-evidence fallback.
- Context multiplier logic.
- Judge deduplication.
- Report JSON schema validation.
