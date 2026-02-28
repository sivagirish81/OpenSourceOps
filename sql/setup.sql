-- OpenSourceOps Snowflake schema setup
CREATE DATABASE IF NOT EXISTS OPENSOURCEOPS COMMENT = 'OpenSourceOps analytics and agent outputs';
CREATE SCHEMA IF NOT EXISTS OPENSOURCEOPS.PUBLIC COMMENT = 'Primary schema for OpenSourceOps';

USE DATABASE OPENSOURCEOPS;
USE SCHEMA PUBLIC;

CREATE TABLE IF NOT EXISTS RUN_CONTEXT (
  run_id STRING COMMENT 'Unique run id',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Run creation time',
  persona STRING COMMENT 'Contributor or Adopter',
  domain STRING COMMENT 'Domain input',
  intent STRING COMMENT 'Intent message',
  constraints VARIANT COMMENT 'Constraint inputs',
  cache_key STRING COMMENT 'Deterministic cache key',
  optimizer VARIANT COMMENT 'Query optimizer output JSON',
  previous_run_id STRING COMMENT 'Run id used as refinement baseline'
);

CREATE TABLE IF NOT EXISTS RUN_FEEDBACK (
  run_id STRING COMMENT 'Run id',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Feedback timestamp',
  section STRING COMMENT 'Section being refined',
  feedback_text STRING COMMENT 'User feedback text'
);

CREATE TABLE IF NOT EXISTS REPOS (
  run_id STRING COMMENT 'Run id',
  cache_key STRING COMMENT 'Cache key',
  persona STRING COMMENT 'Contributor or Adopter',
  domain STRING COMMENT 'Domain context',
  language STRING COMMENT 'Optional language filter',
  full_name STRING COMMENT 'GitHub org/repo',
  html_url STRING COMMENT 'Repository URL',
  description STRING COMMENT 'Repository description',
  topics ARRAY COMMENT 'GitHub topics list',
  repo_language STRING COMMENT 'Primary repository language',
  stargazers_count NUMBER COMMENT 'Star count',
  forks_count NUMBER COMMENT 'Fork count',
  open_issues_count NUMBER COMMENT 'Open issue count',
  watchers_count NUMBER COMMENT 'Watcher/subscriber count',
  license STRING COMMENT 'License identifier',
  pushed_at TIMESTAMP_NTZ COMMENT 'Last push timestamp',
  created_at TIMESTAMP_NTZ COMMENT 'Repository creation timestamp',
  contributors_count NUMBER COMMENT 'Best-effort contributor count',
  fetched_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Ingestion timestamp'
);

CREATE TABLE IF NOT EXISTS REPO_SCORES (
  run_id STRING COMMENT 'Run id',
  cache_key STRING COMMENT 'Cache key',
  persona STRING COMMENT 'Contributor or Adopter',
  full_name STRING COMMENT 'GitHub org/repo',
  intent_match_score FLOAT COMMENT 'Intent relevance score',
  activity_score FLOAT COMMENT 'Recency score',
  adoption_score FLOAT COMMENT 'Adoption score',
  maintenance_score FLOAT COMMENT 'Maintenance score',
  community_score FLOAT COMMENT 'Community score',
  health_score FLOAT COMMENT 'Health composite score',
  risk_score FLOAT COMMENT 'Risk score',
  final_score FLOAT COMMENT 'Final ranking score',
  scored_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Scoring timestamp'
);

CREATE TABLE IF NOT EXISTS REPO_AI (
  run_id STRING COMMENT 'Run id',
  cache_key STRING COMMENT 'Cache key',
  persona STRING COMMENT 'Contributor or Adopter',
  domain STRING COMMENT 'Domain context',
  full_name STRING COMMENT 'GitHub org/repo',
  maturity STRING COMMENT 'AI-estimated maturity',
  summary STRING COMMENT 'AI summary',
  risks ARRAY COMMENT 'AI risks',
  recommended_use STRING COMMENT 'AI recommended use',
  engagement_plan STRING COMMENT 'AI engagement plan',
  raw_json VARIANT COMMENT 'Raw AI JSON',
  generated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Generation timestamp'
);

CREATE TABLE IF NOT EXISTS DOMAIN_AI (
  run_id STRING COMMENT 'Run id',
  persona STRING COMMENT 'Contributor or Adopter',
  domain STRING COMMENT 'Domain context',
  cache_key STRING COMMENT 'Cache key',
  playbook VARIANT COMMENT 'Structured output JSON (playbook or contribution plan)',
  playbook_md STRING COMMENT 'Markdown-rendered output',
  generated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Generation timestamp'
);

CREATE TABLE IF NOT EXISTS PLAYBOOK_STORAGE (
  run_id STRING COMMENT 'Run id',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Persist timestamp',
  payload VARIANT COMMENT 'Stored playbook payload'
);

CREATE TABLE IF NOT EXISTS REPO_EVIDENCE_INDEX (
  repo_id STRING COMMENT 'Deterministic repo id derived from repo_url@commit',
  repo_url STRING COMMENT 'GitHub repository URL',
  commit_sha STRING COMMENT 'Indexed commit SHA',
  file_path STRING COMMENT 'Relative file path in repo',
  start_line NUMBER COMMENT 'Chunk start line',
  end_line NUMBER COMMENT 'Chunk end line',
  heading STRING COMMENT 'Document heading, when applicable',
  symbol_name STRING COMMENT 'Code symbol name, when applicable',
  content_type STRING COMMENT 'doc|code|config',
  chunk_text STRING COMMENT 'Evidence chunk text',
  chunk_meta VARIANT COMMENT 'Extra metadata including source URL and embedding',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Index write timestamp'
);

CREATE TABLE IF NOT EXISTS CHAT_TRACES (
  trace_id STRING COMMENT 'Unique chat trace id',
  repo_id STRING COMMENT 'Indexed repo id',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Trace timestamp',
  mode STRING COMMENT 'grounded_qa|scenario_brainstorm|risk_review',
  question STRING COMMENT 'User question',
  answer STRING COMMENT 'Assistant answer',
  confidence STRING COMMENT 'High|Medium|Low',
  citations ARRAY COMMENT 'Citation objects used in response',
  next_checks ARRAY COMMENT 'Follow-up checks suggested by verifier',
  question_type STRING COMMENT 'A|B|C|D|E'
);

CREATE TABLE IF NOT EXISTS CHAT_EVAL_LOGS (
  eval_id STRING COMMENT 'Unique eval event id',
  repo_id STRING COMMENT 'Indexed repo id',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Eval timestamp',
  metric STRING COMMENT 'Evaluation metric name',
  value FLOAT COMMENT 'Metric value',
  detail STRING COMMENT 'Metric detail'
);

CREATE TABLE IF NOT EXISTS ISSUES (
  run_id STRING COMMENT 'Run id',
  cache_key STRING COMMENT 'Cache key',
  persona STRING COMMENT 'Contributor or Adopter',
  domain STRING COMMENT 'Domain context',
  full_name STRING COMMENT 'Repository full name',
  issue_id NUMBER COMMENT 'GitHub issue id',
  title STRING COMMENT 'Issue title',
  body STRING COMMENT 'Issue body',
  labels ARRAY COMMENT 'Normalized labels',
  comments NUMBER COMMENT 'Comment count',
  created_at TIMESTAMP_NTZ COMMENT 'Issue creation timestamp',
  updated_at TIMESTAMP_NTZ COMMENT 'Issue update timestamp',
  html_url STRING COMMENT 'Issue URL',
  fetched_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Ingestion timestamp'
);

CREATE TABLE IF NOT EXISTS CONTRIBUTION_OPPORTUNITIES (
  run_id STRING COMMENT 'Run id',
  domain STRING COMMENT 'Domain context',
  persona STRING COMMENT 'Contributor mode',
  cache_key STRING COMMENT 'Cache key',
  repo_full_name STRING COMMENT 'Repository full name',
  item_type STRING COMMENT 'ISSUE or DISCUSSION',
  item_id STRING COMMENT 'Issue/discussion id',
  title STRING COMMENT 'Title',
  body STRING COMMENT 'Body excerpt',
  labels ARRAY COMMENT 'Labels/categories',
  comments NUMBER COMMENT 'Engagement count',
  updated_at TIMESTAMP_NTZ COMMENT 'Last update timestamp',
  url STRING COMMENT 'Item URL',
  engagement_score FLOAT COMMENT 'Engagement score',
  scope_score FLOAT COMMENT 'Scope score',
  relevance_score FLOAT COMMENT 'Intent relevance score',
  ease_score FLOAT COMMENT 'Ease score',
  final_score FLOAT COMMENT 'Final opportunity score',
  why STRING COMMENT 'Why this opportunity matters',
  suggested_next_action STRING COMMENT 'Suggested engagement action',
  difficulty STRING COMMENT 'Estimated difficulty',
  computed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Computed timestamp'
);

CREATE TABLE IF NOT EXISTS RUN_ACTIONS (
  run_id STRING COMMENT 'Run id',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Action timestamp',
  action_type STRING COMMENT 'Action type',
  status STRING COMMENT 'SUCCESS|FAILED|DRY_RUN',
  target_url STRING COMMENT 'External URL if available',
  detail STRING COMMENT 'Action details'
);

CREATE TABLE IF NOT EXISTS RUN_LOG (
  run_id STRING COMMENT 'Run id',
  ts TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Log timestamp',
  agent STRING COMMENT 'Agent/component name',
  step STRING COMMENT 'Step name',
  status STRING COMMENT 'STARTED|SUCCESS|FAILED|INFO',
  detail STRING COMMENT 'Human-readable detail'
);
