-- OpenSourceOps Snowflake schema setup
CREATE DATABASE IF NOT EXISTS OPENSOURCEOPS COMMENT = 'OpenSourceOps analytics and agent outputs';
CREATE SCHEMA IF NOT EXISTS OPENSOURCEOPS.PUBLIC COMMENT = 'Primary schema for OpenSourceOps';

USE DATABASE OPENSOURCEOPS;
USE SCHEMA PUBLIC;

CREATE TABLE IF NOT EXISTS REPOS (
  run_id STRING COMMENT 'Unique run identifier for this pipeline execution',
  domain STRING COMMENT 'User-provided domain query',
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
  license STRING COMMENT 'License identifier if available',
  pushed_at TIMESTAMP_NTZ COMMENT 'Last push timestamp',
  created_at TIMESTAMP_NTZ COMMENT 'Repository creation timestamp',
  contributors_count NUMBER COMMENT 'Best-effort contributor count',
  fetched_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Ingestion timestamp'
);

CREATE TABLE IF NOT EXISTS REPO_SCORES (
  run_id STRING COMMENT 'Run id',
  full_name STRING COMMENT 'GitHub org/repo',
  activity_score FLOAT COMMENT 'Recency score based on pushed_at',
  adoption_score FLOAT COMMENT 'Log-normalized stars/forks score',
  maintenance_score FLOAT COMMENT 'Issue-load maintenance score',
  community_score FLOAT COMMENT 'Community engagement score',
  health_score FLOAT COMMENT 'Weighted health composite',
  risk_score FLOAT COMMENT 'Inverse reliability risk score',
  scored_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Scoring timestamp'
);

CREATE TABLE IF NOT EXISTS REPO_AI (
  run_id STRING COMMENT 'Run id',
  domain STRING COMMENT 'Domain context',
  full_name STRING COMMENT 'GitHub org/repo',
  maturity STRING COMMENT 'AI-estimated maturity bucket',
  summary STRING COMMENT 'AI summary text',
  risks ARRAY COMMENT 'AI identified risks',
  recommended_use STRING COMMENT 'AI recommended usage',
  engagement_plan STRING COMMENT 'AI suggested engagement approach',
  raw_json VARIANT COMMENT 'Raw parsed JSON response from AI_COMPLETE',
  generated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Generation timestamp'
);

CREATE TABLE IF NOT EXISTS DOMAIN_AI (
  run_id STRING COMMENT 'Run id',
  domain STRING COMMENT 'Domain context',
  language STRING COMMENT 'Language context',
  playbook_json VARIANT COMMENT 'Structured adoption playbook JSON',
  playbook_md STRING COMMENT 'Markdown-rendered playbook',
  generated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Generation timestamp'
);

CREATE TABLE IF NOT EXISTS ISSUES (
  run_id STRING COMMENT 'Run id',
  domain STRING COMMENT 'Domain context',
  full_name STRING COMMENT 'Repository full name',
  issue_id NUMBER COMMENT 'GitHub issue id',
  title STRING COMMENT 'Issue title',
  body STRING COMMENT 'Issue body excerpt',
  labels ARRAY COMMENT 'Normalized labels',
  comments NUMBER COMMENT 'Comment count',
  created_at TIMESTAMP_NTZ COMMENT 'Issue creation timestamp',
  updated_at TIMESTAMP_NTZ COMMENT 'Issue update timestamp',
  html_url STRING COMMENT 'Issue URL',
  fetched_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Ingestion timestamp'
);

CREATE TABLE IF NOT EXISTS ISSUE_SCORES (
  run_id STRING COMMENT 'Run id',
  full_name STRING COMMENT 'Repository full name',
  issue_id NUMBER COMMENT 'GitHub issue id',
  impact_score FLOAT COMMENT 'Deterministic issue impact score',
  difficulty_score FLOAT COMMENT 'Deterministic issue difficulty estimate',
  reputation_score FLOAT COMMENT 'Contribution reputation potential score',
  scored_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Scoring timestamp'
);

CREATE TABLE IF NOT EXISTS RUN_LOG (
  run_id STRING COMMENT 'Run id across all agent steps',
  ts TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Log timestamp',
  agent STRING COMMENT 'Agent or component name',
  step STRING COMMENT 'Step name',
  status STRING COMMENT 'STARTED|SUCCESS|FAILED|INFO',
  detail STRING COMMENT 'Human-readable detail for UI/audit'
);
