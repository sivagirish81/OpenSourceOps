-- OSS Due Diligence AI Snowflake schema
CREATE DATABASE IF NOT EXISTS OPENSOURCEOPS COMMENT='OSS Due Diligence AI system of record';
CREATE SCHEMA IF NOT EXISTS OPENSOURCEOPS.PUBLIC COMMENT='Primary application schema';

USE DATABASE OPENSOURCEOPS;
USE SCHEMA PUBLIC;

CREATE TABLE IF NOT EXISTS COMPANY_PROFILES (
  company_id STRING COMMENT 'Deterministic company profile id',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Profile creation time',
  updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Last update time',
  profile VARIANT COMMENT 'Complete onboarding profile JSON'
);

CREATE TABLE IF NOT EXISTS SCOUTING_RUNS (
  scout_run_id STRING COMMENT 'Scouting run id',
  company_id STRING COMMENT 'Owning company profile id',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Run creation time',
  requirements STRING COMMENT 'Requirements entered by user',
  constraints VARIANT COMMENT 'Derived constraints from company context',
  query_pack VARIANT COMMENT 'Search query plan used for GitHub scouting',
  top3 VARIANT COMMENT 'Top 3 ranked repositories with scoring breakdown'
);

CREATE TABLE IF NOT EXISTS REPO_CANDIDATES (
  candidate_id STRING COMMENT 'Candidate row id',
  scout_run_id STRING COMMENT 'Scouting run id',
  company_id STRING COMMENT 'Company id',
  repo_full_name STRING COMMENT 'owner/repo',
  repo_url STRING COMMENT 'Repo URL',
  score_breakdown VARIANT COMMENT 'Fit/Maturity/Health/Risk/Context breakdown',
  pros VARIANT COMMENT 'Pros with citations',
  cons VARIANT COMMENT 'Cons with citations',
  warnings VARIANT COMMENT 'Constraint warnings',
  final_score FLOAT COMMENT 'Final rank score',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Inserted at'
);

CREATE TABLE IF NOT EXISTS REPO_INGESTIONS (
  ingest_run_id STRING COMMENT 'Ingestion run id',
  scout_run_id STRING COMMENT 'Source scouting run id',
  company_id STRING COMMENT 'Owning company profile id',
  repo_full_name STRING COMMENT 'owner/repo',
  repo_url STRING COMMENT 'Repository URL',
  commit_sha STRING COMMENT 'Indexed commit SHA',
  files_indexed NUMBER COMMENT 'Number of indexed files',
  chunks_indexed NUMBER COMMENT 'Number of indexed chunks',
  budget_usage VARIANT COMMENT 'Skyfire budget usage info',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Ingestion timestamp',
  file_manifest VARIANT COMMENT 'Indexed file list'
);

CREATE TABLE IF NOT EXISTS INGESTION_JOBS (
  job_id STRING COMMENT 'Background ingestion job id',
  ingest_run_id STRING COMMENT 'Target ingestion run id',
  company_id STRING COMMENT 'Company id',
  repo_full_name STRING COMMENT 'owner/repo',
  repo_url STRING COMMENT 'Repository URL',
  status STRING COMMENT 'QUEUED|RUNNING|COMPLETE|FAILED',
  progress FLOAT COMMENT '0..1 progress',
  message STRING COMMENT 'Latest status message',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Job creation time',
  updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Last status update',
  error STRING COMMENT 'Failure details if any'
);

CREATE TABLE IF NOT EXISTS EVIDENCE_CHUNKS (
  chunk_id STRING COMMENT 'Chunk id',
  ingest_run_id STRING COMMENT 'Ingestion run id',
  repo_url STRING COMMENT 'Repository URL',
  commit_sha STRING COMMENT 'Indexed commit SHA',
  file_path STRING COMMENT 'File path',
  start_line NUMBER COMMENT 'Start line',
  end_line NUMBER COMMENT 'End line',
  heading STRING COMMENT 'Doc heading if present',
  symbol_name STRING COMMENT 'Code symbol if present',
  content_type STRING COMMENT 'doc|code|config',
  chunk_text STRING COMMENT 'Chunk text content',
  chunk_pointer STRING COMMENT 'Pointer URL to source',
  metadata VARIANT COMMENT 'Additional metadata',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Creation timestamp'
);

CREATE TABLE IF NOT EXISTS REPO_EVIDENCE (
  evidence_id STRING COMMENT 'Evidence id',
  ingest_run_id STRING COMMENT 'Ingestion run id',
  repo_full_name STRING COMMENT 'owner/repo',
  repo_url STRING COMMENT 'Repo URL',
  commit_or_ref STRING COMMENT 'Commit SHA or ref',
  source_type STRING COMMENT 'github_repo|github_release|github_file|github_workflow|github_issue_sample',
  file_path STRING COMMENT 'File path if applicable',
  source_url STRING COMMENT 'Deep link URL',
  api_ref STRING COMMENT 'GitHub API reference',
  retrieved_at TIMESTAMP_NTZ COMMENT 'Collection timestamp',
  content STRING COMMENT 'Evidence content (bounded)',
  line_map VARIANT COMMENT 'Line map for line-aware citations',
  sha STRING COMMENT 'Content SHA',
  metadata VARIANT COMMENT 'Structured metadata payload',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Inserted at'
);

CREATE TABLE IF NOT EXISTS REPO_SIGNALS (
  signal_id STRING COMMENT 'Signal id',
  ingest_run_id STRING COMMENT 'Ingestion run id',
  repo_full_name STRING COMMENT 'owner/repo',
  category STRING COMMENT 'docs|security|license|ci|ops|release',
  signal_name STRING COMMENT 'Signal name',
  signal_value VARIANT COMMENT 'Signal value',
  evidence_id STRING COMMENT 'Source evidence id',
  citations VARIANT COMMENT 'Citations backing the signal',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Inserted at'
);

CREATE TABLE IF NOT EXISTS DUE_DILIGENCE_RUNS (
  dd_run_id STRING COMMENT 'Due diligence run id',
  company_id STRING COMMENT 'Owning company profile id',
  scout_run_id STRING COMMENT 'Source scouting run',
  ingest_run_id STRING COMMENT 'Source ingestion run',
  repo_full_name STRING COMMENT 'Selected repository',
  repo_url STRING COMMENT 'Selected repository URL',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Run creation time',
  status STRING COMMENT 'RUNNING|COMPLETE|FAILED',
  decision STRING COMMENT 'GO|CONDITIONAL_GO|NO_GO',
  report_json VARIANT COMMENT 'Final validated report JSON',
  report_md STRING COMMENT 'Rendered markdown report'
);

CREATE TABLE IF NOT EXISTS REPORTS (
  report_id STRING COMMENT 'Report id',
  dd_run_id STRING COMMENT 'Due diligence run id',
  decision STRING COMMENT 'GO|CONDITIONAL_GO|NO_GO',
  report_json VARIANT COMMENT 'Final report JSON',
  report_md STRING COMMENT 'Final report markdown',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Inserted at'
);

CREATE TABLE IF NOT EXISTS FINDINGS (
  finding_id STRING COMMENT 'Unique finding id',
  dd_run_id STRING COMMENT 'Due diligence run id',
  agent_name STRING COMMENT 'Producing agent',
  category STRING COMMENT 'security|license|reliability|architecture|community',
  claim STRING COMMENT 'Finding statement',
  confidence STRING COMMENT 'High|Medium|Low',
  citations VARIANT COMMENT 'Citation list',
  base_severity FLOAT COMMENT 'Base severity 0..1',
  context_multiplier FLOAT COMMENT 'Context multiplier',
  adjusted_severity FLOAT COMMENT 'Final adjusted severity',
  multiplier_rationale STRING COMMENT 'Rationale for multiplier',
  next_checks VARIANT COMMENT 'Next checks list',
  owner STRING COMMENT 'Eng|Security|Legal|SRE',
  effort STRING COMMENT 'S|M|L',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Finding creation time'
);

CREATE TABLE IF NOT EXISTS AGENT_TRANSCRIPTS (
  transcript_id STRING COMMENT 'Transcript event id',
  dd_run_id STRING COMMENT 'Due diligence run id',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Event time',
  agent_name STRING COMMENT 'Agent name',
  step STRING COMMENT 'Step name',
  status STRING COMMENT 'INFO|SUCCESS|FAILED',
  payload VARIANT COMMENT 'Structured transcript payload'
);

CREATE TABLE IF NOT EXISTS REPORT_EXPORTS (
  export_id STRING COMMENT 'Export id',
  dd_run_id STRING COMMENT 'Due diligence run id',
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'Export time',
  export_type STRING COMMENT 'json|markdown',
  content STRING COMMENT 'Exported content'
);
