from __future__ import annotations

from src.due_diligence_agents import ScoutingAgent


class DummyGitHub:
    def search_repositories(self, *args, **kwargs):
        return []

    def fetch_repository(self, full_name):
        catalog = {
            "temporalio/temporal": {
                "full_name": "temporalio/temporal",
                "html_url": "https://github.com/temporalio/temporal",
                "description": "Durable execution platform for workflows",
                "topics": ["durable-execution", "workflow", "orchestration"],
                "language": "Go",
                "stargazers_count": 15000,
                "forks_count": 1200,
                "open_issues_count": 400,
                "watchers_count": 900,
                "license": "MIT",
                "pushed_at": "2026-02-20T00:00:00Z",
            },
            "cadence-workflow/cadence": {
                "full_name": "cadence-workflow/cadence",
                "html_url": "https://github.com/cadence-workflow/cadence",
                "description": "Workflow orchestration engine",
                "topics": ["workflow", "durable"],
                "language": "Go",
                "stargazers_count": 9000,
                "forks_count": 1000,
                "open_issues_count": 250,
                "watchers_count": 600,
                "license": "Apache-2.0",
                "pushed_at": "2026-02-20T00:00:00Z",
            },
            "Netflix/conductor": {
                "full_name": "Netflix/conductor",
                "html_url": "https://github.com/Netflix/conductor",
                "description": "Workflow orchestration platform",
                "topics": ["workflow", "orchestration"],
                "language": "Java",
                "stargazers_count": 12000,
                "forks_count": 1500,
                "open_issues_count": 350,
                "watchers_count": 700,
                "license": "Apache-2.0",
                "pushed_at": "2026-02-20T00:00:00Z",
            },
        }
        return catalog.get(full_name)


def test_scouting_filters_irrelevant_list_repos_and_prefers_fit():
    agent = ScoutingAgent(DummyGitHub())
    repos = [
        {
            "full_name": "aneasystone/github-trending",
            "html_url": "https://github.com/aneasystone/github-trending",
            "description": "track github trending projects",
            "topics": ["trending", "github"],
            "language": "Python",
            "stargazers_count": 5000,
            "forks_count": 200,
            "open_issues_count": 5,
            "watchers_count": 300,
            "license": "MIT",
            "pushed_at": "2026-01-10T00:00:00Z",
        },
        {
            "full_name": "temporalio/temporal",
            "html_url": "https://github.com/temporalio/temporal",
            "description": "Durable execution and workflow orchestration platform",
            "topics": ["workflow", "orchestration", "durable-execution"],
            "language": "Go",
            "stargazers_count": 15000,
            "forks_count": 1200,
            "open_issues_count": 400,
            "watchers_count": 900,
            "license": "MIT",
            "pushed_at": "2026-02-20T00:00:00Z",
        },
    ]
    constraints = {"primary_languages": [], "maturity_preference": "Stable only"}
    out = agent._rank_top3(repos, constraints, "durable workflow orchestration platform")
    names = [r["full_name"] for r in out]
    assert "temporalio/temporal" in names
    assert "aneasystone/github-trending" not in names


def test_scouting_filters_malicious_results_for_payment_requirements():
    agent = ScoutingAgent(DummyGitHub())
    repos = [
        {
            "full_name": "lunnar211/fb-brute",
            "html_url": "https://github.com/lunnar211/fb-brute",
            "description": "facebook cracker and brute force tool",
            "topics": ["hacking", "brute"],
            "language": "Python",
            "stargazers_count": 1000,
            "forks_count": 50,
            "open_issues_count": 2,
            "watchers_count": 20,
            "license": "MIT",
            "pushed_at": "2026-02-20T00:00:00Z",
        },
        {
            "full_name": "juspay/hyperswitch",
            "html_url": "https://github.com/juspay/hyperswitch",
            "description": "Open source payments switch",
            "topics": ["payments", "gateway", "fintech"],
            "language": "Rust",
            "stargazers_count": 18000,
            "forks_count": 2200,
            "open_issues_count": 120,
            "watchers_count": 600,
            "license": "Apache-2.0",
            "pushed_at": "2026-02-20T00:00:00Z",
        },
    ]
    constraints = {"primary_languages": [], "maturity_preference": "Stable only"}
    out = agent._rank_top3(repos, constraints, "I want to use an open source payment provider")
    names = [r["full_name"] for r in out]
    assert "juspay/hyperswitch" in names
    assert "lunnar211/fb-brute" not in names


def test_scouting_enforces_min_stars(monkeypatch):
    monkeypatch.setenv("SCOUT_MIN_STARS", "200")
    agent = ScoutingAgent(DummyGitHub())
    repos = [
        {
            "full_name": "acme/low-stars",
            "html_url": "https://github.com/acme/low-stars",
            "description": "payment provider sdk",
            "topics": ["payments"],
            "language": "Go",
            "stargazers_count": 99,
            "forks_count": 10,
            "open_issues_count": 2,
            "watchers_count": 20,
            "license": "MIT",
            "pushed_at": "2026-02-20T00:00:00Z",
        },
        {
            "full_name": "acme/high-stars",
            "html_url": "https://github.com/acme/high-stars",
            "description": "open source payment gateway platform",
            "topics": ["payments", "gateway"],
            "language": "Go",
            "stargazers_count": 500,
            "forks_count": 50,
            "open_issues_count": 8,
            "watchers_count": 60,
            "license": "MIT",
            "pushed_at": "2026-02-20T00:00:00Z",
        },
    ]
    constraints = {"primary_languages": [], "maturity_preference": "Stable only"}
    out = agent._rank_top3(repos, constraints, "open source payment provider")
    names = [r["full_name"] for r in out]
    assert "acme/high-stars" in names
    assert "acme/low-stars" not in names


def test_durable_execution_augments_with_canonical_engines():
    agent = ScoutingAgent(DummyGitHub())
    constraints = {"primary_languages": [], "maturity_preference": "Stable only"}
    repos = [
        {
            "full_name": "tmgthb/Autonomous-Agents",
            "html_url": "https://github.com/tmgthb/Autonomous-Agents",
            "description": "Autonomous Agents research papers. Updated daily.",
            "topics": ["agents", "research"],
            "language": "Python",
            "stargazers_count": 5000,
            "forks_count": 200,
            "open_issues_count": 20,
            "watchers_count": 100,
            "license": "MIT",
            "pushed_at": "2026-02-20T00:00:00Z",
        }
    ]
    augmented = agent._augment_with_canonical_repos(
        repos, "Need a durable workflow orchestration platform"
    )
    ranked = agent._rank_top3(augmented, constraints, "Need a durable workflow orchestration platform")
    names = [r["full_name"] for r in ranked]
    assert "temporalio/temporal" in names
    assert names[0] in {"temporalio/temporal", "cadence-workflow/cadence", "Netflix/conductor"}
