#!/usr/bin/env python3
"""
launchers/run_crew.py

Entry point for Coverage_Crew. Runs the four-agent pipeline against
every asset currently in the registry and writes an HTML report to
reports/.

Usage:
    python launchers/run_crew.py --seed-assets data/assets_seed.yaml
    python launchers/run_crew.py --model qwen3:4b
    python launchers/run_crew.py --provider anthropic
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewai import Task, Crew, Process

from core import db
from core.report import write_html_report
from core.pricing import estimate_cost_usd, PRICING_SOURCE_NOTE
from agents.agents import (
    build_llm,
    build_ingest_agent,
    build_relevance_agent,
    build_exploitability_agent,
    build_communicator_agent,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


class Timer:
    """
    Tracks wall-clock time per agent/task plus the end-to-end total.

    Attaches as a `callback` on each Task -- CrewAI calls this the moment
    a task finishes, passing the TaskOutput (which includes `.agent`, the
    agent's role string). Recording a timestamp per completed task and
    diffing consecutive timestamps gives an accurate per-agent duration
    for a sequential crew without needing to instrument the agents
    themselves.
    """

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.splits: list[tuple[str, float]] = []  # (agent_role, finished_at)

    def start(self) -> None:
        self.start_time = time.monotonic()
        self.splits = []

    def task_finished(self, task_output) -> None:
        self.splits.append((task_output.agent, time.monotonic()))

    def summary(self) -> dict:
        """Returns {"per_agent": {role: seconds}, "total": seconds}."""
        if self.start_time is None or not self.splits:
            return {"per_agent": {}, "total": 0.0}
        per_agent = {}
        prev = self.start_time
        for role, finished_at in self.splits:
            per_agent[role] = round(finished_at - prev, 1)
            prev = finished_at
        total = round(self.splits[-1][1] - self.start_time, 1)
        return {"per_agent": per_agent, "total": total}

    def print_summary(self) -> None:
        s = self.summary()
        print("\n=== Performance Summary ===")
        for role, seconds in s["per_agent"].items():
            print(f"{role:35s}: {seconds:7.1f}s")
        print("-" * 44)
        print(f"{'Total end-to-end':35s}: {s['total']:7.1f}s")


def print_usage_summary(usage: dict, model: str) -> None:
    """
    Prints token usage from crew.usage_metrics, plus a rough $ cost
    estimate when the model is a recognized paid provider (never for
    Ollama, which is always local/free).
    """
    print("\n=== Token Usage ===")
    print(f"{'Prompt tokens':35s}: {usage.get('prompt_tokens', 0):>10,}")
    print(f"{'Completion tokens':35s}: {usage.get('completion_tokens', 0):>10,}")
    print(f"{'Total tokens':35s}: {usage.get('total_tokens', 0):>10,}")
    print(f"{'Successful LLM requests':35s}: {usage.get('successful_requests', 0):>10,}")

    cost = estimate_cost_usd(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    if cost is not None:
        print(f"{'Estimated cost (USD)':35s}: {'$' + format(cost, '.4f'):>10}")
        print(f"\n({PRICING_SOURCE_NOTE})")
    else:
        print(f"{'Estimated cost':35s}: {'local/free':>10}")


def print_ground_truth_summary() -> None:
    """
    Prints what's ACTUALLY in the database right now, independent of
    anything any agent claimed in its final answer text. Local models
    can narrate a tool action without invoking it, or describe a tool
    result inaccurately in their own summary -- this print statement
    reads the database directly so a mismatch between what an agent
    said happened and what really happened is visible immediately in
    the console, without needing to scroll back through the full log.
    """
    findings = db.list_findings()
    by_status: dict[str, int] = {}
    for f in findings:
        by_status[f["status"]] = by_status.get(f["status"], 0) + 1

    print("\n=== Ground Truth (direct from database) ===")
    print(f"{'Assets tracked':35s}: {len(db.list_assets()):>10,}")
    print(f"{'Total findings recorded':35s}: {len(findings):>10,}")
    for status in ("new", "prioritized", "reported", "reviewed"):
        print(f"{'  status=' + status:35s}: {by_status.get(status, 0):>10,}")
    if not findings:
        print("(No findings exist in the database. If any agent's final "
              "answer claimed findings were recorded or prioritized, "
              "that claim did not actually happen.)")


def seed_assets_from_yaml(path: str) -> None:
    """Optional convenience loader -- avoids requiring PyYAML by using a
    minimal hand-rolled parser for the simple flat structure we need."""
    import re
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"^\s*-\s+name:", text, flags=re.M)[1:]
    for block in blocks:
        block = "name:" + block
        fields = {}
        for line in block.splitlines():
            line = line.strip()
            if ":" in line:
                k, _, v = line.partition(":")
                fields[k.strip()] = v.strip().strip('"').strip("'")
        db.add_asset(
            name=fields.get("name", "unnamed"),
            vendor=fields.get("vendor", ""),
            product=fields.get("product", ""),
            version=fields.get("version", ""),
            exposure=fields.get("exposure", "internal"),
            criticality=fields.get("criticality", "medium"),
            notes=fields.get("notes", ""),
        )


def build_tasks(agents: dict, timer: Timer) -> list[Task]:
    t_ingest = Task(
        description=(
            "Call list_assets to get every tracked asset. For each asset, "
            "call search_recent_cves using the asset's vendor and product "
            "as the keyword (try a couple of reasonable keyword variants "
            "if the first search returns nothing), looking back 30 days. "
            "Collect all candidate CVE IDs across all assets, then call "
            "check_kev once with the full list to see which are actively "
            "exploited. Report the full candidate list with KEV status "
            "attached, grouped by asset."
        ),
        expected_output=(
            "For each asset: asset_id, and a list of candidate CVEs each "
            "with cve_id, cvss, severity, summary, and in_kev true/false."
        ),
        agent=agents["ingest"],
        callback=timer.task_finished,
    )

    t_relevance = Task(
        description=(
            "For each candidate CVE against each asset from the ingest "
            "step, decide whether it is GENUINELY relevant -- check "
            "whether the product name in the CVE summary actually matches "
            "the asset's product (not just a substring match), and whether "
            "the asset's version falls in the affected range if one is "
            "mentioned.\n\n"
            "Build a complete list of every CVE you judge genuinely "
            "relevant, across ALL assets, then call bulk_record_findings "
            "EXACTLY ONCE with that complete list -- do not call "
            "record_finding repeatedly in a loop, and do not describe "
            "what you would record without actually calling the tool. "
            "Simply writing out a list of findings in your final answer "
            "does NOT record them -- only an actual bulk_record_findings "
            "tool call writes anything to the database. Skip CVEs that "
            "only keyword-matched but clearly apply to a different "
            "product or an unaffected version, and note why in your "
            "summary instead of including them."
        ),
        expected_output=(
            "The JSON result of the bulk_record_findings call, showing "
            "recorded_finding_ids and any skipped entries, plus a short "
            "note on any candidates that were deliberately excluded and "
            "why. Your summary text must accurately reflect what the "
            "tool actually returned -- never state something was "
            "recorded if the tool result shows it was skipped."
        ),
        agent=agents["relevance"],
        context=[t_ingest],
        callback=timer.task_finished,
    )

    t_exploitability = Task(
        description=(
            "Call list_findings with status='new' YOURSELF to get every "
            "newly-recorded finding -- do not reuse any numbers "
            "mentioned in earlier context as finding_ids; the only valid "
            "finding_ids are the 'id' values list_findings actually "
            "returns to you in this task. For EACH ONE, decide a "
            "priority_tier (HIGH/MEDIUM/LOW) by weighing: CVSS score, "
            "whether in_kev is true (this should push toward HIGH almost "
            "regardless of CVSS), and the asset's own exposure "
            "('internet-facing' assets deserve higher priority than "
            "'internal' ones at the same CVSS) and criticality.\n\n"
            "Once you have a decision for every single finding_id "
            "returned by list_findings, call bulk_update_priority EXACTLY "
            "ONCE with the complete list of decisions -- do not call "
            "update_priority repeatedly in a loop, and do not stop after "
            "only some findings. If list_findings returned 15 findings, "
            "your bulk_update_priority call must contain 15 entries, one "
            "per finding_id, with no exceptions."
        ),
        expected_output=(
            "The JSON result of the bulk_update_priority call, showing "
            "every finding_id that was successfully applied. The "
            "'applied' list length must equal the number of findings "
            "returned by list_findings(status='new'). Your summary text "
            "must exactly match what the tool actually returned -- if "
            "the tool result shows an entry under 'skipped', you must "
            "report it as skipped, never as applied."
        ),
        agent=agents["exploitability"],
        context=[t_relevance],
        callback=timer.task_finished,
    )

    t_communicate = Task(
        description=(
            "Call list_findings with status='prioritized' and "
            "min_priority='MEDIUM' so you have the real data in front of "
            "you. A complete, guaranteed-accurate technical table of "
            "every one of these findings is ALREADY generated separately "
            "by code and will be shown to the reader automatically -- "
            "you do NOT need to, and must NOT, reproduce a full "
            "CVE-by-CVE technical table yourself. Do not group findings "
            "under a range like 'CVE-X to CVE-Y' and do not summarize "
            "away individual items -- simply don't attempt a full "
            "enumeration at all, since it already exists elsewhere.\n\n"
            "Your ONLY job is to write ONE executive summary: a short, "
            "plain-language narrative for non-technical stakeholders. No "
            "CVE IDs, no CVSS numbers, no jargon in the main text. "
            "Cover: what's genuinely at risk across the environment right "
            "now, which 1-3 issues are the most urgent and why (you may "
            "name a CVE ID here if it's one of the few most critical "
            "ones -- that's fine, just don't attempt to list everything), "
            "how bad it could get if ignored, and what happens if nothing "
            "is done. Keep it to a few short paragraphs -- this is a "
            "summary for someone who will never look at the technical "
            "table, not a restatement of it.\n\n"
            "Do NOT call any tool other than list_findings. You do not "
            "need to mark anything as reported -- that bookkeeping is "
            "handled automatically in code after you finish. Your "
            "output should be the summary prose and nothing else."
        ),
        expected_output=(
            "One short, plain-language executive summary of a few "
            "paragraphs. No CVE-by-CVE table, no JSON, no tool-call "
            "confirmations -- just the narrative."
        ),
        agent=agents["communicator"],
        context=[t_exploitability],
        callback=timer.task_finished,
    )

    return [t_ingest, t_relevance, t_exploitability, t_communicate]


def main():
    parser = argparse.ArgumentParser(description="Run the Coverage_Crew pipeline")
    parser.add_argument("--provider", default="ollama", choices=["anthropic", "openai", "ollama"])
    parser.add_argument("--model", default=None,
                         help="Model name override, e.g. 'llama3.1' for ollama, "
                              "'claude-sonnet-4-6' for anthropic")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                         help="Base URL for a local or remote Ollama server")
    parser.add_argument("--seed-assets", default=None, help="Optional YAML file of assets to seed before running")
    args = parser.parse_args()

    db.init_db()

    if args.seed_assets:
        seed_assets_from_yaml(args.seed_assets)
        print(f"Seeded assets from {args.seed_assets}")

    existing = db.list_assets()
    if not existing:
        print("No assets in the registry. Add some first via the MCP server "
              "(add_asset) or --seed-assets, then re-run.")
        sys.exit(1)

    llm = build_llm(args.provider, model=args.model, base_url=args.ollama_url)
    agents = {
        "ingest": build_ingest_agent(llm),
        "relevance": build_relevance_agent(llm),
        "exploitability": build_exploitability_agent(llm),
        "communicator": build_communicator_agent(llm),
    }
    timer = Timer()
    tasks = build_tasks(agents, timer)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    timer.start()
    result = crew.kickoff()
    timer.print_summary()

    # Post-run bookkeeping, done in code rather than by an agent.
    # Both of these are mechanical questions with a single correct
    # answer that the database (or a published catalog) already knows,
    # and both were previously delegated to the Communicator with poor
    # results: KEV membership was being asserted for CVEs not in the
    # catalog, and "mark these N findings reported" reliably came back
    # partial (2 of 18 in the last observed run). Neither needs an LLM.
    kev_sync = db.sync_kev_flags()
    if kev_sync.get("catalog_unavailable"):
        print("\n[warn] CISA KEV catalog unavailable -- in_kev flags left as-is.")
    elif kev_sync.get("corrected"):
        print(f"\n[info] Corrected {kev_sync['corrected']} in_kev flag(s) against the real CISA KEV catalog.")

    reported = db.mark_all_reported(min_priority="MEDIUM")
    if reported.get("applied"):
        print(f"[info] Marked {len(reported['applied'])} finding(s) as reported.")

    print_ground_truth_summary()

    resolved_model = args.model or (
        "qwen3:8b" if args.provider == "ollama" else
        "claude-sonnet-4-6" if args.provider == "anthropic" else "gpt-4o"
    )

    usage_dict = {}
    if crew.usage_metrics:
        usage_dict = {
            "prompt_tokens": crew.usage_metrics.prompt_tokens,
            "completion_tokens": crew.usage_metrics.completion_tokens,
            "total_tokens": crew.usage_metrics.total_tokens,
            "successful_requests": crew.usage_metrics.successful_requests,
        }
        print_usage_summary(usage_dict, resolved_model)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "latest_briefing.html"
    write_html_report(
        str(result),
        provider=args.provider,
        model=resolved_model,
        out_path=out_path,
        timings=timer.summary(),
        usage=usage_dict,
    )
    print(f"\nBriefing written to {out_path}")


if __name__ == "__main__":
    main()
