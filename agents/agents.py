"""
agents/agents.py

Four agents, each with a narrow job: small single-responsibility agents
coordinated sequentially, rather than one do-everything agent.

The division of labour follows one rule learned the hard way in testing:
the LLM is given the genuinely judgment-shaped questions (does this CVE
actually apply to this asset? how urgent is it really, given exposure?
what does a non-technical reader need to hear?), and everything with a
single authoritative answer -- CVSS, KEV membership, status bookkeeping,
exhaustive enumeration -- is handled in code. Where an agent must touch
many records, it is given a *bulk* tool, because asking a local model to
issue the same tool call 15+ times in a row reliably produces partial
results or narration instead of execution.
"""
from __future__ import annotations

import os

from crewai import Agent, LLM

from tools.crew_tools import (
    ListAssetsTool,
    AddAssetTool,
    SearchCvesTool,
    CheckKevTool,
    RecordFindingTool,
    BulkRecordFindingsTool,
    UpdatePriorityTool,
    BulkUpdatePriorityTool,
    ListFindingsTool,
    MarkReportedTool,
    BulkMarkReportedTool,
)


def build_llm(provider: str = "ollama", model: str | None = None,
              base_url: str = "http://localhost:11434") -> LLM:
    """
    provider: 'ollama' (default, local/free), 'anthropic', or 'openai'.

    Model resolution order: explicit `model` arg > OLLAMA_MODEL /
    ANTHROPIC_MODEL / OPENAI_MODEL env var > built-in default below.
    This means you can switch models with zero code changes:
        --model qwen3:8b
        OLLAMA_MODEL=qwen3:8b python launchers/run_crew.py

    Tool-calling reliability on Ollama varies a lot by model, and this
    entire crew depends on tool calls -- every agent's job IS calling
    tools. From a typical `ollama list`:

        qwen3:8b / qwen3:4b   -- confirmed native tool-calling support
                                  in Ollama's docs; 8b is the safer
                                  default, 4b is faster for iterating.
        llama3.1 / llama3.2   -- also confirmed tool-calling support,
                                  if you have them.
        llama3.2:1b           -- too small (1B) for reliable multi-step
                                  tool use; expect it to narrate tool
                                  calls as text instead of invoking them.
        gemma2/gemma3 family  -- tool-calling support is inconsistent
                                  across Gemma versions in Ollama; test
                                  before relying on it.
        FenkoHQ/Foundation-Sec-8B -- a security-domain fine-tune of
                                  Llama-3.1-8B; may have degraded
                                  tool-calling relative to base Llama
                                  3.1 since fine-tuning wasn't tool-call
                                  focused. Worth trying, verify it
                                  actually invokes tools rather than
                                  describing them in prose.
    """
    if provider == "ollama":
        resolved_model = model or os.environ.get("OLLAMA_MODEL", "qwen3:8b")
        return LLM(model=f"ollama/{resolved_model}", base_url=base_url, temperature=0.2)
    if provider == "anthropic":
        resolved = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return LLM(model=resolved, temperature=0.2)
    if provider == "openai":
        resolved = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        return LLM(model=resolved, temperature=0.2)
    raise ValueError(f"unknown provider: {provider}")


def build_ingest_agent(llm) -> Agent:
    return Agent(
        role="Threat Intel Ingest Analyst",
        goal=(
            "For every tracked asset, pull recent CVEs that might apply "
            "based on vendor/product keywords, and check each candidate "
            "against the CISA KEV catalog for active-exploitation status."
        ),
        backstory=(
            "A methodical researcher who casts a slightly wide net on "
            "purpose -- better to hand the Relevance Analyst a few "
            "borderline candidates than to silently filter out something "
            "that turns out to matter."
        ),
        tools=[ListAssetsTool(), SearchCvesTool(), CheckKevTool()],
        llm=llm,
        verbose=True,
    )


def build_relevance_agent(llm) -> Agent:
    return Agent(
        role="Relevance Analyst",
        goal=(
            "For each candidate CVE, judge whether it genuinely applies "
            "to a specific asset -- checking version ranges and product "
            "name variants carefully rather than relying on a keyword "
            "match alone -- and record only the ones that are truly "
            "relevant, with a clear rationale, via a single bulk call."
        ),
        backstory=(
            "Has seen too many alert queues drowned in CVEs that "
            "technically matched a keyword search but applied to a "
            "completely different product or an unaffected version range. "
            "Would rather record fewer, correct findings than flood the "
            "registry with noise. Also knows that describing what should "
            "be recorded is not the same as actually recording it -- the "
            "database only reflects what was truly saved through a tool "
            "call, not what was mentioned in a summary."
        ),
        tools=[ListAssetsTool(), BulkRecordFindingsTool(), RecordFindingTool()],
        llm=llm,
        verbose=True,
    )


def build_exploitability_agent(llm) -> Agent:
    return Agent(
        role="Exploitability & Exposure Analyst",
        goal=(
            "For every newly recorded finding, weigh CVSS score, KEV "
            "(active exploitation) status, and the asset's own exposure "
            "and criticality together to assign a HIGH/MEDIUM/LOW "
            "priority tier -- not just a mechanical CVSS threshold -- "
            "then apply every decision in a single bulk_update_priority "
            "call."
        ),
        backstory=(
            "A vulnerability management veteran who has watched teams "
            "burn a week patching a 9.8 CVSS bug on an isolated internal "
            "dev box while a 6.5 CVSS bug with a public exploit sat "
            "unpatched on an internet-facing production host. Prioritizes "
            "based on real-world risk, not just the raw score. Also "
            "knows better than to fire off dozens of individual update "
            "calls one at a time when a single batched call does the "
            "same job far more reliably."
        ),
        tools=[ListFindingsTool(), BulkUpdatePriorityTool(), UpdatePriorityTool()],
        llm=llm,
        verbose=True,
    )


def build_communicator_agent(llm) -> Agent:
    return Agent(
        role="Executive Communicator",
        goal=(
            "Write a short, plain-language executive summary naming the "
            "1-3 most critical issues, for non-technical stakeholders. A "
            "complete technical table of every finding is already "
            "generated separately -- do not attempt to reproduce or "
            "enumerate it, and do not do any status bookkeeping."
        ),
        backstory=(
            "Has learned that a non-technical reader needs 'are we at "
            "risk, and what happens if we do nothing,' not a CVE-by-CVE "
            "table -- so leaves the exhaustive technical enumeration to "
            "the system that already generates it perfectly, and focuses "
            "purely on synthesizing the few things that matter most. "
            "Deliberately carries a single read-only tool: bookkeeping "
            "like marking findings reported is mechanical work the "
            "database can do correctly on its own, and attempting it by "
            "hand only ever produced partial results."
        ),
        tools=[ListFindingsTool()],
        llm=llm,
        verbose=True,
    )
