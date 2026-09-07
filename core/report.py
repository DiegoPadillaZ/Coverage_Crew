"""
core/report.py

Renders the crew's output as a single styled HTML report, with:

  1. A run-summary header pulled straight from the database --
     timestamp, model used, asset count, and a findings breakdown -- so
     the report is self-describing without cross-referencing the
     terminal log.

  2. A deterministic, code-generated "Verified Findings" table (see
     _render_verified_findings_section) -- the actual fix for the
     Communicator repeatedly under-enumerating or skipping findings
     entirely. The Communicator's own task was deliberately narrowed to
     writing ONLY a short executive-summary narrative (naming at most
     the 1-3 most critical issues) rather than exhaustively listing
     every finding in prose -- local models reliably shortcut on
     tedious, repetitive enumeration tasks, the same weakness that
     motivated the bulk_record_findings/bulk_update_priority tools
     upstream. Since there's no tool-call equivalent for "write
     complete prose", the fix here is architectural instead: stop
     asking the LLM to redundantly re-enumerate data that can be
     rendered perfectly and completely by code. This table is generated
     directly from the database, unconditionally, independent of
     anything in the model's narrative, and is what a human should
     treat as authoritative.

  3. A performance section showing how long each agent took and the
     total end-to-end time, when timing data is supplied.

  4. A data-integrity guard on the LLM's own narrative, in two
     directions: fabricating findings that don't exist when none were
     ever prioritized, and the mirror image -- claiming no findings
     exist at all when real ones do. There is deliberately no "X of Y
     findings not individually named" check: under the current design,
     the narrative is *supposed* to name only a handful of top issues,
     with the Verified Findings table (item 2) carrying full
     completeness -- flagging partial narrative coverage would treat
     the intended design as a bug on every run.

  5. A CVE-to-asset correspondence guard, applied at both layers: the
     database now rejects a misattributed finding at insertion time
     (see core.db.record_finding), and this module re-checks again at
     display time as a backstop for any record that predates that check
     or slipped through a failed lookup. A real CVE ID for one product
     has been observed recorded against a completely different asset
     (e.g. a Fortinet FortiOS CVE filed under a Windows RDP gateway),
     with a rationale carefully worded to sound plausible without ever
     naming the true source product -- so checking the model's own
     wording isn't sufficient. Both layers re-fetch the CVE's
     authoritative description directly from NVD by ID (independent of
     anything any agent saw) and check whether the asset's own product
     name actually appears in it. A lookup failure (offline,
     rate-limited) is treated as "unverifiable", never as a mismatch.
"""
from __future__ import annotations

import html as _html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import markdown as _markdown

from core import db
from core.pricing import estimate_cost_usd, PRICING_SOURCE_NOTE
from tools.nvd_tools import fetch_cve_by_id, verify_cve_product_match

CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{3,8}")
FINDING_ID_PATTERN = re.compile(r"Finding\s*ID:?\s*\**\s*(\d+)", re.IGNORECASE)

CSS = """
:root{
  --bg:#0b0f19;--surface:#131a2b;--border:#232f47;
  --accent:#00d4ff;--accent2:#7c3aed;--success:#10b981;--danger:#ef4444;--warn:#f59e0b;
  --text:#e2e8f0;--muted:#8b98ac;
  --sans:'Segoe UI',system-ui,-apple-system,sans-serif;--mono:'Consolas','SF Mono',monospace;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:var(--sans);margin:0;padding:0;line-height:1.6}
.wrap{max-width:900px;margin:0 auto;padding:2.5rem 1.5rem}
.hdr{border-bottom:1px solid var(--border);padding-bottom:1.5rem;margin-bottom:2rem}
.hdr h1{margin:0 0 .3rem;font-size:26px}
.hdr .sub{color:var(--muted);font-size:13px;font-family:var(--mono)}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.75rem;margin:1.5rem 0 2rem}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.9rem;text-align:center}
.stat .val{font-size:22px;font-weight:600;font-family:var(--mono);color:var(--accent)}
.stat .val.high{color:var(--danger)}
.stat .val.medium{color:var(--warn)}
.stat .val.low{color:var(--success)}
.stat .lbl{font-size:11px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.5px}
.warning-banner{background:rgba(239,68,68,0.1);border:1px solid var(--danger);border-radius:8px;padding:1rem 1.25rem;margin-bottom:1.5rem}
.warning-banner .wb-title{color:var(--danger);font-weight:600;font-size:13px;margin-bottom:.5rem}
.warning-banner ul{margin:0;padding-left:1.25rem}
.warning-banner li{font-size:12.5px;color:var(--text);margin-bottom:.4rem;font-family:var(--mono)}
.perf-section{margin-bottom:1.5rem}
.perf-title{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:.6rem}
.perf-table{width:100%;border-collapse:collapse;font-size:13px}
.perf-table td{padding:.4rem .7rem;border-bottom:1px solid var(--border)}
.perf-table td.agent-name{color:var(--text)}
.perf-table td.agent-time{text-align:right;font-family:var(--mono);color:var(--accent)}
.perf-table tr.perf-total td{border-top:1px solid var(--border);border-bottom:none;font-weight:600;color:var(--text)}
.perf-table tr.perf-total td.agent-time{color:var(--success)}
.usage-section{margin-bottom:1.5rem}
.usage-title{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:.6rem}
.usage-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.6rem}
.usage-stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.7rem;text-align:center}
.usage-stat .uval{font-size:17px;font-weight:600;font-family:var(--mono);color:var(--accent)}
.usage-stat .ulbl{font-size:10px;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:.5px}
.usage-note{font-size:10.5px;color:var(--muted);margin-top:.5rem;font-style:italic}
.assets-section{margin-bottom:1.5rem}
.assets-title{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:.6rem}
.assets-table{width:100%;border-collapse:collapse;font-size:12.5px}
.assets-table th{background:#0d1420;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.5px;padding:.5rem .6rem;text-align:left;border-bottom:1px solid var(--border)}
.assets-table td{padding:.5rem .6rem;border-bottom:1px solid var(--border);color:var(--text)}
.assets-table tr:last-child td{border-bottom:none}
.assets-table td.num{text-align:center;font-family:var(--mono)}
.assets-table td.num.nonzero{color:var(--accent);font-weight:600}
.exposure-tag{font-size:10px;padding:2px 7px;border-radius:4px;font-family:var(--mono)}
.exposure-tag.internet-facing{background:rgba(239,68,68,0.15);color:var(--danger)}
.exposure-tag.internal{background:rgba(139,152,172,0.15);color:var(--muted)}
.verified-section{margin-bottom:1.5rem;background:var(--surface);border:1px solid var(--accent2);border-radius:8px;padding:1rem 1.25rem}
.verified-title{font-size:12px;color:var(--accent2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:.75rem;font-weight:600}
.verified-table{width:100%;border-collapse:collapse;font-size:12.5px}
.verified-table th{background:#0d1420;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.5px;padding:.5rem .6rem;text-align:left;border-bottom:1px solid var(--border)}
.verified-table td{padding:.5rem .6rem;border-bottom:1px solid var(--border);color:var(--text);vertical-align:top}
.verified-table tr:last-child td{border-bottom:none}
.verified-table td.tier-high{color:var(--danger);font-weight:600}
.verified-table td.tier-medium{color:var(--warn);font-weight:600}
.verified-table td.tier-low{color:var(--success);font-weight:600}
.mismatch-flag{display:inline-block;margin-left:6px;font-size:10px;color:var(--danger);font-family:var(--mono)}
.verified-note{font-size:10.5px;color:var(--muted);margin-top:.75rem;font-style:italic}
.verified-empty{font-size:13px;color:var(--muted)}
.content{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:2rem}
.content h1,.content h2,.content h3{color:var(--text)}
.content h1{font-size:20px;border-bottom:1px solid var(--border);padding-bottom:.5rem;margin-top:2rem}
.content h2{font-size:17px;margin-top:1.75rem;color:var(--accent)}
.content h3{font-size:14px;margin-top:1.25rem}
.content p,.content li{font-size:14px;color:var(--text)}
.content code{background:#0d1420;padding:2px 6px;border-radius:4px;font-family:var(--mono);font-size:12px;color:var(--accent)}
.content table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:13px}
.content th,.content td{border:1px solid var(--border);padding:.5rem .7rem;text-align:left}
.content th{background:#0d1420;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.content tr:nth-child(even){background:rgba(255,255,255,0.02)}
.content strong{color:var(--accent)}
.footer{margin-top:2rem;text-align:center;color:var(--muted);font-size:11px;font-family:var(--mono)}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Coverage_Crew — Run Summary</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
  <div class="hdr">
    <h1>Coverage_Crew — Run Summary</h1>
    <div class="sub">Generated {timestamp} UTC &middot; provider: {provider} &middot; model: {model}</div>
  </div>

  {warning_banner}

  {verified_findings_section}

  <div class="summary-grid">
    <div class="stat"><div class="val">{asset_count}</div><div class="lbl">Assets tracked</div></div>
    <div class="stat"><div class="val">{total_findings}</div><div class="lbl">Total findings</div></div>
    <div class="stat"><div class="val high">{high_count}</div><div class="lbl">High priority</div></div>
    <div class="stat"><div class="val medium">{medium_count}</div><div class="lbl">Medium priority</div></div>
    <div class="stat"><div class="val low">{low_count}</div><div class="lbl">Low priority</div></div>
    <div class="stat"><div class="val">{kev_count}</div><div class="lbl">In CISA KEV</div></div>
  </div>

  {assets_section}

  {performance_section}

  {usage_section}

  <div class="content">
    {body}
  </div>

  <div class="footer">Coverage_Crew &middot; findings require human verification before action</div>
</div>
</body>
</html>
"""


def _run_summary_stats() -> dict:
    assets = db.list_assets()
    findings = db.list_findings()
    tier_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    kev_count = 0
    for f in findings:
        tier = (f.get("priority_tier") or "").upper()
        if tier in tier_counts:
            tier_counts[tier] += 1
        if f.get("in_kev"):
            kev_count += 1
    return {
        "asset_count": len(assets),
        "total_findings": len(findings),
        "high_count": tier_counts["HIGH"],
        "medium_count": tier_counts["MEDIUM"],
        "low_count": tier_counts["LOW"],
        "kev_count": kev_count,
    }


NO_FINDINGS_CLAIM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bno\s+(?:new\s+|critical\s+|high[- ]priority\s+|prioriti[sz]ed\s+)?"
        r"(?:findings?|vulnerabilit(?:y|ies)|issues?|risks?)\b",
        r"\bno\s+findings?\s+(?:were\s+)?(?:found|identified|matched|reported)\b",
        r"\bnothing\s+(?:was\s+)?(?:found|identified|to\s+report)\b",
        r"\ball\s+(?:systems|assets)\s+are\s+(?:currently\s+)?secure\b",
        r"\bno\s+action\s+(?:is\s+)?(?:required|needed)\b",
        r"\bdid\s+not\s+(?:find|identify)\s+any\b",
    )
]


def _claims_no_findings(markdown_text: str) -> bool:
    """
    True when the narrative explicitly asserts that nothing was found.

    Used instead of "the narrative mentions no CVE IDs", because the
    Communicator is now deliberately instructed to keep CVE IDs out of
    the executive summary -- absence of identifiers is the intended
    shape of a correct report, whereas an explicit all-clear claim
    while real findings sit in the registry is a genuine failure.
    """
    return any(p.search(markdown_text) for p in NO_FINDINGS_CLAIM_PATTERNS)


def _detect_data_integrity_issues(markdown_text: str) -> list[str]:
    """
    Cross-checks CVE IDs and Finding IDs mentioned in the report text
    against what actually exists in the registry. Returns a list of
    human-readable warning strings; empty if nothing looks fabricated.
    """
    warnings: list[str] = []

    all_findings = db.list_findings()
    all_cve_ids = {f["cve_id"] for f in all_findings}
    all_finding_ids = {f["id"] for f in all_findings}
    # Deliberately not filtered by status='prioritized': findings are
    # moved to 'reported' by the post-run bookkeeping step in the
    # launcher, which may already have run by the time a report is
    # rendered. A status filter here would treat a fully successful run
    # (findings correctly prioritized AND correctly marked reported) as
    # if nothing had ever been prioritized. What actually matters is
    # whether the priority tier was ever set, regardless of which stage
    # the finding has since moved to.
    reportable = db.list_findings(min_priority="MEDIUM")

    mentioned_cves = set(CVE_PATTERN.findall(markdown_text))
    mentioned_finding_ids = {int(m) for m in FINDING_ID_PATTERN.findall(markdown_text)}

    if not reportable and (mentioned_cves or mentioned_finding_ids):
        warnings.append(
            "No findings in the database have ever reached MEDIUM or "
            "higher priority, but the text below references "
            f"{len(mentioned_cves)} CVE ID(s) and {len(mentioned_finding_ids)} "
            "Finding ID(s). This content was very likely fabricated by the "
            "model rather than grounded in actual tool output -- treat "
            "everything below as unverified until confirmed against the "
            "database directly."
        )
        return warnings

    # The mirror-image failure: real, genuinely qualifying findings exist
    # in the database, but the narrative explicitly asserts that nothing
    # was found -- an incorrect "all clear" claim. Observed in practice:
    # the Communicator skipping list_findings entirely and defaulting to
    # a plausible-sounding empty-report template. This is treated as more
    # severe than over-reporting, since a security tool silently claiming
    # safety when real findings exist is the worst possible failure.
    #
    # Deliberately keyed on an explicit no-findings CLAIM rather than on
    # the mere absence of CVE IDs: the Communicator's task now forbids
    # CVE IDs in the executive narrative by design (the Verified Findings
    # table above carries the identifiers), so "no CVE IDs mentioned" is
    # the expected shape of a correct report, not a symptom of failure.
    if reportable and _claims_no_findings(markdown_text):
        warnings.append(
            f"The database has {len(reportable)} finding(s) at MEDIUM or "
            "higher priority, but the narrative below states that no "
            "findings exist. This is very likely a failure to retrieve "
            "real findings, not a genuinely clean result -- the Verified "
            "Findings table above reflects the actual current state."
        )
        return warnings

    # Note: there is deliberately no "X of Y findings not mentioned"
    # partial-coverage check here. The Communicator's task now
    # intentionally names only the 1-3 most critical issues in its
    # narrative rather than enumerating every finding -- the Verified
    # Findings table above is the completeness mechanism, not the
    # narrative. Flagging partial narrative coverage would treat the
    # intended design as a bug on every single run.

    unknown_cves = mentioned_cves - all_cve_ids
    if unknown_cves:
        warnings.append(
            f"{len(unknown_cves)} CVE ID(s) mentioned in this report do not "
            f"exist in any recorded finding: {', '.join(sorted(unknown_cves))}. "
            "Verify these independently before acting on them."
        )

    unknown_finding_ids = mentioned_finding_ids - all_finding_ids
    if unknown_finding_ids:
        ids_str = ", ".join(str(i) for i in sorted(unknown_finding_ids))
        warnings.append(
            f"Finding ID(s) referenced in this report do not exist in the "
            f"database: {ids_str}."
        )

    warnings.extend(_check_cve_asset_correspondence(mentioned_cves, all_findings))

    return warnings


def _product_terms(product: str) -> list[str]:
    """
    Generates a few normalized candidate strings to search for in an
    NVD description. Handles the common mismatches between how a
    product is stored (e.g. 'http_server', 'struts2') and how NVD's own
    prose describes it (e.g. 'HTTP Server', 'Struts 2' or 'Struts'):
    underscores become spaces, and a trailing version-like digit is
    also tried stripped off.
    """
    if not product:
        return []
    normalized = product.replace("_", " ").strip().lower()
    terms = {normalized, normalized.replace(" ", "")}
    stripped = re.sub(r"\s*\d+$", "", normalized).strip()
    if stripped:
        terms.add(stripped)
    return [t for t in terms if t]


def _check_cve_asset_correspondence(mentioned_cves: set[str], all_findings: list[dict]) -> list[str]:
    """
    For every FINDING (not every unique CVE) whose CVE is mentioned,
    re-fetches its authoritative description directly from NVD
    (independent of anything any agent saw) and checks whether that
    finding's own asset product name actually appears in it. This is a
    display-time backstop for the record-time check now in core.db --
    it catches a bad record that predates that check, or one where the
    verification lookup happened to fail when the finding was first
    recorded.

    Deliberately iterates every matching finding row rather than
    collapsing to one entry per CVE ID first: the same CVE ID can
    legitimately (or erroneously) be recorded against more than one
    asset -- e.g. a correct record from an earlier run coexisting with
    a fresh misattribution from a later one. Collapsing by CVE ID first
    would silently validate only whichever record happened to survive
    the collapse and never even look at the other, which is exactly how
    a real misattribution slipped through undetected in practice: the
    correct record for the CVE overwrote the wrong one in a dict keyed
    by cve_id, and only the correct one ever got checked.

    NVD lookups are still cached per CVE ID so duplicate-asset findings
    for the same CVE don't trigger redundant network calls. A lookup
    failure (offline, rate-limited, unknown ID) is skipped rather than
    treated as a mismatch, since "couldn't verify" and "verified wrong"
    are different things and only one of them is actionable.
    """
    warnings: list[str] = []
    cve_cache: dict[str, object] = {}
    matching_findings = [f for f in all_findings if f["cve_id"] in mentioned_cves]

    for finding in matching_findings:
        cve_id = finding["cve_id"]
        product = finding.get("asset_product") or ""
        asset_name = finding.get("asset_name") or "unknown asset"
        candidates = _product_terms(product)
        if not candidates:
            continue

        if cve_id not in cve_cache:
            cve_cache[cve_id] = fetch_cve_by_id(cve_id)
        record = cve_cache[cve_id]
        if record is None or not record.summary:
            continue  # unverifiable, not a mismatch

        desc = record.summary.lower()
        if not any(term in desc for term in candidates):
            warnings.append(
                f"{cve_id} is filed under '{asset_name}' ({product}), but "
                "NVD's own description of this CVE does not mention that "
                "product. This looks like a real CVE ID matched to the "
                "wrong asset -- verify independently before acting on it."
            )

    return warnings


def _render_verified_findings_section() -> str:
    """
    A deterministic, code-generated table of every finding at MEDIUM+
    priority, built directly from the database rather than trusted to
    the Communicator's own prose. This exists because the Communicator
    has been observed skipping list_findings entirely and defaulting to
    an incorrect "no findings" narrative, or writing up only a fraction
    of what actually qualified -- generating this section in code
    guarantees completeness by construction, rather than depending on
    an LLM's willingness to enumerate everything correctly every time.
    It renders unconditionally, independent of anything in the model's
    own markdown output, and is the section a human should treat as
    authoritative.

    Each row is independently re-verified against NVD's own CVE
    description as a second line of defense against misattribution that
    slipped past the record-time check in core.db (e.g. data recorded
    before that check existed, or a lookup that failed at record time
    but succeeds now) -- a verified mismatch is flagged inline rather
    than hidden.
    """
    findings = db.list_findings(min_priority="MEDIUM")
    if not findings:
        return (
            '<div class="verified-section">'
            '<div class="verified-title">Verified Findings — source of truth</div>'
            '<p class="verified-empty">No findings currently meet MEDIUM or higher priority.</p>'
            "</div>"
        )

    tier_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    findings_sorted = sorted(
        findings,
        key=lambda f: tier_order.get((f.get("priority_tier") or "").upper(), 9),
    )

    rows = []
    for f in findings_sorted:
        cve_id = f["cve_id"]
        product = f.get("asset_product") or ""
        match = verify_cve_product_match(cve_id, product)
        flag = '<span class="mismatch-flag">⚠ unverified match</span>' if match is False else ""
        tier = (f.get("priority_tier") or "").upper()
        cvss_str = f"{f['cvss']:.1f}" if f.get("cvss") is not None else "—"
        rationale = (f.get("relevance_rationale") or "")[:150]
        rows.append(
            "<tr>"
            f'<td class="tier-{tier.lower()}">{_html.escape(tier)}</td>'
            f"<td>{_html.escape(cve_id)}</td>"
            f"<td>{_html.escape(f.get('asset_name', ''))}{flag}</td>"
            f"<td>{cvss_str}</td>"
            f"<td>{'Yes' if f.get('in_kev') else 'No'}</td>"
            f"<td>{_html.escape(rationale)}</td>"
            "</tr>"
        )

    return (
        '<div class="verified-section">'
        f'<div class="verified-title">Verified Findings ({len(findings)}) — source of truth</div>'
        '<table class="verified-table">'
        "<tr><th>Priority</th><th>CVE ID</th><th>Asset</th><th>CVSS</th><th>KEV</th><th>Rationale</th></tr>"
        f"{''.join(rows)}"
        "</table>"
        '<p class="verified-note">Generated directly from the registry, independent of the narrative below -- '
        "always complete and current regardless of what the model wrote.</p>"
        "</div>"
    )


def _render_warning_banner(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li>{_html.escape(w)}</li>" for w in warnings)
    return (
        '<div class="warning-banner">'
        '<div class="wb-title">⚠ Data Integrity Warning</div>'
        f"<ul>{items}</ul>"
        "</div>"
    )


def _render_performance_section(timings: Optional[dict]) -> str:
    if not timings or not timings.get("per_agent"):
        return ""
    rows = "".join(
        f'<tr><td class="agent-name">{_html.escape(role)}</td>'
        f'<td class="agent-time">{seconds:.1f}s</td></tr>'
        for role, seconds in timings["per_agent"].items()
    )
    total = timings.get("total", 0.0)
    rows += (
        '<tr class="perf-total"><td>Total end-to-end</td>'
        f'<td class="agent-time">{total:.1f}s</td></tr>'
    )
    return (
        '<div class="perf-section">'
        '<div class="perf-title">Performance</div>'
        f'<table class="perf-table">{rows}</table>'
        "</div>"
    )


def _render_usage_section(usage: Optional[dict], model: str) -> str:
    """
    usage is expected to have: prompt_tokens, completion_tokens,
    total_tokens, successful_requests (matching CrewAI's UsageMetrics
    shape). Cost is estimated via core.pricing and only shown for
    recognized non-Ollama models.
    """
    if not usage:
        return ""
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    requests = usage.get("successful_requests", 0)

    stats = [
        (f"{prompt_tokens:,}", "Prompt tokens"),
        (f"{completion_tokens:,}", "Completion tokens"),
        (f"{total_tokens:,}", "Total tokens"),
        (f"{requests:,}", "LLM requests"),
    ]

    cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
    note = ""
    if cost is not None:
        stats.append((f"${cost:.4f}", "Est. cost (USD)"))
        note = f'<div class="usage-note">{_html.escape(PRICING_SOURCE_NOTE)}</div>'
    else:
        note = '<div class="usage-note">Local Ollama model -- no API cost.</div>'

    grid = "".join(
        f'<div class="usage-stat"><div class="uval">{val}</div><div class="ulbl">{lbl}</div></div>'
        for val, lbl in stats
    )
    return (
        '<div class="usage-section">'
        '<div class="usage-title">Token Usage</div>'
        f'<div class="usage-grid">{grid}</div>'
        f"{note}"
        "</div>"
    )


def _render_assets_section() -> str:
    """
    Lists every asset currently tracked, with a per-asset findings
    breakdown. Exists so a report is always cross-checkable at a glance:
    if the narrative text mentions an asset name that doesn't appear in
    this table, that's an immediate, visible sign something was
    fabricated -- no need to go dig through the database separately.
    """
    assets = db.list_assets()
    if not assets:
        return ""

    all_findings = db.list_findings()
    by_asset: dict[int, dict] = {}
    for f in all_findings:
        aid = f["asset_id"]
        bucket = by_asset.setdefault(aid, {"total": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0})
        bucket["total"] += 1
        tier = (f.get("priority_tier") or "").upper()
        if tier in ("HIGH", "MEDIUM", "LOW"):
            bucket[tier] += 1

    def cell(n: int) -> str:
        cls = "num nonzero" if n else "num"
        return f'<td class="{cls}">{n}</td>'

    rows = []
    for a in assets:
        counts = by_asset.get(a["id"], {"total": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0})
        exposure = a.get("exposure", "internal")
        exposure_html = f'<span class="exposure-tag {_html.escape(exposure)}">{_html.escape(exposure)}</span>'
        version = f" {a['version']}" if a.get("version") else ""
        product_str = f"{_html.escape(a.get('vendor', ''))} {_html.escape(a.get('product', ''))}{_html.escape(version)}"
        rows.append(
            "<tr>"
            f"<td>{_html.escape(a['name'])}</td>"
            f"<td>{product_str}</td>"
            f"<td>{exposure_html}</td>"
            f"<td>{_html.escape(a.get('criticality', ''))}</td>"
            f"{cell(counts['total'])}"
            f"{cell(counts['HIGH'])}"
            f"{cell(counts['MEDIUM'])}"
            f"{cell(counts['LOW'])}"
            "</tr>"
        )

    return (
        '<div class="assets-section">'
        f'<div class="assets-title">Evaluated Assets ({len(assets)})</div>'
        '<table class="assets-table">'
        "<tr><th>Asset</th><th>Product</th><th>Exposure</th><th>Criticality</th>"
        "<th>Findings</th><th>High</th><th>Medium</th><th>Low</th></tr>"
        f"{''.join(rows)}"
        "</table>"
        "</div>"
    )


def render_html_report(markdown_text: str, provider: str, model: str,
                        timings: Optional[dict] = None,
                        usage: Optional[dict] = None) -> str:
    """
    Converts the crew's markdown output to HTML and wraps it with a
    run-summary header, a deterministic Verified Findings table (always
    complete and correct, independent of the LLM's own output), a table
    of every evaluated asset with its findings breakdown, an optional
    performance breakdown, an optional token-usage/cost breakdown, and a
    data-integrity warning banner if the LLM's narrative text appears to
    disagree with what's actually in the registry.
    """
    body_html = _markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    stats = _run_summary_stats()
    warnings = _detect_data_integrity_issues(markdown_text)
    return HTML_TEMPLATE.format(
        css=CSS,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        provider=_html.escape(provider),
        model=_html.escape(model),
        warning_banner=_render_warning_banner(warnings),
        verified_findings_section=_render_verified_findings_section(),
        assets_section=_render_assets_section(),
        performance_section=_render_performance_section(timings),
        usage_section=_render_usage_section(usage, model),
        body=body_html,
        **stats,
    )


def write_html_report(markdown_text: str, provider: str, model: str, out_path: Path,
                       timings: Optional[dict] = None,
                       usage: Optional[dict] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_html_report(markdown_text, provider, model, timings=timings, usage=usage),
        encoding="utf-8",
    )
