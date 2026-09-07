#!/usr/bin/env python3
"""
launchers/mcp_server.py

Exposes the asset + findings registry as MCP tools so you can manage it
conversationally from Claude Desktop -- "add an asset", "what's new since
last week", "mark this finding reviewed" -- without touching SQLite
directly, and without re-running the full crew for a quick registry edit.

Run:
    python launchers/mcp_server.py

Claude Desktop config:
    {
      "mcpServers": {
        "coverage_crew": {
          "command": "python",
          "args": ["/absolute/path/to/launchers/mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP

from core import db
from tools.nvd_tools import search_recent_cves
from tools.kev_tools import check_kev

db.init_db()
mcp = FastMCP("CoverageCrew")


@mcp.tool()
def add_asset(name: str, vendor: str, product: str, version: str = "",
              exposure: str = "internal", criticality: str = "medium",
              notes: str = "") -> dict:
    """Add a new asset to the registry.

    exposure should be 'internet-facing' or 'internal'.
    criticality should be 'high', 'medium', or 'low'.
    """
    asset_id = db.add_asset(name, vendor, product, version, exposure, criticality, notes)
    return {"status": "added", "asset_id": asset_id}


@mcp.tool()
def list_assets() -> list[dict]:
    """List every asset currently tracked in the registry."""
    return db.list_assets()


@mcp.tool()
def search_cves(keyword: str, days_back: int = 30) -> list[dict]:
    """Search NVD for CVEs matching a keyword, published in the last N days."""
    return [r.to_dict() for r in search_recent_cves(keyword, days_back=days_back)]


@mcp.tool()
def check_kev_status(cve_ids: list[str]) -> dict:
    """Check whether CVE IDs appear in CISA's Known Exploited Vulnerabilities catalog."""
    return check_kev(cve_ids)


@mcp.tool()
def record_finding(asset_id: int, cve_id: str, relevance_score: float,
                    relevance_rationale: str, cvss: float | None = None,
                    severity: str | None = None) -> dict:
    """Record that a CVE is relevant to a specific asset."""
    finding_id = db.record_finding(asset_id, cve_id, relevance_score,
                                    relevance_rationale, cvss, severity)
    return {"status": "recorded", "finding_id": finding_id}


@mcp.tool()
def list_findings(status: str | None = None, min_priority: str | None = None) -> list[dict]:
    """List findings, optionally filtered by status ('new', 'prioritized',
    'reported', 'reviewed') and/or minimum priority ('LOW', 'MEDIUM', 'HIGH')."""
    return db.list_findings(status=status, min_priority=min_priority)


@mcp.tool()
def update_finding_priority(finding_id: int, priority_tier: str, in_kev: bool) -> dict:
    """Set the priority tier (HIGH/MEDIUM/LOW) and KEV flag for a finding."""
    db.update_priority(finding_id, priority_tier.upper(), in_kev)
    return {"status": "updated", "finding_id": finding_id}


@mcp.tool()
def mark_finding_reviewed(finding_id: int) -> dict:
    """Mark a finding as reviewed by a human -- distinct from 'reported',
    for tracking which findings someone has actually looked at."""
    db.mark_reviewed(finding_id)
    return {"status": "marked_reviewed", "finding_id": finding_id}


@mcp.tool()
def whats_new(since_status: str = "new") -> list[dict]:
    """Convenience tool: returns findings that haven't been reported yet,
    across all priority tiers. Good for 'what's new since last time'."""
    return db.list_findings(status=since_status)


if __name__ == "__main__":
    mcp.run()
