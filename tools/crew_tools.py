"""
tools/crew_tools.py

CrewAI BaseTool wrappers around core.db, tools.nvd_tools, and
tools.kev_tools. As with the underlying modules, no CrewAI import exists
in those modules themselves -- this file is the only place framework
and business logic meet, which keeps the same functions reusable for the
MCP server.
"""
from __future__ import annotations

import json
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from core import db
from tools.nvd_tools import search_recent_cves
from tools.kev_tools import check_kev


# ---------------------------------------------------------------- Assets ----

class ListAssetsInput(BaseModel):
    pass


class ListAssetsTool(BaseTool):
    name: str = "list_assets"
    description: str = (
        "List every asset currently tracked in the registry, with its "
        "vendor, product, version, exposure, and criticality. Call this "
        "first to know what to search CVEs for."
    )
    args_schema: Type[BaseModel] = ListAssetsInput

    def _run(self) -> str:
        return json.dumps(db.list_assets())


class AddAssetInput(BaseModel):
    name: str = Field(..., description="Short human name for the asset, e.g. 'web-prod-01'")
    vendor: str = Field(..., description="Vendor name, e.g. 'Apache'")
    product: str = Field(..., description="Product name, matching how NVD would refer to it, e.g. 'http_server'")
    version: str = Field("", description="Version string, e.g. '2.4.49'")
    exposure: str = Field("internal", description="'internet-facing' or 'internal'")
    criticality: str = Field("medium", description="'high', 'medium', or 'low'")
    notes: str = Field("", description="Free-text notes")


class AddAssetTool(BaseTool):
    name: str = "add_asset"
    description: str = "Add a new asset to the registry."
    args_schema: Type[BaseModel] = AddAssetInput

    def _run(self, name: str, vendor: str, product: str, version: str = "",
              exposure: str = "internal", criticality: str = "medium", notes: str = "") -> str:
        asset_id = db.add_asset(name, vendor, product, version, exposure, criticality, notes)
        return json.dumps({"status": "added", "asset_id": asset_id})


# ------------------------------------------------------------------ NVD ----

class SearchCvesInput(BaseModel):
    keyword: str = Field(..., description="Product/vendor keyword to search, e.g. 'jenkins'")
    days_back: int = Field(30, description="How many days back to search for published CVEs")


class SearchCvesTool(BaseTool):
    name: str = "search_recent_cves"
    description: str = (
        "Search NVD for CVEs matching a keyword (usually a product name) "
        "published within the given number of days. Returns CVE ID, "
        "published date, CVSS score, severity, and a summary."
    )
    args_schema: Type[BaseModel] = SearchCvesInput

    def _run(self, keyword: str, days_back: int = 30) -> str:
        records = search_recent_cves(keyword, days_back=days_back)
        return json.dumps([r.to_dict() for r in records])


# ------------------------------------------------------------------ KEV ----

class CheckKevInput(BaseModel):
    cve_ids: list[str] = Field(..., description="List of CVE IDs to check against CISA KEV")


class CheckKevTool(BaseTool):
    name: str = "check_kev"
    description: str = (
        "Check whether one or more CVE IDs appear in CISA's Known "
        "Exploited Vulnerabilities catalog -- the strongest available "
        "signal that a vulnerability is being actively exploited in the "
        "wild right now, not just theoretically dangerous."
    )
    args_schema: Type[BaseModel] = CheckKevInput

    def _run(self, cve_ids: list[str]) -> str:
        return json.dumps(check_kev(cve_ids))


# -------------------------------------------------------------- Findings ----

class RecordFindingInput(BaseModel):
    asset_id: int = Field(..., description="ID of the asset this CVE was judged relevant to")
    cve_id: str = Field(..., description="CVE identifier, e.g. 'CVE-2021-41773'")
    relevance_score: float = Field(..., description="0.0-1.0 confidence this CVE actually applies to this asset")
    relevance_rationale: str = Field(..., description="Why this CVE was judged relevant (or not) -- version match, product name variant, etc.")
    cvss: Optional[float] = Field(None, description="CVSS base score if known")
    severity: Optional[str] = Field(None, description="Severity label if known")


class RecordFindingTool(BaseTool):
    name: str = "record_finding"
    description: str = (
        "Record that a CVE is relevant to a specific asset, with a "
        "rationale. Only call this for CVEs you have judged genuinely "
        "relevant -- do not record every CVE that merely keyword-matches. "
        "Prefer bulk_record_findings when you have more than one finding "
        "to record -- calling this tool once per CVE, many times in a "
        "row, is unreliable and easy to lose track of partway through."
    )
    args_schema: Type[BaseModel] = RecordFindingInput

    def _run(self, asset_id: int, cve_id: str, relevance_score: float,
              relevance_rationale: str, cvss: Optional[float] = None,
              severity: Optional[str] = None) -> str:
        finding_id = db.record_finding(asset_id, cve_id, relevance_score,
                                        relevance_rationale, cvss, severity)
        return json.dumps({"status": "recorded", "finding_id": finding_id})


class FindingRecordItem(BaseModel):
    asset_id: int = Field(..., description="ID of the asset this CVE was judged relevant to")
    cve_id: str = Field(..., description="CVE identifier, e.g. 'CVE-2021-41773'")
    relevance_score: float = Field(..., description="0.0-1.0 confidence this CVE actually applies")
    relevance_rationale: str = Field(..., description="Why this CVE was judged relevant")
    cvss: Optional[float] = Field(None, description="CVSS base score if known")
    severity: Optional[str] = Field(None, description="Severity label if known")


class BulkRecordFindingsInput(BaseModel):
    items: list[FindingRecordItem] = Field(
        ...,
        description="One entry per CVE you have judged genuinely relevant, across all assets. Include everything you decided to record in this single call -- do not split this across multiple tool calls.",
    )


class BulkRecordFindingsTool(BaseTool):
    name: str = "bulk_record_findings"
    description: str = (
        "Record MANY relevant findings in a single call. This is the "
        "PREFERRED way to record findings: after reviewing all candidate "
        "CVEs across all assets, call this tool exactly once with every "
        "CVE you judged genuinely relevant. Do not call record_finding "
        "in a loop instead -- this tool exists precisely so you don't "
        "have to, and calling it once is far more reliable than many "
        "individual calls."
    )
    args_schema: Type[BaseModel] = BulkRecordFindingsInput

    def _run(self, items: list) -> str:
        normalized = [
            i.model_dump() if hasattr(i, "model_dump") else dict(i)
            for i in items
        ]
        result = db.bulk_record_findings(normalized)
        return json.dumps(result)


class UpdatePriorityInput(BaseModel):
    finding_id: int = Field(..., description="ID of the finding to update")
    priority_tier: str = Field(..., description="'HIGH', 'MEDIUM', or 'LOW'")
    in_kev: bool = Field(..., description="Whether this CVE is in the CISA KEV catalog")


class UpdatePriorityTool(BaseTool):
    name: str = "update_priority"
    description: str = (
        "Set the priority tier and KEV status for ONE finding. Prefer "
        "bulk_update_priority when you have more than one finding to "
        "update -- calling this tool once per finding, many times in a "
        "row, is unreliable and easy to lose track of partway through."
    )
    args_schema: Type[BaseModel] = UpdatePriorityInput

    def _run(self, finding_id: int, priority_tier: str, in_kev: bool) -> str:
        db.update_priority(finding_id, priority_tier.upper(), in_kev)
        return json.dumps({"status": "updated", "finding_id": finding_id})


class PriorityUpdateItem(BaseModel):
    finding_id: int = Field(..., description="ID of the finding to update")
    priority_tier: str = Field(..., description="'HIGH', 'MEDIUM', or 'LOW'")
    in_kev: bool = Field(..., description="Whether this CVE is in the CISA KEV catalog")


class BulkUpdatePriorityInput(BaseModel):
    updates: list[PriorityUpdateItem] = Field(
        ...,
        description="One entry per finding you are prioritizing. Include EVERY finding_id from list_findings(status='new') in this single call -- do not split this across multiple tool calls.",
    )


class BulkUpdatePriorityTool(BaseTool):
    name: str = "bulk_update_priority"
    description: str = (
        "Set the priority tier and KEV status for MANY findings in a "
        "single call. This is the PREFERRED way to prioritize findings: "
        "call list_findings(status='new') once, decide a priority_tier "
        "and in_kev value for every single finding_id it returns, then "
        "call this tool exactly once with the complete list. Do not call "
        "update_priority in a loop instead -- this tool exists precisely "
        "so you don't have to."
    )
    args_schema: Type[BaseModel] = BulkUpdatePriorityInput

    def _run(self, updates: list) -> str:
        # updates may arrive as list[PriorityUpdateItem] or list[dict]
        # depending on how the framework serializes tool args.
        normalized = [
            u.model_dump() if hasattr(u, "model_dump") else dict(u)
            for u in updates
        ]
        result = db.bulk_update_priority(normalized)
        return json.dumps(result)


class ListFindingsInput(BaseModel):
    status: Optional[str] = Field(None, description="Filter by status: 'new', 'prioritized', 'reported', 'reviewed'")
    min_priority: Optional[str] = Field(None, description="Filter to at least this priority: 'LOW', 'MEDIUM', 'HIGH'")


class ListFindingsTool(BaseTool):
    name: str = "list_findings"
    description: str = "List findings, optionally filtered by status and/or minimum priority tier."
    args_schema: Type[BaseModel] = ListFindingsInput

    def _run(self, status: Optional[str] = None, min_priority: Optional[str] = None) -> str:
        return json.dumps(db.list_findings(status=status, min_priority=min_priority))


class MarkReportedInput(BaseModel):
    finding_id: int = Field(..., description="ID of the finding to mark as reported")


class MarkReportedTool(BaseTool):
    name: str = "mark_reported"
    description: str = (
        "Mark ONE finding as 'reported'. Prefer bulk_mark_reported when "
        "you have more than one finding to mark -- calling this tool "
        "once per finding, many times in a row, is unreliable and easy "
        "to lose track of partway through."
    )
    args_schema: Type[BaseModel] = MarkReportedInput

    def _run(self, finding_id: int) -> str:
        db.mark_reported(finding_id)
        return json.dumps({"status": "marked_reported", "finding_id": finding_id})


class BulkMarkReportedInput(BaseModel):
    finding_ids: list[int] = Field(
        ...,
        description="Every finding_id from list_findings that should be marked reported. Include ALL of them in this single call -- do not split across multiple calls.",
    )


class BulkMarkReportedTool(BaseTool):
    name: str = "bulk_mark_reported"
    description: str = (
        "Mark MANY findings as 'reported' in a single call. This is the "
        "PREFERRED way to mark findings reported: call list_findings "
        "once, take every finding_id it returned, then call this tool "
        "exactly once with the complete list. Do not call mark_reported "
        "in a loop instead -- this tool exists precisely so you don't "
        "have to, and calling it once is far more reliable than many "
        "individual calls."
    )
    args_schema: Type[BaseModel] = BulkMarkReportedInput

    def _run(self, finding_ids: list[int]) -> str:
        result = db.bulk_mark_reported(finding_ids)
        return json.dumps(result)
