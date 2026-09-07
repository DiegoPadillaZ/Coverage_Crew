"""
core/db.py

Persistent asset + vulnerability-findings registry for Coverage_Crew.

This exists because the actual problem is *lack of visibility*: there is
no ready-made vulnerability exposure feed to consume, so the automation
has to build and maintain that visibility itself, one crew run at a time.

Two tables:

  assets    -- what you actually run. Seeded once by hand (see
               data/assets_seed.yaml), added to over time via the MCP
               server or the CLI as your environment changes.

  findings  -- every CVE the crew has ever judged relevant to a specific
               asset, with the relevance rationale, CVSS, whether it's in
               CISA's Known Exploited Vulnerabilities catalog, an assigned
               priority tier, and a review status so the same finding is
               never reported twice.

A recurring theme in this module: facts that have a single authoritative
source are taken from that source in code, not from whatever the model
asserted. CVSS and severity come from NVD, KEV membership comes from
CISA's published catalog, and "which findings belong in this report" is
computed by query. The LLM's judgment is reserved for the genuinely
judgment-shaped question -- does this CVE actually apply to this asset.

No ORM -- this is intentionally small and dependency-light so it's easy
to inspect (`sqlite3 data/threatlens.db`).
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from tools.nvd_tools import fetch_and_verify
from tools.kev_tools import check_kev

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "threatlens.db"

# A real CVE ID is CVE-YYYY-NNNN (4+ digits after the year). This exists
# because a local model has been observed writing placeholder IDs like
# "CVE-2026-XXXX" into a finding instead of the actual number it found --
# not a plausible-but-wrong fabrication, a literal unfilled template.
# Validating the format here means such a record can never be persisted
# in the first place, rather than trying to catch every possible
# placeholder pattern later in report text.
CVE_ID_FORMAT = re.compile(r"^CVE-\d{4}-\d{4,8}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    vendor      TEXT NOT NULL,
    product     TEXT NOT NULL,
    version     TEXT,
    exposure    TEXT NOT NULL DEFAULT 'internal',   -- 'internet-facing' | 'internal'
    criticality TEXT NOT NULL DEFAULT 'medium',      -- 'high' | 'medium' | 'low'
    notes       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS findings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id            INTEGER NOT NULL REFERENCES assets(id),
    cve_id              TEXT NOT NULL,
    relevance_score     REAL,
    relevance_rationale TEXT,
    cvss                REAL,
    severity            TEXT,
    in_kev              INTEGER NOT NULL DEFAULT 0,
    priority_tier       TEXT,                        -- 'HIGH' | 'MEDIUM' | 'LOW'
    status              TEXT NOT NULL DEFAULT 'new',  -- 'new' -> 'prioritized' -> 'reported'
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    UNIQUE(asset_id, cve_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------- Assets ----

def get_asset_by_name(name: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM assets WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None


def add_asset(name: str, vendor: str, product: str, version: str = "",
              exposure: str = "internal", criticality: str = "medium",
              notes: str = "") -> int:
    """
    Insert a new asset, or no-op (returning the existing id) if an asset
    with this exact name already exists. This makes re-running
    --seed-assets on an already-seeded registry safe -- without this,
    every re-seed created a fresh duplicate row per asset, which both
    inflated the asset count and made asset_id references inconsistent
    across runs (finding rows recorded against an id that later pointed
    at a different asset once duplicates were interleaved).
    """
    existing = get_asset_by_name(name)
    if existing:
        return existing["id"]
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO assets (name, vendor, product, version, exposure, "
            "criticality, notes, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (name, vendor, product, version, exposure, criticality, notes, _now()),
        )
        return cur.lastrowid


def list_assets() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM assets ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def get_asset(asset_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
        return dict(row) if row else None


# ------------------------------------------------------------- Findings ----

def record_finding(asset_id: int, cve_id: str, relevance_score: float,
                    relevance_rationale: str, cvss: Optional[float] = None,
                    severity: Optional[str] = None) -> Optional[int]:
    """
    Insert a new finding, or no-op if this (asset_id, cve_id) pair already
    exists -- prevents the same CVE being re-recorded against the same
    asset on every crew run.

    Returns None (rejects the insert) if cve_id doesn't match the real
    CVE-YYYY-NNNN format -- guards against a placeholder like
    "CVE-2026-XXXX" being written to the database as if it were a real
    finding -- or if the CVE is independently verified against NVD's
    own description as not actually being about this asset's product.
    That second check exists because a real CVE ID for one product has
    been observed recorded against a completely different asset (e.g.
    a Fortinet FortiOS CVE filed under a Windows RDP gateway), with a
    rationale that sounds plausible but doesn't hold up against the
    CVE's actual authoritative content. Validating at insertion time
    stops that record from ever entering the registry, rather than
    only flagging it later when a report is generated. A CVE that
    can't be verified (NVD unreachable, unknown ID) is allowed through
    rather than rejected -- "couldn't check" is not "confirmed wrong".

    The same NVD lookup also supplies cvss/severity, overriding whatever
    the caller passed: an agent has been observed reporting CVSS 9.8 for
    a CVE whose real base score is 8.8, and the round-trip is already
    being paid for, so there is no reason to keep the model's number.
    """
    if not CVE_ID_FORMAT.match(cve_id or ""):
        return None
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM findings WHERE asset_id = ? AND cve_id = ?",
            (asset_id, cve_id),
        ).fetchone()
        if existing:
            return existing["id"]
        asset_row = conn.execute("SELECT product FROM assets WHERE id = ?", (asset_id,)).fetchone()
        verdict, record = fetch_and_verify(cve_id, asset_row["product"] if asset_row else "")
        if verdict is False:
            return None
        # Prefer NVD's own CVSS/severity over whatever the model supplied.
        if record is not None:
            if record.cvss is not None:
                cvss = record.cvss
            if record.severity:
                severity = record.severity
        cur = conn.execute(
            "INSERT INTO findings (asset_id, cve_id, relevance_score, "
            "relevance_rationale, cvss, severity, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?, 'new', ?, ?)",
            (asset_id, cve_id, relevance_score, relevance_rationale, cvss,
             severity, _now(), _now()),
        )
        return cur.lastrowid


def bulk_record_findings(items: list[dict]) -> dict:
    """
    Record many findings in a single call.

    Exists for the exact same reason as bulk_update_priority: asking a
    local LLM to call record_finding once per relevant CVE, potentially
    30+ times in a row, is where small models tend to switch from
    actually invoking tools to just narrating what they *would* record
    in their final text answer -- producing a plausible-looking summary
    that never touched the database. One bulk call removes that failure
    mode by turning a long repetitive tool-call sequence into a single
    JSON payload.

    Each dict in `items` needs: asset_id, cve_id, relevance_score,
    relevance_rationale, and optionally cvss, severity. Invalid entries
    are skipped and reported back rather than failing the whole batch:
    missing required fields, an asset_id that doesn't exist, a cve_id
    that doesn't match the real CVE-YYYY-NNNN format, or a CVE
    independently verified against NVD's own description as not
    actually being about that asset's product -- see record_finding's
    docstring for why this check exists, why an unverifiable CVE is
    allowed through rather than rejected, and why NVD's own CVSS wins
    over the model's.
    """
    recorded, skipped = [], []
    with _connect() as conn:
        asset_products = {
            row["id"]: row["product"]
            for row in conn.execute("SELECT id, product FROM assets").fetchall()
        }
        for item in items:
            asset_id = item.get("asset_id")
            cve_id = item.get("cve_id")
            rationale = item.get("relevance_rationale")
            score = item.get("relevance_score")
            if (asset_id not in asset_products or not cve_id or
                    rationale is None or score is None or
                    not CVE_ID_FORMAT.match(cve_id)):
                skipped.append(item)
                continue
            existing = conn.execute(
                "SELECT id FROM findings WHERE asset_id = ? AND cve_id = ?",
                (asset_id, cve_id),
            ).fetchone()
            if existing:
                recorded.append(existing["id"])
                continue
            verdict, record = fetch_and_verify(cve_id, asset_products[asset_id])
            if verdict is False:
                skipped.append(item)
                continue
            # Prefer NVD's own CVSS/severity over whatever the model supplied.
            cvss = item.get("cvss")
            severity = item.get("severity")
            if record is not None:
                if record.cvss is not None:
                    cvss = record.cvss
                if record.severity:
                    severity = record.severity
            cur = conn.execute(
                "INSERT INTO findings (asset_id, cve_id, relevance_score, "
                "relevance_rationale, cvss, severity, status, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?, 'new', ?, ?)",
                (asset_id, cve_id, score, rationale, cvss, severity, _now(), _now()),
            )
            recorded.append(cur.lastrowid)
    return {"recorded_finding_ids": recorded, "skipped": skipped}


def update_priority(finding_id: int, priority_tier: str, in_kev: bool) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE findings SET priority_tier = ?, in_kev = ?, status = "
            "'prioritized', updated_at = ? WHERE id = ?",
            (priority_tier, int(in_kev), _now(), finding_id),
        )


def bulk_update_priority(updates: list[dict]) -> dict:
    """
    Apply priority_tier + in_kev to many findings in a single call.

    Exists because asking a local LLM to reliably execute one tool call
    per finding, N times in a row, is a much harder completion task than
    asking it to produce one JSON array covering all of them -- smaller
    models tend to reason through a long list narratively and then never
    actually invoke the tool for most of it. Each dict in `updates` needs
    keys: finding_id, priority_tier, in_kev. Unknown finding_ids are
    skipped and reported back rather than raising, so one bad entry
    doesn't fail the whole batch.

    Note that the in_kev value supplied here is provisional: sync_kev_flags()
    overwrites it from CISA's real catalog after the run, because KEV
    membership is a set-membership fact rather than a judgment call.
    """
    applied, skipped = [], []
    with _connect() as conn:
        for u in updates:
            fid = u.get("finding_id")
            tier = (u.get("priority_tier") or "").upper()
            in_kev = bool(u.get("in_kev", False))
            if fid is None or tier not in ("HIGH", "MEDIUM", "LOW"):
                skipped.append(u)
                continue
            cur = conn.execute(
                "UPDATE findings SET priority_tier = ?, in_kev = ?, status = "
                "'prioritized', updated_at = ? WHERE id = ?",
                (tier, int(in_kev), _now(), fid),
            )
            if cur.rowcount == 0:
                skipped.append(u)
            else:
                applied.append(fid)
    return {"applied": applied, "skipped": skipped}


def mark_reported(finding_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE findings SET status = 'reported', updated_at = ? WHERE id = ?",
            (_now(), finding_id),
        )


def bulk_mark_reported(finding_ids: list[int]) -> dict:
    """
    Mark many findings as reported in a single call. Unknown finding_ids
    are skipped and reported back rather than raising, so one bad ID
    doesn't fail the whole batch.
    """
    applied, skipped = [], []
    with _connect() as conn:
        for fid in finding_ids:
            cur = conn.execute(
                "UPDATE findings SET status = 'reported', updated_at = ? WHERE id = ?",
                (_now(), fid),
            )
            if cur.rowcount == 0:
                skipped.append(fid)
            else:
                applied.append(fid)
    return {"applied": applied, "skipped": skipped}


def sync_kev_flags() -> dict:
    """
    Set every finding's in_kev flag from CISA's actual KEV catalog,
    overwriting whatever an agent asserted.

    Exists because in_kev was previously whatever the Exploitability
    agent claimed, and it has been observed asserting "explicitly
    listed in CISA KEV" for CVEs that are not in the catalog at all --
    inventing the single strongest urgency signal in the whole system.
    KEV membership is a plain set-membership question against a
    published catalog, so there is no reason to let a model answer it:
    this reads the real catalog and corrects every row.

    Fails safe: if the catalog can't be fetched, flags are left exactly
    as they were rather than being cleared, since wrongly clearing a
    true KEV hit would be worse than leaving a stale one.
    """
    findings = list_findings()
    if not findings:
        return {"checked": 0, "corrected": 0, "catalog_unavailable": False}

    cve_ids = sorted({f["cve_id"] for f in findings})
    try:
        kev_map = check_kev(cve_ids)
    except Exception:
        return {"checked": 0, "corrected": 0, "catalog_unavailable": True}

    corrected = 0
    with _connect() as conn:
        for f in findings:
            truth = 1 if kev_map.get(f["cve_id"]) else 0
            if int(f.get("in_kev") or 0) != truth:
                conn.execute(
                    "UPDATE findings SET in_kev = ?, updated_at = ? WHERE id = ?",
                    (truth, _now(), f["id"]),
                )
                corrected += 1
    return {"checked": len(findings), "corrected": corrected, "catalog_unavailable": False}


def mark_all_reported(min_priority: str = "MEDIUM") -> dict:
    """
    Mark every currently-prioritized finding at or above `min_priority`
    as reported, in code.

    Exists because "mark each of these N findings reported" is pure
    bookkeeping with a single correct answer, and handing it to an LLM
    reliably produced partial results: given 18 qualifying findings, the
    Communicator called bulk_mark_reported with only the 2 finding_ids
    it happened to name in its narrative, silently leaving 16 stuck in
    'prioritized' forever. Since the set of findings that belong in the
    report is already computed deterministically for the Verified
    Findings table, the matching status update belongs in code too --
    the model has no information the database doesn't already have.
    """
    targets = [f["id"] for f in list_findings(status="prioritized", min_priority=min_priority)]
    if not targets:
        return {"applied": [], "skipped": []}
    return bulk_mark_reported(targets)


def mark_reviewed(finding_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE findings SET status = 'reviewed', updated_at = ? WHERE id = ?",
            (_now(), finding_id),
        )


def list_findings(status: Optional[str] = None, min_priority: Optional[str] = None) -> list[dict]:
    tier_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT f.*, a.name AS asset_name, a.exposure, a.criticality, "
                "a.vendor AS asset_vendor, a.product AS asset_product "
                "FROM findings f JOIN assets a ON a.id = f.asset_id "
                "WHERE f.status = ? ORDER BY f.created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT f.*, a.name AS asset_name, a.exposure, a.criticality, "
                "a.vendor AS asset_vendor, a.product AS asset_product "
                "FROM findings f JOIN assets a ON a.id = f.asset_id "
                "ORDER BY f.created_at DESC"
            ).fetchall()
        result = [dict(r) for r in rows]
    if min_priority:
        threshold = tier_rank.get(min_priority.upper(), 0)
        result = [r for r in result if tier_rank.get((r.get("priority_tier") or ""), 0) >= threshold]
    return result


def get_finding(finding_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
        return dict(row) if row else None
