"""
tools/kev_tools.py

Pulls CISA's Known Exploited Vulnerabilities (KEV) catalog -- the U.S.
government's public list of CVEs confirmed to be actively exploited in
the wild. This is the single strongest free "is this actually urgent"
signal available, and it's what the Exploitability & Exposure agent
leans on most heavily when assigning a priority tier: a CVE in KEV
against an internet-facing asset should never sit at LOW priority
regardless of its base CVSS score.

The catalog is a few MB of JSON updated regularly by CISA, so it's
cached locally with a TTL rather than re-downloaded on every tool call.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "kev_cache.json"
CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours


@dataclass
class KevEntry:
    cve_id: str
    vendor_project: str
    product: str
    date_added: str
    short_description: str
    known_ransomware_use: str

    def to_dict(self) -> dict:
        return asdict(self)


def _fetch_kev_json(timeout: int = 20) -> dict:
    headers = {"User-Agent": "ThreatLensCrew/1.0 (SEC598 coursework)"}
    req = Request(KEV_URL, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def load_kev_catalog(force_refresh: bool = False) -> Dict[str, KevEntry]:
    """
    Returns a dict keyed by CVE ID for O(1) lookup. Uses a local cache
    file to avoid re-downloading the (multi-MB) catalog on every call;
    refreshes automatically once the cache is older than CACHE_TTL_SECONDS.
    """
    use_cache = (
        not force_refresh
        and CACHE_PATH.exists()
        and (time.time() - CACHE_PATH.stat().st_mtime) < CACHE_TTL_SECONDS
    )

    if use_cache:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    else:
        try:
            raw = _fetch_kev_json()
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(json.dumps(raw), encoding="utf-8")
        except (URLError, HTTPError, TimeoutError):
            if CACHE_PATH.exists():
                raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            else:
                raise

    catalog: Dict[str, KevEntry] = {}
    for item in raw.get("vulnerabilities", []):
        entry = KevEntry(
            cve_id=item.get("cveID", ""),
            vendor_project=item.get("vendorProject", ""),
            product=item.get("product", ""),
            date_added=item.get("dateAdded", ""),
            short_description=item.get("shortDescription", "")[:300],
            known_ransomware_use=item.get("knownRansomwareCampaignUse", "Unknown"),
        )
        if entry.cve_id:
            catalog[entry.cve_id] = entry
    return catalog


def check_kev(cve_ids: List[str]) -> Dict[str, Optional[dict]]:
    """Return {cve_id: kev_entry_dict_or_None} for each requested CVE."""
    catalog = load_kev_catalog()
    return {cid: (catalog[cid].to_dict() if cid in catalog else None) for cid in cve_ids}


if __name__ == "__main__":
    import sys
    ids = sys.argv[1:] or ["CVE-2021-44228"]
    for cve_id, hit in check_kev(ids).items():
        print(cve_id, "->", "IN KEV: " + str(hit) if hit else "not in KEV")
