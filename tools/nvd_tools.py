"""
tools/nvd_tools.py

Search recent CVEs from the NVD REST API by keyword (product name), with
optional publish-date bounding. No API key required for light use;
NVD_API_KEY is honored if set, for a higher rate limit.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


@dataclass
class CveRecord:
    cve_id: str
    published: Optional[str]
    cvss: Optional[float]
    severity: Optional[str]
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


def search_recent_cves(keyword: str, days_back: int = 7, max_results: int = 20,
                        timeout: int = 15) -> List[CveRecord]:
    """
    Search NVD for CVEs matching `keyword` (typically a product name, e.g.
    'apache http server' or 'jenkins') published within the last
    `days_back` days.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    params = {
        "keywordSearch": keyword,
        "pubStartDate": start.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "pubEndDate": end.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "resultsPerPage": str(max_results),
    }
    url = f"{NVD_URL}?{urlencode(params)}"
    headers = {"User-Agent": "CoverageCrew/1.0"}
    api_key = os.environ.get("NVD_API_KEY")
    if api_key:
        headers["apiKey"] = api_key

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as e:
        raise RuntimeError(f"NVD search failed for '{keyword}': {e}")

    records: List[CveRecord] = []
    for item in data.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cve_id = cve.get("id", "UNKNOWN")
        descriptions = cve.get("descriptions", [])
        summary = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
        metrics = cve.get("metrics", {})
        cvss = None
        severity = None
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics and metrics[key]:
                cvss_data = metrics[key][0]["cvssData"]
                cvss = cvss_data.get("baseScore")
                severity = cvss_data.get("baseSeverity") or metrics[key][0].get("baseSeverity")
                break
        records.append(CveRecord(
            cve_id=cve_id,
            published=cve.get("published"),
            cvss=cvss,
            severity=severity,
            summary=summary[:400],
        ))

    time.sleep(0.6 if not api_key else 0.1)
    return records


def fetch_cve_by_id(cve_id: str, timeout: int = 15) -> Optional[CveRecord]:
    """
    Fetch a single CVE's authoritative record directly from NVD by ID,
    independent of any candidate list an agent may have seen or reasoned
    about. Used to verify a finding's actual subject matter against the
    asset it was filed under -- a local model's own rationale text can
    describe a plausible-sounding but false justification without ever
    naming the true source product, so checking the model's wording
    alone isn't sufficient. Returns None on any lookup failure (network
    issue, unknown CVE, rate limit) so callers can skip verification
    gracefully rather than treating "couldn't check" as "mismatch".
    """
    params = {"cveId": cve_id}
    url = f"{NVD_URL}?{urlencode(params)}"
    headers = {"User-Agent": "CoverageCrew/1.0"}
    api_key = os.environ.get("NVD_API_KEY")
    if api_key:
        headers["apiKey"] = api_key

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return None

    cve = vulns[0].get("cve", {})
    descriptions = cve.get("descriptions", [])
    summary = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
    metrics = cve.get("metrics", {})
    cvss = None
    severity = None
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics and metrics[key]:
            cvss_data = metrics[key][0]["cvssData"]
            cvss = cvss_data.get("baseScore")
            severity = cvss_data.get("baseSeverity") or metrics[key][0].get("baseSeverity")
            break

    time.sleep(0.6 if not api_key else 0.1)
    return CveRecord(
        cve_id=cve.get("id", cve_id),
        published=cve.get("published"),
        cvss=cvss,
        severity=severity,
        summary=summary[:600],
    )


def product_terms(product: str) -> list[str]:
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


def verify_cve_product_match(cve_id: str, product: str) -> Optional[bool]:
    """
    Checks whether NVD's own authoritative description of `cve_id`
    actually mentions `product`. Returns:
        True  -- verified match
        False -- verified mismatch (NVD's description doesn't mention
                 the product at all)
        None  -- unverifiable (network issue, unknown CVE, or no
                 product terms to check against)

    Deliberately three-valued rather than boolean: a caller that treats
    "couldn't verify" as "verified wrong" would reject or flag findings
    just because NVD was briefly unreachable, which is a worse failure
    than letting an unverifiable finding through. Only a definite False
    should ever block or flag something.
    """
    verdict, _record = fetch_and_verify(cve_id, product)
    return verdict


def fetch_and_verify(cve_id: str, product: str) -> tuple[Optional[bool], Optional["CveRecord"]]:
    """
    Same check as verify_cve_product_match, but also hands back the
    authoritative CveRecord it fetched.

    Exists so a caller that is already paying for the network round-trip
    to verify attribution can *also* use NVD's own CVSS and severity for
    the record it stores, instead of trusting the numbers an LLM
    supplied. Observed in practice: an agent reporting CVSS 9.8 for a
    CVE whose real NVD base score is 8.8, and asserting CISA-KEV
    membership for CVEs that are not in the catalog at all. Fetching
    once and using the result for both purposes closes that gap for
    free.

    Returns (verdict, record) where verdict follows the same
    three-valued convention as verify_cve_product_match, and record is
    None whenever the lookup itself failed.
    """
    candidates = product_terms(product)
    record = fetch_cve_by_id(cve_id)
    if record is None or not record.summary:
        return None, record
    if not candidates:
        return None, record
    desc = record.summary.lower()
    return any(term in desc for term in candidates), record


if __name__ == "__main__":
    import sys
    kw = sys.argv[1] if len(sys.argv) > 1 else "apache http server"
    for r in search_recent_cves(kw, days_back=30):
        print(r.to_dict())
