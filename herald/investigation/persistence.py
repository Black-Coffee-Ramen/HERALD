from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from herald.investigation.targets import safe_path_fragment


def create_evidence_dir(trace_id: str, domain: str, root: str = "evidence") -> Path:
    path = Path(root) / f"{trace_id}_{safe_path_fragment(domain)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_investigation(result: dict[str, Any], evidence_dir: str) -> dict[str, str]:
    base = Path(evidence_dir)
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / "investigation.json"
    md_path = base / "report.md"
    result.setdefault("visual", {}).setdefault("report_paths", {"json": os.fspath(json_path), "markdown": os.fspath(md_path)})

    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")

    latest_dir = Path("evidence")
    latest_dir.mkdir(exist_ok=True)
    index_path = latest_dir / "investigations.jsonl"
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "trace_id": result["trace_id"],
            "domain": result["domain"],
            "verdict": result["verdict"],
            "phishing_score": result["phishing_score"],
            "evidence_dir": os.fspath(base),
            "completed_at": result["completed_at"],
        }, sort_keys=True) + "\n")

    return {"json": os.fspath(json_path), "markdown": os.fspath(md_path)}


def find_report(trace_id: str, root: str = "evidence") -> dict[str, Any] | None:
    root_path = Path(root)
    if not root_path.exists():
        return None
    for report_path in root_path.glob(f"{trace_id}*/investigation.json"):
        return json.loads(report_path.read_text(encoding="utf-8"))
    return None


def render_markdown(result: dict[str, Any]) -> str:
    dns = result.get("dns", {})
    tls = result.get("tls", {})
    visual = result.get("visual", {})
    summary = result.get("summary", {})
    risk_factors = result.get("risk_factors") or []
    report_paths = visual.get("report_paths") or {}
    lines = [
        "# HERALD Investigation Report",
        "",
        "## Executive Summary",
        "",
        summary.get("headline") or f"{result['domain']} is {result['verdict']} with score {result['phishing_score']}.",
        "",
        f"- Trace ID: `{result['trace_id']}`",
        f"- Target: `{result['url']}`",
        f"- Domain: `{result['domain']}`",
        f"- Verdict: **{result['verdict']}**",
        f"- Phishing score: `{result['phishing_score']}`",
        f"- Evidence directory: `{result['evidence_dir']}`",
        f"- Completed: `{result['completed_at']}`",
        "",
        "## Why Flagged",
        "",
    ]
    for reason in summary.get("why_flagged", []):
        lines.append(f"- {reason}")

    lines.extend([
        "",
        "## Risk Factors",
        "",
        "| Factor | Severity | Impact | Explanation |",
        "|---|---:|---:|---|",
    ])
    if risk_factors:
        for factor in risk_factors:
            lines.append(
                f"| {factor.get('name', 'Unknown')} | {factor.get('severity', 'unknown')} | "
                f"{factor.get('score_impact', 0)} | {factor.get('detail', '')} |"
            )
    else:
        lines.append("| None | info | 0 | No notable risk factors recorded. |")

    lines.extend([
        "",
        "## DNS and WHOIS",
        "",
        "| Signal | Value |",
        "|---|---|",
        f"| Registrar | `{dns.get('registrar') or 'unknown'}` |",
        f"| Creation date | `{dns.get('creation_date') or 'unknown'}` |",
        f"| Domain age | `{dns.get('domain_age_days') if dns.get('domain_age_days') is not None else 'unknown'}` |",
        f"| Nameservers | `{', '.join(dns.get('nameservers') or []) or 'unknown'}` |",
        f"| A records | `{', '.join(dns.get('a_records') or []) or 'unknown'}` |",
        f"| MX records | `{', '.join(dns.get('mx_records') or []) or 'unknown'}` |",
        "",
        "## TLS",
        "",
        "| Signal | Value |",
        "|---|---|",
        f"| Has TLS | `{tls.get('has_tls')}` |",
        f"| Issuer | `{tls.get('issuer') or 'unknown'}` |",
        f"| Subject | `{tls.get('subject') or 'unknown'}` |",
        f"| SAN match | `{tls.get('domain_in_san')}` |",
        f"| Days remaining | `{tls.get('days_remaining') if tls.get('days_remaining') is not None else 'unknown'}` |",
        "",
        "## Evidence",
        "",
        f"- Screenshot: `{visual.get('screenshot_path') or 'not captured'}`",
        f"- OCR suspicious phrases: `{', '.join((visual.get('ocr_findings') or {}).get('phrases_found') or []) or 'none'}`",
        f"- JSON report: `{report_paths.get('json') or 'investigation.json'}`",
        f"- Markdown report: `{report_paths.get('markdown') or 'report.md'}`",
        "",
        "## Lifecycle",
        "",
        "| Stage | Status | Duration |",
        "|---|---:|---:|",
    ])
    for stage in result.get("stages", []):
        lines.append(f"| {stage['name']} | `{stage['status']}` | `{stage['duration_ms']}ms` |")

    if result.get("errors"):
        lines.extend(["", "## Degraded Notes", ""])
        for error in result["errors"]:
            lines.append(f"- {error}")

    return "\n".join(lines) + "\n"
