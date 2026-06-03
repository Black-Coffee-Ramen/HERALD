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
    html_path = base / "report.html"
    result.setdefault("visual", {}).setdefault("report_paths", {"json": os.fspath(json_path), "markdown": os.fspath(md_path), "html": os.fspath(html_path)})

    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    html_path.write_text(render_html(result), encoding="utf-8")

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

    return {"json": os.fspath(json_path), "markdown": os.fspath(md_path), "html": os.fspath(html_path)}


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

def render_html(result: dict[str, Any]) -> str:
    """Generate a clean, professional HTML report from the investigation result."""
    dns = result.get("dns", {})
    tls = result.get("tls", {})
    visual = result.get("visual", {})
    summary = result.get("summary", {})
    risk_factors = result.get("risk_factors") or []
    
    # Base styling
    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        f"    <title>HERALD Report: {result.get('domain', 'Unknown')}</title>",
        "    <style>",
        "        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 2rem; background-color: #f9f9fb; }",
        "        h1, h2, h3 { color: #1a1a2e; }",
        "        .header { background-color: #ffffff; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 2rem; border-top: 5px solid #2ecc71; }",
        "        .header.Phishing { border-top-color: #e74c3c; }",
        "        .header.Suspected { border-top-color: #f1c40f; }",
        "        .section { background-color: #ffffff; padding: 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1.5rem; }",
        "        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }",
        "        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eaeaea; }",
        "        th { background-color: #f8f9fa; font-weight: 600; color: #555; }",
        "        .badge { display: inline-block; padding: 0.25em 0.6em; font-size: 0.85em; font-weight: 700; line-height: 1; text-align: center; white-space: nowrap; vertical-align: baseline; border-radius: 0.25rem; }",
        "        .badge.high { background-color: #fee2e2; color: #991b1b; }",
        "        .badge.medium { background-color: #fef3c7; color: #92400e; }",
        "        .badge.low { background-color: #f3f4f6; color: #374151; }",
        "        .badge.info { background-color: #e0f2fe; color: #075985; }",
        "        .screenshot-container { max-width: 100%; margin-top: 1rem; border: 1px solid #eee; border-radius: 4px; overflow: hidden; }",
        "        .screenshot-container img { max-width: 100%; height: auto; display: block; }",
        "    </style>",
        "</head>",
        "<body>",
    ]
    
    headline = summary.get('headline') or f"{result.get('domain')} is {result.get('verdict')} with score {result.get('phishing_score')}"
    
    html.extend([
        f"    <div class='header {result.get('verdict', 'Clean')}'>",
        "        <h1>HERALD Investigation Report</h1>",
        f"        <p style='font-size: 1.2rem;'><strong>{headline}</strong></p>",
        "        <table style='width: 100%; background: #fdfdfd;'>",
        f"            <tr><td><strong>Trace ID:</strong></td><td><code>{result.get('trace_id')}</code></td><td><strong>Verdict:</strong></td><td><strong>{result.get('verdict')}</strong></td></tr>",
        f"            <tr><td><strong>Domain:</strong></td><td><code>{result.get('domain')}</code></td><td><strong>Score:</strong></td><td><code>{result.get('phishing_score')}</code></td></tr>",
        f"            <tr><td><strong>Completed:</strong></td><td>{result.get('completed_at')}</td><td><strong>Target:</strong></td><td><code>{result.get('url')}</code></td></tr>",
        "        </table>",
        "    </div>",
    ])

    # Why Flagged
    html.append("<div class='section'><h2>Why Flagged</h2><ul>")
    for reason in summary.get("why_flagged", ["No strong phishing indicators were observed."]):
        html.append(f"<li>{reason}</li>")
    html.append("</ul></div>")

    # Risk Factors
    html.append("<div class='section'><h2>Risk Factors</h2><table>")
    html.append("<tr><th>Factor</th><th>Severity</th><th>Impact</th><th>Explanation</th></tr>")
    if risk_factors:
        for factor in risk_factors:
            sev = factor.get('severity', 'info')
            html.append(f"<tr><td>{factor.get('name', 'Unknown')}</td><td><span class='badge {sev}'>{sev}</span></td><td>{factor.get('score_impact', 0)}</td><td>{factor.get('detail', '')}</td></tr>")
    else:
        html.append("<tr><td colspan='4'>No notable risk factors recorded.</td></tr>")
    html.append("</table></div>")

    # DNS and TLS
    html.append("<div style='display: flex; gap: 1.5rem;'>")
    
    # DNS
    html.append("<div class='section' style='flex: 1;'><h2>DNS & WHOIS</h2><table>")
    html.append(f"<tr><th>Registrar</th><td>{dns.get('registrar') or 'unknown'}</td></tr>")
    html.append(f"<tr><th>Domain Age</th><td>{dns.get('domain_age_days') if dns.get('domain_age_days') is not None else 'unknown'} days</td></tr>")
    html.append(f"<tr><th>A Records</th><td>{', '.join(dns.get('a_records') or []) or 'none'}</td></tr>")
    html.append(f"<tr><th>MX Records</th><td>{', '.join(dns.get('mx_records') or []) or 'none'}</td></tr>")
    html.append("</table></div>")

    # TLS
    html.append("<div class='section' style='flex: 1;'><h2>TLS Certificate</h2><table>")
    html.append(f"<tr><th>Has TLS</th><td>{tls.get('has_tls', False)}</td></tr>")
    html.append(f"<tr><th>Issuer</th><td>{tls.get('issuer') or 'unknown'}</td></tr>")
    html.append(f"<tr><th>SAN Match</th><td>{tls.get('domain_in_san', False)}</td></tr>")
    html.append(f"<tr><th>Days Remaining</th><td>{tls.get('days_remaining') if tls.get('days_remaining') is not None else 'unknown'}</td></tr>")
    html.append("</table></div>")
    
    html.append("</div>") # End flex

    # Evidence / Screenshot
    html.append("<div class='section'><h2>Visual Evidence</h2>")
    ocr = ", ".join((visual.get('ocr_findings') or {}).get('phrases_found') or []) or "none"
    html.append(f"<p><strong>OCR Suspicious Phrases:</strong> {ocr}</p>")
    
    screenshot_path = visual.get('screenshot_path')
    if screenshot_path:
        # Resolve path relative to the HTML report (which is in the same dir as the screenshots folder)
        rel_path = "screenshots/homepage.png" if "homepage.png" in screenshot_path else screenshot_path.split("/")[-1]
        html.append(f"<div class='screenshot-container'><img src='{rel_path}' alt='Screenshot of {result.get('domain')}'></div>")
    else:
        html.append("<p><em>Screenshot not captured.</em></p>")
    html.append("</div>")

    html.append("</body></html>")
    return "\n".join(html)
