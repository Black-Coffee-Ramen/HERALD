from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys

from rich.console import Console
from rich import box
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from herald.core.security import SSRFProtectionError
from herald.investigation.persistence import find_report
from herald.investigation.pipeline import InvestigationPipeline


console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "investigate":
        return run_investigate(args)
    if args.command == "analyze":
        return run_analyze(args)
    if args.command == "screenshot":
        return run_screenshot(args)
    if args.command == "report":
        return run_report(args)

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="herald", description="HERALD phishing investigation CLI")
    subparsers = parser.add_subparsers(dest="command")

    investigate = subparsers.add_parser("investigate", help="Run a full URL investigation")
    investigate.add_argument("target")
    investigate.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    investigate.add_argument("--no-visual", action="store_true", help="Skip screenshot and OCR")
    investigate.add_argument("--allow-private", action="store_true", help="Allow targets that resolve to private/internal IPs")

    analyze = subparsers.add_parser("analyze", help="Run domain intelligence without screenshot/OCR")
    analyze.add_argument("domain")
    analyze.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    analyze.add_argument("--allow-private", action="store_true", help="Allow targets that resolve to private/internal IPs")

    screenshot = subparsers.add_parser("screenshot", help="Capture screenshot and OCR evidence")
    screenshot.add_argument("target")
    screenshot.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    screenshot.add_argument("--allow-private", action="store_true", help="Allow targets that resolve to private/internal IPs")

    report = subparsers.add_parser("report", help="Show a persisted investigation by trace ID")
    report.add_argument("trace_id")
    report.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    return parser


def run_investigate(args: argparse.Namespace) -> int:
    pipeline = InvestigationPipeline()
    try:
        result = execute_pipeline(pipeline, args.target, include_visual=not args.no_visual, allow_private=args.allow_private, quiet=args.json)
    except SSRFProtectionError as exc:
        return emit_ssrf_error(exc, args.target, "investigate", as_json=args.json)
    return emit_result(result.to_dict(), as_json=args.json)


def run_analyze(args: argparse.Namespace) -> int:
    pipeline = InvestigationPipeline()
    try:
        result = execute_pipeline(pipeline, args.domain, include_visual=False, allow_private=args.allow_private, quiet=args.json)
    except SSRFProtectionError as exc:
        return emit_ssrf_error(exc, args.domain, "analyze", as_json=args.json)
    return emit_result(result.to_dict(), as_json=args.json)


def run_screenshot(args: argparse.Namespace) -> int:
    pipeline = InvestigationPipeline()
    try:
        result = execute_pipeline(pipeline, args.target, include_visual=True, allow_private=args.allow_private, quiet=args.json)
    except SSRFProtectionError as exc:
        return emit_ssrf_error(exc, args.target, "screenshot", as_json=args.json)
    visual_only = {
        "trace_id": result.trace_id,
        "domain": result.domain,
        "evidence_dir": result.evidence_dir,
        "visual": result.visual,
        "stages": [stage.to_dict() for stage in result.stages if stage.name == "Screenshot and OCR"],
    }
    if args.json:
        console.print_json(json.dumps(visual_only))
    else:
        console.print(Panel.fit(f"[bold]Screenshot evidence[/bold]\nTrace: {result.trace_id}\nPath: {result.visual.get('screenshot_path') or 'not captured'}"))
    return 0


def run_report(args: argparse.Namespace) -> int:
    report = find_report(args.trace_id)
    if not report:
        console.print(f"[red]No persisted report found for trace ID[/red] {args.trace_id}")
        return 1
    return emit_result(report, as_json=args.json)


def emit_result(result: dict, *, as_json: bool) -> int:
    if as_json:
        console.print_json(json.dumps(result))
        return 0

    color = verdict_color(result["verdict"])
    summary = result.get("summary") or {}
    console.print(Panel(
        f"[bold]{summary.get('headline') or result['domain']}[/bold]\n"
        f"Trace: [cyan]{result['trace_id']}[/cyan]\n"
        f"Verdict: [{color}]{result['verdict']}[/{color}]  Score: [bold]{result['phishing_score']}[/bold]\n"
        f"Evidence: [dim]{result['evidence_dir']}[/dim]",
        title="HERALD Investigation",
        border_style=color,
        box=box.ROUNDED,
    ))

    why = Table(title="Why Flagged", show_header=True, header_style="bold cyan", box=box.SIMPLE_HEAVY)
    why.add_column("Reason")
    for reason in summary.get("why_flagged", ["No strong phishing indicators were observed."]):
        why.add_row(reason)
    console.print(why)

    intel = Table(title="Intelligence", show_header=True, header_style="bold cyan", box=box.SIMPLE)
    intel.add_column("Signal")
    intel.add_column("Value")
    dns = result.get("dns", {})
    tls = result.get("tls", {})
    lexical = result.get("lexical", {})
    visual = result.get("visual", {})
    intel.add_row("Lexical keywords", ", ".join(lexical.get("suspicious_keywords") or []) or "none")
    intel.add_row("Registrar", str(dns.get("registrar") or "unknown"))
    intel.add_row("Domain age", str(dns.get("domain_age_days") if dns.get("domain_age_days") is not None else "unknown"))
    intel.add_row("A records", ", ".join(dns.get("a_records") or []) or "unknown")
    intel.add_row("TLS issuer", str(tls.get("issuer") or "unknown"))
    intel.add_row("TLS suspicious", str(tls.get("suspicious")))
    intel.add_row("Screenshot", str(visual.get("screenshot_path") or "not captured"))
    intel.add_row("OCR phrases", ", ".join((visual.get("ocr_findings") or {}).get("phrases_found") or []) or "none")
    console.print(intel)

    risk = Table(title="Risk Factors", show_header=True, header_style="bold cyan", box=box.SIMPLE)
    risk.add_column("Factor")
    risk.add_column("Severity")
    risk.add_column("Impact", justify="right")
    risk.add_column("Explanation")
    for factor in result.get("risk_factors") or []:
        risk.add_row(
            str(factor.get("name", "Unknown")),
            format_severity(str(factor.get("severity", "unknown"))),
            str(factor.get("score_impact", 0)),
            str(factor.get("detail", "")),
        )
    console.print(risk)

    stages = Table(title="Lifecycle", show_header=True, header_style="bold cyan", box=box.SIMPLE)
    stages.add_column("Stage")
    stages.add_column("Status")
    stages.add_column("Time")
    for stage in result.get("stages", []):
        stages.add_row(stage["name"], stage["status"], f"{stage['duration_ms']}ms")
    console.print(stages)
    return 0


def emit_ssrf_error(exc: SSRFProtectionError, target: str, command: str, *, as_json: bool) -> int:
    is_metadata_block = "metadata" in (exc.reason or "").lower()
    suggested = f"herald {command} {target} --allow-private"
    payload = {
        "error": "ssrf_blocked",
        "message": "Metadata endpoint blocked" if is_metadata_block else "Target resolves to private/internal infrastructure",
        "target": target,
        "hostname": exc.hostname,
        "resolved_ip": exc.resolved_ip,
        "reason": exc.reason,
        "suggested_override": None if is_metadata_block else suggested,
    }
    if as_json:
        console.print_json(json.dumps(payload))
        return 2

    console.print(Panel(
        f"[bold]{payload['message']}[/bold]\n"
        f"Target: [cyan]{target}[/cyan]\n"
        f"Resolved IP: [yellow]{exc.resolved_ip or 'unknown'}[/yellow]\n"
        f"Reason: {exc.reason}"
        + ("" if is_metadata_block else f"\n\nUse [bold]{suggested}[/bold] if this is intentional."),
        title="SSRF Protection Blocked Target",
        border_style="red",
        box=box.ROUNDED,
    ))
    return 2


def execute_pipeline(pipeline: InvestigationPipeline, target: str, *, include_visual: bool, allow_private: bool, quiet: bool):
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return pipeline.investigate(target, include_visual=include_visual, allow_private=allow_private)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task("Investigating target...", total=None)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return pipeline.investigate(target, include_visual=include_visual, allow_private=allow_private)


def verdict_color(verdict: str) -> str:
    if verdict == "Phishing":
        return "red"
    if verdict == "Suspected":
        return "yellow"
    return "green"


def format_severity(severity: str) -> str:
    colors = {
        "high": "red",
        "medium": "yellow",
        "low": "bright_black",
        "info": "cyan",
    }
    color = colors.get(severity, "white")
    return f"[{color}]{severity}[/{color}]"


if __name__ == "__main__":
    sys.exit(main())
