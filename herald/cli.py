"""Command-line interface for HERALD.

Usage examples::

    # Analyse a URL
    herald url "http://paypa1-secure.xyz/login"

    # Analyse email headers from a file
    herald headers --file suspicious.eml

    # Analyse email headers inline
    herald headers --from "Security Team <noreply@paypa1-secure.xyz>" \\
                   --subject "URGENT: Verify your account"

    # Full email analysis from a raw .eml file
    herald email suspicious.eml

    # Adjust detection threshold
    herald url "http://example.com" --threshold 0.4

    # Output as JSON
    herald url "http://paypa1-secure.xyz" --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

from herald.config import Config
from herald.detector import PhishingDetector
from herald.logger import configure_logging, get_logger

logger = get_logger("cli")


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _cmd_url(args: argparse.Namespace, detector: PhishingDetector) -> int:
    """Handle the ``url`` sub-command."""
    result = detector.analyze_url(args.url)
    _print_result(result, args.format)
    return 1 if result.is_phishing else 0


def _cmd_text(args: argparse.Namespace, detector: PhishingDetector) -> int:
    """Handle the ``text`` sub-command."""
    if args.file:
        try:
            with open(args.file, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            logger.error("Cannot read file '%s': %s", args.file, exc)
            return 2
    else:
        text = args.text or ""

    result = detector.analyze_text(text)
    _print_result(result, args.format)
    return 1 if result.is_phishing else 0


def _cmd_headers(args: argparse.Namespace, detector: PhishingDetector) -> int:
    """Handle the ``headers`` sub-command."""
    headers: Dict[str, str] = {}

    if args.file:
        try:
            with open(args.file, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            import email as email_lib
            import email.policy as email_policy

            msg = email_lib.message_from_string(raw, policy=email_policy.default)
            headers = {k: str(v) for k, v in msg.items()}
        except OSError as exc:
            logger.error("Cannot read file '%s': %s", args.file, exc)
            return 2
    else:
        if args.from_addr:
            headers["From"] = args.from_addr
        if args.reply_to:
            headers["Reply-To"] = args.reply_to
        if args.subject:
            headers["Subject"] = args.subject
        if args.return_path:
            headers["Return-Path"] = args.return_path

    if not headers:
        logger.error("No headers provided.  Use --file or supply --from/--subject.")
        return 2

    result = detector.analyze_headers(headers)
    _print_result(result, args.format)
    return 1 if result.is_phishing else 0


def _cmd_email(args: argparse.Namespace, detector: PhishingDetector) -> int:
    """Handle the ``email`` sub-command."""
    try:
        with open(args.file, encoding="utf-8", errors="replace") as fh:
            raw_message = fh.read()
    except OSError as exc:
        logger.error("Cannot read file '%s': %s", args.file, exc)
        return 2

    result = detector.analyze_email(raw_message=raw_message)
    _print_result(result, args.format)
    return 1 if result.is_phishing else 0


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_result(result, output_format: str) -> None:  # noqa: ANN001
    if output_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="herald",
        description="HERALD: AI-Based Phishing Detection for Critical Infrastructure",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Phishing score threshold (0–1).  Overrides HERALD_THRESHOLD env var.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Path to a saved model file.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- url ---
    url_p = sub.add_parser("url", help="Analyse a single URL.")
    url_p.add_argument("url", help="URL to analyse.")

    # --- text ---
    text_p = sub.add_parser("text", help="Analyse message body text.")
    text_g = text_p.add_mutually_exclusive_group(required=True)
    text_g.add_argument("--file", "-f", help="Path to a text file to analyse.")
    text_g.add_argument("text", nargs="?", help="Text string to analyse.")

    # --- headers ---
    hdr_p = sub.add_parser("headers", help="Analyse email headers.")
    hdr_src = hdr_p.add_mutually_exclusive_group()
    hdr_src.add_argument(
        "--file", "-f", help="Path to a raw .eml file (headers are extracted)."
    )
    hdr_p.add_argument("--from-addr", metavar="FROM", help="From: header value.")
    hdr_p.add_argument("--reply-to", metavar="REPLY-TO", help="Reply-To: header value.")
    hdr_p.add_argument("--subject", metavar="SUBJECT", help="Subject: header value.")
    hdr_p.add_argument(
        "--return-path", metavar="RETURN-PATH", help="Return-Path: header value."
    )

    # --- email ---
    email_p = sub.add_parser("email", help="Full analysis of a raw .eml file.")
    email_p.add_argument("file", help="Path to the .eml file.")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Returns:
        Exit code: 0 = legitimate, 1 = phishing detected, 2 = error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    config = Config.from_env()
    if args.threshold is not None:
        config.phishing_threshold = args.threshold

    detector = PhishingDetector(config=config, model_path=args.model)

    dispatch = {
        "url": _cmd_url,
        "text": _cmd_text,
        "headers": _cmd_headers,
        "email": _cmd_email,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 2

    return handler(args, detector)


if __name__ == "__main__":
    sys.exit(main())
