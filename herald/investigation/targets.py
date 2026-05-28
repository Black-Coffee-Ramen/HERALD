from __future__ import annotations

from urllib.parse import urlparse


def normalize_target(target: str) -> tuple[str, str]:
    raw = target.strip()
    if not raw:
        raise ValueError("target cannot be empty")

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    domain = (parsed.hostname or "").lower().removeprefix("www.")
    if not domain:
        raise ValueError(f"could not determine domain from target: {target}")

    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""
    url = f"{parsed.scheme}://{domain}{path}{query}"
    return url, domain


def safe_path_fragment(value: str) -> str:
    return "".join(char if char.isalnum() or char in ".-" else "_" for char in value)[:120]

