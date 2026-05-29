from __future__ import annotations

import socket
import ssl
from datetime import datetime, timezone
from typing import Any

import dns.resolver
import whois


def _first_datetime(value: Any) -> datetime | None:
    if isinstance(value, list):
        value = next((item for item in value if item), None)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return None


def _short_error(exc: Exception) -> str:
    return str(exc).splitlines()[0][:240]


def collect_dns_intelligence(domain: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "registrar": None,
        "creation_date": None,
        "domain_age_days": None,
        "nameservers": [],
        "a_records": [],
        "mx_records": [],
        "txt_records": [],
        "asn": None,
        "ip_metadata": [],
        "errors": [],
    }

    try:
        w = whois.whois(domain)
        created_at = _first_datetime(getattr(w, "creation_date", None))
        result["registrar"] = getattr(w, "registrar", None)
        result["creation_date"] = created_at.isoformat(timespec="seconds") if created_at else None
        if created_at:
            result["domain_age_days"] = (datetime.now(timezone.utc) - created_at).days
        nameservers = getattr(w, "name_servers", None) or []
        if isinstance(nameservers, str):
            nameservers = [nameservers]
        result["nameservers"] = sorted({str(ns).rstrip(".").lower() for ns in nameservers if ns})
    except whois.parser.PywhoisError as exc:
        result["errors"].append(f"whois: {_short_error(exc)}")

    resolver = dns.resolver.Resolver()
    resolver.timeout = 3
    resolver.lifetime = 5

    for record_type, key in [("A", "a_records"), ("MX", "mx_records"), ("TXT", "txt_records")]:
        try:
            answers = resolver.resolve(domain, record_type)
            result[key] = [str(answer).strip('"') for answer in answers]
        except dns.exception.DNSException as exc:
            result["errors"].append(f"{record_type}: {_short_error(exc)}")

    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(domain, 443, type=socket.SOCK_STREAM):
            ip = sockaddr[0]
            if not any(item["ip"] == ip for item in result["ip_metadata"]):
                result["ip_metadata"].append({"ip": ip, "family": "IPv6" if family == socket.AF_INET6 else "IPv4"})
    except socket.gaierror as exc:
        result["errors"].append(f"ip: {_short_error(exc)}")

    return result


def collect_tls_intelligence(domain: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "has_tls": False,
        "issuer": None,
        "subject": None,
        "not_before": None,
        "not_after": None,
        "days_remaining": None,
        "san": [],
        "domain_in_san": False,
        "suspicious": False,
        "errors": [],
    }

    try:
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as tls_sock:
                cert = tls_sock.getpeercert()
    except (socket.timeout, socket.gaierror, ssl.SSLError, ConnectionError) as exc:
        result["errors"].append(_short_error(exc))
        return result

    result["has_tls"] = True
    issuer_parts = cert.get("issuer", ())
    subject_parts = cert.get("subject", ())
    result["issuer"] = ", ".join("=".join(item) for part in issuer_parts for item in part)
    result["subject"] = ", ".join("=".join(item) for part in subject_parts for item in part)
    result["not_before"] = cert.get("notBefore")
    result["not_after"] = cert.get("notAfter")

    san = [value for kind, value in cert.get("subjectAltName", []) if kind == "DNS"]
    result["san"] = san
    result["domain_in_san"] = any(_san_matches(domain, item) for item in san)

    try:
        expires = datetime.fromtimestamp(ssl.cert_time_to_seconds(cert["notAfter"]), tz=timezone.utc)
        result["days_remaining"] = (expires - datetime.now(timezone.utc)).days
    except ValueError:
        pass

    issuer = (result["issuer"] or "").lower()
    result["suspicious"] = bool(
        result["has_tls"]
        and (not result["domain_in_san"] or (result["days_remaining"] is not None and result["days_remaining"] < 7))
    )
    if "let's encrypt" in issuer and result["days_remaining"] is not None and result["days_remaining"] < 15:
        result["suspicious"] = True

    return result


def _san_matches(domain: str, san: str) -> bool:
    san = san.lower()
    domain = domain.lower()
    if san.startswith("*."):
        return domain.endswith(san[1:])
    return domain == san
