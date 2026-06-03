import ipaddress
import socket
from urllib.parse import urlparse
import structlog

logger = structlog.get_logger(__name__)

class SSRFProtectionError(Exception):
    def __init__(self, message: str, *, hostname: str | None = None, resolved_ip: str | None = None, reason: str | None = None):
        super().__init__(message)
        self.hostname = hostname
        self.resolved_ip = resolved_ip
        self.reason = reason or message


METADATA_HOSTNAMES = {"metadata.google.internal"}
METADATA_IPS = {"169.254.169.254"}


def validate_url_safe(
    url: str,
    *,
    allow_private: bool = False,
    trace_id: str | None = None,
    private_overrides: list[dict[str, str | None]] | None = None,
) -> str:
    """
    Validates a URL against SSRF (Server-Side Request Forgery).
    Raises SSRFProtectionError if the URL is unsafe (e.g. localhost, private IP).
    Returns the normalized domain.
    """
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            raise SSRFProtectionError(f"Invalid URL or missing hostname: {url}")

        if hostname.lower() in METADATA_HOSTNAMES or hostname in METADATA_IPS:
            raise SSRFProtectionError(
                "Metadata endpoints are blocked.",
                hostname=hostname,
                resolved_ip=hostname if hostname in METADATA_IPS else None,
                reason="Cloud metadata endpoints are always blocked",
            )
            
        # Extract IP addresses
        addresses = socket.getaddrinfo(hostname, None)
        for result in addresses:
            ip_str = result[4][0]
            ip_obj = ipaddress.ip_address(ip_str)

            if ip_str in METADATA_IPS:
                raise SSRFProtectionError(
                    "Metadata endpoints are blocked.",
                    hostname=hostname,
                    resolved_ip=ip_str,
                    reason="Cloud metadata endpoints are always blocked",
                )
            
            # Block loopback, private, multicast, reserved
            if ip_obj.is_loopback or ip_obj.is_private or ip_obj.is_multicast or ip_obj.is_reserved:
                if allow_private:
                    if private_overrides is not None:
                        private_overrides.append({
                            "target": url,
                            "hostname": hostname,
                            "resolved_ip": str(ip_obj),
                            "trace_id": trace_id,
                            "reason": _blocked_ip_reason(ip_obj),
                        })
                    logger.warning(
                        "ssrf_private_override_used",
                        url=url,
                        resolved_ip=str(ip_obj),
                        hostname=hostname,
                        trace_id=trace_id,
                    )
                    continue

                logger.warning("ssrf_blocked", url=url, resolved_ip=str(ip_obj), hostname=hostname)
                raise SSRFProtectionError(
                    "Target resolves to private/internal infrastructure.",
                    hostname=hostname,
                    resolved_ip=ip_str,
                    reason=_blocked_ip_reason(ip_obj),
                )
            
        return hostname

    except socket.gaierror as e:
        # If it doesn't resolve, we let it pass the SSRF check as it will just fail DNS later anyway
        # but to be safe, we could reject it. Let's allow it to fail later to get a proper error trace.
        logger.info("ssrf_dns_resolution_failed", url=url, error=str(e))
        return parsed.hostname
    except ValueError as e:
        raise SSRFProtectionError(f"Invalid URL formatting: {str(e)}")


def _blocked_ip_reason(ip_obj: ipaddress._BaseAddress) -> str:
    if ip_obj.is_loopback:
        return "loopback address"
    if ip_obj.is_private:
        return "private/internal address"
    if ip_obj.is_multicast:
        return "multicast address"
    if ip_obj.is_reserved:
        return "reserved address"
    return "blocked internal address"
