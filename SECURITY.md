# Security Policy

## Supported Versions

The following table lists the versions of HERALD that are currently being supported with security updates. Since the project is in early active development, only the latest minor release line receives security patches.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.x | :x:                |

## Reporting a Vulnerability

We take the security of this threat intelligence platform very seriously. If you discover a vulnerability in HERALD (such as an SSRF bypass, a remote code execution vector in the Playwright worker, or an authentication bypass in the FastAPI endpoints), please report it responsibly.

**How to report:**
Please email **athiyo22118@iiitd.ac.in** with the subject line `[SECURITY VULNERABILITY] HERALD`. 

**What to include:**
- A description of the vulnerability and its potential impact.
- Step-by-step instructions to reproduce the issue.
- Any relevant logs, screenshots, or proof-of-concept code.

**What to expect:**
- You will receive an acknowledgment of your report within 48 hours.
- We will evaluate the vulnerability and determine the severity.
- If accepted, we will work on a patch and notify you before it is publicly released.
- You will receive credit in the release notes for the responsible disclosure (unless you prefer to remain anonymous).

Please **do not** open public GitHub issues for security vulnerabilities until they have been patched.
