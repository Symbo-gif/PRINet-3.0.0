# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 3.0.x   | ✅ Yes     |
| < 3.0   | ❌ No      |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in PRINet, please report it by emailing:

**therealmichaelmaillet@gmail.com**

Include the following in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested mitigations you are aware of

You will receive an acknowledgement within **48 hours** and a full response with remediation timeline within **7 days**.

## Scope

Security reports are appropriate for:
- Arbitrary code execution via deserialization of model checkpoints
- CUDA kernel memory safety issues that could affect the host system
- Dependency vulnerabilities with known CVEs affecting the public API

Out of scope:
- Numerical precision issues in oscillator simulations
- Performance regressions
- Issues requiring physical access to the machine
- Social engineering attacks

## Preferred Languages

We prefer reports in English.

## Disclosure Policy

- We ask that you give us reasonable time to address the vulnerability before public disclosure.
- We will credit reporters in the release notes unless you request otherwise.
- We follow responsible disclosure practices aligned with the CVD (Coordinated Vulnerability Disclosure) standard.
