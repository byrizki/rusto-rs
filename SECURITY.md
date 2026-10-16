# Security Policy

The RustO! project takes security and safety very seriously. As a pure-Rust optical character recognition (OCR) engine and multi-platform toolkit deployed across desktop, mobile, browser (WASM), and server environments, ensuring memory safety, dependency integrity, and vulnerability-free code is a top priority.

## Supported Versions

Only the latest active minor release line receives security updates and patches.

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| < 0.2.0 | :x:                |

## Continuous Security Monitoring

RustO! is actively secured and monitored in partnership with the **[Snyk Open Source Program](https://snyk.io/open-source/)**. Snyk provides automated vulnerability scanning across:
- **Rust dependencies** (`Cargo.lock`, crates.io packages)
- **Node.js dependencies** (`yarn.lock`, npm packages under `@rustors/*`)
- **Container and CI/CD environments**

Check our real-time security status:
[![Known Vulnerabilities](https://snyk.io/test/github/byrizki/rusto-rs/badge.svg)](https://snyk.io/test/github/byrizki/rusto-rs)

## Reporting a Vulnerability

If you discover a security vulnerability or suspect a security flaw in RustO!, please **do not open a public GitHub issue**. Instead, follow responsible disclosure practices:

1. **GitHub Private Vulnerability Reporting**:
   Navigate to the [Security Advisories tab](https://github.com/byrizki/rusto-rs/security/advisories) on GitHub and click **"Report a vulnerability"**.

2. **Email Disclosure**:
   If you cannot submit via GitHub Advisories, email the maintainer directly at:
   - **Email**: `rizki@quadrant.id` or via GitHub profile [@byrizki](https://github.com/byrizki)
   - Include a detailed description of the vulnerability, affected components/platforms, steps to reproduce, and any proof-of-concept (PoC) code or inputs.

### Our Response Commitment

- **Acknowledgment**: We aim to acknowledge receipt of security reports within **48 hours**.
- **Assessment**: We will investigate, verify severity, and discuss mitigation steps with the reporter.
- **Fix & Advisory**: Once a patch is developed and verified across supported platforms, we will publish a coordinated advisory and release updated packages to crates.io, npm, NuGet, CocoaPods, and JitPack.

## Security Practices

- **Zero C++ / OpenCV Runtime Dependency**: RustO! v0.3+ is written in 100% memory-safe Rust with the RTen inference engine, eliminating traditional memory corruption attack surfaces (buffer overflows, use-after-free) common in native C/C++ computer vision libraries.
- **Model Integrity**: Model files (`.rten` and `.onnx`) are verified with SHA-256 checksums during automated downloads.
- **Automated CI Scanning**: All commits and pull requests are tested and monitored for dependency vulnerabilities.
