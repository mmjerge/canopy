# Security Policy

## Supported versions

Canopy is an early-stage research project. Security fixes are applied to the latest
released version and the `dev` branch.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

Please **do not open a public issue** for security vulnerabilities.

Instead, report privately using GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability):
go to the repository's **Security** tab and choose **Report a vulnerability**. If that
is unavailable, contact the maintainer (@mmjerge) directly through GitHub.

When reporting, please include:

- a description of the issue and its potential impact,
- steps to reproduce or a proof of concept,
- any suggested mitigation, if known.

We aim to acknowledge reports within a few days and will keep you updated on progress.
Please give us a reasonable window to address the issue before any public disclosure.

## Scope notes

This library makes outbound calls to third-party model providers (AWS Bedrock, OpenAI)
only through the optional `canopy.llm` clients, and only when you supply credentials.
Credentials are read from your environment / standard provider credential chains and are
never logged or committed. Never commit API keys or `.env` files.
