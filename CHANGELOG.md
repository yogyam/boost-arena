# Changelog

Dates are when the change reached `main`. Versions of the interface, task set and duel set are tracked separately in each result file.

## Unreleased

### Rules (version 2)
- Three scorings per person per 30 days, counting updates; one GitHub account per person.
- Seasons: the scoring seed changes each season and every bot is scored again. Season 1 uses seed 0, so existing scores stand.
- Entries under question are flagged on the leaderboard (`flags.json`).

### Submissions
- The manifest names the GitHub login that opens the pull request, and the pull request check compares them.
- Sealed files are bound to the submission's slug and login (format `BOOSTARENA-SEALED-2`); files sealed with the old format must be sealed again.
- A model or name already on the leaderboard is refused.

### Scoring service
- The private key is used only to open models; scoring and duels run without it, each model in its own process with a time and memory limit.
- A model that fails while running is recorded as not scored instead of stopping the run.
- Models are also checked for the size of their intermediate values and the time one decision takes; `Dropout` is no longer accepted.
- Everything published is checked against the repository: result and duel files must match the manifests on `main`, their numbers must agree, and only official results are published.
- Downloads follow only public https addresses, with a time limit.
- Actions pinned to commits; Dependabot; timeouts on every job.

### Website
- Redesigned leaderboard; three.js is served by the site itself, with a content security policy.

### Project
- The service installs from a hash-locked `requirements.lock`, and every result records the library versions it was made with.
- Lint and format checks (ruff), tests on Linux, macOS and Windows for Python 3.11 and 3.12, and the starter kit tested in CI. Python 3.13 is not supported until rlgym-rocket-league allows numpy 2.
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, CITATION.cff, `.editorconfig`.
- The starter kit uses the curriculum phase's learning rate and discount when a run starts or resumes.
- The simulator version in results is read from the installed package rather than a constant.

## 0.1.0 — 2026-10-01

First public version: interface 1, task set 1, duel set 1, sealed submissions, automatic scoring, leaderboard with replays, starter kit.
