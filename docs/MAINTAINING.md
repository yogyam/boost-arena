# Maintaining the scoring service

Notes for whoever holds the keys. Entrants don't need this page.

## How a submission gets scored

1. A pull request adds or changes `submissions/<slug>/submission.json`. The `Check submission` workflow (no secrets) verifies the manifest, downloads the sealed file to check its hash, refuses copies of other entries, and checks the login and the scoring limit. The pull request must change nothing outside `submissions/`.
2. A maintainer reads the manifest and merges. Only merge submission pull requests that pass the check; never merge one that also touches code or workflows.
3. `Score submissions` runs on the push to `main`:
   - `open-submissions` is the only step with `BOOST_ARENA_PRIVATE_KEY`. It opens every pending model into `$RUNNER_TEMP/opened_models` (outside the checkout, never uploaded) and removes the key from its environment. A submission that cannot be opened gets an error result.
   - `process-submissions` and `process-duels` run each model, or each pair, in a child process with a 45-minute and 8 GB limit. A crash, timeout or memory blow-up becomes an error result; a half-played duel is dropped.
   - The `publish` job validates every file against the repository (`boost-arena validate-results`), commits `results/`, `replays/` and `duels/`, and deploys the site. If validation fails, nothing is committed; look at the job log.
4. The weekly cron plays duels still outstanding, 15 pairs per run.

## When a run fails

- **Validation failed in publish**: the scoring job produced something inconsistent. The artifact `new-results` holds what it made (7 days). Fix the cause, then re-run the workflow from the Actions tab; the submission is still pending so it is scored again.
- **The job timed out** (5 hours): too many pending submissions or duels. Re-run; already-scored results are kept once published, and the run picks up the rest. Lower `--max-pairs` if duels are the problem.
- **A bot's sealed file is gone**: its result records the download error. The entrant fixes the address and opens a pull request; or if the bot had a score, it keeps it but sits out new duels and re-scorings.

## Rotating the private key

`boost-arena keygen` prints a new pair. Replace `PUBLIC_KEY` in the repository, the `BOOST_ARENA_PRIVATE_KEY` secret, and the offline backup. Announce that every entrant must seal their model again (`boost-arena submit`) and update their manifest. Existing scores stay.

Keep a copy of the private key off GitHub (a password manager is fine). A GitHub secret cannot be read back, and without the key no submission can be opened.

## Changing a season

1. Announce it two weeks ahead (issue plus the README status line).
2. Archive the leaderboard: copy `results/`, `duels/` and the built site into `archive/season-N/` and commit.
3. Add the new seed to `OFFICIAL_SEEDS` in `src/boost_arena/runner.py`, bump `SEASON`, add a changelog entry, and merge.
4. Run `Score submissions` by hand. Every bot is re-scored (its result's season no longer matches) and every pair is re-played. With many bots this takes several runs; the cron finishes the duels.

## Flagging an entry

Add `"<slug>": "https://github.com/yogyam/boost-arena/issues/<n>"` to `flags.json`; the leaderboard shows a "Flagged" link. Remove the line to clear it, or remove the submission folder and its result files to delist the entry.

## Branch protection

`main` should require the `Check submission` and `Tests` checks to pass, and a review from a code owner for anything outside `submissions/` (see `.github/CODEOWNERS`). Set this in the repository settings; it is not in the code.
