# Rules

Version 2. Changes are recorded in [CHANGELOG.md](CHANGELOG.md) and in this file's history.

## The spirit of the project

Boost Arena exists so people can learn reinforcement learning, compare methods fairly, and enjoy watching their bots improve. It is for bots that play in a simulator.

## What you may not do

1. **No online play against people.** Nothing from this project may be used to play Rocket League online, in ranked, casual, private or tournament matches. Using a bot against people who haven't agreed to it is cheating.
2. **No cheat tools.** Don't submit, link to or discuss tools for getting a bot into online play.
3. **No money.** The project is free and non-commercial. No entry fees, paid features or selling of bots through it.

## Submissions

1. **One ONNX file per bot**, following [the interface](docs/INTERFACE.md). No code is submitted.
2. **The file must pass `boost-arena check`.** It must be self-contained, feed-forward, under 64 MB, and have at most 16 million parameters.
3. **Train however you like.** Any framework, method, reward design or hardware.
4. **Your own work.** Submit bots you trained, or have permission to submit. Say so in the description if you started from someone else's model.
5. **One name per bot.** Names and descriptions must be civil and must not impersonate anyone. A name already on the leaderboard cannot be used for another bot.
6. **Up to three bots per person on the leaderboard.** Update one instead of adding a fourth.
7. **Up to three scorings per person in any 30 days**, counting new bots and updates together. Test locally first; the official run is not for finding out whether a model works.
8. **One GitHub account per person.** The manifest names the account that opens the pull request, the sealed file is tied to it, and scorings are counted per account. Second accounts are removed along with their entries.

## Model files

1. **The project does not publish model files.** Scores and replays are public. Your sealed model is opened only to score it and is then discarded.
2. **You may publish your own model** if you want to. That is your decision and your responsibility, and the rules above still apply to it.

## Strength

The bot-making community limits the strength of the bots it shares, because strong public bots have been used to cheat. Boost Arena follows the same principle, and holds no model files, which is the main safeguard.

1. **Bots that score 90 or more overall are reviewed before they are listed.** The maintainers may ask the entrant how the bot was trained and may decline to list it.
2. **Don't publish a model that is stronger than the bots the RLBot community shares.** Their rule of thumb is a Grand Champion level bot; a bot that reliably beats every baseline here by a wide margin is likely past it.

## Scoring

1. **Official scores come only from the project's scoring service.** Scores from your own machine are for your own use.
2. **Every bot faces the same situations** within a season.
3. **A score stands once given.** Different machines give slightly different results, so a bot is scored once, when it is submitted or updated. The confidence interval says how precise a score is.
4. **Seasons.** The situations come from a seed that changes each season, so a bot tuned to the exact situations of one season gains nothing in the next. A season lasts at least three months. When it ends, the leaderboard is archived, and every bot whose sealed file is still available is scored again on the new seed, free of charge against the scoring limit. Season changes are announced on the repository two weeks ahead.
5. **Results are tied to versions.** Each result records the season and the interface, simulator and task-set versions it was made with. When any of them changes, a new leaderboard starts and the old one is kept for reference.
6. **Scoring is done in isolation.** Each model runs in its own process with a time and memory limit. A model that fails, stalls or exhausts memory is listed as not scored, and that scoring counts towards the limit.

## Integrity

1. **Only the scoring service writes scores.** Every published result is checked against the submission it is for before it is published; a score cannot be edited into the leaderboard by hand or by a pull request.
2. **A sealed file belongs to one submission.** Pointing a manifest at someone else's sealed file, or submitting a model already on the leaderboard, is refused.
3. **Entries may be flagged.** If an entry is under question, the maintainers mark it on the leaderboard with a link to the discussion, and remove it if the question is not answered.

## Maintainers

The maintainers may refuse or remove any entry that breaks these rules, and may change the rules. Rule changes are announced on the repository before they take effect, except for fixes to loopholes. They aim to answer submission problems within a week, but this is a volunteer project.
