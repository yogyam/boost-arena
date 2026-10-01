# Rules

Version 1. Changes are recorded in this file's history.

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
5. **One name per bot.** Names and descriptions must be civil and must not impersonate anyone.
6. **Up to three bots per person on the leaderboard.** Update one instead of adding a fourth.

## Model files

1. **The project does not publish model files.** Scores and replays are public. Your sealed model is opened only to score it and is then discarded.
2. **You may publish your own model** if you want to. That is your decision and your responsibility, and the rules above still apply to it.

## Strength

The bot-making community limits the strength of the bots it shares, because strong public bots have been used to cheat. Boost Arena follows the same principle, and holds no model files, which is the main safeguard.

1. **Bots that score 90 or more overall are reviewed before they are listed.** The maintainers may ask the entrant how the bot was trained and may decline to list it.
2. **Don't publish a model that is stronger than the bots the RLBot community shares.** Their rule of thumb is a Grand Champion level bot; a bot that reliably beats every baseline here by a wide margin is likely past it.

## Scoring

1. **Official scores come only from the project's scoring service.** Scores from your own machine are for your own use.
2. **Every bot faces the same situations** within a task set.
3. **A score stands once given.** Different machines give slightly different results, so a bot is scored once, when it is submitted or updated. The confidence interval says how precise a score is.
4. **Results are tied to versions.** Each result records the interface, simulator and task-set versions it was made with. When any of them changes, a new leaderboard starts and the old one is kept for reference.

## Maintainers

The maintainers may refuse or remove any entry that breaks these rules, and may change the rules. They aim to answer submission problems within a week, but this is a volunteer project.
