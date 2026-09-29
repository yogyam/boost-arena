# Rules

> **Draft.** These rules are not final and submissions are not open yet.

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
4. **Your own work.** Submit bots you trained, or have permission to submit. Say so if you started from someone else's model.
5. **One name per bot.** Keep names and descriptions civil.

## Model files

1. **The project does not publish model files.** Scores and replays are public. Your model is used for scoring and then discarded.
2. **You may publish your own model** if you want to. That is your decision and your responsibility.

## Strength

The bot-making community limits the strength of the bots it shares, because strong public bots have been used to cheat. Boost Arena follows the same principle.

1. **Bots at or above the level of top human players are not listed.**
2. **The maintainers decide**, and may remove an entry for this reason.

## Scoring

1. **Official scores come only from the project's scoring service.** Scores from your own machine are for your own use.
2. **Every bot faces the same situations** within a season.
3. **Results are tied to versions.** Each result records the interface, simulator and task versions it was made with. Results from different versions are not compared.

## Maintainers

The maintainers may refuse or remove any entry that breaks these rules, and may change the rules. Changes are recorded in this file's history.
