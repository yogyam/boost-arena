# Boost Arena

An open benchmark for car-football bots. Train a bot however you like, submit the trained model, and see how it scores on a set of fixed tasks.

Bots are scored in [RocketSim](https://github.com/ZealanL/RocketSim), an open-source simulator of Rocket League's physics. The game itself is never run.

> **Status: early development.** The scoring tool works and has one task. Submissions, the leaderboard and replays are not built yet.

## How it works

1. **Everyone uses the same interface.** A bot sees 53 numbers describing the ball and cars, and picks one of 90 actions, 15 times a second. See [docs/INTERFACE.md](docs/INTERFACE.md).
2. **You train your own way.** Any framework, any method, any hardware.
3. **You submit one model file**, in ONNX format. No code is submitted or run.
4. **Every bot faces the same situations** and gets a success rate and a time to score for each task.

## Try it

Needs Python 3.11 or newer. Works on Linux, macOS (Apple Silicon) and Windows.

```bash
git clone https://github.com/yogyam/boost-arena.git
cd boost-arena
pip install -e .

boost-arena tasks                 # List the tasks
boost-arena check my_bot.onnx     # Is this file an acceptable submission?
boost-arena score my_bot.onnx     # Score it
```

Example output:

```
Empty-net finish
  Success rate     80.2%  (95% interval 76.5% to 83.5%)
  Time to score    5.4 s on average, limit 20 s
  Episodes         500: 401 scored, 1 own goals, 98 ran out of time
```

Scores from your own machine are for your own use. Official leaderboard scores will come from the project's scoring service, because results can differ slightly between computers.

## Tasks

| Task | The bot must | Time limit |
|---|---|---|
| Empty-net finish | Score from a still ball in the attacking half, with nobody in goal | 20 s |

More are planned: finishing a pass, a falling ball, an aerial cross, making a save, and a penalty shoot-out against a keeper.

## Making a model file

If you trained with GigaLearnCPP, convert a checkpoint with:

```bash
pip install torch
python tools/convert_gigalearn_checkpoint.py path/to/checkpoint my_bot.onnx
```

From any other framework, export your policy network to ONNX with an input of shape (N, 53) and an output of shape (N, 90). The output is one logit per action. Masking and choosing the action are done by the benchmark.

## Rules

See [RULES.md](RULES.md). In short: the project is free and non-commercial, bots are for the simulator and for research, and nothing here may be used to play online against people.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Credits and licences

Boost Arena is released under the [MIT licence](LICENSE). It builds on the work of others: see [CREDITS.md](CREDITS.md).

## Disclaimer

Boost Arena is a fan project. It is not affiliated with Psyonix or Epic Games.

Portions of the materials used are trademarks and/or copyrighted works of Epic Games, Inc. All rights reserved by Epic. This material is not official and is not endorsed by Epic.
