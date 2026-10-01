# Contributing

Thanks for taking an interest. Two kinds of contribution come in through pull requests, and they are handled differently.

## Entering a bot

That is a submission, not a code change: see [docs/SUBMITTING.md](docs/SUBMITTING.md). A submission pull request changes only `submissions/<slug>/submission.json` and is merged once its check passes.

## Changing the project

Bug reports and ideas go in [issues](https://github.com/yogyam/boost-arena/issues); there are templates for a submission problem and for task or rules feedback. For a change of any size beyond a typo, open an issue first so the shape of it can be agreed before you spend time on it.

### Setting up

```bash
git clone https://github.com/yogyam/boost-arena.git
cd boost-arena
pip install -e ".[dev]"
pytest -q
ruff check src tests tools dev && ruff format --check src tests tools dev
```

The tests run in about half a minute and need no network. `pip install -e ".[train]"` adds PyTorch and the RLGym training tools for the starter kit's tests.

### What a pull request needs

- Tests for what it changes. The suite is the project's evidence that scoring is fair; anything that touches `interface.py`, `sim.py`, `tasks.py` or `runner.py` must keep `tests/test_interface.py` and `tests/test_stepping.py` passing unchanged, because those pin the interface to recorded data.
- `ruff check` and `ruff format` clean.
- Prose in the style of the existing docs: short, concrete, and honest about limits.

### Changes that affect scores

Some changes make old results incomparable with new ones. Each has a version number that must be bumped, and a changelog entry:

| Change | Bump |
|---|---|
| What a bot sees or can do, the tick timing, the car, the physics settings | `INTERFACE_VERSION` in `interface.py` |
| A task's situations, time limit or outcome rule, or the task list | `TASK_SET_VERSION` in `tasks.py` |
| A duel's setup, time limit or episode count | `DUEL_SET_VERSION` in `duels.py` |
| The scoring seed | A new season: `SEASON` and `OFFICIAL_SEEDS` in `runner.py`, see `docs/MAINTAINING.md` |

A new task set or interface version starts a new leaderboard and archives the old one. Propose such changes in an issue first; they are announced before they take effect.

### Proposing a task

A good task measures one skill, can be set up from a seed so every bot faces the same situations, and has a clear success. Open an issue with the task or rules feedback template describing the situation, the success condition and why existing tasks do not cover it. A prototype in `dev/` is welcome.

## Code of conduct

Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
