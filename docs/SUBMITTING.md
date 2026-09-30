# Submitting a bot

Four steps: seal your model, host the sealed file, open a pull request, and wait for the score.

Your model is sealed with the project's public key before it leaves your computer. Only the scoring service can open it. The project publishes your scores and never publishes your model.

## 1. Seal the model

You need a clone of this repository, with the tool installed:

```bash
git clone https://github.com/yogyam/boost-arena.git
cd boost-arena
pip install -e .
```

Check your model, then seal it:

```bash
boost-arena check my_bot.onnx
boost-arena submit my_bot.onnx --name "My Bot" --author "your name or handle"
```

Optional flags: `--description` (one line about the bot), `--homepage` (an https link), and `--public-model-url` if you publish the model yourself and want it linked.

This writes a folder `my_submission/my-bot/` with two files:

| File | What it is |
|---|---|
| `my-bot.sealed` | Your model, sealed. About the size of the model. |
| `submission.json` | The manifest: name, author, checksums, and where the sealed file will be. |

## 2. Host the sealed file

Put `my-bot.sealed` somewhere public with an `https://` address that serves the file directly. A release on one of your GitHub repositories works well and is free:

1. Go to any repository of yours, open **Releases**, and create a release.
2. Attach `my-bot.sealed` to it.
3. After publishing, right-click the attachment and copy its link.

Then open `submission.json` and put that link in `model_url`. Keep the file where it is: the scoring service downloads it once, when it scores your bot.

## 3. Open a pull request

Fork this repository, then add your manifest at `submissions/my-bot/submission.json`, where `my-bot` is the `slug` from the manifest. Nothing else goes in the folder, and the pull request must change nothing else.

Open the pull request. A check runs within a minute or two and reports whether the manifest and the sealed file are in order. Fix anything it reports by updating the pull request.

## 4. Wait for the score

A maintainer looks over the pull request and merges it. Scoring then runs on its own: it takes a few minutes, and the result appears on the leaderboard and in `results/my-bot.json`. If the model couldn't be scored, the reason appears on the leaderboard under "Not scored".

## Updating a bot

Seal the new model, host the new sealed file, and open a pull request that updates `submissions/my-bot/submission.json`. It is scored again and the leaderboard shows the new result. Want both versions listed? Give the new one a different name.

## What's checked

- The manifest has the right fields, and the sealed file is where it says, with the right checksum.
- The opened model is a valid ONNX file with only feed-forward operations, under 64 MB, with at most 16 million parameters. Run `boost-arena check` first to be sure.
- Scores come from 1,000 episodes per task with the standard settings; see [TASKS.md](TASKS.md).

## Trying the flow before you have a bot

```bash
boost-arena make-random-bot random.onnx
boost-arena score random.onnx --episodes 100
```

## Questions

Open an issue on the repository.
