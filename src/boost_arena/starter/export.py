"""Export a trained bot as a Boost Arena submission file.

    python -m boost_arena.starter.export --run my_first_bot my_first_bot.onnx

Uses the newest checkpoint of the run unless --checkpoint names one.
"""

import argparse
import os
import sys

import torch

from .actor import MaskedDiscreteFF
from .train import RUNS_FOLDER

ACTOR_CRITIC_FILE = "actor_critic.pt"


def newest_checkpoint(run_folder: str) -> str:
    """The most recently saved checkpoint folder under a run, or None."""
    candidates = []
    for folder, _, files in os.walk(os.path.join(run_folder, "checkpoints")):
        if ACTOR_CRITIC_FILE in files:
            path = os.path.join(folder, ACTOR_CRITIC_FILE)
            candidates.append((os.path.getmtime(path), os.path.dirname(folder)))  # The folder above ppo_learner/
    return max(candidates)[1] if candidates else None


def find_actor_file(checkpoint: str) -> str:
    for folder, _, files in os.walk(checkpoint):
        if ACTOR_CRITIC_FILE in files:
            return os.path.join(folder, ACTOR_CRITIC_FILE)
    raise SystemExit(f"No {ACTOR_CRITIC_FILE} under {checkpoint}")


def export(checkpoint: str, output: str, layer_sizes, layer_norm=True) -> int:
    state = torch.load(find_actor_file(checkpoint), map_location="cpu", weights_only=True)
    actor_state = {key[len("actor.") :]: value for key, value in state.items() if key.startswith("actor.")}

    actor = MaskedDiscreteFF(layer_sizes, layer_norm, torch.float32, torch.device("cpu"))
    try:
        actor.load_state_dict(actor_state)
    except RuntimeError as e:
        raise SystemExit(f"The checkpoint does not match --layers {','.join(map(str, layer_sizes))}: {e}") from None

    data = actor.export()
    with open(output, "wb") as f:
        f.write(data)
    return len(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a trained bot as a Boost Arena submission file")
    parser.add_argument("output", help="The ONNX file to write")
    parser.add_argument("--run", default="bot", help="Run name, as given to train")
    parser.add_argument("--checkpoint", help="A checkpoint folder, instead of the run's newest")
    parser.add_argument("--layers", default="256,256,256", help="The hidden layer sizes the run was trained with")
    parser.add_argument("--no-layer-norm", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint or newest_checkpoint(os.path.join(RUNS_FOLDER, args.run))
    if checkpoint is None:
        raise SystemExit(f"No checkpoints under {os.path.join(RUNS_FOLDER, args.run)}")
    layer_sizes = tuple(int(size) for size in args.layers.split(","))
    size = export(checkpoint, args.output, layer_sizes, not args.no_layer_norm)
    print(f"Wrote {args.output} ({size / 1e6:.1f} MB) from {checkpoint}")
    print(f"Check it with: boost-arena check {args.output}")


if __name__ == "__main__":
    sys.exit(main())
