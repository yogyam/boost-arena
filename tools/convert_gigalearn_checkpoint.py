#!/usr/bin/env python3
"""Converts a GigaLearnCPP checkpoint into a Boost Arena submission (one ONNX file).

A checkpoint folder holds SHARED_HEAD.lt and POLICY.lt. Both are read, joined into one
network and written as ONNX. The critic and the optimiser files are not needed.

    python tools/convert_gigalearn_checkpoint.py checkpoints/1300280064 my_bot.onnx

Only convert checkpoints you trained yourself or fully trust: the .lt format can run
code when it is opened. The ONNX file this writes cannot, which is why submissions use it.

Needs PyTorch:  pip install torch
"""

import argparse
import os
import sys

from boost_arena.export import ACTIVATIONS, check_and_serialize, read_gigalearn_layers


def main():
    parser = argparse.ArgumentParser(description="Convert a GigaLearnCPP checkpoint into a Boost Arena ONNX file")
    parser.add_argument("checkpoint", help="Checkpoint folder holding SHARED_HEAD.lt and POLICY.lt")
    parser.add_argument("output", help="ONNX file to write")
    parser.add_argument(
        "--activation",
        choices=sorted(ACTIVATIONS),
        default="leaky_relu",
        help="The activation the network was trained with (default: leaky_relu)",
    )
    args = parser.parse_args()

    layers = []
    for name in ("SHARED_HEAD.lt", "POLICY.lt"):
        path = os.path.join(args.checkpoint, name)
        if os.path.exists(path):
            layers += read_gigalearn_layers(path)
        elif name == "POLICY.lt":
            raise SystemExit(f"No POLICY.lt in {args.checkpoint}")

    try:
        data = check_and_serialize(layers, args.activation)
    except ValueError as e:
        raise SystemExit(f"{e}. Not written.") from None

    with open(args.output, "wb") as f:
        f.write(data)

    parameters = sum(weight.size + bias.size for _, weight, bias in layers)
    print(f"Wrote {args.output}: {parameters:,} parameters, {len(data) / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
