# Credits

Boost Arena depends on these projects. None of their code is copied into this repository; they are installed as packages.

| Project | Used for | Licence |
|---|---|---|
| [RocketSim](https://github.com/ZealanL/RocketSim) by ZealanL, Python package by [mtheall](https://github.com/mtheall/RocketSim) | The physics simulator | MIT |
| [RLGym](https://github.com/RLGym/rlgym) | Supplies the arena collision files the simulator needs | Apache 2.0 |
| [ONNX](https://github.com/onnx/onnx) | Reading and checking model files | Apache 2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Running models | MIT |
| [NumPy](https://numpy.org) | Arithmetic | BSD 3-Clause |

## The interface

Interface version 1 follows the setup described in:

> A. Coelho, D. Amaral, M. Camocho Carvalho, G. Leão, A. Sousa and L. P. Reis, "PISTY: Curriculum-Based Deep Reinforcement Learning of Ground and Aerial Striking in 1v1 Rocket League", 2026.

The action table is a masked variant of the lookup table introduced by the Necto bot and used across the RLGym community.

## Arena collision files

The simulator needs files describing the shape of the arena. They are not part of this repository. They are installed with the `rlgym-rocket-league` package.
