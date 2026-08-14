# Third-party notices

This repository contains an independent MLX implementation and does not distribute
MiniMax Music 3 model weights.

## MiniMax Music 3

The official checkpoint, configuration, tokenizer, examples, and model license are
published at:

- https://huggingface.co/MiniMaxAI/MiniMax-Music3
- https://github.com/MiniMax-AI/MiniMax-Music3

The weights are governed by the MiniMax-Music3 Community License. That license is
separate from this repository's MIT license and includes attribution, acceptable-use,
and commercial terms. Users are responsible for reviewing the current upstream
license before downloading or using the checkpoint.

## Reference implementations

Reference checkouts are local development inputs and are not included in source or
wheel distributions:

- Hugging Face Diffusers: Apache License 2.0
- SGLang-Omni: Apache License 2.0
- MLX-VLM, MLX-LM, and related Apple MLX examples: MIT License
- ComfyUI: GNU General Public License version 3

Direct adaptations must retain all notices required by the applicable upstream
license. ComfyUI is used only to compare behavior and results; its implementation
code is not copied into this package.
