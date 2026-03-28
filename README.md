# GPT-2 Implementation (PyTorch)

## Overview

Rather than relying on high-level pre-packaged layers like `nn.Transformer`, this project implements the core components at the tensor/matrix-math level. It features custom implementations of:

* Multi-Head Causal Self-Attention
* Transformer Blocks
* Positional Embeddings
* The training loop and text generation sequence

Strongly inspired by: https://github.com/karpathy/minGPT
