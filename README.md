# Altera ML Interview

## Problem 
Your goal is to implement [Contrastive Decoding](https://arxiv.org/abs/2210.15097) with HuggingFace transformers and PyTorch.

Your code should use `Qwen/Qwen2.5-3B-Instruct` as the large model and `Qwen/Qwen2.5-Coder-0.5B-Instruct` as the small model and be implemented in `main.py`.

Your code should be correct first, but also efficient. Implement the token-level algorithm, rather than the beam search algorithm.

In addition to implementing main.py, please answer the following questions in `response.md`:

1. What should you do if the two models have different tokenizers?
2. Do you think contrastive decoding is used in practice?