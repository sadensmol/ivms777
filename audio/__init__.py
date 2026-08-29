"""Models-service-only audio models (design §5.1, §11.1).

Deliberately EMPTY. The modules below import torch through `kokoro` and
`transformers`, and `TorchWorker` resolves them by name inside a spawned child
— so importing this package must never pull either in (CLAUDE.md § "One model
process"). Anything added here is imported by every process that touches `audio.*`.
"""
