# Forward-Pass Chain-of-Thought

Linear probes on Qwen3-0.6B hidden states to detect whether intermediate arithmetic results become decodable across layers — evidence for implicit multi-step reasoning in a single forward pass.

Inspired by [Ryan Greenblatt's blogpost](https://www.redwoodresearch.com/blog/filler-tokens) on filler tokens and no-CoT math.

## Setup

```
uv sync
```

## Usage

Run `notebook.ipynb`. Model is configured via `MODEL_NAME` in `lib.py`.
