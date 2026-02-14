"""
Backend library for decoding multi-hop arithmetic intermediates from hidden states.
"""

import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ========================== MODEL CONFIG ==========================
MODEL_NAME = "Qwen/Qwen3-4B"
# ==================================================================


@dataclass
class Example:
    expression: str
    prompt: str
    nhops: int
    intermediates: list[int]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _build_expression(nhops: int) -> Example:
    n_operands = nhops + 1
    operands = [random.randint(2, 9) for _ in range(n_operands)]

    intermediates = []
    val = operands[0]
    expr = str(operands[0])
    for i in range(nhops):
        expr = f"({expr} + {operands[i+1]})"
        val = val + operands[i+1]
        intermediates.append(val)

    prompt = f"Compute: {expr} ="
    return Example(expression=expr, prompt=prompt, nhops=nhops, intermediates=intermediates)


def generate_data(n_per_hop: int = 10000, seed: int = 42) -> list[Example]:
    random.seed(seed)
    examples = []
    for nhops in tqdm([2, 3, 4, 5], desc="Generating data"):
        for _ in range(n_per_hop):
            examples.append(_build_expression(nhops))
    return examples


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(examples: list[Example], batch_size: int = 64,
                          cache_path: str = "data/hidden_states.pt") -> dict:
    if os.path.exists(cache_path):
        print(f"Loading cached hidden states from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size

    all_hidden = torch.zeros(len(examples), n_layers, hidden_dim, dtype=torch.float32)
    all_predictions = []

    for start in tqdm(range(0, len(examples), batch_size), desc="Extracting hidden states"):
        batch = examples[start:start + batch_size]
        prompts = [ex.prompt for ex in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx in range(n_layers):
            hs = outputs.hidden_states[layer_idx]
            all_hidden[start:start + len(batch), layer_idx] = hs[:, -1, :].float().cpu()

        gen_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        for i, ex in enumerate(batch):
            input_len = inputs["input_ids"].shape[1]
            generated = tokenizer.decode(gen_ids[i, input_len:], skip_special_tokens=True).strip()
            all_predictions.append(generated)

    result = {
        "hidden_states": all_hidden,
        "predictions": all_predictions,
        "intermediates": [ex.intermediates for ex in examples],
        "nhops": [ex.nhops for ex in examples],
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(result, cache_path)
    print(f"Saved hidden states to {cache_path}")
    return result


# ---------------------------------------------------------------------------
# Accuracy summary
# ---------------------------------------------------------------------------

def compute_accuracy(examples: list[Example], predictions: list[str]) -> dict[int, tuple[int, int]]:
    """Returns {nhops: (n_correct, n_total)}."""
    correct_by_nhops: dict[int, list[bool]] = {}
    for ex, pred in zip(examples, predictions):
        expected = str(ex.intermediates[-1])
        correct_by_nhops.setdefault(ex.nhops, []).append(pred.startswith(expected))
    return {nh: (sum(vals), len(vals)) for nh, vals in sorted(correct_by_nhops.items())}


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------

NUM_CLASSES = 100


def train_probes(data: dict, max_hop: int = 5, n_epochs: int = 100,
                 lr: float = 1e-2, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Returns (acc_matrix, loss_curves) where loss_curves is (n_layers, max_hop, n_epochs)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_states = data["hidden_states"]
    intermediates = data["intermediates"]
    nhops_list = data["nhops"]

    n_examples, n_layers, hidden_dim = hidden_states.shape
    acc_matrix = np.full((n_layers, max_hop), np.nan)
    loss_curves = np.full((n_layers, max_hop, n_epochs), np.nan)

    for hop_level in tqdm(range(1, max_hop + 1), desc="Hop levels"):
        mask = [i for i, nh in enumerate(nhops_list) if nh >= hop_level]
        if len(mask) < 50:
            continue

        y_all = torch.tensor([intermediates[i][hop_level - 1] for i in mask], dtype=torch.long, device=device)
        X_all = hidden_states[mask]  # keep on CPU

        train_idx, test_idx = train_test_split(range(len(mask)), test_size=0.2, random_state=42)
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        for layer_idx in tqdm(range(n_layers), desc=f"  Layers (hop {hop_level})", leave=False):
            X_tr = X_all[train_idx, layer_idx].to(device)
            X_te = X_all[test_idx, layer_idx].to(device)

            probe = nn.Linear(hidden_dim, NUM_CLASSES).to(device)
            optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(n_epochs):
                probe.train()
                perm = torch.randperm(X_tr.shape[0], device=device)
                for start in range(0, X_tr.shape[0], batch_size):
                    idx = perm[start:start + batch_size]
                    optimizer.zero_grad()
                    loss_fn(probe(X_tr[idx]), y_train[idx]).backward()
                    optimizer.step()

                probe.eval()
                with torch.no_grad():
                    loss_curves[layer_idx, hop_level - 1, epoch] = loss_fn(probe(X_te), y_test).item()

            with torch.no_grad():
                acc_matrix[layer_idx, hop_level - 1] = (probe(X_te).argmax(1) == y_test).float().mean().item()

    return acc_matrix, loss_curves
