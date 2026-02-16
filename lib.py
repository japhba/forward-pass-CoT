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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ========================== CONFIG ==========================
NUM_CLASSES = 100  # probe output classes (intermediates live in 0–99)
MAX_LAYERS = 10    # subsample layers to at most this many
CACHE_DIR = os.path.join(os.environ["HF_HOME"], "forward-pass-CoT")
# ============================================================


@dataclass
class Example:
    expression: str
    prompt: str
    nhops: int
    intermediates: list[int]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

NHOPS = 4
OPERAND_RANGE = (2, 16)  # max intermediate = 16*5 = 80 < NUM_CLASSES
TOLERANCES = [0, 2, 5]


def _build_expression() -> Example:
    n_operands = NHOPS + 1
    operands = [random.randint(*OPERAND_RANGE) for _ in range(n_operands)]

    intermediates = []
    val = operands[0]
    expr = str(operands[0])
    for i in range(NHOPS):
        expr = f"({expr} + {operands[i+1]})"
        val = val + operands[i+1]
        intermediates.append(val)

    prompt = f"Compute: {expr} ="
    return Example(expression=expr, prompt=prompt, nhops=NHOPS, intermediates=intermediates)


def generate_data(n: int = 10_000, seed: int = 42) -> list[Example]:
    random.seed(seed)
    # 15^5 ≈ 760k unique expressions, so n=10k is duplicate-free
    seen = set()
    examples = []
    while len(examples) < n:
        ex = _build_expression()
        if ex.expression not in seen:
            seen.add(ex.expression)
            examples.append(ex)
    return examples


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, device


def _pick_layers(n_layers: int) -> list[int]:
    """Evenly subsample layer indices down to MAX_LAYERS, always including first and last."""
    if n_layers <= MAX_LAYERS:
        return list(range(n_layers))
    return sorted(set(np.linspace(0, n_layers - 1, MAX_LAYERS, dtype=int).tolist()))


def extract_hidden_states(model_name: str, examples: list[Example], batch_size: int = 64) -> dict:
    cache_path = os.path.join(CACHE_DIR, f"hidden_states_{model_name.replace('/', '_')}.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached hidden states from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    tokenizer, model, device = load_model(model_name)

    n_layers_total = model.config.num_hidden_layers + 1  # +1 for embedding layer
    layer_indices = _pick_layers(n_layers_total)
    hidden_dim = model.config.hidden_size
    print(f"Extracting {len(layer_indices)}/{n_layers_total} layers: {layer_indices}")

    all_hidden = torch.zeros(len(examples), len(layer_indices), hidden_dim, dtype=torch.float32)
    all_predictions = []

    for start in tqdm(range(0, len(examples), batch_size), desc="Extracting hidden states"):
        batch = examples[start:start + batch_size]
        prompts = [ex.prompt for ex in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for i, layer_idx in enumerate(layer_indices):
            hs = outputs.hidden_states[layer_idx]
            all_hidden[start:start + len(batch), i] = hs[:, -1, :].float().cpu()

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
        "layer_indices": layer_indices,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(result, cache_path)
    print(f"Saved hidden states to {cache_path}")
    return result


# ---------------------------------------------------------------------------
# Accuracy summary
# ---------------------------------------------------------------------------

def compute_accuracy(examples: list[Example], predictions: list[str]) -> tuple[int, int]:
    """Returns (n_correct, n_total)."""
    correct = sum(pred.startswith(str(ex.intermediates[-1])) for ex, pred in zip(examples, predictions))
    return correct, len(examples)


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------

def train_probes(data: dict, max_hop: int = NHOPS, n_epochs: int = 100,
                 lr: float = 1e-2, batch_size: int = 512,
                 cache_name: str = "probes") -> tuple[np.ndarray, np.ndarray]:
    """Returns (acc_matrix, loss_curves).
    acc_matrix: (n_layers, max_hop, len(TOLERANCES)) — accuracy at each tolerance.
    loss_curves: (n_layers, max_hop, n_epochs).
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached probe results from {cache_path}")
        cached = np.load(cache_path)
        return cached["acc_matrix"], cached["loss_curves"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_states = data["hidden_states"]
    intermediates = data["intermediates"]
    nhops_list = data["nhops"]

    n_examples, n_layers, hidden_dim = hidden_states.shape
    n_tol = len(TOLERANCES)
    acc_matrix = np.full((n_layers, max_hop, n_tol), np.nan)
    loss_curves = np.full((n_layers, max_hop, n_epochs), np.nan)

    for hop_level in tqdm(range(1, max_hop + 1), desc="Hop levels"):
        mask = [i for i, nh in enumerate(nhops_list) if nh >= hop_level]
        y_all = torch.tensor([intermediates[i][hop_level - 1] for i in mask], dtype=torch.long, device=device)
        X_all = hidden_states[mask]

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
                preds = probe(X_te).argmax(1)
                error = (preds - y_test).abs()
                for ti, tol in enumerate(TOLERANCES):
                    acc_matrix[layer_idx, hop_level - 1, ti] = (error <= tol).float().mean().item()

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, acc_matrix=acc_matrix, loss_curves=loss_curves)
    print(f"Saved probe results to {cache_path}")
    return acc_matrix, loss_curves
