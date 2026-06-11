"""Utility functions for dataset generation and saving."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np


def powerset(items: list) -> list[tuple]:
    """Return all 2^N subsets (as tuples) of items, ordered by size then value."""
    patterns = []
    for k in range(len(items) + 1):
        patterns.extend(combinations(items, k))
    return patterns


def save_split(data: np.ndarray | list, labels: np.ndarray, meta: list, name: str,
               out_dir: Path | str):
    """Save one dataset split to disk as .npy (signals) and .json (labels + metadata).

    Args:
        data: Signal array of shape (N, sample_length).
        labels: Multi-label array of shape (N, num_discriminative_frequencies).
        meta: List of N metadata dicts with ground-truth explanation entries.
        name: Split name used as filename prefix, e.g. ``"train"``, ``"val"``, ``"test"``.
        out_dir: Output directory. Created if it does not exist.
    """
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}_data.npy", data)
    label_json = {
        "label": {str(i): labels[i].tolist() for i in range(len(labels))},
        "metadata": {
            "explanation_ground_truth": {
                "time_domain": {str(i): meta[i] for i in range(len(meta))}
            }
        },
    }
    (out_dir / f"{name}_labels.json").write_text(json.dumps(label_json))
