"""Generate, split, and save the benchmark dataset."""

import argparse

import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split

from data.benchmark import Benchmark
from utils import save_split


def train_test_val_split(indices: ndarray,
                         stratify_keys: list|None = None,
                         train_ratio: float = 0.7,
                         test_ratio: float = 0.15,
                         seed: int = 42):
    """Split data / indices into train, test and validation sets."""

    # SPLIT 1: Training vs. remaining data
    train_size = round(train_ratio * len(indices))  # avoid rounding issues in train_test_split
    temp_size = len(indices) - train_size

    train_indices, temp_indices = train_test_split(
        indices, test_size=temp_size,
        stratify=stratify_keys, random_state=seed
    )

    # SPLIT 2: Validation vs. test data
    if stratify_keys is not None:
        stratify_keys_temp = [stratify_keys[i] for i in temp_indices]
    else:
        stratify_keys_temp = None

    split_ratio = test_ratio / (1 - train_ratio)
    test_size = round(split_ratio * len(temp_indices))  # avoid rounding issues in train_test_split

    val_indices, test_indices = train_test_split(
        temp_indices, test_size=test_size,
        stratify=stratify_keys_temp, random_state=seed
    )

    return train_indices, test_indices, val_indices


def main(args):

    # generate dataset
    dataset = Benchmark(random_seed=args.seed,
                        num_full_distractions=args.num_full_distractions,
                        num_window_distractions=args.num_window_distractions,)

    data = dataset.data
    labels = np.array(dataset.labels).astype(int)
    metadata = dataset.meta

    # split dataset (stratified split)
    indices = np.arange(len(labels))
    stratify_keys = [str(label) for label in labels]

    train_indices, test_indices, val_indices = train_test_val_split(indices=indices,
                                                                    stratify_keys=stratify_keys,
                                                                    train_ratio=args.train_split,
                                                                    test_ratio=args.test_split,
                                                                    seed=args.seed)

    train_data, train_labels, train_metadata = data[train_indices], labels[train_indices], [metadata[i] for i in train_indices]
    val_data, val_labels, val_metadata = data[val_indices], labels[val_indices], [metadata[i] for i in val_indices]
    test_data, test_labels, test_metadata = data[test_indices], labels[test_indices], [metadata[i] for i in test_indices]

    # save dataset (in splits)
    save_split(data=train_data, labels=train_labels, meta=train_metadata, name="train", out_dir=args.out_dir)
    save_split(data=val_data, labels=val_labels, meta=val_metadata, name="val", out_dir=args.out_dir)
    save_split(data=test_data, labels=test_labels, meta=test_metadata, name="test", out_dir=args.out_dir)

    print(f"\nSaved  →  {args.out_dir}")
    print(f"  train={len(train_indices)}  val={len(val_indices)}  test={len(test_indices)}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_split", required=False, type=float, default=0.7)
    parser.add_argument("--test_split", required=False, type=float, default=0.15)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--out_dir", required=False, type=str, default="./dataset_temp/")
    parser.add_argument("--num_full_distractions", required=False, type=int, default=3)
    parser.add_argument("--num_window_distractions", required=False, type=int, default=3)

    args = parser.parse_args()

    main(args)
