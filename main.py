"""Generate, split, and save the benchmark dataset."""

import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from data.benchmark import Benchmark
from utils import save_split


def main(args):

    # generate dataset
    dataset = Benchmark(random_seed=args.seed)

    data = dataset.data
    labels = np.array(dataset.labels).astype(int)
    metadata = dataset.meta

    # split dataset (stratified split)
    indices = np.arange(len(labels))
    stratify_keys = [str(label) for label in labels]
    train_indices, temp_indices = train_test_split(
        indices, test_size=1 - args.train_split,
        stratify=stratify_keys, random_state=args.seed
    )
    stratify_keys_temp = [stratify_keys[i] for i in temp_indices]
    split_ratio = args.test_split / (1 - args.train_split)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=split_ratio,
        stratify=stratify_keys_temp, random_state=args.seed
    )

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
    parser.add_argument("--out_dir", required=False, type=str, default="./dataset/")

    args = parser.parse_args()

    main(args)
