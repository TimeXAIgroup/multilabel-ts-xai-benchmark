# Synthetic Multi-Label Time Series Classification Dataset for XAI Evaluation

[![Dataset](https://img.shields.io/badge/Dataset-10.5281%2Fzenodo.20744119-1682D4)](https://doi.org/10.5281/zenodo.20744119)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green)
![Python](https://img.shields.io/badge/python-3.11-blue)

This repository accompanies a synthetic multi-label time series classification dataset designed to support the 
evaluation of eXplainable AI (XAI) methods using ground-truth explanations. The dataset provides a controlled benchmark 
in which multiple class labels and their corresponding discriminative patterns may co-occur within the same sample, 
enabling quantitative evaluation of XAI methods in the time, frequency, and time-frequency domains.

The dataset contains 4000 univariate time series of 500 time steps each. Each 
sample contains zero or more of three class labels, defined by the presence of class-specific frequency windows 
with random location. All eight co-occurring combinations are evenly distributed (500 samples each). Samples 
additionally contain Gaussian noise and non-discriminative frequency components (both windows and full-time-series 
frequencies) as distractors. Co-occurring class-specific frequency windows overlap between 25% and 50% of the 
window length, allowing evaluation of explanation methods under feature overlap.

Every sample is accompanied by full ground-truth annotations containing the position, length, frequency, and phase of 
each discriminative component. The data are provided in NumPy format with labels and ground-truth annotations in JSON 
format. The dataset and trained baseline models are deposited on [Zenodo](https://doi.org/10.5281/zenodo.20744119). The 
data generation and model training code are available in this repository.

![Sample_6](./plots/sample_6.png)

- [ ] TODO
Specifically, the dataset allows evaluation of whether an explanation:
- identifies the correct temporal region,
- identifies the correct frequency component,
- distinguishes discriminative from non-discriminative signals,
- remains robust under overlapping discriminative patterns,
- recovers explanations in time, frequency, and time-frequency domains.

## Dataset Description
### Dataset at a glance

| Property                  | Value                           |
|---------------------------|---------------------------------|
| Samples                   | 4000                            |
| Signal length             | 500 time steps                  |
| Channels                  | 1                               |
| Labels                    | 3 binary classes                |
| Label combinations        | 8                               |
| Sampling rate             | 100 Hz                          |
| Ground-truth explanations | ✓                               |
| Domains                   | Time, Frequency, Time-Frequency |
| Format                    | NumPy + JSON                    |

### Signal composition
Each sample is constructed as the superposition of:
- 0-3 discriminative frequency windows at class-specific frequencies (7, 19, or 37Hz), placed at random positions 
within the signal with an overlap between 25% and 50% of the window length
- 3 distractor frequency windows at non-discriminative frequencies, placed at random positions
- 3 full-signal frequency distractors spanning all 500 time steps at non-discriminative frequencies
- Gaussian noise with a standard deviation of 0.1

### Signal structure
The dataset contains 4000 univariate time series of 500 time steps each, stored as NumPy array of shape 
`(4000, 1, 500)` with dtype `float32`. Signal values are the superposition of frequency windows and Gaussian noise, 
with typical amplitudes in the range of $\pm$3.

### Label structure
Labels are provided as multi-hot binary matrix of shape `(4000, 3)` with integer values, where index 0 corresponds to 
7Hz, index 1 to 19Hz and index 2 to 37Hz. Three binary labels may co-occur, yielding 8 possible label combinations:

| Label vector | Active classes    |
|--------------|-------------------|
| `[0, 0, 0]`  | none              |
| `[0, 0, 1]`  | 37Hz              |
| `[0, 1, 0]`  | 19Hz              |
| `[0, 1, 1]`  | 19 and 37Hz       |
| `[1, 0, 0]`  | 7Hz               |
| `[1, 0, 1]`  | 7 and 37Hz        |
| `[1, 1, 0]`  | 7 and 19Hz        |
| `[1, 1, 1]`  | 7 and 19 and 37Hz |

### Ground-truth annotation
Each sample is accompanied by full ground-truth annotations in a JSON file. The top-level structure contains a 
`"label"` and a `"metadata"` dictionary keyed by sample index (as a string). Each metadata entry contains two lists: 
`"ground_truth"` for discriminative components and `"non-discriminative"` for distractor components. Each component 
entry has the following fields:

| Field     | Description                                  |
|-----------|----------------------------------------------|
| `channel` | Always 0 due to univariate nature of dataset |
| `pos`     | Start position of the component window       |
| `len`     | Length of the window                         |
| `freq`    | Frequency in Hz                              |
| `phase`   | Phase offset                                 |

A minimal example (label `[0, 1, 0]`, 19Hz only):
```
"5": {
    "ground_truth": [
        {"channel": 0,  "pos": 264, "len": 150, "freq": 19, "phase": 0.340}
    ],
    "non_discriminative": [
        {"channel": 0,  "pos": 141, "len": 150, "freq": 32, "phase": 5.608},
        {"channel": 0,  "pos": 0,   "len": 500, "freq": 10, "phase": 5.535}
    ]
}
```
Samples with label `[0, 0, 0]` have an empty `"ground_truth"` list. Non-discriminative components include both 
full-signal sinusoids (`"len": 500`) and windowed sinusoids (`"len": 150`).

### Dataset splits
With default settings (`--train_split 0.7`, `--test_split 0.15`), the dataset contains 2800 training, 600 validation, 
and 600 test samples. A stratified split is performed to keep the distribution of label combinations similar in each 
split.

### Download
Direct link to Zenodo record. What files are in the archive, their format and size. Minimal loading example:
pythonimport numpy as np, json
data   = np.load("train_data.npy")
labels = json.load(open("train_labels.json"))

## Quick Start
Generate the benchmark:
```bash
python generate_dataset.py
```

Visualize samples:
- [ ] TODO: args

Train a baseline model:
```bash
python train.py --model inceptiontime
```

## Repository Structure

```text
./
├── data/
│   └── benchmark.py        # dataset logic
├── models/
│   └── cnn.py
│   └── inceptiontime.py
│   └── minirocket.py
│   └── moment.py
│   └── resnet.py
│   └── tcn.py
├── generate_dataset.py     # create benchmark dataset
├── train.py                # train baseline models 
├── utils.py
└── visualize.py            # visualize signals and explanations
```

## Installation

Recommended Python Version: Python 3.11

```bash
pip install -r requirements.txt
```

## Generate a Dataset

```bash
python generate_dataset.py [--out_dir ./dataset/] [--train_split 0.7] [--test_split 0.15] [--seed 42]
```

Saves `{split}_data.npy` and `{split}_labels.json` for each of the three splits (`train`, `val`, `test`).

**Options**

```
options:
  -h, --help            show this help message and exit
  --train_split TRAIN_SPLIT
  --test_split TEST_SPLIT
  --seed SEED
  --out_dir OUT_DIR
  --num_full_distractions NUM_FULL_DISTRACTIONS
  --num_window_distractions NUM_WINDOW_DISTRACTIONS
```

| Argument | Default | Description |
|---|---|---|
| `--train_split` | `0.7` | Fraction of data used for training |
| `--test_split` | `0.15` | Fraction of data used for testing (remainder → val) |
| `--seed` | `42` | Random seed for reproducible splits |
| `--out_dir` | `./dataset/` | Output directory |

### Advanced: Custom Dataset Configuration

To customise the dataset itself, instantiate `Benchmark` directly:

```python
from data.benchmark import Benchmark

dataset = Benchmark(
    sample_length=500,               # time steps per sample
    sampling_rate=100,               # Hz — also determines the distractor pool
    discriminative_frequencies=[7, 19, 37],  # frequencies that define class labels
    frequency_window_length=150,     # length (samples) of each discriminative window
    num_samples_per_pattern=500,     # samples generated per class
    overlap_min=0.25,                # min fractional overlap between windows
    overlap_max=0.5,                 # max fractional overlap between windows
    noise_std=0.1,                   # std of background Gaussian noise
    sin_amplitude=1.0,               # amplitude of all sinusoidal components
    num_full_distractions=3,         # distractor sinusoids spanning the full signal
    num_window_distractions=3,       # distractor sinusoids in random windows
    random_seed=42,
)
```

With the default settings, the dataset has **8 classes** (power set of 3 frequencies) × 500 samples = **4 000 samples** total.

## Visualize the Dataset and Ground Truth


## Train a Model

Requires a dataset generated by `generate_dataset.py`.

```bash
python train.py [--model resnet] [--data_dir ./dataset/] [--epochs 20] [--batch_size 64] [--lr 1e-3]
```

The best checkpoint (by val loss) is saved to `OUT_DIR/best_{model}.pt`.
Final val and test metrics (loss, exact-match accuracy, macro F1) are printed after training
and stored in `OUT_DIR/best_{model}_val_metrics.json` / `OUT_DIR/best_{model}_test_metrics.json`.

**Training options**

```
options:
  -h, --help            show this help message and exit
  --model {cnn,tcn,inceptiontime,resnet,moment,minirocket}
                        Model architecture
  --data_dir DATA_DIR   Directory containing the dataset splits
  --out_dir OUT_DIR     Directory to save the model checkpoints
  --epochs EPOCHS       Number of training epochs
  --batch_size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --patience PATIENCE   Patience for early stopping
  --delta DELTA         Delta for early stopping
```

## Citation
If you use this dataset in your work, please cite the Zenodo record:
```
@dataset{Fischer2026,
  author    = {Fischer, Jana and Gumpfer, Nils and Hannig, Jennifer},
  title     = {Synthetic Multi-Label Time Series Classification Dataset for XAI Evaluation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20744119},
  url       = {https://doi.org/10.5281/zenodo.20744119},
}
```

## License
The code in this repository is licensed under the BSD 3-Clause License.
The dataset published on Zenodo is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
