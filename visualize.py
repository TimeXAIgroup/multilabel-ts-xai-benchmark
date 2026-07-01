"""Visualisation utilities for the Benchmark dataset."""
import argparse
import itertools
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerBase

from train import load_split

_FREQS          = [7, 19, 37]
_CLASS_COLORS   = ["#5B9BD5", "#70AD47", "#E06060"]   # blue, green, red

CLASS_COLORS = _CLASS_COLORS[:len(_FREQS)] if len(_FREQS) <= 3 else [
    plt.get_cmap("tab10")(i) for i in range(len(_FREQS))
]

_NPERSEG = 50    # STFT window length (samples)
_GT_BW   = 0.08  # normalised-frequency bandwidth per class band in the GT mask


class _HandlerSideBySide(HandlerBase):
    """Draws several Patches next to each other in legend icon."""
    def __init__(self, gap=2.0, **kwargs):
        self.gap = gap  # Abstand zwischen den Patches in Punkten
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle,
                        xdescent, ydescent, width, height, fontsize, trans):
        n = len(orig_handle)
        gap = self.gap
        total_gap = gap * (n - 1)
        seg_w = (width - total_gap) / n

        artists = []
        x = xdescent
        for patch in orig_handle:
            r = mpatches.Rectangle(
                (x, ydescent), seg_w, height,
                facecolor=patch.get_facecolor(),
                edgecolor=patch.get_edgecolor(),
                transform=trans,
            )
            artists.append(r)
            x += seg_w + gap
        return artists


def plot_sample(data,
                labels,
                metadata,
                sample_index: int,
                sampling_rate= 100,
                save_dir = "./plots/",
                save_name_addition = None):
    """Plot signal, spectrogram, and ground-truth mask for one benchmark sample.

    Layout:
        Top row    — time-domain signal (full width) with legend on right.
        Bottom row — STFT spectrogram (left) and ground-truth mask (right).

    Args:
        data: Data
        labels: Labels
        metadata: Metadata (contains ground truth)
        sample_index: Sample index to plot.
        sampling_rate: Sampling rate of data.
        save_dir: Directory to save plot.
        save_name_addition: Name addition for the plot name.
    """
    signal       = data[sample_index]

    fig = plt.figure(figsize=(12, 7))
    fig.subplots_adjust(left=0.08, right=0.72, top=0.93, bottom=0.10, hspace=0.45)
    gs      = fig.add_gridspec(2, 2, wspace=0.35)
    ax_sig  = fig.add_subplot(gs[0, :])
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_mask = fig.add_subplot(gs[1, 1])

    # ── Collect discriminative windows ──────────────────────────────────────
    freq_windows = _get_frequency_windows(metadata, sample_index)

    # ── Three panels ────────────────────────────────────────────────────────
    _plot_signal(ax_sig, signal, freq_windows, _FREQS, CLASS_COLORS)
    _plot_spectrogram(ax_spec, signal)
    _plot_gtmask(ax_mask, list(zip(_FREQS, CLASS_COLORS)), freq_windows,
                 sampling_rate, len(signal))

    # ── Legend and label vector (right of signal panel) ─────────────────────
    _attach_legend(ax_sig, _FREQS, CLASS_COLORS, freq_windows, _find_overlaps(freq_windows))
    _draw_label_vector(ax_mask, labels[sample_index], CLASS_COLORS)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(Path(save_dir) / f"sample{f'_{save_name_addition}' if save_name_addition is not None else ''}_{sample_index}.png", bbox_inches="tight", dpi=600)


def _get_frequency_windows(metadata, sample_index: int) -> dict[Any, Any]:
    freq_windows = {}
    for entry in metadata[str(sample_index)].get("ground_truth", []):
        freq_windows.setdefault(entry["freq"], []).append(
            (entry["pos"], entry["pos"] + entry["len"])
        )
    return freq_windows


# ── Overlap geometry ──────────────────────────────────────────────────────────

def _first_overlap_range(wins_i, wins_j):
    """Return the first (o_start, o_end) where windows from two groups overlap, else None."""
    for s1, e1 in wins_i:
        for s2, e2 in wins_j:
            o_start, o_end = max(s1, s2), min(e1, e2)
            if o_start < o_end:
                return o_start, o_end
    return None


def _find_overlaps(freq_windows):
    """Return (freq_i, freq_j, (o_start, o_end)) for each pair with a temporal overlap."""
    result = []
    keys, vals = list(freq_windows.keys()), list(freq_windows.values())
    for i, wins_i in enumerate(vals):
        for j, wins_j in enumerate(vals[i + 1:], i + 1):
            pair_range = _first_overlap_range(wins_i, wins_j)
            if pair_range:
                result.append((keys[i], keys[j], pair_range))
    return result


def _blend_color(freq_i, freq_j, freqs, class_colors, alpha=0.25):
    """Visible RGB when the two class axvspans (alpha each) are composited on white."""
    r, g, b = 1.0, 1.0, 1.0
    for freq in (freq_i, freq_j):
        cr, cg, cb = to_rgb(class_colors[freqs.index(freq)])
        r = (1 - alpha) * r + alpha * cr
        g = (1 - alpha) * g + alpha * cg
        b = (1 - alpha) * b + alpha * cb
    return r, g, b


# ── Panel A — signal ──────────────────────────────────────────────────────────

def _plot_signal(ax, signal, freq_windows, freqs, class_colors):
    """Time-domain signal with shaded class windows."""
    ax.grid(True, linestyle="--", linewidth=0.5, color="#CCCCCC", zorder=0)
    ax.plot(np.arange(len(signal)), signal, color="black", linewidth=0.8, zorder=4)

    for freq, windows in freq_windows.items():
        color = class_colors[freqs.index(freq)]
        for s, e in windows:
            ax.axvspan(s, e, alpha=0.25, facecolor=color, edgecolor='none', zorder=1)

    ax.set_xlim(0, len(signal))
    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title("Signal", fontsize=10)


# ── Panel B — spectrogram ─────────────────────────────────────────────────────

def _plot_spectrogram(ax, signal):
    """STFT power spectrogram computed with a Hann window via numpy FFT."""
    hop      = _NPERSEG // 4
    n_frames = (len(signal) - _NPERSEG) // hop + 1
    window   = np.hanning(_NPERSEG)

    sxx = np.array([
        np.abs(np.fft.rfft(signal[k * hop: k * hop + _NPERSEG] * window)) ** 2
        for k in range(n_frames)
    ]).T                                                  # (n_freqs, n_frames)
    f_norm = np.fft.rfftfreq(_NPERSEG)                   # 0 … 0.5 (normalised)
    t_samp = np.arange(n_frames) * hop + _NPERSEG // 2   # frame centres (samples)

    ax.pcolormesh(
        t_samp, f_norm, sxx / sxx.max(),
        cmap="viridis", shading="auto", vmin=0, vmax=1, rasterized=True,
    )

    ax.set_xlim(0, len(signal))
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(["0", "10", "20", "30", "40", "50"])
    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title("Time-frequency representation", fontsize=10)


# ── Panel C — ground-truth mask ───────────────────────────────────────────────

def _plot_gtmask(ax, freq_colors, freq_windows, sampling_rate, sample_length):
    """Ground-truth mask: one coloured band per present class in time-freq space."""
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.3, color="#CCCCCC", zorder=0)

    bw = _GT_BW
    for freq, color in freq_colors:
        if freq not in freq_windows:
            continue
        f_ctr = freq / sampling_rate
        for s, e in freq_windows[freq]:
            ax.add_patch(mpatches.Rectangle(
                (s, f_ctr - bw / 2), e - s, bw,
                facecolor=color, alpha=0.35, edgecolor="none", zorder=1,
            ))

    ax.set_xlim(0, sample_length)
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(["0", "10", "20", "30", "40", "50"])
    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title("Explanation ground-truth mask", fontsize=10)


# ── Legend and label vector ───────────────────────────────────────────────────

def _attach_legend(ax, freqs, class_colors, freq_windows, overlaps):
    """Right-side legend attached to the signal axes."""
    handles = [
        mlines.Line2D([0], [0], color="black", linewidth=0.8, label="Signal (observed)"),
        mpatches.Patch(facecolor="none", edgecolor="none", label="Ground-truth regions"),
    ]
    labels = [h.get_label() for h in handles]
    for i, freq in enumerate(freqs):
        if freq not in freq_windows:
            continue
        handles.append(mpatches.Patch(
            facecolor=class_colors[i], alpha=0.35, edgecolor="none",
            label=f"{freqs[i]}Hz window",
        ))
        labels.append(f"{freqs[i]}Hz window")

    if len(overlaps) == 1:
        freq_i, freq_j, _ = overlaps[0]
        handles.append(mpatches.Patch(
            facecolor=_blend_color(freq_i, freq_j, freqs, class_colors),
            edgecolor="none",
            label="Overlap (co-occurrence)",
        ))
        labels.append("Overlap (co-occurrence)")
    elif len(overlaps) == 2:
        patches_for_group = []
        for freq_i, freq_j, _ in overlaps:
            patches_for_group.append(mpatches.Patch(
                facecolor=_blend_color(freq_i, freq_j, freqs, class_colors),
                edgecolor="none",
            ))
        handles.append(tuple(patches_for_group))
        labels.append("Overlap (co-occurrence)")
    elif len(overlaps) < 1:
        pass
    else:
        warnings.warn("Legend patch not implemented for 3 or more overlaps.")

    leg = ax.legend(
        handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1.02, 1.02),
        borderaxespad=0, frameon=True, fontsize=8.5, borderpad=0.8,
        handlelength=1.5, framealpha=1.0, edgecolor="#AAAAAA",
        handler_map={tuple: _HandlerSideBySide(gap=0.0)}
    )
    leg.texts[1].set_fontweight("bold")


def _draw_label_vector(ax, label_vec, colors):
    """Bordered label-vector box to the right of the signal axes."""
    bx, by, bw, bh = 1.02, 0.02, 0.4, 0.28

    ax.add_patch(mpatches.FancyBboxPatch(
        (bx, by), bw, bh, boxstyle="square,pad=0",
        facecolor="white", edgecolor="#333333", linewidth=0.8,
        transform=ax.transAxes, clip_on=False, zorder=10,
    ))
    ax.text(
        bx + bw / 2, by + bh * 0.72, "Label vector",
        transform=ax.transAxes, fontsize=9, fontweight="bold",
        ha="center", va="center", clip_on=False, zorder=11,
    )

    char_w = 0.025
    cx     = bx + 0.025
    cy     = by + bh * 0.27

    def _put(s, color="black", bold=False):
        nonlocal cx
        ax.text(
            cx, cy, s, transform=ax.transAxes, fontsize=9,
            ha="left", va="center", fontfamily="monospace",
            color=color, fontweight="bold" if bold else "normal",
            clip_on=False, zorder=11,
        )
        cx += len(s) * char_w

    _put("y = [")
    for k, lv in enumerate(label_vec):
        _put(str(int(lv)), color=colors[k] if lv else "#888888", bold=True)
        if k < len(label_vec) - 1:
            _put(", ")
    _put("]")


# ── Per-class signal thumbnails ───────────────────────────────────────────────


def _get_sample_index(labels, pattern_idx: int):
    n_classes = labels.shape[1]
    patterns = list(itertools.product([0, 1], repeat=n_classes))
    pattern = patterns[pattern_idx]
    pattern_indices = np.all(labels == pattern, axis=1)
    matches = np.where(pattern_indices)[0]

    if len(matches) == 0:
        raise ValueError(
            f"No sample found for label pattern {pattern} (pattern_idx={pattern_idx})."
        )

    return matches[0]


def plot_thumbnail(data,
                   labels,
                   metadata,
                   pattern_idx: int,
                   sampling_rate: int = 100,
                   save_dir: str = "./plots/"):
    """Combined signal + ground-truth mask thumbnail for one label-combination pattern.

    Top: time-domain signal with shaded class windows.
    Bottom: ground-truth mask in time-frequency domain.

    Args:
        data: Dataset.
        labels: Labels array.
        metadata: Metadata dict.
        pattern_idx: Index in all binary label combinations (0 = all absent, max = all present).
        sampling_rate: Sampling rate of the time series data.
        save_dir: Directory to save the plot.
    """
    sample_index = _get_sample_index(labels, pattern_idx)
    freq_windows = _get_frequency_windows(metadata, sample_index)

    signal_length = len(data[0])
    signal = data[sample_index]
    mid = signal_length // 2

    fig, (ax_signal, ax_mask) = plt.subplots(nrows=2, ncols=1,
                                             figsize=(2.0, 0.8),
                                             sharex=True,
                                             gridspec_kw={"height_ratios": [1.4, 1.0]})
    fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.10, hspace=0.1)

    # --- Signal ---
    ax_signal.plot(np.arange(signal_length), signal, color="black", linewidth=0.2)
    for freq, windows in freq_windows.items():
        color = CLASS_COLORS[_FREQS.index(freq)]
        for s, e in windows:
            ax_signal.axvspan(s, e, facecolor=color, edgecolor='none', alpha=0.25, zorder=1)
    ax_signal.set_xlim(0, signal_length)
    ax_signal.set_xticks([0, mid, signal_length])
    ax_signal.set_xticklabels(["0", str(mid), str(signal_length)])
    ax_signal.set_yticks([-1, 0, 1])

    # --- Ground truth mask ---
    for freq, color in zip(_FREQS, CLASS_COLORS):
        if freq not in freq_windows:
            continue
        f_ctr = freq / sampling_rate
        for s, e in freq_windows[freq]:
            ax_mask.add_patch(mpatches.Rectangle(
                (s, f_ctr - _GT_BW / 2), e - s, _GT_BW,
                facecolor=color, alpha=0.35, edgecolor="none",
            ))
    ax_mask.set_xlim(0, signal_length)
    ax_mask.set_ylim(0, 0.5)
    ax_mask.set_xticks([0, mid, signal_length])
    ax_mask.set_xticklabels(["0", str(mid), str(signal_length)])
    ax_mask.set_yticks([0, 0.5])
    ax_mask.set_yticklabels(["0", "50"])

    # --- Shared styling ---
    for ax, spine_width in [(ax_signal, 0.2), (ax_mask, 0.2)]:
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(labelsize=3, length=1, width=0.2, pad=1)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(Path(save_dir) / f"thumbnail_{pattern_idx}.png",
                bbox_inches="tight", dpi=600, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./experiments/dataset/", help="Directory containing the dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"], help="Split of the dataset")
    parser.add_argument("--out_dir", type=str, default="./experiments/dataset/plots/", help="Directory to save the plots")
    parser.add_argument("--sample_idx", type=int, default=6, help="Index of the sample to plot")

    args = parser.parse_args()

    data, labels, metadata = load_split(args.data_dir, args.split)
    plot_sample(data, labels, metadata, sample_index=args.sample_idx, save_dir=args.out_dir)
