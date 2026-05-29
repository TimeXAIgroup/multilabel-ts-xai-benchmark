from typing import Any

import numpy as np
from numpy import dtype, float64, ndarray
from torch.utils.data import Dataset

from utils import powerset


class Benchmark(Dataset):
    def __init__(self,
                 sample_length:int=500,
                 sampling_rate:int=100,
                 discriminative_frequencies:list=[7,19,37],
                 frequency_window_length:int=150,
                 num_samples_per_pattern:int=500,
                 overlap_min:float=0.25,
                 overlap_max:float=0.5,
                 noise_std:float=0.1,
                 sin_amplitude:float=1.0,
                 num_full_distractions:int=3,
                 num_window_distractions:int=3,
                 random_seed:int=42,
                 ):
        super(Benchmark, self).__init__()

        self.sample_length = sample_length
        self.sampling_rate = sampling_rate
        self.discriminative_frequencies = discriminative_frequencies
        self.frequency_window_length = frequency_window_length
        self.num_samples_per_pattern = num_samples_per_pattern
        self.overlap_min = overlap_min
        self.overlap_max = overlap_max
        self.noise_std = noise_std
        self.sin_amplitude = sin_amplitude
        self.num_full_distractions = num_full_distractions
        self.num_window_distractions = num_window_distractions

        self.nyquist_frequency = int(self.sampling_rate / 2)

        self.t = np.arange(self.sample_length) / self.sampling_rate

        self.rng = np.random.default_rng(random_seed)

        self.patterns = powerset(self.discriminative_frequencies)
        self.pattern_labels = [[int(freq in pattern) for freq in self.discriminative_frequencies] for pattern in self.patterns]
        self.distractor_pool = np.array([freq for freq in range(1, self.nyquist_frequency) if freq not in self.discriminative_frequencies])

        labels, meta = [], []
        for pattern, label_vector in zip(self.patterns, self.pattern_labels):
            pattern = np.array(pattern)
            for _ in range(self.num_samples_per_pattern):
                labels.append(label_vector)
                if len(pattern) > 1:
                    self.rng.shuffle(pattern)
                windows = self._place_windows(len(pattern))
                discriminative_metadata = [{"channel": 0,
                                            "pos": window[0],
                                            "len": window[1]-window[0],
                                            "freq": freq.item(),
                                            "phase": self.rng.uniform(0, 2 * np.pi)
                                            } for freq, window in zip(pattern, windows)]

                full_distractions = self.rng.choice(self.distractor_pool, size=self.num_full_distractions,
                                                    replace=False)
                noise_full_distractions =  [{"channel": 0,
                                             "pos": 0,
                                             "len": self.sample_length,
                                             "freq": freq.item(),
                                             "phase": self.rng.uniform(0, 2 * np.pi)
                                             } for freq in full_distractions]

                window_distractions = self.rng.choice(self.distractor_pool, size=self.num_window_distractions,
                                                      replace=False)
                windows = self._place_windows(len(window_distractions))
                noise_window_distractions = [{"channel": 0,
                                              "pos": window[0],
                                              "len": window[1]-window[0],
                                              "freq": freq.item(),
                                              "phase": self.rng.uniform(0, 2 * np.pi)
                                              } for freq, window in zip(window_distractions, windows)]
                meta.append({"discriminative": discriminative_metadata, "noise": noise_full_distractions + noise_window_distractions})


        self.labels = labels
        self.meta = meta

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_metadata = self.meta[idx]
        return self._make_sample(sample_metadata)

    def _make_sample(self, metadata):
        signal = self.rng.normal(0, self.noise_std, self.sample_length).astype(np.float32)

        metadata = metadata.get("discriminative", []) + metadata.get("noise", [])
        if len(metadata) > 0:
            frequencies = [window.get("freq") for window in metadata]
            positions = [window.get("pos") for window in metadata]
            lengths = [window.get("len") for window in metadata]
            phases = [window.get("phase") for window in metadata]
            signal = self._add_windows(frequencies, signal, positions, lengths, phases)

        return signal


    def _add_windows(self, frequencies, signal: ndarray[tuple[Any, ...], dtype[float64]], positions, lengths, phases):
        windows = zip(positions, lengths)
        for (start, length), freq, phase in zip(windows, frequencies, phases):
            end = min(start+length, self.sample_length)
            # envelope = self._window_envelope(end - start) # TODO: add other envelopes for damping
            envelope = np.ones_like(signal[start:end])
            signal[start:end] += self.sin_amplitude * envelope * np.sin(
                2 * np.pi * freq * self.t[start:end] + phase
            )

        return signal

    def _place_windows(self, num_windows: int):
        if num_windows == 0:
            return []
        if num_windows == 1:
            start = int(self.rng.integers(0, max(1, self.sample_length - self.frequency_window_length + 1)))
            return [(start, start + self.frequency_window_length)]

        hop_min = max(1, int(self.frequency_window_length * (1 - self.overlap_max)))
        hop_max = max(hop_min, int(self.frequency_window_length * (1 - self.overlap_min)))

        hops = self.rng.integers(hop_min, hop_max + 1, size=num_windows - 1)

        span = int(hops.sum()) + self.frequency_window_length
        if span > self.sample_length:
            scale = (self.sample_length - self.frequency_window_length) / max(1, int(hops.sum()))
            hops = np.maximum(1, (hops * scale)).astype(int)
            span = int(hops.sum()) + self.frequency_window_length

        t0 = int(self.rng.integers(0, max(1, self.sample_length - span + 1)))
        starts = [t0] + [t0 + int(hops[:k].sum()) for k in range(1, num_windows)]
        return [(s, s + self.frequency_window_length) for s in starts]
