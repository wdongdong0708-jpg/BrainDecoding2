"""Stimulus-independent pulse streams for word-window timing controls.

No neural sample is read. Shared and independent controls have the same pulse
law, padded stream length, normalization and dataset-specific window transform.
Only the identity of the underlying stream changes between word windows.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch
from torch.utils.data import Dataset


CONDITIONS = ("shared", "independent")
GENERATOR_VERSION = "smooth_pulses_50hz_v1"


@lru_cache(maxsize=3)
def _gaussian_kernel(sigma):
    radius = int(4 * sigma + .5)
    impulse = np.zeros(2 * radius + 1, dtype=np.float64)
    impulse[radius] = 1
    return gaussian_filter1d(impulse, sigma=sigma, mode="constant", truncate=4.0)


def stable_seed(*parts):
    """Do not use Python's process-randomized hash or words as RNG inputs."""
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:16], "little")


def synthetic_stream(samples, channels, sfreq, rng, *, pad_seconds=1.0):
    """Generate three 1 Hz pulse processes, normalized before window cutting.

    A one-second margin exceeds the widest Gaussian kernel's 4-sigma support.
    Pulse positions are shared across channels; Gaussian amplitudes are not.
    Word onsets and labels are never provided to this generator.
    """
    samples, channels = int(samples), int(channels)
    if samples <= 0 or channels <= 0 or sfreq <= 0 or pad_seconds < .96:
        raise ValueError("Invalid synthetic stream shape, sampling rate or padding")
    pad = int(np.ceil(pad_seconds * sfreq))
    n = samples + 2 * pad
    signal = np.zeros((channels, n), dtype=np.float32)
    for sigma_ms in (60, 120, 240):
        count = min(int(rng.poisson(n / sfreq)), n)
        positions = rng.choice(n, count, replace=False)
        amplitudes = rng.standard_normal((channels, count)).astype(np.float32)
        kernel = _gaussian_kernel(sigma_ms / 1000 * sfreq)
        radius = len(kernel) // 2
        component = np.zeros_like(signal)
        # Sparse convolution is the same Gaussian smoothing operation. Applying
        # only nonzero impulses avoids filtering hundreds of zero samples for
        # every independent word, which otherwise dominates CNN training time.
        for pulse, position in enumerate(positions):
            left, right = max(0, position - radius), min(n, position + radius + 1)
            weights = kernel[left - position + radius:right - position + radius]
            component[:, left:right] += amplitudes[:, pulse:pulse + 1] * weights[None, :]
        signal += component
    signal /= float(signal.std(dtype=np.float64)) + 1e-6
    return np.ascontiguousarray(signal[:, pad:pad + samples])


def transform_window(window, *, baseline_samples=0, clamp=5.0):
    """Match the existing post-cut transform, identically for both controls."""
    result = np.array(window, dtype=np.float32, copy=True)
    if baseline_samples:
        result -= result[:, :int(baseline_samples)].mean(axis=-1, keepdims=True)
    if clamp is not None:
        np.clip(result, -float(clamp), float(clamp), out=result)
    return result


class SyntheticTimingDataset(Dataset):
    """Reuse frozen labels/groups/geometry, but never call base.__getitem__.

    Sampling indices are computed in the original recording clock, *then*
    translated by an integer segment origin. Rounding relative seconds would
    change some intervals at half-sample boundaries.
    """
    def __init__(self, base, condition, *, split, training_seed=0,
                 evaluation_seed=20261008, cache_streams=32):
        if condition not in CONDITIONS or split not in ("train", "val", "test"):
            raise ValueError("Unknown synthetic condition or split")
        if cache_streams < 1:
            raise ValueError("At least one stream cache slot is required")
        self.base, self.condition, self.split = base, condition, split
        self.table = base.table
        self.training_seed, self.evaluation_seed = int(training_seed), int(evaluation_seed)
        self.epoch, self.cache_streams = 0, int(cache_streams)
        self._streams = OrderedDict()
        self.sfreq = float(base.dataset_config["target_sampling_rate_hz"])
        if self.sfreq != 50.0 or base.window_samples != 50:
            raise ValueError("This control is frozen to the existing 50 Hz, 1 s inputs")
        self.signal_key = "meg" if "runtime_context_uid" in self.table else "eeg"
        group_column = "runtime_context_uid" if self.signal_key == "meg" else "sentence_uid"
        self.baseline_samples = (int(round(float(base.dataset_config["baseline_seconds"]) * self.sfreq))
                                 if self.signal_key == "meg" else 0)
        self.clamp = base.dataset_config.get("clamp", 5.0)
        self.words = self.table.normalized_word.astype(str).tolist()
        self.event_ids = self.table.event_id.astype(str).tolist()
        self.recording_ids = self.table.recording_id.astype(str).tolist()
        self.subject_indices = np.asarray([base.subject_map[str(s)] if self.signal_key == "meg"
            else base.subject_to_index[str(s)] for s in self.table.subject_id], dtype=np.int64)
        self.group_keys = list(zip(self.table.subject_id.astype(str),
                                   self.table.recording_id.astype(str), self.table[group_column].astype(str)))
        if self.signal_key == "meg":
            from braindecoding.data.pallier2025 import recording_onset_to_sample
            offset = float(base.dataset_config.get("window_start_offset_seconds", 0))
            absolute = np.asarray([recording_onset_to_sample(float(onset) + offset,
                source_first_samp=base.signal_metadata[recording]["source_first_samp"],
                source_sampling_rate_hz=base.signal_metadata[recording]["source_sampling_rate_hz"],
                output_sampling_rate_hz=self.sfreq)
                for onset, recording in zip(self.table.onset_seconds, self.recording_ids)], dtype=np.int64)
        else:
            absolute = self.table.window_start_target_sample.to_numpy(dtype=np.int64)
        self.starts = absolute.copy()
        groups = {}
        for index, key in enumerate(self.group_keys):
            groups.setdefault(key, []).append(index)
        self.stream_lengths = {}
        for key, indices in groups.items():
            values = absolute[indices]
            if (np.diff(values) < 0).any():
                raise ValueError("Onsets within a segment must be chronological")
            self.starts[indices] -= values.min()
            self.stream_lengths[key] = int(values.max() - values.min() + base.window_samples)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __len__(self):
        return len(self.table)

    def set_epoch(self, epoch):
        if self.split == "train" and int(epoch) != self.epoch:
            self.epoch = int(epoch)
            self._streams.clear()

    def _stream(self, index):
        group = self.group_keys[index]
        identity = (group, self.event_ids[index] if self.condition == "independent" else "shared_stream")
        if identity not in self._streams:
            seed = (stable_seed(GENERATOR_VERSION, self.condition, "train", self.training_seed,
                                self.epoch, identity) if self.split == "train" else
                    stable_seed(GENERATOR_VERSION, self.condition, self.split, self.evaluation_seed, identity))
            self._streams[identity] = synthetic_stream(self.stream_lengths[group], self.channel_count,
                                                       self.sfreq, np.random.default_rng(seed))
            if len(self._streams) > self.cache_streams:
                self._streams.popitem(last=False)
        self._streams.move_to_end(identity)
        return self._streams[identity]

    def raw_window(self, index):
        index = int(index)
        start = int(self.starts[index])
        return self._stream(index)[:, start:start + self.window_samples].copy()

    def __getitem__(self, index):
        index = int(index)
        window = transform_window(self.raw_window(index), baseline_samples=self.baseline_samples,
                                  clamp=self.clamp)
        return {self.signal_key: torch.from_numpy(window),
                "text_embedding": torch.from_numpy(np.array(self.embedding_map[self.words[index]],
                                                               dtype=np.float32, copy=True)),
                "subject_index": torch.tensor(self.subject_indices[index], dtype=torch.long),
                "sentence_index": torch.tensor(self.sentence_indices[index], dtype=torch.long),
                "word": self.words[index], "event_id": self.event_ids[index],
                "recording_id": self.recording_ids[index]}


def overlap_statistics(table, *, window_seconds=1.0):
    """Count true adjacent eligible words within subject/recording/segment.

    Never join over an excluded word, a split, or a language-segment boundary.
    Report physical-time overlap, plus its quantized 50 Hz counterpart.
    """
    aliases = {"subject_id": "受试者", "recording_id": "记录编号",
               "sentence_uid": "上下文编号", "split": "数据划分", "onset_seconds": "开始时间",
               "order": "记录内序号", "is_trainable": "是否可训练"}
    source = table.copy()
    for target, origin in aliases.items():
        if target not in source:
            source[target] = source[origin]
    eligible = source.is_trainable.astype(str).str.lower().isin(("true", "1"))
    source = source[eligible].copy()
    keys = ["split", "subject_id", "recording_id", "sentence_uid"]
    source = source.sort_values(keys + ["order"], kind="stable")
    groups = source.groupby(keys, sort=False)
    delta = groups.onset_seconds.diff()
    adjacent = groups.order.diff().eq(1)
    valid = adjacent & delta.notna()
    if delta[valid].lt(0).any():
        raise ValueError("Negative adjacent onset interval")
    source["delta"] = delta
    source["adjacent"] = valid
    source["sample_delta"] = groups.onset_seconds.transform(lambda x: np.round(x * 50)).groupby(
        [source[k] for k in keys], sort=False).diff()
    rows = []
    for (split, subject), part in source.groupby(["split", "subject_id"], sort=False):
        intervals = part.loc[part.adjacent, "delta"].to_numpy(dtype=np.float64)
        sample_intervals = part.loc[part.adjacent, "sample_delta"].to_numpy(dtype=np.float64)
        if len(intervals):
            rows.append({"split": str(split), "subject_id": str(subject), "pairs": len(intervals),
                "overlapping_pairs": int((intervals < window_seconds).sum()),
                "overlap_probability": float((intervals < window_seconds).mean()),
                "mean_shared_fraction": float(np.maximum(0, 1 - intervals / window_seconds).mean()),
                "mean_interval_seconds": float(intervals.mean()),
                "median_interval_seconds": float(np.median(intervals)),
                "zero_interval_pairs": int((intervals == 0).sum()),
                "sample_overlap_probability": float((sample_intervals < 50 * window_seconds).mean()),
                "sample_mean_shared_fraction": float(np.maximum(0, 1 - sample_intervals / (50 * window_seconds)).mean())})
    total = sum(r["pairs"] for r in rows)
    return {"window_seconds": window_seconds, "pairs": total,
            "overlap_probability": sum(r["overlapping_pairs"] for r in rows) / total if total else None,
            "mean_shared_fraction": sum(r["pairs"] * r["mean_shared_fraction"] for r in rows) / total if total else None,
            "skipped_nonadjacent_gaps": int((delta.notna() & ~adjacent).sum()),
            "aggregation": "eligible adjacent pairs; subject repetitions retained; no segment boundary joins",
            "by_subject_split": rows}
