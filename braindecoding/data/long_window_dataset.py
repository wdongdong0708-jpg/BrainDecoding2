"""Single continuous word-onset windows from unchanged canonical signals."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def window_spec(seconds, sfreq=50.):
    seconds, sfreq = float(seconds), float(sfreq)
    if not np.isfinite(seconds) or seconds <= 0 or sfreq != 50.:
        raise ValueError("Expected a positive duration and the frozen 50 Hz signals")
    samples = int(round(seconds * sfreq))
    if samples not in (50, 64, 75):
        raise ValueError("Only the 1 s control and requested 1.27/1.5 s windows are supported")
    return dict(requested_seconds=seconds, samples=samples, actual_seconds=samples / sfreq,
                sample_rate_hz=sfreq, interval="half_open", start_offset_seconds=0.)


@lru_cache(maxsize=4)
def open_signal(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


class SingleWordLongWindowDataset(Dataset):
    """Extend a single cut; do not assemble neighbouring word representations.

    Original event eligibility, targets, geometry and sampling groups are reused.
    Bounds are checked for every event, without padding or dropping any row.
    The CNN must be configured without a cross-word Transformer by the caller.
    """
    def __init__(self, base, seconds):
        self.base, self.table = base, base.table
        self.spec = window_spec(seconds, base.dataset_config["target_sampling_rate_hz"])
        self.window_samples = self.spec["samples"]
        if base.window_samples != 50 or float(base.dataset_config.get("window_start_offset_seconds", 0.)) != 0.:
            raise ValueError("The canonical source must use 1 s windows at word onset")
        self.signal_key = "meg" if hasattr(base, "signal_paths") else "eeg"
        self.baseline_samples = 25 if self.signal_key == "meg" else 0
        self.clamp = base.dataset_config.get("clamp", 5.)
        self.words = self.table.normalized_word.astype(str).tolist()
        self.event_ids = self.table.event_id.astype(str).tolist()
        self.recordings = self.table.recording_id.astype(str).tolist()
        self.sentence_indices = np.asarray(base.sentence_indices, dtype=np.int64)
        if self.signal_key == "meg":
            self.paths = [str(Path(base.signal_paths[r]).resolve()) for r in self.recordings]
            self.subject_indices = np.asarray(base.subject_indices, dtype=np.int64)
            onset = self.table.onset_seconds.to_numpy(dtype=np.float64)
            origins = np.asarray([base.signal_metadata[r]["source_first_samp"] /
                                  base.signal_metadata[r]["source_sampling_rate_hz"] for r in self.recordings])
            self.starts = np.rint((onset - origins) * self.spec["sample_rate_hz"]).astype(np.int64)
            if float(base.dataset_config["baseline_seconds"]) != .5:
                raise ValueError("Pallier's baseline must remain the first 0.5 s")
        else:
            self.paths = [str(Path(base.recording_cache_paths[str(r)]).resolve()) for r in self.table.vhdr_relpath]
            self.subject_indices = np.asarray([base.subject_to_index[str(s)] for s in self.table.subject_id], dtype=np.int64)
            self.starts = self.table.window_start_target_sample.to_numpy(dtype=np.int64)
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("Duplicate single-word events")
        self.bounds_audit = self.audit_bounds()

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def __len__(self):
        return len(self.table)

    def audit_bounds(self):
        path_array = np.asarray(self.paths)
        shapes, minimum_margin = {}, None
        for path in sorted(set(self.paths)):
            signal = open_signal(path)
            if signal.ndim != 2 or signal.shape[0] != self.base.channel_count:
                raise ValueError(f"Continuous signal geometry changed: {path}")
            selected = np.flatnonzero(path_array == path)
            stops = self.starts[selected] + self.window_samples
            valid = (self.starts[selected] >= 0) & (stops <= signal.shape[1])
            if not valid.all():
                event = self.event_ids[selected[np.flatnonzero(~valid)[0]]]
                raise ValueError(f"Long window exceeds recording; no row may be silently dropped: {event}")
            margin = int((signal.shape[1] - stops).min())
            minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
            shapes[path] = list(signal.shape)
        return dict(rows=len(self), recordings=len(shapes), signal_shapes=shapes,
                    out_of_bounds=0, dropped_events=0, padded_events=0,
                    minimum_recording_end_margin_samples=minimum_margin)

    def raw_window(self, index):
        index = int(index)
        start = int(self.starts[index])
        return np.array(open_signal(self.paths[index])[:, start:start + self.window_samples],
                        dtype=np.float32, copy=True)

    def __getitem__(self, index):
        index = int(index)
        window = self.raw_window(index)
        if self.baseline_samples:
            window -= window[:, :self.baseline_samples].mean(axis=1, keepdims=True)
        if self.clamp is not None:
            np.clip(window, -float(self.clamp), float(self.clamp), out=window)
        return {self.signal_key: torch.from_numpy(window),
                "text_embedding": torch.from_numpy(np.array(self.embedding_map[self.words[index]], dtype=np.float32, copy=True)),
                "subject_index": torch.tensor(self.subject_indices[index], dtype=torch.long),
                "sentence_index": torch.tensor(self.sentence_indices[index], dtype=torch.long),
                "word": self.words[index], "event_id": self.event_ids[index],
                "recording_id": self.recordings[index]}
