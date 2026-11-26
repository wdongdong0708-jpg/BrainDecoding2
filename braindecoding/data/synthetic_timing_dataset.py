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
from torch.nn import functional as F
from torch.utils.data import Dataset


CONDITIONS = ("shared", "independent")
GENERATOR_VERSION = "smooth_pulses_50hz_v1"
_GPU_CHUNK_MAX_STREAMS = 256
_GPU_CHUNK_MAX_PADDED_SAMPLES = 65_536
_GPU_KERNEL_CACHE: dict[tuple[float, str], tuple[tuple[int, torch.Tensor], ...]] = {}


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


def _gpu_gaussian_kernels(
    sfreq: float, device: torch.device
) -> tuple[tuple[int, torch.Tensor], ...]:
    cache_key = (round(float(sfreq), 6), str(device))
    cached = _GPU_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    built: list[tuple[int, torch.Tensor]] = []
    for sigma_ms in (60, 120, 240):
        k_np = _gaussian_kernel(sigma_ms / 1000.0 * float(sfreq)).astype(np.float32)
        radius = len(k_np) // 2
        k_tensor = torch.from_numpy(k_np).to(
            device=device, dtype=torch.float32
        ).view(1, 1, -1)
        built.append((radius, k_tensor))
    result = tuple(built)
    _GPU_KERNEL_CACHE[cache_key] = result
    return result


def synthesize_streams_batch_gpu(
    lengths,
    channels: int,
    sfreq: float,
    generator: torch.Generator,
    *,
    device: torch.device,
    pad_seconds: float = 1.0,
) -> torch.Tensor:
    """在 GPU 上批量合成与 synthetic_stream 同分布的平滑高斯脉冲流 [G, C, L_max]。

    每个流独立按自身有效填充长度 n_g = L_g + 2*pad 采样三尺度 1 Hz 泊松无放回脉冲、
    通道独立标准正态振幅与精确 1D 高斯卷积，并按各流总体标准差（ddof=0）归一化后切去两端 pad。
    """
    channels = int(channels)
    sfreq = float(sfreq)
    if channels <= 0 or sfreq <= 0 or pad_seconds < 0.96:
        raise ValueError("Invalid synthetic stream shape, sampling rate or padding")
    lengths_t = torch.as_tensor(lengths, device=device, dtype=torch.long).view(-1)
    if lengths_t.numel() == 0 or bool((lengths_t <= 0).any().item()):
        raise ValueError("Invalid synthetic stream shape, sampling rate or padding")

    g_count = int(lengths_t.numel())
    pad = int(np.ceil(pad_seconds * sfreq))
    n_t = lengths_t + 2 * pad
    n_max = int(n_t.max().item())
    l_max = int(lengths_t.max().item())

    pos_grid = torch.arange(n_max, device=device, dtype=torch.long).unsqueeze(0)
    valid_n = pos_grid < n_t.unsqueeze(1)
    signal = torch.zeros(
        (g_count, channels, n_max), device=device, dtype=torch.float32
    )
    kernels = _gpu_gaussian_kernels(sfreq, device)

    for radius, k_1d in kernels:
        counts = torch.minimum(
            torch.poisson(
                n_t.to(dtype=torch.float32) / sfreq, generator=generator
            ).to(dtype=torch.long),
            n_t,
        )
        k_max = int(counts.max().item())
        if k_max <= 0:
            continue
        u = torch.rand(
            (g_count, n_max),
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        u.masked_fill_(~valid_n, 2.0)
        _, top_pos = torch.topk(
            u, k=k_max, dim=-1, largest=False, sorted=True
        )
        active = (
            torch.arange(k_max, device=device, dtype=torch.long).unsqueeze(0)
            < counts.unsqueeze(1)
        ).unsqueeze(1)
        pulse_amps = torch.randn(
            (g_count, channels, k_max),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        pulse_amps.masked_fill_(~active, 0.0)
        impulses = torch.zeros(
            (g_count, channels, n_max), device=device, dtype=torch.float32
        )
        impulses.scatter_add_(
            2, top_pos.unsqueeze(1).expand(g_count, channels, k_max), pulse_amps
        )
        smoothed = F.conv1d(
            impulses.reshape(g_count * channels, 1, n_max),
            k_1d,
            padding=radius,
        ).view(g_count, channels, n_max)
        signal.add_(smoothed)

    signal.masked_fill_(~valid_n.unsqueeze(1), 0.0)
    denom = (n_t * channels).to(dtype=torch.float64).view(g_count, 1, 1)
    mean_g = signal.sum(dim=(1, 2), keepdim=True, dtype=torch.float64) / denom
    sq_mean_g = (
        (signal * signal).sum(dim=(1, 2), keepdim=True, dtype=torch.float64)
        / denom
    )
    var_g = torch.clamp(sq_mean_g - mean_g.pow(2), min=0.0)
    std_g = (torch.sqrt(var_g) + 1e-6).to(dtype=torch.float32)
    signal.div_(std_g)
    return signal[:, :, pad : pad + l_max]


def transform_window_gpu(
    windows: torch.Tensor, *, baseline_samples: int = 0, clamp: float | None = 5.0
) -> torch.Tensor:
    """在 GPU 张量 [B, C, W] 上执行与 transform_window 完全一致的基线均值扣除与截断。"""
    result = windows
    if baseline_samples:
        result = result - result[:, :, : int(baseline_samples)].mean(
            dim=-1, keepdim=True
        )
    if clamp is not None:
        result = result.clamp(-float(clamp), float(clamp))
    return result


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
                 evaluation_seed=20261008, cache_streams=32,
                 include_onset_seconds=None,
                 temporal_shift_seconds=0.0,
                 shift_mode="preroll",
                 preroll_seconds=45.0):
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
        self.sampling_rate_hz = self.sfreq
        self.temporal_shift_seconds = float(temporal_shift_seconds)
        self.temporal_shift_samples = int(round(self.temporal_shift_seconds * self.sfreq))
        self.shift_mode = str(shift_mode)
        self.preroll_seconds = float(preroll_seconds)
        if self.sfreq != 50.0 or base.window_samples != 50:
            raise ValueError("This control is frozen to the existing 50 Hz, 1 s inputs")
        self.signal_key = "meg" if "runtime_context_uid" in self.table else "eeg"
        group_column = "runtime_context_uid" if self.signal_key == "meg" else "sentence_uid"
        self.baseline_samples = (int(round(float(base.dataset_config["baseline_seconds"]) * self.sfreq))
                                 if self.signal_key == "meg" else 0)
        self.baseline_seconds = float(base.dataset_config.get("baseline_seconds", 0.5))
        self.causal_highpass_mode = str(
            base.dataset_config.get("causal_highpass_mode", "moving_average")
            or "moving_average"
        ).strip().lower()
        self.causal_highpass_seconds = float(
            base.dataset_config.get("causal_highpass_seconds", 0.0)
        )
        self.causal_highpass_samples = int(
            round(self.causal_highpass_seconds * self.sfreq)
        )
        self.causal_highpass_hz = float(
            base.dataset_config.get("causal_highpass_hz", 0.5)
        )
        self.causal_highpass_order = int(
            base.dataset_config.get("causal_highpass_order", 2)
        )
        self._butterworth_active = (
            self.signal_key == "meg"
            and self.causal_highpass_mode == "butterworth_iir"
            and self.causal_highpass_hz > 0.0
        )
        self._moving_avg_active = (
            self.signal_key == "meg"
            and self.causal_highpass_mode == "moving_average"
            and self.causal_highpass_samples > 1
        )
        self.clamp = base.dataset_config.get("clamp", 5.0)
        self.words = self.table.normalized_word.astype(str).tolist()
        self.event_ids = self.table.event_id.astype(str).tolist()
        self.recording_ids = self.table.recording_id.astype(str).tolist()
        self.include_onset_seconds = (
            bool(getattr(base, "include_onset_seconds", False))
            if include_onset_seconds is None
            else bool(include_onset_seconds)
        )
        if "onset_seconds" in self.table.columns:
            self.onset_seconds = self.table["onset_seconds"].to_numpy(dtype=np.float32)
        elif "开始时间" in self.table.columns:
            self.onset_seconds = self.table["开始时间"].to_numpy(dtype=np.float32)
        else:
            self.onset_seconds = None
        self.subject_indices = np.asarray([base.subject_map[str(s)] if self.signal_key == "meg"
            else base.subject_to_index[str(s)] for s in self.table.subject_id], dtype=np.int64)
        self.group_keys = list(zip(self.table.subject_id.astype(str),
                                   self.table.recording_id.astype(str), self.table[group_column].astype(str)))
        if self.signal_key == "meg":
            from braindecoding.data.word_window_dataset import recording_onset_to_sample
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
        self._gpu_stream_plan = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __len__(self):
        return len(self.table)

    def ensure_preroll_buffers(self, *args, **kwargs):
        """合成流模式下的空操作占位：严禁回落至 base 读取真实 MEG 信号。"""
        del args, kwargs
        return {}

    def set_epoch(self, epoch):
        if self.split == "train" and int(epoch) != self.epoch:
            self.epoch = int(epoch)
            self._streams.clear()

    def _filter_synthetic_stream_np(self, stream: np.ndarray) -> np.ndarray:
        """对连续合成脉冲流施加与真实 MEG 严格同构的因果高通滤波（零标签依赖）。"""
        if self._butterworth_active:
            from braindecoding.data.word_window_dataset import (
                apply_causal_butterworth_highpass,
            )

            return apply_causal_butterworth_highpass(
                stream,
                sampling_rate_hz=self.sfreq,
                cutoff_hz=self.causal_highpass_hz,
                order=self.causal_highpass_order,
            )
        if self._moving_avg_active:
            from braindecoding.data.word_window_dataset import (
                _apply_causal_moving_average_highpass_full,
            )

            return _apply_causal_moving_average_highpass_full(
                stream,
                highpass_samples=self.causal_highpass_samples,
            )
        return stream

    def _filter_synthetic_streams_gpu(self, sig: torch.Tensor) -> torch.Tensor:
        """在 GPU 批量合成路径中对连续脉冲流 [G, C, L] 施加与 CPU 严格一致的因果高通滤波。"""
        if self._butterworth_active:
            from braindecoding.data.word_window_dataset import (
                apply_causal_butterworth_highpass,
            )

            g_cnt, ch_cnt, l_max = sig.shape
            sig_np = sig.detach().cpu().numpy().reshape(g_cnt * ch_cnt, l_max)
            filt_np = apply_causal_butterworth_highpass(
                sig_np,
                sampling_rate_hz=self.sfreq,
                cutoff_hz=self.causal_highpass_hz,
                order=self.causal_highpass_order,
            )
            return torch.from_numpy(filt_np.reshape(g_cnt, ch_cnt, l_max)).to(
                device=sig.device, dtype=sig.dtype
            )
        if self._moving_avg_active:
            l_max = int(sig.shape[-1])
            hp_w = int(self.causal_highpass_samples)
            sig64 = sig.to(dtype=torch.float64)
            cumsum = torch.zeros(
                (*sig.shape[:-1], l_max + 1),
                device=sig.device,
                dtype=torch.float64,
            )
            torch.cumsum(sig64, dim=-1, out=cumsum[..., 1:])
            right_idx = torch.arange(1, l_max + 1, device=sig.device, dtype=torch.long)
            left_idx = torch.clamp(right_idx - hp_w, min=0)
            denom = (right_idx - left_idx).to(dtype=torch.float64).view(1, 1, l_max)
            moving_avg = (
                cumsum.index_select(-1, right_idx) - cumsum.index_select(-1, left_idx)
            ) / denom
            return (sig64 - moving_avg).to(dtype=sig.dtype)
        return sig

    def _stream(self, index):
        group = self.group_keys[index]
        base_identity = (group, self.event_ids[index] if self.condition == "independent" else "shared_stream")
        shift_key = round(float(self.temporal_shift_seconds), 4)
        is_shifted = abs(shift_key) > 1e-9
        cache_key = (base_identity, "shift", shift_key) if is_shifted else base_identity
        if cache_key not in self._streams:
            if self.split == "train":
                seed_parts = (
                    (GENERATOR_VERSION, self.condition, "train", self.training_seed,
                     self.epoch, base_identity, "shift", shift_key)
                    if is_shifted else
                    (GENERATOR_VERSION, self.condition, "train", self.training_seed,
                     self.epoch, base_identity)
                )
            else:
                seed_parts = (
                    (GENERATOR_VERSION, self.condition, self.split, self.evaluation_seed,
                     base_identity, "shift", shift_key)
                    if is_shifted else
                    (GENERATOR_VERSION, self.condition, self.split, self.evaluation_seed,
                     base_identity)
                )
            seed = stable_seed(*seed_parts)
            raw_stream = synthetic_stream(
                self.stream_lengths[group],
                self.channel_count,
                self.sfreq,
                np.random.default_rng(seed),
            )
            self._streams[cache_key] = self._filter_synthetic_stream_np(raw_stream)
            if len(self._streams) > self.cache_streams:
                self._streams.popitem(last=False)
        self._streams.move_to_end(cache_key)
        return self._streams[cache_key]

    def raw_window(self, index):
        index = int(index)
        start = int(self.starts[index])
        return self._stream(index)[:, start:start + self.window_samples].copy()

    def __getitem__(self, index):
        index = int(index)
        window = transform_window(self.raw_window(index), baseline_samples=self.baseline_samples,
                                  clamp=self.clamp)
        sample = {
                self.signal_key: torch.from_numpy(window),
                "text_embedding": torch.from_numpy(np.array(self.embedding_map[self.words[index]],
                                                               dtype=np.float32, copy=True)),
                "subject_index": torch.tensor(self.subject_indices[index], dtype=torch.long),
                "sentence_index": torch.tensor(self.sentence_indices[index], dtype=torch.long),
                "word": self.words[index], "event_id": self.event_ids[index],
                "recording_id": self.recording_ids[index]}
        if self.include_onset_seconds and self.onset_seconds is not None:
            sample["onset_seconds"] = torch.tensor(
                float(self.onset_seconds[index]), dtype=torch.float32
            )
        return sample

    def _get_gpu_stream_plan(self) -> list[dict]:
        """按句段长度排序并自适应分块，缓存零标签依赖的批量脉冲流索引计划。"""
        if self._gpu_stream_plan is not None:
            return self._gpu_stream_plan

        pad = int(np.ceil(1.0 * self.sfreq))
        units: list[tuple[int, str, np.ndarray]] = []
        if self.condition == "shared":
            grouped: dict[tuple[str, str, str], list[int]] = {}
            for row_idx, key in enumerate(self.group_keys):
                grouped.setdefault(key, []).append(row_idx)
            for key, indices in grouped.items():
                stream_len = int(self.stream_lengths[key])
                ident_str = f"{key[0]}\x1f{key[1]}\x1f{key[2]}\x1fshared_stream"
                units.append(
                    (stream_len, ident_str, np.asarray(indices, dtype=np.int64))
                )
        else:
            for row_idx, key in enumerate(self.group_keys):
                stream_len = int(self.stream_lengths[key])
                ident_str = (
                    f"{key[0]}\x1f{key[1]}\x1f{key[2]}\x1f"
                    f"{self.event_ids[row_idx]}\x1f{row_idx}"
                )
                units.append(
                    (stream_len, ident_str, np.asarray([row_idx], dtype=np.int64))
                )

        units.sort(key=lambda item: (item[0], item[1]))
        plan: list[dict] = []
        cur: list[tuple[int, str, np.ndarray]] = []
        for unit in units:
            next_padded = int(unit[0]) + 2 * pad
            if cur and (
                len(cur) >= _GPU_CHUNK_MAX_STREAMS
                or (len(cur) + 1) * next_padded > _GPU_CHUNK_MAX_PADDED_SAMPLES
            ):
                plan.append(self._pack_gpu_chunk_spec(cur))
                cur = []
            cur.append(unit)
        if cur:
            plan.append(self._pack_gpu_chunk_spec(cur))

        self._gpu_stream_plan = plan
        return plan

    def _pack_gpu_chunk_spec(
        self, chunk_units: list[tuple[int, str, np.ndarray]]
    ) -> dict:
        digest = hashlib.sha256(
            "\n".join(u_key for _, u_key, _ in chunk_units).encode("utf-8")
        ).hexdigest()[:32]
        lengths_np = np.asarray(
            [u_len for u_len, _, _ in chunk_units], dtype=np.int64
        )
        rows_np = np.concatenate(
            [u_rows for _, _, u_rows in chunk_units]
        ).astype(np.int64, copy=False)
        local_g_np = np.repeat(
            np.arange(len(chunk_units), dtype=np.int64),
            [len(u_rows) for _, _, u_rows in chunk_units],
        )
        return {
            "identity_digest": digest,
            "lengths": lengths_np,
            "rows": rows_np,
            "local_g": local_g_np,
            "starts": self.starts[rows_np].astype(np.int64, copy=False),
            "subjects": self.subject_indices[rows_np].astype(
                np.int64, copy=False
            ),
        }

    def _chunk_gpu_seed(self, identity_digest: str, shift_key: float) -> int:
        is_shifted = abs(shift_key) > 1e-9
        if self.split == "train":
            seed_parts = (
                (
                    GENERATOR_VERSION,
                    "gpu_batch_v1",
                    self.condition,
                    "train",
                    self.training_seed,
                    self.epoch,
                    identity_digest,
                    "shift",
                    shift_key,
                )
                if is_shifted
                else (
                    GENERATOR_VERSION,
                    "gpu_batch_v1",
                    self.condition,
                    "train",
                    self.training_seed,
                    self.epoch,
                    identity_digest,
                )
            )
        else:
            seed_parts = (
                (
                    GENERATOR_VERSION,
                    "gpu_batch_v1",
                    self.condition,
                    self.split,
                    self.evaluation_seed,
                    identity_digest,
                    "shift",
                    shift_key,
                )
                if is_shifted
                else (
                    GENERATOR_VERSION,
                    "gpu_batch_v1",
                    self.condition,
                    self.split,
                    self.evaluation_seed,
                    identity_digest,
                )
            )
        return stable_seed(*seed_parts) % (1 << 63)

    @torch.no_grad()
    def materialize_windows_gpu(
        self,
        *,
        device: torch.device,
        apply_transform: bool = True,
    ) -> torch.Tensor:
        """在指定设备上批量合成并按原始行序切出全部词窗 [N, C, W]（供测试与小规模核验）。"""
        n_rows = len(self.table)
        ch_count = int(self.channel_count)
        win_len = int(self.window_samples)
        if n_rows == 0:
            return torch.empty(
                (0, ch_count, win_len), device=device, dtype=torch.float32
            )

        out = torch.empty(
            (n_rows, ch_count, win_len), device=device, dtype=torch.float32
        )
        plan = self._get_gpu_stream_plan()
        shift_key = round(float(self.temporal_shift_seconds), 4)
        win_arange = torch.arange(
            win_len, device=device, dtype=torch.long
        ).unsqueeze(0)
        gen = torch.Generator(device=device)

        for chunk in plan:
            gen.manual_seed(
                self._chunk_gpu_seed(chunk["identity_digest"], shift_key)
            )
            sig = synthesize_streams_batch_gpu(
                chunk["lengths"],
                ch_count,
                self.sfreq,
                gen,
                device=device,
            )
            sig = self._filter_synthetic_streams_gpu(sig)
            g_chunk, _, l_max = sig.shape
            sig_flat_tc = (
                sig.transpose(1, 2).contiguous().view(g_chunk * l_max, ch_count)
            )
            rows_dev = torch.as_tensor(
                chunk["rows"], device=device, dtype=torch.long
            )
            local_g_dev = torch.as_tensor(
                chunk["local_g"], device=device, dtype=torch.long
            )
            starts_dev = torch.as_tensor(
                chunk["starts"], device=device, dtype=torch.long
            )
            linear_gt = (
                local_g_dev.unsqueeze(1) * l_max
                + starts_dev.unsqueeze(1)
                + win_arange
            )
            windows = (
                sig_flat_tc.index_select(0, linear_gt.reshape(-1))
                .view(int(rows_dev.numel()), win_len, ch_count)
                .transpose(1, 2)
                .contiguous()
            )
            if apply_transform:
                windows = transform_window_gpu(
                    windows,
                    baseline_samples=self.baseline_samples,
                    clamp=self.clamp,
                )
            out.index_copy_(0, rows_dev, windows)
        return out

    @torch.no_grad()
    def extract_cnn_embeddings_gpu(
        self,
        model,
        *,
        device: torch.device,
        batch_size: int = 512,
        amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        show_progress: bool = True,
        progress_desc: str = "Pre-extract CNN",
        progress_wrap=None,
        out_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """在 GPU 上直接合成平滑脉冲流、切窗归一化并前向提取冻结 CNN 特征 [N, D]。"""
        n_rows = len(self.table)
        if n_rows == 0:
            if out_tensor is not None:
                return out_tensor
            return torch.empty((0, 0), dtype=torch.float32)

        model.brain_encoder.eval()
        plan = self._get_gpu_stream_plan()
        shift_key = round(float(self.temporal_shift_seconds), 4)
        ch_count = int(self.channel_count)
        win_len = int(self.window_samples)
        eff_bs = max(1, int(batch_size))
        use_amp = (
            bool(amp)
            and device.type == "cuda"
            and amp_dtype in {torch.float16, torch.bfloat16}
        )
        win_arange = torch.arange(
            win_len, device=device, dtype=torch.long
        ).unsqueeze(0)
        gen = torch.Generator(device=device)
        out_cpu: torch.Tensor | None = None

        if progress_wrap is not None:
            iterable, pbar = progress_wrap(
                plan,
                enabled=show_progress,
                desc=progress_desc,
                total=len(plan),
            )
        else:
            iterable, pbar = plan, None

        try:
            for chunk in iterable:
                gen.manual_seed(
                    self._chunk_gpu_seed(chunk["identity_digest"], shift_key)
                )
                sig = synthesize_streams_batch_gpu(
                    chunk["lengths"],
                    ch_count,
                    self.sfreq,
                    gen,
                    device=device,
                )
                sig = self._filter_synthetic_streams_gpu(sig)
                g_chunk, _, l_max = sig.shape
                sig_flat_tc = (
                    sig.transpose(1, 2)
                    .contiguous()
                    .view(g_chunk * l_max, ch_count)
                )
                rows_np = chunk["rows"]
                local_g_dev = torch.as_tensor(
                    chunk["local_g"], device=device, dtype=torch.long
                )
                starts_dev = torch.as_tensor(
                    chunk["starts"], device=device, dtype=torch.long
                )
                subj_dev = torch.as_tensor(
                    chunk["subjects"], device=device, dtype=torch.long
                )
                rows_dev = (
                    torch.as_tensor(
                        rows_np, device=out_tensor.device, dtype=torch.long
                    )
                    if (out_tensor is not None and out_tensor.device.type != "cpu")
                    else None
                )

                w_total = int(len(rows_np))
                for sub_start in range(0, w_total, eff_bs):
                    sub_end = min(sub_start + eff_bs, w_total)
                    sub_b = sub_end - sub_start
                    linear_gt = (
                        local_g_dev[sub_start:sub_end].unsqueeze(1) * l_max
                        + starts_dev[sub_start:sub_end].unsqueeze(1)
                        + win_arange
                    )
                    windows = (
                        sig_flat_tc.index_select(0, linear_gt.reshape(-1))
                        .view(sub_b, win_len, ch_count)
                        .transpose(1, 2)
                        .contiguous()
                    )
                    windows = transform_window_gpu(
                        windows,
                        baseline_samples=self.baseline_samples,
                        clamp=self.clamp,
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        z_sub = F.normalize(
                            model.brain_encoder(
                                windows, subj_dev[sub_start:sub_end]
                            ),
                            dim=-1,
                        )
                    if out_tensor is not None:
                        if out_tensor.device.type == "cpu":
                            out_tensor[rows_np[sub_start:sub_end]] = z_sub.to(
                                device="cpu", dtype=out_tensor.dtype
                            )
                        else:
                            out_tensor.index_copy_(
                                0,
                                rows_dev[sub_start:sub_end],
                                z_sub.to(
                                    device=out_tensor.device,
                                    dtype=out_tensor.dtype,
                                ),
                            )
                    else:
                        if out_cpu is None:
                            out_cpu = torch.empty(
                                (n_rows, int(z_sub.shape[-1])),
                                dtype=torch.float32,
                                device="cpu",
                            )
                        out_cpu[rows_np[sub_start:sub_end]] = (
                            z_sub.float().cpu()
                        )
                del sig, sig_flat_tc
        finally:
            if pbar is not None:
                pbar.close()

        if out_tensor is not None:
            return out_tensor
        assert out_cpu is not None
        return out_cpu


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
