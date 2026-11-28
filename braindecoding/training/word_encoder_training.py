"""词级训练入口共同使用的批处理与编码函数。"""

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler


def brain_encoder_frozen_for_epoch(epoch_index, freeze_epochs):
    """按零基 epoch 冻结 brain encoder；不把 epoch 换算为 update。"""
    return int(epoch_index) < int(freeze_epochs)


def validation_patience_exhausted(epochs_without_improvement, patience):
    """validation 连续未改善轮数达到 patience 时停止。"""
    return int(epochs_without_improvement) >= int(patience)


class SentenceBatchSampler(Sampler):
    """打乱序列组，同时严格限制实际批次大小。

    超过 ``batch_size`` 的组会被切成连续块。这样既保留块内顺序，也不会让
    较长的记录组或故事组在无提示的情况下形成超大批次。
    """

    def __init__(
        self,
        sentence_uids,
        batch_size,
        shuffle,
        seed,
        preserve_complete_groups=False,
    ):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.preserve_complete_groups = bool(preserve_complete_groups)
        self.epoch = 0
        groups = {}
        for index, value in enumerate(sentence_uids):
            groups.setdefault(str(value), []).append(index)
        self.groups = list(groups.values())
        largest_group = max((len(group) for group in self.groups), default=0)
        if self.preserve_complete_groups and largest_group > self.batch_size:
            raise ValueError(
                "完整上下文组不能跨批次："
                f"最大组长 {largest_group}，batch_size={self.batch_size}。"
            )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _ordered_groups(self):
        order = np.arange(len(self.groups))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        return [self.groups[index] for index in order]

    def __iter__(self):
        batch = []
        for group in self._ordered_groups():
            chunks = (
                [group]
                if self.preserve_complete_groups
                else [
                    group[start : start + self.batch_size]
                    for start in range(0, len(group), self.batch_size)
                ]
            )
            for chunk in chunks:
                if batch and len(batch) + len(chunk) > self.batch_size:
                    yield batch
                    batch = []
                if len(chunk) == self.batch_size:
                    if batch:
                        yield batch
                        batch = []
                    yield chunk
                else:
                    batch.extend(chunk)
        if batch:
            yield batch

    def __len__(self):
        count = 0
        batch_size = 0
        for group in self.groups:
            chunks = (
                [group]
                if self.preserve_complete_groups
                else [
                    group[start : start + self.batch_size]
                    for start in range(0, len(group), self.batch_size)
                ]
            )
            for chunk in chunks:
                if batch_size and batch_size + len(chunk) > self.batch_size:
                    count += 1
                    batch_size = 0
                if len(chunk) == self.batch_size:
                    if batch_size:
                        count += 1
                        batch_size = 0
                    count += 1
                else:
                    batch_size += len(chunk)
        return count + int(batch_size > 0)


class MinGapBatchSampler(Sampler):
    """按记录内最小时间间隔贪心分泳道的去重叠词级批采样器。

    将同一记录内事件按 ``onset_seconds`` 升序贪心分配到互不重叠的泳道，保证同一泳道内
    任意来自相同记录的两个词起始时间间隔不小于 ``min_gap_seconds``；跨记录事件可安全共享
    同一泳道，从而在保持 100% 训练词覆盖率的前提下，彻底消除同批对比负样本的时间窗物理重叠。
    """

    def __init__(
        self,
        recording_ids,
        onset_seconds,
        batch_size,
        *,
        min_gap_seconds=2.5,
        shuffle=True,
        seed=0,
    ):
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数：{self.batch_size}")
        self.min_gap_seconds = float(min_gap_seconds)
        if self.min_gap_seconds <= 0.0:
            raise ValueError(f"min_gap_seconds 必须为正数：{self.min_gap_seconds}")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        self._recording_ids = [str(r) for r in recording_ids]
        self._onset_seconds = np.asarray(onset_seconds, dtype=np.float64)
        if len(self._recording_ids) != len(self._onset_seconds):
            raise ValueError("recording_ids 与 onset_seconds 长度不一致。")

        by_recording = {}
        for idx, rec in enumerate(self._recording_ids):
            by_recording.setdefault(rec, []).append(idx)

        global_lanes = []
        for rec in sorted(by_recording):
            rec_indices = sorted(
                by_recording[rec], key=lambda i: (float(self._onset_seconds[i]), i)
            )
            rec_lane_last_onset = []
            for idx in rec_indices:
                onset = float(self._onset_seconds[idx])
                placed = False
                for lane_idx, last_onset in enumerate(rec_lane_last_onset):
                    if onset - last_onset >= self.min_gap_seconds:
                        rec_lane_last_onset[lane_idx] = onset
                        global_lanes[lane_idx].append(idx)
                        placed = True
                        break
                if not placed:
                    lane_idx = len(rec_lane_last_onset)
                    rec_lane_last_onset.append(onset)
                    if lane_idx >= len(global_lanes):
                        global_lanes.append([])
                    global_lanes[lane_idx].append(idx)

        self.lanes = [lane for lane in global_lanes if lane]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _can_insert_into_batch(self, batch, candidate_idx):
        cand_rec = self._recording_ids[candidate_idx]
        cand_onset = float(self._onset_seconds[candidate_idx])
        for existing_idx in batch:
            if (
                self._recording_ids[existing_idx] == cand_rec
                and abs(float(self._onset_seconds[existing_idx]) - cand_onset)
                < self.min_gap_seconds
            ):
                return False
        return True

    def _build_batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        singletons = []
        for lane in self.lanes:
            items = list(lane)
            if self.shuffle and len(items) > 1:
                order = rng.permutation(len(items))
                items = [items[int(i)] for i in order]
            n = len(items)
            if n == 1:
                singletons.append(items[0])
                continue
            cursor = 0
            while cursor < n:
                remaining = n - cursor
                if (
                    remaining == self.batch_size + 1
                    and self.batch_size >= 3
                ):
                    batches.append(items[cursor : cursor + self.batch_size - 1])
                    batches.append(items[cursor + self.batch_size - 1 : n])
                    cursor = n
                else:
                    stop = min(cursor + self.batch_size, n)
                    batches.append(items[cursor:stop])
                    cursor = stop

        for single_idx in singletons:
            placed = False
            for batch in batches:
                if len(batch) < self.batch_size and self._can_insert_into_batch(
                    batch, single_idx
                ):
                    batch.append(single_idx)
                    placed = True
                    break
            if not placed:
                batches.append([single_idx])

        if self.shuffle and len(batches) > 1:
            order = rng.permutation(len(batches))
            batches = [batches[int(i)] for i in order]
        return batches

    def __iter__(self):
        yield from self._build_batches()

    def __len__(self):
        return len(self._build_batches())


class MultiTrialPrototypeBatchSampler(Sampler):
    """第三阶梯 Stage-1 抗坍缩多试次原型批采样器（100% 轮次覆盖 + 零物理窗口重叠）。

    每轮将训练集内同一词的出现无放回划分为大小不超过 ``proto_group_size`` 的零重叠小组
    （同录音任意两试次间隔 ``>= proto_min_gap_seconds``，默认 10.0s；跨录音天然零重叠），
    再将各词小组打包装入 ``batch_size`` 批次，同时严格保证：
      1. 同批任意同录音样本起始间隔 ``>= min_gap_seconds``（默认 2.5s）；
      2. 同批任意同词同录音样本起始间隔 ``>= proto_min_gap_seconds``（默认 10.0s）；
      3. 全轮 100% 不重不漏覆盖全部训练样本，并规避单样本尾批防止 BatchNorm1d 崩溃。
    """

    def __init__(
        self,
        words,
        recording_ids,
        onset_seconds,
        batch_size,
        *,
        proto_group_size=4,
        min_gap_seconds=2.5,
        proto_min_gap_seconds=10.0,
        shuffle=True,
        seed=0,
        max_open_batches=32,
    ):
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数：{self.batch_size}")
        self.proto_group_size = int(proto_group_size)
        if self.proto_group_size <= 0:
            raise ValueError(
                f"proto_group_size 必须为正整数：{self.proto_group_size}"
            )
        self.min_gap_seconds = float(min_gap_seconds)
        self.proto_min_gap_seconds = max(
            float(proto_min_gap_seconds), self.min_gap_seconds
        )
        if self.min_gap_seconds <= 0.0:
            raise ValueError(f"min_gap_seconds 必须为正数：{self.min_gap_seconds}")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_open_batches = max(4, int(max_open_batches))
        self.epoch = 0
        self._cached_epoch = None
        self._cached_batches = None

        n_rows = len(words)
        if len(recording_ids) != n_rows or len(onset_seconds) != n_rows:
            raise ValueError("words、recording_ids 与 onset_seconds 长度必须一致。")

        word_map: dict[str, int] = {}
        rec_map: dict[str, int] = {}
        word_ids = np.empty(n_rows, dtype=np.int64)
        rec_ids = np.empty(n_rows, dtype=np.int64)
        indices_by_word: list[list[int]] = []
        for idx in range(n_rows):
            w_str = str(words[idx])
            wid = word_map.get(w_str)
            if wid is None:
                wid = len(word_map)
                word_map[w_str] = wid
                indices_by_word.append([])
            word_ids[idx] = wid
            indices_by_word[wid].append(idx)

            r_str = str(recording_ids[idx])
            rid = rec_map.get(r_str)
            if rid is None:
                rid = len(rec_map)
                rec_map[r_str] = rid
            rec_ids[idx] = rid

        self._word_ids = word_ids.tolist()
        self._rec_ids = rec_ids.tolist()
        self._onset_seconds = np.asarray(onset_seconds, dtype=np.float64).tolist()
        self._indices_by_word = [
            np.asarray(items, dtype=np.int64) for items in indices_by_word
        ]

    def set_epoch(self, epoch):
        new_epoch = int(epoch)
        if new_epoch != self.epoch:
            self.epoch = new_epoch
            self._cached_epoch = None
            self._cached_batches = None

    def _can_add_item(self, rec_dict, word_rec_dict, idx_val):
        rid = self._rec_ids[idx_val]
        wid = self._word_ids[idx_val]
        onset = self._onset_seconds[idx_val]
        existing_r = rec_dict.get(rid)
        if existing_r is not None:
            min_gap = self.min_gap_seconds
            for prev_t in existing_r:
                if abs(onset - prev_t) < min_gap:
                    return False
        existing_wr = word_rec_dict.get((wid, rid))
        if existing_wr is not None:
            p_gap = self.proto_min_gap_seconds
            for prev_t in existing_wr:
                if abs(onset - prev_t) < p_gap:
                    return False
        return True

    def _build_batches(self):
        if self._cached_epoch == self.epoch and self._cached_batches is not None:
            return self._cached_batches

        rng = np.random.default_rng(self.seed + self.epoch)
        proto_gap = self.proto_min_gap_seconds
        group_k = self.proto_group_size
        rec_ids = self._rec_ids
        onsets = self._onset_seconds

        # 阶段 1：每词内部按 >= proto_min_gap_seconds 无放回划分最多 K 试次小组
        groups: list[tuple[int, list[int]]] = []
        for wid, raw_idx in enumerate(self._indices_by_word):
            n_w = len(raw_idx)
            if n_w == 1 or group_k == 1:
                for item in raw_idx.tolist():
                    groups.append((wid, [int(item)]))
                continue
            perm_idx = raw_idx[rng.permutation(n_w)] if self.shuffle else raw_idx
            open_groups: list[list[int]] = []
            open_rec_onsets: list[dict[int, list[float]]] = []
            for idx_val in perm_idx.tolist():
                rid = rec_ids[idx_val]
                onset = onsets[idx_val]
                placed = False
                for g_i in range(len(open_groups)):
                    rec_map = open_rec_onsets[g_i]
                    existing = rec_map.get(rid)
                    conflict = False
                    if existing is not None:
                        for prev_t in existing:
                            if abs(onset - prev_t) < proto_gap:
                                conflict = True
                                break
                    if not conflict:
                        open_groups[g_i].append(idx_val)
                        if existing is None:
                            rec_map[rid] = [onset]
                        else:
                            existing.append(onset)
                        if len(open_groups[g_i]) >= group_k:
                            groups.append((wid, open_groups.pop(g_i)))
                            open_rec_onsets.pop(g_i)
                        placed = True
                        break
                if not placed:
                    open_groups.append([idx_val])
                    open_rec_onsets.append({rid: [onset]})
            for g in open_groups:
                groups.append((wid, g))

        if self.shuffle and len(groups) > 1:
            order = rng.permutation(len(groups))
            groups = [groups[int(i)] for i in order]

        # 阶段 2：将各词小组打包装入 batch_size 批次并严格守卫同录音 2.5s 与同词 10.0s 间隔
        open_batches: list[list[int]] = []
        open_b_recs: list[dict[int, list[float]]] = []
        open_b_wrecs: list[dict[tuple[int, int], list[float]]] = []
        batches: list[list[int]] = []
        final_b_recs: list[dict[int, list[float]]] = []
        final_b_wrecs: list[dict[tuple[int, int], list[float]]] = []

        bs = self.batch_size
        min_gap = self.min_gap_seconds
        max_open = self.max_open_batches

        for wid, grp in groups:
            g_len = len(grp)
            placed = False
            for b_i in range(len(open_batches)):
                if len(open_batches[b_i]) + g_len > bs:
                    continue
                b_recs = open_b_recs[b_i]
                b_wrecs = open_b_wrecs[b_i]
                conflict = False
                for idx_val in grp:
                    rid = rec_ids[idx_val]
                    onset = onsets[idx_val]
                    existing_r = b_recs.get(rid)
                    if existing_r is not None:
                        for prev_t in existing_r:
                            if abs(onset - prev_t) < min_gap:
                                conflict = True
                                break
                        if conflict:
                            break
                    existing_wr = b_wrecs.get((wid, rid))
                    if existing_wr is not None:
                        for prev_t in existing_wr:
                            if abs(onset - prev_t) < proto_gap:
                                conflict = True
                                break
                        if conflict:
                            break
                if not conflict:
                    open_batches[b_i].extend(grp)
                    for idx_val in grp:
                        rid = rec_ids[idx_val]
                        onset = onsets[idx_val]
                        b_recs.setdefault(rid, []).append(onset)
                        b_wrecs.setdefault((wid, rid), []).append(onset)
                    if len(open_batches[b_i]) >= bs:
                        batches.append(open_batches.pop(b_i))
                        final_b_recs.append(open_b_recs.pop(b_i))
                        final_b_wrecs.append(open_b_wrecs.pop(b_i))
                    placed = True
                    break

            if not placed:
                if g_len >= bs:
                    b_recs_new: dict[int, list[float]] = {}
                    b_wrecs_new: dict[tuple[int, int], list[float]] = {}
                    for idx_val in grp:
                        rid = rec_ids[idx_val]
                        onset = onsets[idx_val]
                        b_recs_new.setdefault(rid, []).append(onset)
                        b_wrecs_new.setdefault((wid, rid), []).append(onset)
                    batches.append(list(grp))
                    final_b_recs.append(b_recs_new)
                    final_b_wrecs.append(b_wrecs_new)
                else:
                    if len(open_batches) >= max_open:
                        fullest_idx = max(
                            range(len(open_batches)),
                            key=lambda i: len(open_batches[i]),
                        )
                        batches.append(open_batches.pop(fullest_idx))
                        final_b_recs.append(open_b_recs.pop(fullest_idx))
                        final_b_wrecs.append(open_b_wrecs.pop(fullest_idx))
                    b_recs_new = {}
                    b_wrecs_new = {}
                    for idx_val in grp:
                        rid = rec_ids[idx_val]
                        onset = onsets[idx_val]
                        b_recs_new.setdefault(rid, []).append(onset)
                        b_wrecs_new.setdefault((wid, rid), []).append(onset)
                    open_batches.append(list(grp))
                    open_b_recs.append(b_recs_new)
                    open_b_wrecs.append(b_wrecs_new)

        singletons: list[int] = []
        for b_items, b_r, b_wr in zip(open_batches, open_b_recs, open_b_wrecs):
            if len(b_items) == 1:
                singletons.append(b_items[0])
            else:
                batches.append(b_items)
                final_b_recs.append(b_r)
                final_b_wrecs.append(b_wr)

        # 尽量吸收单样本尾批，防止 BatchNorm1d 单样本方差异常
        for single_idx in singletons:
            placed = False
            for b_i, b_items in enumerate(batches):
                if len(b_items) < bs and self._can_add_item(
                    final_b_recs[b_i], final_b_wrecs[b_i], single_idx
                ):
                    b_items.append(single_idx)
                    rid = rec_ids[single_idx]
                    wid = self._word_ids[single_idx]
                    onset = onsets[single_idx]
                    final_b_recs[b_i].setdefault(rid, []).append(onset)
                    final_b_wrecs[b_i].setdefault((wid, rid), []).append(onset)
                    placed = True
                    break
            if not placed and bs >= 2:
                s_rid = rec_ids[single_idx]
                s_wid = self._word_ids[single_idx]
                s_onset = onsets[single_idx]
                for b_i, b_items in enumerate(batches):
                    if len(b_items) < 3:
                        continue
                    for pos in range(len(b_items) - 1, -1, -1):
                        donor = b_items[pos]
                        d_rid = rec_ids[donor]
                        d_wid = self._word_ids[donor]
                        d_onset = onsets[donor]
                        if d_rid == s_rid and abs(d_onset - s_onset) < (
                            proto_gap if d_wid == s_wid else min_gap
                        ):
                            continue
                        b_items.pop(pos)
                        new_b = [single_idx, donor]
                        new_r: dict[int, list[float]] = {}
                        new_wr: dict[tuple[int, int], list[float]] = {}
                        for v in new_b:
                            new_r.setdefault(rec_ids[v], []).append(onsets[v])
                            new_wr.setdefault((self._word_ids[v], rec_ids[v]), []).append(
                                onsets[v]
                            )
                        batches.append(new_b)
                        final_b_recs.append(new_r)
                        final_b_wrecs.append(new_wr)
                        placed = True
                        break
                    if placed:
                        break
            if not placed:
                batches.append([single_idx])

        if self.shuffle and len(batches) > 1:
            order = rng.permutation(len(batches))
            batches = [batches[int(i)] for i in order]

        self._cached_epoch = self.epoch
        self._cached_batches = batches
        return batches

    def __iter__(self):
        yield from self._build_batches()

    def __len__(self):
        return len(self._build_batches())


def _wrap_progress(iterable, *, enabled=False, desc=None, total=None):
    """在启用进度条时使用 tqdm 包装迭代器，若未安装 tqdm 则优雅降级。"""
    if not enabled:
        return iterable, None
    try:
        import os
        from tqdm import tqdm

        bar = tqdm(
            iterable,
            total=total if total is not None else len(iterable),
            desc=desc or "Progress",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.4,
            ascii=os.name == "nt",
            bar_format="{l_bar}{bar:24}{r_bar}",
        )
        return bar, bar
    except Exception:
        return iterable, None


def shutdown_dataloader_iterator(loader) -> None:
    """显式关闭 DataLoader 内部的多进程迭代器与锁页内存线程，防止跨轮次/跨种子泄漏。"""
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown_fn = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_fn):
            try:
                shutdown_fn()
            except Exception:
                pass
        try:
            loader._iterator = None
        except Exception:
            pass


def make_loader(
    dataset,
    training_config,
    shuffle,
    group_column="sentence_uid",
    *,
    batch_size=None,
    persistent_workers=None,
):
    """按指定句组列构建保持现有批次语义的 DataLoader。"""
    if group_column not in dataset.table:
        raise KeyError(f"Dataset 缺少批处理分组列：{group_column}")
    resolved_batch_size = int(
        batch_size if batch_size is not None else training_config["batch_size"]
    )
    preserve_groups = bool(training_config.get("preserve_complete_groups", False))
    word_sampling = str(training_config.get("word_sampling", "") or "").strip().lower()
    if (
        bool(shuffle)
        and not preserve_groups
        and word_sampling in {"min_gap_sampler", "multi_trial_prototype_sampler"}
    ):
        rec_col = (
            "recording_id"
            if "recording_id" in dataset.table.columns
            else "记录编号"
        )
        onset_col = (
            "onset_seconds"
            if "onset_seconds" in dataset.table.columns
            else "开始时间"
        )
        if rec_col not in dataset.table.columns or onset_col not in dataset.table.columns:
            raise KeyError(
                f"{word_sampling} 需要 Dataset 包含 {rec_col!r} 与 {onset_col!r} 列。"
            )
        if word_sampling == "multi_trial_prototype_sampler":
            word_col = (
                "normalized_word"
                if "normalized_word" in dataset.table.columns
                else ("word" if "word" in dataset.table.columns else "规范词")
            )
            if word_col not in dataset.table.columns:
                raise KeyError(
                    f"multi_trial_prototype_sampler 需要 Dataset 包含 {word_col!r} 列。"
                )
            sampler = MultiTrialPrototypeBatchSampler(
                dataset.table[word_col].astype(str).tolist(),
                dataset.table[rec_col].astype(str).tolist(),
                dataset.table[onset_col].to_numpy(dtype=np.float64),
                batch_size=resolved_batch_size,
                proto_group_size=int(training_config.get("proto_group_size", 4)),
                min_gap_seconds=float(training_config.get("min_gap_seconds", 2.5)),
                proto_min_gap_seconds=float(
                    training_config.get("proto_min_gap_seconds", 10.0)
                ),
                shuffle=shuffle,
                seed=int(training_config["seed"]),
            )
        else:
            sampler = MinGapBatchSampler(
                dataset.table[rec_col].astype(str).tolist(),
                dataset.table[onset_col].to_numpy(dtype=np.float64),
                batch_size=resolved_batch_size,
                min_gap_seconds=float(training_config.get("min_gap_seconds", 2.5)),
                shuffle=shuffle,
                seed=int(training_config["seed"]),
            )
    else:
        sampler = SentenceBatchSampler(
            dataset.table[group_column].astype(str).tolist(),
            batch_size=resolved_batch_size,
            shuffle=shuffle,
            seed=int(training_config["seed"]),
            preserve_complete_groups=preserve_groups,
        )
    num_workers = int(training_config.get("num_workers", 0))
    loader_kwargs = {
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        if persistent_workers is not None:
            use_persistent = bool(persistent_workers)
        elif shuffle:
            use_persistent = bool(
                training_config.get("persistent_workers", True)
            )
        else:
            use_persistent = False
        loader_kwargs["persistent_workers"] = use_persistent
        loader_kwargs["prefetch_factor"] = int(
            training_config.get("prefetch_factor", 4)
        )
    loader = DataLoader(dataset, **loader_kwargs)
    return loader, sampler


def move_batch(batch, device):
    """移动词级任务的脑信号批次字段。"""
    signal_key = "meg" if "meg" in batch else "eeg"
    return (
        batch[signal_key].to(device, non_blocking=True),
        batch["text_embedding"].to(device, non_blocking=True),
        batch["subject_index"].to(device, non_blocking=True),
        batch["sentence_index"].to(device, non_blocking=True),
    )


def event_times_for_model(model, batch, device):
    """只为显式声明需要物理时间坐标的模型移动 onset。"""
    if not bool(getattr(model, "requires_event_times", False)):
        return None
    if "onset_seconds" not in batch:
        raise KeyError("启用物理时间偏置的模型要求批次包含 onset_seconds。")
    return batch["onset_seconds"].to(
        device=device, dtype=torch.float64, non_blocking=True
    )


def local_context_internal_mask(group_indices, radius=1):
    """标记批次内同时具有指定半径左右邻居的词位置。"""
    if group_indices.ndim != 1:
        raise ValueError("group_indices 必须是一维张量。")
    radius = int(radius)
    if radius < 1:
        raise ValueError("局部上下文半径必须大于零。")
    mask = torch.zeros_like(group_indices, dtype=torch.bool)
    groups = {}
    for row_index, group in enumerate(group_indices.detach().cpu().tolist()):
        groups.setdefault(int(group), []).append(row_index)
    for indices in groups.values():
        if len(indices) >= 2 * radius + 1:
            internal = indices[radius:-radius]
            mask[torch.as_tensor(internal, device=mask.device)] = True
    return mask


def three_state_internal_mask(group_indices):
    """兼容旧调用：三状态对应半径 1。"""
    return local_context_internal_mask(group_indices, radius=1)


def resolve_amp_dtype(config_or_dtype=None):
    """解析混合精度训练/推理使用的 autocast dtype（默认 float16，支持 bf16-mixed）。"""
    if isinstance(config_or_dtype, torch.dtype):
        return config_or_dtype
    raw = config_or_dtype
    if isinstance(config_or_dtype, dict):
        raw = config_or_dtype.get("amp_dtype", config_or_dtype.get("precision", "fp16"))
    text = str(raw or "fp16").strip().lower()
    if text in {"bf16", "bfloat16", "bf16-mixed", "bf16_mixed"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "16", "16-mixed", "fp16-mixed", "fp16_mixed"}:
        return torch.float16
    if text in {"fp32", "float32", "32", "32-true"}:
        return torch.float32
    raise ValueError(f"不支持的混合精度类型: {raw!r}")


def train_one_epoch(
    model,
    loss_module,
    loader,
    optimizer,
    scaler,
    device,
    config,
    freeze_brain_encoder=False,
    return_stats=False,
    return_diagnostics=False,
    epoch_index=0,
    show_progress=False,
    progress_desc=None,
    step_scheduler=None,
    ema=None,
):
    """执行 LibriBrain100 与 SMN4Lang 已共同使用的 epoch 级训练。"""
    model.train()
    if freeze_brain_encoder:
        model.brain_encoder.eval()
    loss_module.train()
    amp_dtype = resolve_amp_dtype(config)
    use_amp = (
        bool(config.get("amp", True))
        and device.type == "cuda"
        and amp_dtype in {torch.float16, torch.bfloat16}
    )
    loss_sum = 0.0
    sample_count = 0
    optimizer_updates = 0
    batches_seen = 0
    word_loss_sum = 0.0
    relation_loss_sum = 0.0
    relation_margin_sum = 0.0
    relation_correct_count = 0
    relation_pair_count = 0
    relation_enabled = bool(
        getattr(model, "relation_auxiliary_enabled", False)
    )
    relation_weight = float(config.get("relation_auxiliary_weight", 0.1))
    if relation_enabled and relation_weight <= 0.0:
        raise ValueError("relation_auxiliary_weight 必须为正数。")
    relation_generator = torch.Generator(device="cpu")
    relation_generator.manual_seed(
        int(config.get("seed", 0)) * 1_000_003 + int(epoch_index) + 97
    )
    prefix_revision_enabled = bool(
        getattr(model, "prefix_revision_enabled", False)
    )
    prefix_generator = torch.Generator(device="cpu")
    prefix_generator.manual_seed(
        int(config.get("seed", 0)) * 1_000_033 + int(epoch_index) + 193
    )
    revision_loss_sum = 0.0
    revision_supervision_count = 0
    revision_delay_counts = {
        int(delay): 0
        for delay in (
            getattr(model, "prefix_revision_horizons", ())
            if prefix_revision_enabled
            else ()
        )
    }
    revision_delay_loss_sums = {
        delay: 0.0 for delay in revision_delay_counts
    }
    revision_delay_loss_batches = {
        delay: 0 for delay in revision_delay_counts
    }
    total_batches = len(loader)
    current_lr = (
        float(optimizer.param_groups[0].get("lr", 0.0))
        if getattr(optimizer, "param_groups", None)
        else 0.0
    )
    iterable, pbar = _wrap_progress(
        loader,
        enabled=show_progress,
        desc=progress_desc or f"Epoch {int(epoch_index) + 1:02d} [train]",
        total=total_batches,
    )
    try:
        needs_word_ids = bool(getattr(loss_module, "requires_word_ids", False))
        needs_pre_head = bool(
            getattr(loss_module, "requires_pre_head", False)
            and getattr(model, "context_transformer", None) is None
            and not prefix_revision_enabled
            and not relation_enabled
        )
        proto_single_sum = 0.0
        proto_pre_sum = 0.0
        proto_z_sum = 0.0
        proto_active_sum = 0.0
        proto_group_size_sum = 0.0
        proto_batches = 0
        for batch in iterable:
            meg, targets, subject_indices, sentence_indices = move_batch(batch, device)
            event_times = event_times_for_model(model, batch, device)
            batch_word_ids = (
                loss_module.map_words_to_ids(batch["word"], device)
                if needs_word_ids
                else None
            )
            batch_pre_head = None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                if getattr(model, "candidate_conditioned", False):
                    if getattr(model, "prefix_supervised", False):
                        _, supervision_scores = model.prefix_candidate_states(
                            meg,
                            targets,
                            subject_indices,
                            sentence_indices,
                            return_supervision_scores=True,
                        )
                        prefix_losses = torch.stack(
                            [
                                loss_module.forward_scores(scores, targets)
                                for scores in supervision_scores
                            ]
                        )
                        weights = prefix_losses.new_tensor(
                            model.prefix_loss_weights
                        )
                        loss = (prefix_losses * weights).sum() / weights.sum()
                    else:
                        scores = model(meg, targets, subject_indices, sentence_indices)
                        loss = loss_module.forward_scores(scores, targets)
                else:
                    if prefix_revision_enabled:
                        revision_kwargs = {"cutoff_generator": prefix_generator}
                        if "revision_cutoff_policy" in config:
                            revision_kwargs["cutoff_policy"] = config[
                                "revision_cutoff_policy"
                            ]
                        (
                            estimates,
                            revision_rows,
                            revision_delays,
                        ) = model.forward_prefix_revision_training(
                            meg,
                            subject_indices,
                            sentence_indices,
                            **revision_kwargs,
                        )
                    elif relation_enabled:
                        estimates, positive_scores, negative_scores = (
                            model.forward_with_relation_auxiliary(
                                meg,
                                subject_indices,
                                sentence_indices,
                                event_times=event_times,
                                corruption_generator=relation_generator,
                            )
                        )
                    elif needs_pre_head and hasattr(model, "forward_word_with_pre_head"):
                        estimates, batch_pre_head = model.forward_word_with_pre_head(
                            meg, subject_indices
                        )
                    elif event_times is None:
                        estimates = model(meg, subject_indices, sentence_indices)
                    else:
                        estimates = model(
                            meg,
                            subject_indices,
                            sentence_indices,
                            event_times=event_times,
                        )
                    if prefix_revision_enabled:
                        if bool(
                            getattr(
                                model,
                                "prefix_revision_separate_horizon_losses",
                                False,
                            )
                        ):
                            horizon_losses = []
                            for delay in revision_delay_counts:
                                delay_mask = revision_delays == delay
                                if not bool(delay_mask.any()):
                                    continue
                                delay_loss = (
                                    loss_module.forward_rectangular(
                                        estimates[delay_mask],
                                        targets,
                                        revision_rows[delay_mask],
                                        target_word_ids=batch_word_ids[
                                            revision_rows[delay_mask]
                                        ],
                                    )
                                    if batch_word_ids is not None
                                    else loss_module.forward_rectangular(
                                        estimates[delay_mask],
                                        targets,
                                        revision_rows[delay_mask],
                                    )
                                )
                                horizon_losses.append(delay_loss)
                                revision_delay_loss_sums[delay] += float(
                                    delay_loss.detach()
                                )
                                revision_delay_loss_batches[delay] += 1
                            if not horizon_losses:
                                raise RuntimeError("当前 batch 没有可用的滚动回修监督。")
                            revision_loss = torch.stack(horizon_losses).mean()
                        else:
                            revision_loss = (
                                loss_module.forward_rectangular(
                                    estimates,
                                    targets,
                                    revision_rows,
                                    target_word_ids=batch_word_ids[revision_rows],
                                )
                                if batch_word_ids is not None
                                else loss_module.forward_rectangular(
                                    estimates,
                                    targets,
                                    revision_rows,
                                )
                            )
                        word_loss = revision_loss
                        revision_loss_sum += float(
                            revision_loss.detach()
                        ) * len(estimates)
                        revision_supervision_count += len(estimates)
                        for delay in revision_delay_counts:
                            revision_delay_counts[delay] += int(
                                (revision_delays == delay).sum()
                            )
                    else:
                        if batch_pre_head is not None:
                            immediate_loss = loss_module(
                                estimates,
                                targets,
                                target_word_ids=batch_word_ids,
                                pre_head=batch_pre_head,
                            )
                        elif batch_word_ids is not None:
                            immediate_loss = loss_module(
                                estimates,
                                targets,
                                target_word_ids=batch_word_ids,
                            )
                        else:
                            immediate_loss = loss_module(estimates, targets)
                        word_loss = immediate_loss
                    if relation_enabled:
                        margins = positive_scores.float() - negative_scores.float()
                        if len(margins):
                            relation_loss = F.softplus(-margins).mean()
                            relation_loss_sum += float(
                                relation_loss.detach()
                            ) * len(margins)
                            relation_margin_sum += float(margins.detach().sum())
                            relation_correct_count += int(
                                (margins.detach() > 0).sum()
                            )
                            relation_pair_count += len(margins)
                        else:
                            relation_loss = word_loss * 0.0
                        loss = word_loss + relation_weight * relation_loss
                    else:
                        loss = word_loss
                    word_loss_sum += float(word_loss.detach()) * len(meg)
            scaler.scale(loss).backward()
            max_grad_norm = float(config.get("max_grad_norm", 0.0))
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                clip_params = list(model.parameters())
                proto_head = getattr(loss_module, "proto_pre_head", None)
                if proto_head is not None:
                    clip_params.extend(proto_head.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm)
            previous_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            step_succeeded = (
                not use_amp
                or not scaler.is_enabled()
                or float(scaler.get_scale()) >= previous_scale
            )
            if step_succeeded:
                optimizer_updates += 1
                if step_scheduler is not None:
                    step_scheduler.step()
                if ema is not None:
                    ema.update(model)
            loss_sum += float(loss.detach()) * len(meg)
            last_comp = getattr(loss_module, "last_components", None)
            if isinstance(last_comp, dict) and last_comp:
                proto_single_sum += float(last_comp["single_loss"]) * len(meg)
                proto_pre_sum += float(last_comp["proto_pre_loss"]) * len(meg)
                proto_z_sum += float(last_comp["proto_z_loss"]) * len(meg)
                proto_active_sum += float(last_comp["active_prototypes"])
                proto_group_size_sum += float(last_comp["mean_group_size"])
                proto_batches += 1
            sample_count += len(meg)
            batches_seen += 1
            if pbar is not None and (
                batches_seen == 1
                or batches_seen % 5 == 0
                or batches_seen == total_batches
            ):
                running_loss = loss_sum / max(sample_count, 1)
                postfix = {
                    "loss": f"{running_loss:.4f}",
                    "lr": f"{current_lr:.1e}",
                }
                if device.type == "cuda":
                    alloc_mb = torch.cuda.max_memory_allocated(device) / (
                        1024.0 * 1024.0
                    )
                    postfix["mem"] = f"{alloc_mb:.0f}MB"
                pbar.set_postfix(postfix, refresh=False)
    finally:
        if pbar is not None:
            pbar.close()
        if not bool(getattr(loader, "persistent_workers", False)):
            shutdown_dataloader_iterator(loader)
    mean_loss = loss_sum / max(sample_count, 1)
    diagnostics = {
        "word_loss": word_loss_sum / max(sample_count, 1),
        "relation_loss": (
            relation_loss_sum / relation_pair_count
            if relation_pair_count
            else None
        ),
        "relation_pairwise_accuracy": (
            relation_correct_count / relation_pair_count
            if relation_pair_count
            else None
        ),
        "relation_mean_margin": (
            relation_margin_sum / relation_pair_count
            if relation_pair_count
            else None
        ),
        "relation_pair_count": int(relation_pair_count),
        "relation_weight": relation_weight if relation_enabled else None,
        "prefix_revision_loss": (
            revision_loss_sum / revision_supervision_count
            if revision_supervision_count
            else None
        ),
        "prefix_revision_supervision_count": int(
            revision_supervision_count
        ),
        "prefix_revision_delay_counts": revision_delay_counts,
        "prefix_revision_delay_losses": {
            delay: (
                revision_delay_loss_sums[delay]
                / revision_delay_loss_batches[delay]
                if revision_delay_loss_batches[delay]
                else None
            )
            for delay in revision_delay_counts
        },
        "prototype_anti_collapse": (
            {
                "single_loss": proto_single_sum / max(sample_count, 1),
                "proto_pre_loss": proto_pre_sum / max(sample_count, 1),
                "proto_z_loss": proto_z_sum / max(sample_count, 1),
                "mean_active_prototypes": proto_active_sum / max(proto_batches, 1),
                "mean_group_size": proto_group_size_sum / max(proto_batches, 1),
            }
            if proto_batches > 0
            else None
        ),
    }
    if return_stats:
        if return_diagnostics:
            return mean_loss, optimizer_updates, batches_seen, diagnostics
        return mean_loss, optimizer_updates, batches_seen
    return mean_loss


@torch.inference_mode()
def _encode_loader_with_session_calibration(
    model,
    loader,
    device,
    *,
    use_amp: bool,
    resolved_dtype: torch.dtype,
    subject_index_map=None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict:
    """两步无泄漏因果在线会话校准推理（与预提取验证编码严格逐位一致，同时兼容零重叠拼接数据集）。"""
    model.eval()
    ctx = getattr(model, "context_transformer", None)
    assert ctx is not None
    local_context_radius = int(getattr(model, "local_context_radius", 1))
    prefix_revision_enabled = bool(
        getattr(model, "prefix_revision_enabled", False)
    )
    dataset_obj = getattr(loader, "dataset", None)
    is_spliced = bool(
        dataset_obj is not None
        and hasattr(dataset_obj, "base_dataset")
        and hasattr(dataset_obj, "donor_indices")
        and hasattr(dataset_obj, "group_ids")
    )

    if is_spliced:
        base_ds = dataset_obj.base_dataset
        donor_indices = torch.as_tensor(
            dataset_obj.donor_indices, dtype=torch.long
        )
        eval_bs = int(
            getattr(loader, "batch_size", None)
            or getattr(getattr(loader, "batch_sampler", None), "batch_size", None)
            or 512
        )
        base_loader = DataLoader(
            base_ds,
            batch_size=max(1, eval_bs),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        base_z_chunks: list[torch.Tensor] = []
        base_h_pre_chunks: list[torch.Tensor] = []
        base_h_bins_chunks: list[torch.Tensor] = []
        base_targets_chunks: list[torch.Tensor] = []
        base_subj_chunks: list[torch.Tensor] = []
        base_onset_chunks: list[torch.Tensor] = []
        base_words: list[str] = []
        base_event_ids: list[str] = []
        base_recording_ids: list[str] = []
        iterable_b, pbar_b = _wrap_progress(
            base_loader,
            enabled=show_progress,
            desc=(progress_desc or "Validating") + " [base Stage-1]",
            total=len(base_loader),
        )
        try:
            for batch in iterable_b:
                meg, text_emb, subj_idx, _ = move_batch(batch, device)
                if subject_index_map is not None:
                    from braindecoding.subject_generalization import (
                        remap_subject_indices,
                    )

                    subj_idx = remap_subject_indices(subj_idx, subject_index_map)
                with torch.autocast(
                    device_type=device.type,
                    dtype=resolved_dtype,
                    enabled=use_amp,
                ):
                    z_b, h_pre_b, h_bins_b = model.extract_stage1_features(
                        meg, subj_idx
                    )
                base_z_chunks.append(z_b.float().cpu())
                if h_pre_b is not None:
                    base_h_pre_chunks.append(h_pre_b.float().cpu())
                if h_bins_b is not None:
                    base_h_bins_chunks.append(h_bins_b.float().cpu())
                base_targets_chunks.append(text_emb.float().cpu())
                base_subj_chunks.append(subj_idx.detach().cpu())
                if "onset_seconds" in batch:
                    base_onset_chunks.append(
                        batch["onset_seconds"].detach().float().cpu()
                    )
                base_words.extend(str(w) for w in batch["word"])
                base_event_ids.extend(str(v) for v in batch["event_id"])
                base_recording_ids.extend(str(v) for v in batch["recording_id"])
        finally:
            if pbar_b is not None:
                pbar_b.close()

        base_z_all = torch.cat(base_z_chunks, dim=0)
        base_h_pre_all = (
            torch.cat(base_h_pre_chunks, dim=0) if base_h_pre_chunks else None
        )
        base_h_bins_all = (
            torch.cat(base_h_bins_chunks, dim=0) if base_h_bins_chunks else None
        )
        base_subj_all = torch.cat(base_subj_chunks, dim=0)
        base_onset_all = (
            torch.cat(base_onset_chunks, dim=0)
            if base_onset_chunks
            else torch.arange(len(base_z_all), dtype=torch.float32) * 0.35
        )
        base_z_cal, base_self_cal = ctx.calibrate_stage1_features(
            base_z_all,
            base_h_pre_all,
            base_h_bins_all,
            recording_ids=base_recording_ids,
            subject_indices=base_subj_all,
            onset_seconds=base_onset_all,
        )
        base_targets_all = torch.cat(base_targets_chunks, dim=0)

        # 按拼接数据集 batch_sampler 顺序提取每个供体在原始录音中因果校准后的 Stage-1 特征
        all_z_cal = base_z_cal.index_select(0, donor_indices)
        all_self_cal = (
            base_self_cal.index_select(0, donor_indices)
            if base_self_cal is not None
            else None
        )
        all_targets = base_targets_all.index_select(0, donor_indices)
        all_subj = base_subj_all.index_select(0, donor_indices)
        all_sent = torch.as_tensor(dataset_obj.group_ids, dtype=torch.long)
        all_onset = base_onset_all.index_select(0, donor_indices)
        donor_list = donor_indices.tolist()
        all_words = [base_words[d] for d in donor_list]
        all_event_ids = [base_event_ids[d] for d in donor_list]
        all_recording_ids = [base_recording_ids[d] for d in donor_list]

        batch_index_lists = [list(b_idx) for b_idx in loader.batch_sampler]
        order_flat = [i for b_idx in batch_index_lists for i in b_idx]
        words = [all_words[i] for i in order_flat]
        event_ids = [all_event_ids[i] for i in order_flat]
        recording_ids = [all_recording_ids[i] for i in order_flat]
        ordered_idx_t = torch.as_tensor(order_flat, dtype=torch.long)
        targets_ordered = all_targets.index_select(0, ordered_idx_t)
        subj_ordered = all_subj.index_select(0, ordered_idx_t)
        sent_ordered = all_sent.index_select(0, ordered_idx_t)
    else:
        z_chunks: list[torch.Tensor] = []
        h_pre_chunks: list[torch.Tensor] = []
        h_bins_chunks: list[torch.Tensor] = []
        targets_chunks: list[torch.Tensor] = []
        subj_chunks: list[torch.Tensor] = []
        sent_chunks: list[torch.Tensor] = []
        onset_chunks: list[torch.Tensor] = []
        words = []
        event_ids = []
        recording_ids = []
        batch_slices: list[tuple[int, int]] = []
        cursor = 0

        iterable, pbar = _wrap_progress(
            loader,
            enabled=show_progress,
            desc=progress_desc or "Validating",
            total=len(loader),
        )
        try:
            for batch in iterable:
                meg, text_emb, subj_idx, sent_idx = move_batch(batch, device)
                if subject_index_map is not None:
                    from braindecoding.subject_generalization import (
                        remap_subject_indices,
                    )

                    subj_idx = remap_subject_indices(subj_idx, subject_index_map)
                with torch.autocast(
                    device_type=device.type,
                    dtype=resolved_dtype,
                    enabled=use_amp,
                ):
                    z_b, h_pre_b, h_bins_b = model.extract_stage1_features(
                        meg, subj_idx
                    )
                b_len = int(len(z_b))
                batch_slices.append((cursor, cursor + b_len))
                cursor += b_len
                z_chunks.append(z_b.float().cpu())
                if h_pre_b is not None:
                    h_pre_chunks.append(h_pre_b.float().cpu())
                if h_bins_b is not None:
                    h_bins_chunks.append(h_bins_b.float().cpu())
                targets_chunks.append(text_emb.float().cpu())
                subj_chunks.append(subj_idx.detach().cpu())
                sent_chunks.append(sent_idx.detach().cpu())
                if "onset_seconds" in batch:
                    onset_chunks.append(
                        batch["onset_seconds"].detach().float().cpu()
                    )
                words.extend(str(w) for w in batch["word"])
                event_ids.extend(str(v) for v in batch["event_id"])
                recording_ids.extend(str(v) for v in batch["recording_id"])
        finally:
            if pbar is not None:
                pbar.close()
            if not bool(getattr(loader, "persistent_workers", False)):
                shutdown_dataloader_iterator(loader)

        raw_z_all = torch.cat(z_chunks, dim=0)
        raw_h_pre_all = (
            torch.cat(h_pre_chunks, dim=0) if h_pre_chunks else None
        )
        raw_h_bins_all = (
            torch.cat(h_bins_chunks, dim=0) if h_bins_chunks else None
        )
        targets_ordered = torch.cat(targets_chunks, dim=0)
        subj_ordered = torch.cat(subj_chunks, dim=0)
        sent_ordered = torch.cat(sent_chunks, dim=0)
        onset_ordered = (
            torch.cat(onset_chunks, dim=0)
            if onset_chunks
            else torch.arange(len(raw_z_all), dtype=torch.float32) * 0.35
        )
        all_z_cal, all_self_cal = ctx.calibrate_stage1_features(
            raw_z_all,
            raw_h_pre_all,
            raw_h_bins_all,
            recording_ids=recording_ids,
            subject_indices=subj_ordered,
            onset_seconds=onset_ordered,
        )
        all_sent = sent_ordered
        all_onset = onset_ordered
        batch_index_lists = [list(range(s, e)) for s, e in batch_slices]

    predictions: list[torch.Tensor] = []
    internal_masks: list[torch.Tensor] = []
    revision_predictions: dict[int, list[torch.Tensor]] = {}
    revision_availability_masks: dict[int, list[torch.Tensor]] = {}
    has_temporal_bias_head = (
        getattr(model, "temporal_bias_head", None) is not None
    )

    for b_indices in batch_index_lists:
        idx_cpu = torch.as_tensor(b_indices, dtype=torch.long)
        z_batch = all_z_cal.index_select(0, idx_cpu).to(device, non_blocking=True)
        self_batch = (
            all_self_cal.index_select(0, idx_cpu).to(device, non_blocking=True)
            if all_self_cal is not None
            else None
        )
        sent_batch = all_sent.index_select(0, idx_cpu).to(device, non_blocking=True)
        onset_batch = (
            all_onset.index_select(0, idx_cpu).to(device, non_blocking=True)
            if has_temporal_bias_head
            else None
        )
        with torch.autocast(
            device_type=device.type,
            dtype=resolved_dtype,
            enabled=use_amp,
        ):
            if prefix_revision_enabled:
                batch_revisions = model.forward_prefix_revisions_from_stage1_features(
                    z_batch,
                    sent_batch,
                    self_features=self_batch,
                    onset_seconds=onset_batch,
                )
                batch_availability = model.revision_availability_masks(sent_batch)
                for delay, pred in batch_revisions.items():
                    revision_predictions.setdefault(int(delay), []).append(
                        pred.float().cpu()
                    )
                    revision_availability_masks.setdefault(int(delay), []).append(
                        batch_availability[delay].cpu()
                    )
                estimate = batch_revisions[max(batch_revisions)]
            else:
                estimate = model.forward_from_stage1_features(
                    z_batch,
                    sent_batch,
                    self_features=self_batch,
                    onset_seconds=onset_batch,
                )
        predictions.append(estimate.float().cpu())
        internal_masks.append(
            local_context_internal_mask(
                sent_batch, radius=local_context_radius
            ).cpu()
        )

    concatenated_internal_masks = torch.cat(internal_masks)
    encoded = {
        "predictions": torch.cat(predictions),
        "targets": targets_ordered,
        "words": words,
        "event_ids": event_ids,
        "recording_ids": recording_ids,
        "subject_indices": subj_ordered,
        "sentence_indices": sent_ordered,
        "local_context_internal_mask": concatenated_internal_masks,
        "three_state_internal_mask": concatenated_internal_masks,
    }
    if revision_predictions:
        encoded["revision_predictions"] = {
            delay: torch.cat(vals) for delay, vals in revision_predictions.items()
        }
        encoded["revision_availability_masks"] = {
            delay: torch.cat(vals)
            for delay, vals in revision_availability_masks.items()
        }
    return encoded


@torch.inference_mode()
def encode_loader(
    model,
    loader,
    device,
    amp=True,
    subject_index_map=None,
    evaluate_relation_auxiliary=False,
    relation_seed=104729,
    amp_dtype=None,
    show_progress=False,
    progress_desc=None,
):
    """编码完整数据划分并保留现有检索元数据。"""
    resolved_dtype = resolve_amp_dtype(amp_dtype)
    use_amp = (
        bool(amp)
        and device.type == "cuda"
        and resolved_dtype in {torch.float16, torch.bfloat16}
    )
    if (
        not evaluate_relation_auxiliary
        and bool(getattr(model, "requires_session_calibration", False))
    ):
        return _encode_loader_with_session_calibration(
            model,
            loader,
            device,
            use_amp=use_amp,
            resolved_dtype=resolved_dtype,
            subject_index_map=subject_index_map,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )

    model.eval()
    predictions = []
    targets = []
    words = []
    event_ids = []
    recording_ids = []
    subject_indices_all = []
    sentence_indices_all = []
    internal_masks = []
    relation_positive_scores = []
    relation_negative_scores = []
    revision_predictions = {}
    revision_availability_masks = {}
    relation_generator = torch.Generator(device="cpu")
    relation_generator.manual_seed(int(relation_seed))
    local_context_radius = int(getattr(model, "local_context_radius", 1))
    iterable, pbar = _wrap_progress(
        loader,
        enabled=show_progress,
        desc=progress_desc or "Validating",
        total=len(loader),
    )
    try:
        for batch in iterable:
            meg, text_embedding, subject_indices, sentence_indices = move_batch(
                batch, device
            )
            event_times = event_times_for_model(model, batch, device)
            has_temporal_bias_head = (
                getattr(model, "temporal_bias_head", None) is not None
            )
            onset_seconds = (
                batch["onset_seconds"].to(device, non_blocking=True)
                if (has_temporal_bias_head and "onset_seconds" in batch)
                else None
            )
            if subject_index_map is not None:
                from braindecoding.subject_generalization import remap_subject_indices

                subject_indices = remap_subject_indices(
                    subject_indices, subject_index_map
                )
            with torch.autocast(
                device_type=device.type,
                dtype=resolved_dtype,
                enabled=use_amp,
            ):
                if bool(getattr(model, "prefix_revision_enabled", False)):
                    rev_kwargs = (
                        {"onset_seconds": onset_seconds}
                        if onset_seconds is not None
                        else {}
                    )
                    batch_revisions = model.forward_prefix_revisions(
                        meg, subject_indices, sentence_indices, **rev_kwargs
                    )
                    batch_availability = model.revision_availability_masks(
                        sentence_indices
                    )
                    for delay, prediction in batch_revisions.items():
                        revision_predictions.setdefault(int(delay), []).append(
                            prediction.float().cpu()
                        )
                        revision_availability_masks.setdefault(
                            int(delay), []
                        ).append(batch_availability[delay].cpu())
                    maximum_delay = max(batch_revisions)
                    estimate = batch_revisions[maximum_delay]
                elif evaluate_relation_auxiliary:
                    if not bool(
                        getattr(model, "relation_auxiliary_enabled", False)
                    ):
                        raise ValueError("请求关系验证，但模型未启用关系判别头。")
                    estimate, positive_scores, negative_scores = (
                        model.forward_with_relation_auxiliary(
                            meg,
                            subject_indices,
                            sentence_indices,
                            event_times=event_times,
                            corruption_generator=relation_generator,
                        )
                    )
                    relation_positive_scores.append(positive_scores.float().cpu())
                    relation_negative_scores.append(negative_scores.float().cpu())
                elif event_times is None:
                    if onset_seconds is not None:
                        estimate = model(
                            meg,
                            subject_indices,
                            sentence_indices,
                            onset_seconds=onset_seconds,
                        )
                    else:
                        estimate = model(meg, subject_indices, sentence_indices)
                else:
                    if onset_seconds is not None:
                        estimate = model(
                            meg,
                            subject_indices,
                            sentence_indices,
                            event_times=event_times,
                            onset_seconds=onset_seconds,
                        )
                    else:
                        estimate = model(
                            meg,
                            subject_indices,
                            sentence_indices,
                            event_times=event_times,
                        )
            predictions.append(estimate.float().cpu())
            targets.append(text_embedding.float().cpu())
            subject_indices_all.append(subject_indices.detach().cpu())
            sentence_indices_all.append(sentence_indices.detach().cpu())
            words.extend(str(word) for word in batch["word"])
            event_ids.extend(str(value) for value in batch["event_id"])
            recording_ids.extend(str(value) for value in batch["recording_id"])
            internal_masks.append(
                local_context_internal_mask(
                    sentence_indices, radius=local_context_radius
                ).cpu()
            )
    finally:
        if pbar is not None:
            pbar.close()
        if not bool(getattr(loader, "persistent_workers", False)):
            shutdown_dataloader_iterator(loader)
    concatenated_internal_masks = torch.cat(internal_masks)
    encoded = {
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
        "words": words,
        "event_ids": event_ids,
        "recording_ids": recording_ids,
        "subject_indices": torch.cat(subject_indices_all),
        "sentence_indices": torch.cat(sentence_indices_all),
        "local_context_internal_mask": concatenated_internal_masks,
        "three_state_internal_mask": concatenated_internal_masks,
    }
    if evaluate_relation_auxiliary:
        encoded["relation_positive_scores"] = torch.cat(
            relation_positive_scores
        )
        encoded["relation_negative_scores"] = torch.cat(
            relation_negative_scores
        )
    if revision_predictions:
        encoded["revision_predictions"] = {
            delay: torch.cat(values)
            for delay, values in revision_predictions.items()
        }
        encoded["revision_availability_masks"] = {
            delay: torch.cat(values)
            for delay, values in revision_availability_masks.items()
        }
    return encoded


@torch.inference_mode()
def score_loader(
    model, loader, candidate_embeddings, device, amp=True, amp_dtype=None
):
    """让候选条件模型对固定候选集逐项读取完整数据划分。"""
    if not getattr(model, "candidate_conditioned", False):
        raise ValueError("score_loader 只接受候选条件模型。")
    model.eval()
    candidates = torch.as_tensor(candidate_embeddings).to(
        device=device, dtype=torch.float32
    )
    scores = []
    targets = []
    words = []
    event_ids = []
    recording_ids = []
    resolved_dtype = resolve_amp_dtype(amp_dtype)
    use_amp = (
        bool(amp)
        and device.type == "cuda"
        and resolved_dtype in {torch.float16, torch.bfloat16}
    )
    for batch in loader:
        meg, text_embedding, subject_indices, sentence_indices = move_batch(
            batch, device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=resolved_dtype,
            enabled=use_amp,
        ):
            batch_scores = model(
                meg, candidates, subject_indices, sentence_indices
            )
        scores.append(batch_scores.float().cpu())
        targets.append(text_embedding.float().cpu())
        words.extend(str(word) for word in batch["word"])
        event_ids.extend(str(value) for value in batch["event_id"])
        recording_ids.extend(str(value) for value in batch["recording_id"])
    return {
        "scores": torch.cat(scores),
        "targets": torch.cat(targets),
        "words": words,
        "event_ids": event_ids,
        "recording_ids": recording_ids,
    }
