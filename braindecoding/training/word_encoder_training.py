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


def make_loader(
    dataset,
    training_config,
    shuffle,
    group_column="sentence_uid",
    *,
    batch_size=None,
):
    """按指定句组列构建保持现有批次语义的 DataLoader。"""
    if group_column not in dataset.table:
        raise KeyError(f"Dataset 缺少批处理分组列：{group_column}")
    resolved_batch_size = int(
        batch_size if batch_size is not None else training_config["batch_size"]
    )
    sampler = SentenceBatchSampler(
        dataset.table[group_column].astype(str).tolist(),
        batch_size=resolved_batch_size,
        shuffle=shuffle,
        seed=int(training_config["seed"]),
        preserve_complete_groups=bool(
            training_config.get("preserve_complete_groups", False)
        ),
    )
    num_workers = int(training_config.get("num_workers", 0))
    loader_kwargs = {
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            training_config.get("persistent_workers", True)
        )
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
        for batch in iterable:
            meg, targets, subject_indices, sentence_indices = move_batch(batch, device)
            event_times = event_times_for_model(model, batch, device)
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
                                delay_loss = loss_module.forward_rectangular(
                                    estimates[delay_mask],
                                    targets,
                                    revision_rows[delay_mask],
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
                            revision_loss = loss_module.forward_rectangular(
                                estimates,
                                targets,
                                revision_rows,
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
            loss_sum += float(loss.detach()) * len(meg)
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
    }
    if return_stats:
        if return_diagnostics:
            return mean_loss, optimizer_updates, batches_seen, diagnostics
        return mean_loss, optimizer_updates, batches_seen
    return mean_loss


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
    model.eval()
    predictions = []
    targets = []
    words = []
    event_ids = []
    recording_ids = []
    internal_masks = []
    relation_positive_scores = []
    relation_negative_scores = []
    revision_predictions = {}
    revision_availability_masks = {}
    full_context_predictions = []
    relation_generator = torch.Generator(device="cpu")
    relation_generator.manual_seed(int(relation_seed))
    local_context_radius = int(getattr(model, "local_context_radius", 1))
    resolved_dtype = resolve_amp_dtype(amp_dtype)
    use_amp = (
        bool(amp)
        and device.type == "cuda"
        and resolved_dtype in {torch.float16, torch.bfloat16}
    )
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
                    batch_revisions = model.forward_prefix_revisions(
                        meg, subject_indices, sentence_indices
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
                    full_ctx_est = (
                        model(meg, subject_indices, sentence_indices)
                        if event_times is None
                        else model(
                            meg,
                            subject_indices,
                            sentence_indices,
                            event_times=event_times,
                        )
                    )
                    full_context_predictions.append(full_ctx_est.float().cpu())
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
                    estimate = model(meg, subject_indices, sentence_indices)
                else:
                    estimate = model(
                        meg,
                        subject_indices,
                        sentence_indices,
                        event_times=event_times,
                    )
            predictions.append(estimate.float().cpu())
            targets.append(text_embedding.float().cpu())
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
    concatenated_internal_masks = torch.cat(internal_masks)
    encoded = {
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
        "words": words,
        "event_ids": event_ids,
        "recording_ids": recording_ids,
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
    if full_context_predictions:
        encoded["full_context_predictions"] = torch.cat(full_context_predictions)
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
