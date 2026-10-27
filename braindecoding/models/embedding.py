"""主脑电/脑磁词嵌入与局部窗口滚动修订模型（BrainEmbeddingModel）。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from braindecoding.models.brain_encoder import DilatedConvEncoder
from braindecoding.models.transformer import GroupedTransformerEncoder


class PreTransformerTrajectoryResidual(nn.Module):
    """把相邻状态的一阶变化以零门控残差注入 Transformer 输入。"""

    def __init__(self, dimension, config):
        super().__init__()
        self.dimension = int(dimension)
        self.hidden_dimension = int(config.get("hidden_dimension", 512))
        if self.hidden_dimension <= 0:
            raise ValueError("轨迹残差 hidden_dimension 必须为正数。")
        self.projection = nn.Sequential(
            nn.Linear(2 * self.dimension, self.hidden_dimension),
            nn.GELU(),
            nn.Linear(self.hidden_dimension, self.dimension),
        )
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, features, group_indices):
        if group_indices is None:
            raise ValueError("轨迹残差要求提供 group_indices。")
        if features.ndim != 2 or features.shape[-1] != self.dimension:
            raise ValueError("轨迹残差输入必须为 [batch, dimension]。")
        if group_indices.ndim != 1 or len(group_indices) != len(features):
            raise ValueError("group_indices 必须与轨迹残差输入逐行对应。")

        trajectory = torch.zeros_like(features)
        groups = {}
        for row_index, group in enumerate(
            group_indices.detach().cpu().tolist()
        ):
            groups.setdefault(int(group), []).append(row_index)
        for indices in groups.values():
            if len(indices) < 3:
                continue
            index_tensor = torch.tensor(indices, device=features.device)
            grouped = features[index_tensor]
            incoming = grouped[1:-1] - grouped[:-2]
            outgoing = grouped[2:] - grouped[1:-1]
            projected = self.projection(
                torch.cat((incoming, outgoing), dim=-1)
            )
            trajectory[index_tensor[1:-1]] = projected.to(
                dtype=features.dtype
            )
        return features + self.alpha * trajectory


class RelationDiscriminationHead(nn.Module):
    """为整句上下文表示打分，用于真实顺序与同句乱序的配对判别。"""

    def __init__(self, dimension, config):
        super().__init__()
        self.dimension = int(dimension)
        self.hidden_dimension = int(config.get("hidden_dimension", 256))
        self.minimum_group_size = int(config.get("minimum_group_size", 3))
        if self.hidden_dimension <= 0:
            raise ValueError("关系判别头 hidden_dimension 必须为正数。")
        if self.minimum_group_size < 3:
            raise ValueError("关系判别至少要求三个同句状态。")
        self.scorer = nn.Sequential(
            nn.Linear(self.dimension, self.hidden_dimension),
            nn.GELU(),
            nn.Linear(self.hidden_dimension, 1),
        )

    def grouped_scores(self, features, group_indices):
        if features.ndim != 2 or features.shape[-1] != self.dimension:
            raise ValueError("关系判别输入必须为 [batch, dimension]。")
        groups = self._eligible_groups(group_indices)
        if not groups:
            return features.new_empty((0,))
        pooled = torch.stack(
            [features[indices].mean(dim=0) for indices in groups]
        )
        return self.scorer(pooled).squeeze(-1)

    def derange_within_groups(self, features, group_indices, generator=None):
        """只置换同句状态，并保证每个参与位置都不保留原状态。"""
        corrupted = features.clone()
        for indices in self._eligible_groups(group_indices):
            length = len(indices)
            identity = torch.arange(length)
            permutation = None
            for _ in range(64):
                candidate = torch.randperm(length, generator=generator)
                if torch.all(candidate != identity):
                    permutation = candidate
                    break
            if permutation is None:
                permutation = torch.roll(identity, shifts=1)
            device_indices = indices.to(device=features.device)
            corrupted[device_indices] = features[
                device_indices[permutation.to(device=features.device)]
            ]
        return corrupted

    def _eligible_groups(self, group_indices):
        if group_indices is None:
            raise ValueError("关系判别要求提供 group_indices。")
        if group_indices.ndim != 1:
            raise ValueError("group_indices 必须是一维张量。")
        groups = {}
        for row_index, group in enumerate(
            group_indices.detach().cpu().tolist()
        ):
            groups.setdefault(int(group), []).append(row_index)
        return [
            torch.tensor(indices, device=group_indices.device)
            for indices in groups.values()
            if len(indices) >= self.minimum_group_size
        ]


class BrainEmbeddingModel(nn.Module):
    """带可选分组上下文 Transformer 的脑信号编码器。"""

    def __init__(self, channel_positions, subject_count, config):
        super().__init__()
        output_dimension = int(config.get("embedding_dimension", 1024))
        self.brain_encoder = DilatedConvEncoder(
            channel_positions,
            subject_count=subject_count,
            output_dimension=output_dimension,
            config=config.get("conv", config),
        )
        self.use_transformer = bool(config.get("use_transformer", True))
        self.context_mode = str(config.get("context_mode", "grouped"))
        if self.context_mode not in {"grouped", "singleton"}:
            raise ValueError(
                "model.context_mode must be either 'grouped' or 'singleton'."
            )
        self.context_transformer = (
            GroupedTransformerEncoder(
                output_dimension, config.get("transformer", {})
            )
            if self.use_transformer
            else None
        )
        trajectory_config = config.get("pre_transformer_trajectory", {})
        self.pre_transformer_trajectory = None
        if bool(trajectory_config.get("enabled", False)):
            if self.context_transformer is None:
                raise ValueError("Transformer 输入轨迹残差要求启用 Transformer。")
            if self.context_mode != "grouped":
                raise ValueError("Transformer 输入轨迹残差要求 grouped 上下文。")
            # 新支路的初始化不能移动后续训练随机数流；这样同 seed 的
            # Transformer dropout 与原始基线仍是严格配对的。
            with torch.random.fork_rng(devices=[]):
                self.pre_transformer_trajectory = (
                    PreTransformerTrajectoryResidual(
                        output_dimension, trajectory_config
                    )
                )
        relation_config = config.get("relation_auxiliary", {})
        self.relation_discriminator = None
        if bool(relation_config.get("enabled", False)):
            if self.context_transformer is None:
                raise ValueError("关系判别辅助任务要求启用 Transformer。")
            if self.context_mode != "grouped":
                raise ValueError("关系判别辅助任务要求 grouped 上下文。")
            # 新头不能改变同 seed 原始 CNN 与 Transformer 的初始化或随机流。
            with torch.random.fork_rng(devices=[]):
                self.relation_discriminator = RelationDiscriminationHead(
                    output_dimension, relation_config
                )
        self.relation_auxiliary_enabled = self.relation_discriminator is not None
        prefix_revision_config = config.get("prefix_revision", {})
        local_window_revision_config = config.get("local_window_revision", {})
        if bool(prefix_revision_config.get("enabled", False)) and bool(
            local_window_revision_config.get("enabled", False)
        ):
            raise ValueError("完整前缀回修与局部窗口滚动回修不能同时启用。")
        local_window_enabled = bool(
            local_window_revision_config.get("enabled", False)
        )
        revision_config = (
            local_window_revision_config
            if local_window_enabled
            else prefix_revision_config
        )
        raw_history_radius = revision_config.get(
            "history_radius",
            2 if local_window_enabled else None,
        )
        if isinstance(raw_history_radius, str) and raw_history_radius.strip().lower() in {
            "none",
            "null",
            "full",
            "all",
            "full_history",
            "full_prefix",
        }:
            raw_history_radius = None
        self.prefix_revision_history_radius = (
            None if raw_history_radius is None else int(raw_history_radius)
        )
        self.prefix_revision_protocol = (
            "local_window"
            if (local_window_enabled and self.prefix_revision_history_radius is not None)
            else "full_prefix"
        )
        self.prefix_revision_enabled = bool(
            revision_config.get("enabled", False)
        )
        self.prefix_revision_horizons = tuple(
            int(value)
            for value in revision_config.get("horizons", (0, 1, 2))
        )
        self.prefix_revision_validation_query_batch_size = int(
            revision_config.get("validation_query_batch_size", 8)
        )
        self.prefix_revision_separate_horizon_losses = bool(
            revision_config.get(
                "separate_horizon_losses",
                local_window_enabled or self.prefix_revision_protocol == "local_window",
            )
        )
        if self.prefix_revision_enabled:
            if self.relation_auxiliary_enabled:
                raise ValueError("前缀回修不能与关系判别辅助任务同时启用。")
            if self.context_transformer is None or self.context_mode != "grouped":
                raise ValueError("前缀回修要求 grouped 上下文 Transformer。")
            if (
                not self.prefix_revision_horizons
                or self.prefix_revision_horizons[0] != 0
                or tuple(sorted(set(self.prefix_revision_horizons)))
                != self.prefix_revision_horizons
            ):
                raise ValueError("前缀回修 horizons 必须是从 0 开始的严格递增整数。")
            if self.prefix_revision_horizons != (0, 1, 2):
                raise ValueError("首版共享前缀回修固定使用 horizons=[0,1,2]。")
            if self.prefix_revision_validation_query_batch_size <= 0:
                raise ValueError("前缀回修验证 query batch size 必须为正数。")
            if (
                self.prefix_revision_protocol == "local_window"
                and self.prefix_revision_history_radius != 2
            ):
                raise ValueError("首版局部窗口滚动回修固定使用 history_radius=2。")
        self.requires_complete_groups = bool(
            self.relation_auxiliary_enabled or self.prefix_revision_enabled
        )
        self.requires_event_times = bool(
            self.context_transformer is not None
            and (
                self.context_transformer.physical_time_gap_bias is not None
                or self.context_transformer.event_time_interaction_bias
                is not None
            )
        )
        if self.prefix_revision_enabled and self.requires_event_times:
            raise ValueError("首版前缀回修不与物理时间偏置组合。")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_merger_key = prefix + "brain_encoder.channel_merger.heads"
        subject_weights_key = prefix + "brain_encoder.subject_projection.weights"
        if legacy_merger_key in state_dict and subject_weights_key in state_dict:
            legacy_weights = state_dict[subject_weights_key]
            if legacy_weights.ndim != 3 or legacy_weights.shape[-2] != legacy_weights.shape[-1]:
                error_msgs.append(
                    f"{subject_weights_key} cannot be migrated from the legacy "
                    "feature-times-weight projection."
                )
            else:
                state_dict[subject_weights_key] = legacy_weights.transpose(-1, -2).contiguous()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        signals,
        subject_indices=None,
        group_indices=None,
        event_times=None,
        return_brain_embedding=False,
    ):
        brain_embedding = F.normalize(
            self.brain_encoder(signals, subject_indices), dim=-1
        )
        output = brain_embedding
        if self.context_transformer is not None:
            if self.context_mode == "singleton":
                group_indices = torch.arange(
                    len(output), device=output.device, dtype=torch.long
                )
            if self.pre_transformer_trajectory is not None:
                output = self.pre_transformer_trajectory(
                    output, group_indices
                )
            output = self.context_transformer(
                output, group_indices, event_times=event_times
            )
            output = F.normalize(output, dim=-1)
        if return_brain_embedding:
            return output, brain_embedding
        return output

    def forward_with_relation_auxiliary(
        self,
        signals,
        subject_indices,
        group_indices,
        event_times=None,
        corruption_generator=None,
    ):
        """共享一次 CNN 编码，配对比较真实顺序与同句严格乱序。"""
        if self.relation_discriminator is None:
            raise ValueError("模型没有启用关系判别辅助任务。")
        brain_embedding = F.normalize(
            self.brain_encoder(signals, subject_indices), dim=-1
        )
        clean_input = brain_embedding
        if self.pre_transformer_trajectory is not None:
            clean_input = self.pre_transformer_trajectory(
                clean_input, group_indices
            )

        # 保存真实分支进入 Transformer 前的 RNG。乱序分支复用完全相同的
        # dropout 掩码，并在退出后恢复到真实分支结束时的 RNG 状态。
        cpu_rng_state = torch.get_rng_state()
        cuda_device = None
        cuda_rng_state = None
        if clean_input.is_cuda:
            cuda_device = clean_input.device.index
            cuda_rng_state = torch.cuda.get_rng_state(cuda_device)
        clean_context = self.context_transformer(
            clean_input, group_indices, event_times=event_times
        )

        corrupted_input = self.relation_discriminator.derange_within_groups(
            brain_embedding, group_indices, generator=corruption_generator
        )
        if self.pre_transformer_trajectory is not None:
            corrupted_input = self.pre_transformer_trajectory(
                corrupted_input, group_indices
            )
        fork_devices = [] if cuda_device is None else [cuda_device]
        with torch.random.fork_rng(devices=fork_devices):
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, cuda_device)
            corrupted_context = self.context_transformer(
                corrupted_input, group_indices, event_times=event_times
            )

        positive_scores = self.relation_discriminator.grouped_scores(
            clean_context, group_indices
        )
        negative_scores = self.relation_discriminator.grouped_scores(
            corrupted_context, group_indices
        )
        return (
            F.normalize(clean_context, dim=-1),
            positive_scores,
            negative_scores,
        )

    def _ordered_groups(self, group_indices):
        if group_indices is None or group_indices.ndim != 1:
            raise ValueError("前缀回修要求一维 group_indices。")
        groups = {}
        for row_index, group in enumerate(
            group_indices.detach().cpu().tolist()
        ):
            groups.setdefault(int(group), []).append(row_index)
        return list(groups.values())

    def _revision_window_bounds(self, target_position, delay, group_length):
        """返回目标词在指定回修延迟下真实可见窗口的半开边界。"""
        target_position = int(target_position)
        delay = int(delay)
        group_length = int(group_length)
        if not 0 <= target_position < group_length:
            raise ValueError("回修目标位置越界。")
        if delay < 0:
            raise ValueError("回修延迟必须为非负整数。")
        cutoff = min(target_position + delay, group_length - 1)
        start = (
            0
            if self.prefix_revision_history_radius is None
            else max(0, target_position - self.prefix_revision_history_radius)
        )
        return start, cutoff + 1, target_position - start

    def revision_availability_masks(self, group_indices):
        """标记每个目标是否真实拥有各个名义延迟所需的未来词。"""
        if not self.prefix_revision_enabled:
            raise ValueError("模型没有启用共享回修。")
        masks = {
            delay: torch.zeros_like(group_indices, dtype=torch.bool)
            for delay in self.prefix_revision_horizons
        }
        for indices in self._ordered_groups(group_indices):
            for position, row_index in enumerate(indices):
                for delay in self.prefix_revision_horizons:
                    if position + delay < len(indices):
                        masks[delay][row_index] = True
        return masks

    def forward_prefix_revision_training(
        self,
        signals,
        subject_indices,
        group_indices,
        cutoff_generator=None,
        cutoff_policy="sample_one",
    ):
        """按截止位置构造共享模型的延迟 0/1/2 监督。"""
        if not self.prefix_revision_enabled:
            raise ValueError("模型没有启用共享前缀回修训练。")
        if cutoff_policy not in {"sample_one", "all"}:
            raise ValueError("回修截止位置策略必须是 sample_one 或 all。")
        brain_embedding = F.normalize(
            self.brain_encoder(signals, subject_indices), dim=-1
        )
        prefix_parts = []
        prefix_groups = []
        revision_rows = []
        revision_delays = []
        prefix_target_locations = []
        cursor = 0
        virtual_group = 0
        for indices in self._ordered_groups(group_indices):
            cutoffs = (
                range(len(indices))
                if cutoff_policy == "all"
                else (
                    int(
                        torch.randint(
                            0,
                            len(indices),
                            (1,),
                            generator=cutoff_generator,
                        ).item()
                    ),
                )
            )
            for cutoff in cutoffs:
                if self.prefix_revision_protocol == "full_prefix":
                    selected_indices = torch.as_tensor(
                        indices[: cutoff + 1], device=brain_embedding.device
                    )
                    prefix = brain_embedding[selected_indices]
                    prefix_parts.append(prefix)
                    prefix_groups.append(
                        torch.full(
                            (len(prefix),),
                            virtual_group,
                            device=group_indices.device,
                            dtype=group_indices.dtype,
                        )
                    )
                    for delay in self.prefix_revision_horizons:
                        target_position = cutoff - delay
                        if target_position < 0:
                            continue
                        prefix_target_locations.append(cursor + target_position)
                        revision_rows.append(indices[target_position])
                        revision_delays.append(delay)
                    cursor += len(prefix)
                    virtual_group += 1
                    continue

                # 每个延迟视图是独立局部窗口；虚拟组阻止层间交换窗外信息。
                for delay in self.prefix_revision_horizons:
                    target_position = cutoff - delay
                    if target_position < 0:
                        continue
                    start, stop, target_offset = self._revision_window_bounds(
                        target_position, delay, len(indices)
                    )
                    if stop != cutoff + 1:
                        raise RuntimeError("局部窗口滚动回修训练边界不一致。")
                    selected_indices = torch.as_tensor(
                        indices[start:stop], device=brain_embedding.device
                    )
                    window = brain_embedding[selected_indices]
                    prefix_parts.append(window)
                    prefix_groups.append(
                        torch.full(
                            (len(window),),
                            virtual_group,
                            device=group_indices.device,
                            dtype=group_indices.dtype,
                        )
                    )
                    prefix_target_locations.append(cursor + target_offset)
                    revision_rows.append(indices[target_position])
                    revision_delays.append(delay)
                    cursor += len(window)
                    virtual_group += 1

        prefix_features = torch.cat(prefix_parts, dim=0)
        prefix_group_indices = torch.cat(prefix_groups, dim=0)
        revised_context = self.context_transformer(
            prefix_features,
            prefix_group_indices,
            causal_override=False,
        )
        locations = torch.as_tensor(
            prefix_target_locations,
            device=brain_embedding.device,
            dtype=torch.long,
        )
        return (
            F.normalize(revised_context[locations], dim=-1),
            torch.as_tensor(
                revision_rows, device=group_indices.device, dtype=torch.long
            ),
            torch.as_tensor(
                revision_delays, device=group_indices.device, dtype=torch.long
            ),
        )

    @torch.inference_mode()
    def forward_prefix_revisions(
        self,
        signals,
        subject_indices,
        group_indices,
    ):
        """严格截断输入，返回每个目标在延迟 0/1/2 时的预测。"""
        if not self.prefix_revision_enabled:
            raise ValueError("模型没有启用共享前缀回修。")
        brain_embedding = F.normalize(
            self.brain_encoder(signals, subject_indices), dim=-1
        )
        outputs = {}
        groups = self._ordered_groups(group_indices)
        row_to_group = {}
        for group_id, indices in enumerate(groups):
            for position, row_index in enumerate(indices):
                row_to_group[row_index] = (group_id, indices, position)

        if self.prefix_revision_history_radius is None:
            # Stage 2.3 (full_prefix): 同组内相同 window_stop 的不同 (position, delay) 共享完全相同的前缀 [0 : window_stop]
            outputs = {
                delay: torch.empty_like(brain_embedding)
                for delay in self.prefix_revision_horizons
            }
            unique_windows = []
            window_key_to_idx = {}
            query_specs = []
            for delay in self.prefix_revision_horizons:
                for row_index in range(len(brain_embedding)):
                    group_id, indices, position = row_to_group[row_index]
                    window_start, window_stop, target_offset = (
                        self._revision_window_bounds(
                            position, delay, len(indices)
                        )
                    )
                    w_key = (group_id, window_start, window_stop)
                    w_idx = window_key_to_idx.get(w_key)
                    if w_idx is None:
                        w_idx = len(unique_windows)
                        window_key_to_idx[w_key] = w_idx
                        unique_windows.append(indices[window_start:window_stop])
                    query_specs.append((delay, row_index, w_idx, target_offset))

            batch_size = self.prefix_revision_validation_query_batch_size
            window_outputs = [None] * len(unique_windows)
            for start in range(0, len(unique_windows), batch_size):
                stop = min(start + batch_size, len(unique_windows))
                prefix_parts = []
                prefix_groups = []
                lengths = []
                for virtual_group, w_idx in enumerate(range(start, stop)):
                    sel = torch.as_tensor(
                        unique_windows[w_idx], device=brain_embedding.device
                    )
                    prefix = brain_embedding[sel]
                    prefix_parts.append(prefix)
                    lengths.append(len(prefix))
                    prefix_groups.append(
                        torch.full(
                            (len(prefix),),
                            virtual_group,
                            device=group_indices.device,
                            dtype=group_indices.dtype,
                        )
                    )
                revised = F.normalize(
                    self.context_transformer(
                        torch.cat(prefix_parts, dim=0),
                        torch.cat(prefix_groups, dim=0),
                        causal_override=False,
                    ),
                    dim=-1,
                )
                cursor = 0
                for offset_idx, w_len in enumerate(lengths):
                    window_outputs[start + offset_idx] = revised[
                        cursor : cursor + w_len
                    ]
                    cursor += w_len

            for delay, row_index, w_idx, target_offset in query_specs:
                outputs[delay][row_index] = window_outputs[w_idx][target_offset]
            return outputs

        for delay in self.prefix_revision_horizons:
            delayed = torch.empty_like(brain_embedding)
            batch_size = self.prefix_revision_validation_query_batch_size
            for start in range(0, len(brain_embedding), batch_size):
                stop = min(start + batch_size, len(brain_embedding))
                prefix_parts = []
                prefix_groups = []
                target_locations = []
                cursor = 0
                for virtual_group, row_index in enumerate(range(start, stop)):
                    _, indices, position = row_to_group[row_index]
                    window_start, window_stop, target_offset = (
                        self._revision_window_bounds(
                            position, delay, len(indices)
                        )
                    )
                    selected_indices = torch.as_tensor(
                        indices[window_start:window_stop],
                        device=brain_embedding.device,
                    )
                    prefix = brain_embedding[selected_indices]
                    prefix_parts.append(prefix)
                    prefix_groups.append(
                        torch.full(
                            (len(prefix),),
                            virtual_group,
                            device=group_indices.device,
                            dtype=group_indices.dtype,
                        )
                    )
                    target_locations.append(cursor + target_offset)
                    cursor += len(prefix)
                revised = self.context_transformer(
                    torch.cat(prefix_parts, dim=0),
                    torch.cat(prefix_groups, dim=0),
                    causal_override=False,
                )
                delayed[start:stop] = F.normalize(
                    revised[
                        torch.as_tensor(
                            target_locations,
                            device=brain_embedding.device,
                            dtype=torch.long,
                        )
                    ],
                    dim=-1,
                )
            outputs[delay] = delayed
        return outputs


def build_brain_embedding_model(
    channel_count,
    channel_positions,
    subject_count,
    config,
):
    """构建带可选分组上下文的扩张卷积模型。"""
    if len(channel_positions) != int(channel_count):
        raise ValueError("Signal channels and channel_positions do not match.")
    return BrainEmbeddingModel(
        channel_positions=channel_positions,
        subject_count=int(subject_count),
        config=config,
    )
