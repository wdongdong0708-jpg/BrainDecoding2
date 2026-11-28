"""跨任务共享的训练损失。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def directional_multi_positive_loss(logits, positive_mask):
    """允许一个样本在批次内对应多个同文本正样本。"""
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, negative_infinity)
    numerator = torch.logsumexp(positive_logits, dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator - numerator).mean()


def multi_positive_contrastive_loss(eeg_embedding, text_embedding, text_keys, temperature=0.07):
    """计算对称的 EEG—文本多正样本对比损失。"""
    eeg_embedding = F.normalize(eeg_embedding, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)
    logits = eeg_embedding @ text_embedding.T / float(temperature)
    positive_mask = text_keys[:, None].eq(text_keys[None, :])
    eeg_loss = directional_multi_positive_loss(logits, positive_mask)
    text_loss = directional_multi_positive_loss(logits.T, positive_mask.T)
    return 0.5 * (eeg_loss + text_loss)


def siglip_contrastive_loss(
    estimate,
    candidate,
    log_temperature,
    bias,
    identical_candidates_threshold=0.999,
):
    """计算 LibriBrain 基线中可识别重复目标的 SigLIP 目标函数。

    文本向量在目标函数内部归一化。脑信号估计应已由模型完成归一化，
    与源实验的 ``norm_kind='y'`` 约定一致。
    """
    if estimate.ndim != 2 or candidate.ndim != 2:
        raise ValueError("SigLIP 输入必须是二维的 batch×feature 张量。")
    if estimate.shape != candidate.shape:
        raise ValueError(
            f"当前任务要求一一配对的估计与目标，实际为 {estimate.shape} / {candidate.shape}。"
        )
    normalized_candidate = F.normalize(candidate, dim=-1, eps=1e-15)
    scores = estimate @ normalized_candidate.T
    return siglip_score_loss(
        scores,
        candidate,
        log_temperature,
        bias,
        identical_candidates_threshold=identical_candidates_threshold,
    )


def siglip_score_loss(
    scores,
    candidate,
    log_temperature,
    bias,
    identical_candidates_threshold=0.999,
):
    """对候选条件模型直接产生的 batch×candidate 分数应用 SigLIP。"""
    if scores.ndim != 2 or candidate.ndim != 2:
        raise ValueError("SigLIP 分数与候选必须是二维张量。")
    if scores.shape != (len(candidate), len(candidate)):
        raise ValueError(
            "批内候选条件分数必须为 batch×batch，"
            f"实际为 {scores.shape} / candidates={len(candidate)}。"
        )
    normalized_candidate = F.normalize(candidate, dim=-1, eps=1e-15)
    logits = log_temperature.exp() * scores + bias

    if identical_candidates_threshold is None:
        targets = torch.eye(
            len(logits), device=logits.device, dtype=logits.dtype
        )
        weights = None
    else:
        candidate_similarity = normalized_candidate @ normalized_candidate.T
        positives = candidate_similarity >= float(identical_candidates_threshold)
        targets = positives.to(logits.dtype)
        # 重复文本目标属于正样本而非负样本。保留一个对角正样本项，
        # 并屏蔽其余冗余的非对角副本。
        weights = (~positives).to(logits.dtype)
        weights += torch.eye(
            len(logits), device=logits.device, dtype=logits.dtype
        )
    return F.binary_cross_entropy_with_logits(
        logits, targets, weight=weights, reduction="sum"
    ) / max(len(logits), 1)


class SigLipLoss(nn.Module):
    """LibriBrain 词解码使用的可学习温度 SigLIP 损失。"""

    def __init__(
        self,
        initial_temperature=10.0,
        initial_bias=-10.0,
        identical_candidates_threshold=0.999,
    ):
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("SigLIP temperature 必须为正数。")
        self.log_temperature = nn.Parameter(
            torch.tensor(float(initial_temperature)).log()
        )
        self.bias = nn.Parameter(torch.tensor(float(initial_bias)))
        self.identical_candidates_threshold = identical_candidates_threshold

    def forward(self, estimate, candidate):
        return siglip_contrastive_loss(
            estimate,
            candidate,
            self.log_temperature,
            self.bias,
            identical_candidates_threshold=self.identical_candidates_threshold,
        )

    def forward_scores(self, scores, candidate):
        """保留同一温度、全局偏置和重复候选规则来训练直接分数。"""
        return siglip_score_loss(
            scores,
            candidate,
            self.log_temperature,
            self.bias,
            identical_candidates_threshold=self.identical_candidates_threshold,
        )

    def forward_rectangular(self, estimate, candidate_bank, positive_indices):
        """让较少的前缀查询使用原批次全部文本目标作为负样本库。"""
        if estimate.ndim != 2 or candidate_bank.ndim != 2:
            raise ValueError("矩形 SigLIP 输入必须是二维张量。")
        if estimate.shape[-1] != candidate_bank.shape[-1]:
            raise ValueError("矩形 SigLIP 的估计和候选特征维度不一致。")
        positive_indices = torch.as_tensor(
            positive_indices, device=estimate.device, dtype=torch.long
        )
        if positive_indices.ndim != 1 or len(positive_indices) != len(estimate):
            raise ValueError("每个前缀查询必须提供一个原批次正样本索引。")
        if len(estimate) == 0:
            return estimate.sum() * 0.0
        if torch.any(positive_indices < 0) or torch.any(
            positive_indices >= len(candidate_bank)
        ):
            raise ValueError("矩形 SigLIP 正样本索引越界。")

        normalized_bank = F.normalize(candidate_bank, dim=-1, eps=1e-15)
        scores = estimate @ normalized_bank.T
        logits = self.log_temperature.exp() * scores + self.bias
        targets = torch.zeros_like(logits)
        targets.scatter_(1, positive_indices[:, None], 1.0)
        weights = torch.ones_like(logits)
        if self.identical_candidates_threshold is not None:
            positive_candidates = normalized_bank[positive_indices]
            duplicates = (
                positive_candidates @ normalized_bank.T
            ) >= float(self.identical_candidates_threshold)
            weights = (~duplicates).to(logits.dtype)
            weights.scatter_(1, positive_indices[:, None], 1.0)
        return F.binary_cross_entropy_with_logits(
            logits, targets, weight=weights, reduction="sum"
        ) / len(estimate)


class LogitAdjustedFullVocabCELoss(nn.Module):
    """基于训练集全词表的 Logit-Adjusted 交叉熵损失（零均匀标签平滑，杜绝高频/低频偏置）。

    计算：
        logits_v = clamp(exp(log_t), max=max_temperature) * (y_hat · e_v) + alpha_prior * log(pi_v)
    其中 ``pi_v`` 与 ``e_v`` 严格来自训练划分，且以非持久化缓冲区保存（不膨胀检查点体积）。
    默认 ``label_smoothing = 0.0``，保证纯噪声输入下最优解严格对应常数余弦得分，零词频捷径。
    """

    def __init__(
        self,
        initial_temperature: float = 12.0,
        max_temperature: float | None = 18.0,
        alpha_prior: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if float(initial_temperature) <= 0.0:
            raise ValueError("LogitAdjustedFullVocabCELoss initial_temperature 必须为正数。")
        if max_temperature is not None and float(max_temperature) <= 0.0:
            raise ValueError("LogitAdjustedFullVocabCELoss max_temperature 必须为正数。")
        if not (0.0 <= float(label_smoothing) < 1.0):
            raise ValueError("label_smoothing 必须位于 [0, 1) 之间。")
        self.log_temperature = nn.Parameter(
            torch.tensor(float(initial_temperature), dtype=torch.float32).log()
        )
        self.max_log_temperature = (
            None if max_temperature is None else math.log(float(max_temperature))
        )
        self.alpha_prior = float(alpha_prior)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer(
            "vocab_embeddings", torch.empty(0, 0, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "log_prior", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self.word_to_id: dict[str, int] = {}
        self.requires_word_ids: bool = True

    @torch.no_grad()
    def set_vocabulary(
        self,
        words,
        vocab_embeddings: torch.Tensor,
        word_counts,
    ) -> None:
        """严格绑定训练集词表嵌入 [V, D] 与训练集词频先验 log(pi_v)。"""
        word_list = [str(w) for w in words]
        emb = torch.as_tensor(vocab_embeddings, dtype=torch.float32)
        if emb.ndim != 2 or emb.shape[0] != len(word_list) or emb.shape[0] == 0:
            raise ValueError(
                f"vocab_embeddings 形状无效：期望 [{len(word_list)}, D]，收到 {tuple(emb.shape)}"
            )
        counts = torch.as_tensor(word_counts, dtype=torch.float64)
        if counts.ndim != 1 or counts.shape[0] != len(word_list):
            raise ValueError("word_counts 长度必须与 words 一致。")
        counts = counts.clamp(min=1.0)
        probs = counts / counts.sum()
        log_pi = torch.log(probs).to(dtype=torch.float32)

        dev = self.log_temperature.device
        self.vocab_embeddings = F.normalize(emb, dim=-1, eps=1e-12).to(
            device=dev, dtype=torch.float32
        ).contiguous()
        self.log_prior = log_pi.to(device=dev, dtype=torch.float32).contiguous()
        self.word_to_id = {w: idx for idx, w in enumerate(word_list)}

    def map_words_to_ids(self, words, device: torch.device) -> torch.Tensor:
        if not self.word_to_id:
            raise RuntimeError("LogitAdjustedFullVocabCELoss 尚未通过 set_vocabulary 绑定训练集词表。")
        return torch.tensor(
            [self.word_to_id[str(w)] for w in words],
            device=device,
            dtype=torch.long,
        )

    def _effective_temperature(self) -> torch.Tensor:
        log_t = self.log_temperature
        if self.max_log_temperature is not None:
            log_t = log_t.clamp(max=self.max_log_temperature)
        return log_t.exp()

    def _compute_ce(
        self,
        estimate: torch.Tensor,
        target_word_ids: torch.Tensor,
    ) -> torch.Tensor:
        if estimate.ndim != 2:
            raise ValueError("LogitAdjustedFullVocabCELoss 输入估计必须是二维张量。")
        if estimate.shape[0] == 0:
            return estimate.sum() * 0.0
        if self.vocab_embeddings.numel() == 0:
            raise RuntimeError(
                "LogitAdjustedFullVocabCELoss 尚未通过 set_vocabulary 绑定训练集词表。"
            )
        est_f = estimate.float()
        vocab_e = self.vocab_embeddings.to(device=est_f.device, dtype=torch.float32)
        if est_f.shape[-1] != vocab_e.shape[-1]:
            raise ValueError(
                f"估计维度 ({est_f.shape[-1]}) 与词表向量维度 ({vocab_e.shape[-1]}) 不一致。"
            )
        temp = self._effective_temperature().to(device=est_f.device, dtype=torch.float32)
        logits = temp * torch.matmul(est_f, vocab_e.t())
        if self.alpha_prior != 0.0:
            log_pi = self.log_prior.to(device=est_f.device, dtype=torch.float32)
            logits = logits + float(self.alpha_prior) * log_pi.unsqueeze(0)
        targets = torch.as_tensor(
            target_word_ids, device=est_f.device, dtype=torch.long
        )
        if targets.ndim != 1 or targets.shape[0] != est_f.shape[0]:
            raise ValueError("target_word_ids 必须与 estimate 批大小一一对应。")
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

    def forward(self, estimate, candidate, target_word_ids=None):
        del candidate
        if target_word_ids is None:
            raise ValueError("LogitAdjustedFullVocabCELoss.forward 需要提供 target_word_ids。")
        return self._compute_ce(estimate, target_word_ids)

    def forward_rectangular(
        self,
        estimate,
        candidate_bank,
        positive_indices,
        target_word_ids=None,
    ):
        del candidate_bank, positive_indices
        if target_word_ids is None:
            raise ValueError(
                "LogitAdjustedFullVocabCELoss.forward_rectangular 需要提供 target_word_ids。"
            )
        return self._compute_ce(estimate, target_word_ids)


class HybridSigLipFullVocabCELoss(nn.Module):
    """混合损失：siglip_weight * SigLIP + ce_weight * LogitAdjustedFullVocabCE。

    同时兼顾批内 SigLIP 语义锥几何稳定性与全词表 18k 精细判别力。
    """

    def __init__(
        self,
        *,
        siglip_weight: float = 0.5,
        ce_weight: float = 0.5,
        initial_temperature: float = 10.0,
        initial_bias: float = -10.0,
        identical_candidates_threshold: float | None = 0.999,
        ce_initial_temperature: float = 12.0,
        ce_max_temperature: float | None = 18.0,
        alpha_prior: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.siglip_weight = float(siglip_weight)
        self.ce_weight = float(ce_weight)
        self.siglip = SigLipLoss(
            initial_temperature=initial_temperature,
            initial_bias=initial_bias,
            identical_candidates_threshold=identical_candidates_threshold,
        )
        self.full_vocab_ce = LogitAdjustedFullVocabCELoss(
            initial_temperature=ce_initial_temperature,
            max_temperature=ce_max_temperature,
            alpha_prior=alpha_prior,
            label_smoothing=label_smoothing,
        )
        self.requires_word_ids: bool = True

    @property
    def word_to_id(self) -> dict[str, int]:
        return self.full_vocab_ce.word_to_id

    @torch.no_grad()
    def set_vocabulary(self, words, vocab_embeddings, word_counts) -> None:
        self.full_vocab_ce.set_vocabulary(words, vocab_embeddings, word_counts)

    def map_words_to_ids(self, words, device: torch.device) -> torch.Tensor:
        return self.full_vocab_ce.map_words_to_ids(words, device)

    def forward(self, estimate, candidate, target_word_ids=None):
        loss_sig = self.siglip(estimate, candidate)
        loss_ce = self.full_vocab_ce(
            estimate, candidate, target_word_ids=target_word_ids
        )
        return self.siglip_weight * loss_sig + self.ce_weight * loss_ce

    def forward_rectangular(
        self,
        estimate,
        candidate_bank,
        positive_indices,
        target_word_ids=None,
    ):
        loss_sig = self.siglip.forward_rectangular(
            estimate, candidate_bank, positive_indices
        )
        loss_ce = self.full_vocab_ce.forward_rectangular(
            estimate,
            candidate_bank,
            positive_indices,
            target_word_ids=target_word_ids,
        )
        return self.siglip_weight * loss_sig + self.ce_weight * loss_ce


class MultiTrialPrototypeSigLipLoss(nn.Module):
    """第三阶梯 Stage-1 抗坍缩多试次原型对比损失（测试期仍严格单试次 k=1，零 model state_dict 改动）。

    组合三项：
      1. ``single_weight * SigLIP(z_norm, targets)``：保住出口 ``z_out`` 在 k=1 单试次下的最优检索；
      2. ``proto_pre_weight * SigLIP_rect(proto(proto_pre_head(h_pre)), E_U)``：将同词多试次（>=10s 零重叠）
         平均降噪原型梯度送入 ``h_pre`` 与 18M 卷积主干，打破单试次噪声引发的维度坍缩；
      3. ``proto_z_weight * SigLIP_rect(proto(z_norm), E_U)``：温和撑大 ``z_out`` 自身有效信号秩，
         使下游 Stage-2 跨词邻居 ``z_{i±Δ}`` 同步受益。

    辅助线性头 ``proto_pre_head`` 与原型温度/偏置仅存于 ``loss_state``，不污染 ``model.state_dict()``，
    保证产出的 Stage-1 检查点可 ``strict=True`` 零摩擦热启动任意 Stage-2 模型。
    """

    def __init__(
        self,
        *,
        single_weight: float = 1.0,
        proto_pre_weight: float = 0.5,
        proto_z_weight: float = 0.25,
        min_group_size: int = 2,
        pre_head_dimension: int = 1024,
        output_dimension: int = 1024,
        initial_temperature: float = 10.0,
        initial_bias: float = -10.0,
        proto_initial_temperature: float = 12.0,
        proto_initial_bias: float = -10.0,
        identical_candidates_threshold: float | None = 0.999,
    ):
        super().__init__()
        self.single_weight = float(single_weight)
        self.proto_pre_weight = float(proto_pre_weight)
        self.proto_z_weight = float(proto_z_weight)
        self.min_group_size = max(2, int(min_group_size))
        self.pre_head_dimension = int(pre_head_dimension)
        self.output_dimension = int(output_dimension)
        if self.single_weight < 0.0 or self.proto_pre_weight < 0.0 or self.proto_z_weight < 0.0:
            raise ValueError("MultiTrialPrototypeSigLipLoss 各项损失权重不能为负。")
        if self.pre_head_dimension <= 0 or self.output_dimension <= 0:
            raise ValueError("pre_head_dimension 与 output_dimension 必须为正整数。")

        self.siglip = SigLipLoss(
            initial_temperature=initial_temperature,
            initial_bias=initial_bias,
            identical_candidates_threshold=identical_candidates_threshold,
        )
        self.proto_pre_head: nn.Linear | None = (
            nn.Linear(self.pre_head_dimension, self.output_dimension, bias=True)
            if self.proto_pre_weight > 0.0
            else None
        )
        self.proto_pre_siglip: SigLipLoss | None = (
            SigLipLoss(
                initial_temperature=proto_initial_temperature,
                initial_bias=proto_initial_bias,
                identical_candidates_threshold=identical_candidates_threshold,
            )
            if self.proto_pre_weight > 0.0
            else None
        )
        self.proto_z_siglip: SigLipLoss | None = (
            SigLipLoss(
                initial_temperature=proto_initial_temperature,
                initial_bias=proto_initial_bias,
                identical_candidates_threshold=identical_candidates_threshold,
            )
            if self.proto_z_weight > 0.0
            else None
        )
        self.word_to_id: dict[str, int] = {}
        self.requires_word_ids: bool = True
        self.requires_pre_head: bool = bool(self.proto_pre_weight > 0.0)
        self.last_components: dict[str, float] = {}

    @torch.no_grad()
    def set_vocabulary(self, words, vocab_embeddings=None, word_counts=None) -> None:
        """绑定训练集词表映射（零大张量缓冲区，不膨胀检查点体积，绝不触碰 val/test）。"""
        del vocab_embeddings, word_counts
        self.word_to_id = {str(w): idx for idx, w in enumerate(words)}

    def map_words_to_ids(self, words, device: torch.device) -> torch.Tensor:
        if not self.word_to_id:
            # 独立单元测试未调 set_vocabulary 时按批内确定性词表回退映射
            local_map: dict[str, int] = {}
            ids = []
            for w in words:
                key = str(w)
                wid = local_map.get(key)
                if wid is None:
                    wid = len(local_map)
                    local_map[key] = wid
                ids.append(wid)
            return torch.tensor(ids, device=device, dtype=torch.long)
        return torch.tensor(
            [self.word_to_id[str(w)] for w in words],
            device=device,
            dtype=torch.long,
        )

    def _pool_unique_word_prototypes(
        self,
        unit_embeddings: torch.Tensor,
        inverse_idx: torch.Tensor,
        counts: torch.Tensor,
        proto_indices: torch.Tensor,
        num_unique: int,
    ) -> torch.Tensor:
        """按批内唯一词 ID 聚合单位球面试次均值并重归一化（等价于训练期 SimpleB2T 原型平均）。"""
        emb_f = unit_embeddings.float()
        summed = emb_f.new_zeros((num_unique, emb_f.shape[-1]))
        summed.index_add_(0, inverse_idx, emb_f)
        denom = counts[proto_indices].to(dtype=torch.float32).clamp(min=1.0).unsqueeze(-1)
        proto_mean = summed[proto_indices] / denom
        return F.normalize(proto_mean, dim=-1, eps=1e-12)

    def forward(
        self,
        estimate: torch.Tensor,
        candidate: torch.Tensor,
        target_word_ids: torch.Tensor | None = None,
        pre_head: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss_single = (
            self.siglip(estimate, candidate)
            if self.single_weight > 0.0
            else estimate.sum() * 0.0
        )
        loss_proto_pre = loss_single * 0.0
        loss_proto_z = loss_single * 0.0
        active_prototypes = 0
        mean_group_size_t = loss_single.new_zeros(())

        need_proto = (self.proto_pre_weight > 0.0) or (self.proto_z_weight > 0.0)
        if need_proto and target_word_ids is not None and len(estimate) >= self.min_group_size:
            word_ids = torch.as_tensor(
                target_word_ids, device=estimate.device, dtype=torch.long
            )
            unique_ids, inverse_idx, counts = torch.unique(
                word_ids, sorted=False, return_inverse=True, return_counts=True
            )
            num_unique = int(unique_ids.shape[0])
            proto_indices = torch.nonzero(
                counts >= self.min_group_size, as_tuple=False
            ).squeeze(-1)
            active_prototypes = int(proto_indices.numel())
            if active_prototypes > 0:
                mean_group_size_t = counts[proto_indices].float().mean().detach()
                cand_norm = F.normalize(candidate.float(), dim=-1, eps=1e-12)
                target_sum = cand_norm.new_zeros((num_unique, cand_norm.shape[-1]))
                target_sum.index_add_(0, inverse_idx, cand_norm)
                unique_targets = F.normalize(target_sum, dim=-1, eps=1e-12)

                if self.proto_pre_weight > 0.0:
                    if pre_head is None:
                        raise ValueError(
                            "MultiTrialPrototypeSigLipLoss 启用了 proto_pre_weight > 0，"
                            "但 forward 未收到 pre_head 特征。"
                        )
                    assert self.proto_pre_head is not None and self.proto_pre_siglip is not None
                    u_pre = F.normalize(
                        self.proto_pre_head(pre_head.to(dtype=self.proto_pre_head.weight.dtype)),
                        dim=-1,
                        eps=1e-12,
                    )
                    proto_pre = self._pool_unique_word_prototypes(
                        u_pre, inverse_idx, counts, proto_indices, num_unique
                    )
                    loss_proto_pre = self.proto_pre_siglip.forward_rectangular(
                        proto_pre, unique_targets, proto_indices
                    )

                if self.proto_z_weight > 0.0:
                    assert self.proto_z_siglip is not None
                    u_z = F.normalize(estimate.float(), dim=-1, eps=1e-12)
                    proto_z = self._pool_unique_word_prototypes(
                        u_z, inverse_idx, counts, proto_indices, num_unique
                    )
                    loss_proto_z = self.proto_z_siglip.forward_rectangular(
                        proto_z, unique_targets, proto_indices
                    )

        total = (
            self.single_weight * loss_single
            + self.proto_pre_weight * loss_proto_pre
            + self.proto_z_weight * loss_proto_z
        )
        self.last_components = {
            "single_loss": loss_single.detach(),
            "proto_pre_loss": loss_proto_pre.detach(),
            "proto_z_loss": loss_proto_z.detach(),
            "active_prototypes": float(active_prototypes),
            "mean_group_size": mean_group_size_t,
        }
        return total

    def forward_rectangular(
        self,
        estimate,
        candidate_bank,
        positive_indices,
        target_word_ids=None,
    ):
        del target_word_ids
        return self.siglip.forward_rectangular(
            estimate, candidate_bank, positive_indices
        )


def build_siglip_loss(config=None):
    """根据配置构建共享词解码损失（默认保持 LibriBrain SigLipLoss）。"""
    cfg = dict(config or {})
    name = str(cfg.get("name", "siglip") or "siglip").strip().lower()
    if name in {"", "siglip"}:
        return SigLipLoss(
            initial_temperature=float(cfg.get("initial_temperature", 10.0)),
            initial_bias=float(cfg.get("initial_bias", -10.0)),
            identical_candidates_threshold=cfg.get(
                "identical_candidates_threshold", 0.999
            ),
        )
    if name in {"logit_adjusted_full_vocab_ce", "la_full_vocab_ce"}:
        raw_max_t = cfg.get("max_temperature", 18.0)
        return LogitAdjustedFullVocabCELoss(
            initial_temperature=float(cfg.get("initial_temperature", 12.0)),
            max_temperature=None if raw_max_t is None else float(raw_max_t),
            alpha_prior=float(cfg.get("alpha_prior", 1.0)),
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        )
    if name in {"hybrid_siglip_full_vocab_ce", "hybrid_siglip_ce"}:
        raw_max_t = cfg.get("ce_max_temperature", cfg.get("max_temperature", 18.0))
        return HybridSigLipFullVocabCELoss(
            siglip_weight=float(cfg.get("siglip_weight", 0.5)),
            ce_weight=float(cfg.get("ce_weight", 0.5)),
            initial_temperature=float(cfg.get("initial_temperature", 10.0)),
            initial_bias=float(cfg.get("initial_bias", -10.0)),
            identical_candidates_threshold=cfg.get(
                "identical_candidates_threshold", 0.999
            ),
            ce_initial_temperature=float(
                cfg.get("ce_initial_temperature", 12.0)
            ),
            ce_max_temperature=None if raw_max_t is None else float(raw_max_t),
            alpha_prior=float(cfg.get("alpha_prior", 1.0)),
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        )
    if name in {"multi_trial_prototype_siglip", "prototype_siglip", "anticollapse_prototype_siglip"}:
        return MultiTrialPrototypeSigLipLoss(
            single_weight=float(cfg.get("single_weight", 1.0)),
            proto_pre_weight=float(cfg.get("proto_pre_weight", 0.5)),
            proto_z_weight=float(cfg.get("proto_z_weight", 0.25)),
            min_group_size=int(cfg.get("min_group_size", 2)),
            pre_head_dimension=int(cfg.get("pre_head_dimension", 1024)),
            output_dimension=int(cfg.get("output_dimension", 1024)),
            initial_temperature=float(cfg.get("initial_temperature", 10.0)),
            initial_bias=float(cfg.get("initial_bias", -10.0)),
            proto_initial_temperature=float(
                cfg.get("proto_initial_temperature", 12.0)
            ),
            proto_initial_bias=float(cfg.get("proto_initial_bias", -10.0)),
            identical_candidates_threshold=cfg.get(
                "identical_candidates_threshold", 0.999
            ),
        )
    raise ValueError(
        f"不支持的 loss.name={name!r}，可选：'siglip' | "
        "'logit_adjusted_full_vocab_ce' | 'hybrid_siglip_full_vocab_ce' | "
        "'multi_trial_prototype_siglip'。"
    )
