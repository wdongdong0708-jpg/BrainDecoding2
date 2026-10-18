"""跨任务共享的训练损失。"""

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


def build_siglip_loss(config=None):
    """根据普通配置构建共享的 LibriBrain SigLIP 损失。"""
    config = config or {}
    return SigLipLoss(
        initial_temperature=float(config.get("initial_temperature", 10.0)),
        initial_bias=float(config.get("initial_bias", -10.0)),
        identical_candidates_threshold=config.get(
            "identical_candidates_threshold", 0.999
        ),
    )
