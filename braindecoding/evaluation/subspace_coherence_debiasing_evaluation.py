"""方案一（零重训 · 测试期即插即用）：低秩共模子空间投影 + 相干度门控去偏（Subspace-Projected + Coherence-Gated LOO）探索性评估模块。

本模块完全独立于主干训练与主评估流水线，用于在不重新训练模型的前提下，
利用测试期 M=6 个纯历史因果时间错位窗口（-15s, -30s, -45s, -60s, -90s, -120s）提取跨窗口共有的
低秩时间结构偏置子空间 P_bias = U_K U_K^T，并结合跨历史窗口相干度门控执行留一法（LOO）共模去偏：

1. 全样本错位共模均值（完全不接触 y_clean 与测试标签）：
   y_shift_mean_all = (1 / M) * sum_{m=1..M} y_shifts[:, m, :]
2. 低秩时间偏置子空间 SVD 提取（秩 K in {8, 12, 16, 24}）：
   _, _, Vh = svd(y_shift_mean_all - mean(y_shift_mean_all, dim=0))
   U_K = Vh[:K, :].T  # [D, K]
3. 跨历史窗口相干度门控（SimpleB2T 跨观测相消对偶）：
   coherence_i = ||shift_mean_i||_2
   gate_i = clamp((coherence_i - 0.35) / (0.65 - 0.35), min=0.2, max=1.2)
4. Clean 与 Shifted LOO 低秩投影扣除：
   - Clean (0s)   : 仅在低秩子空间 U_K 内扣除 (eta * gate_clean) * (y_shift_mean_all @ U_K @ U_K^T)
   - Shifted (dk) : 仅使用其余 M-1 个错位条件均值在 U_K 内的投影扣除，100% 保护正交的 (D - K) 个语义维度。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
import gc
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from braindecoding.config import (
    PROJECT_ROOT,
    load_yaml_with_extends,
    project_path,
    resolve_config_path,
)
from braindecoding.evaluation.common_mode_subtraction_evaluation import (
    parse_eta_values,
)


DEFAULT_SUBSPACE_ETAS: tuple[float, ...] = (0.8, 1.0, 1.2)
DEFAULT_FULL_LOO_ETAS: tuple[float, ...] = (0.5, 0.8, 1.0)
DEFAULT_SUBSPACE_RANKS: tuple[int, ...] = (8, 12, 16, 24)
DEFAULT_CALIBRATION_BETAS: tuple[float, ...] = (1.0,)
DEFAULT_CALIBRATION_TAUS: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 1.0)
DEFAULT_HISTORICAL_SHIFTS: tuple[float, ...] = (
    -15.0,
    -30.0,
    -45.0,
    -60.0,
    -90.0,
    -120.0,
)

# 轻量级内存缓存：加速同一 (y_clean, y_shifts, vocab_embeds, rank_k, eta) 下跨多组 (beta, tau) 的毫秒级扫描
_RAW_SCORE_CALIBRATION_CACHE: dict[tuple, dict] = {}


def parse_rank_k_values(
    raw_ranks: str | Iterable[int | str] | None,
    *,
    default: Iterable[int] = DEFAULT_SUBSPACE_RANKS,
) -> list[int]:
    """解析低秩子空间维度 rank_k 列表（正整数升序去重）。"""
    if raw_ranks is None:
        tokens: list[int | str] = list(default)
    elif isinstance(raw_ranks, int):
        tokens = [int(raw_ranks)]
    elif isinstance(raw_ranks, str):
        text = raw_ranks.strip()
        if not text:
            tokens = list(default)
        else:
            tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
    else:
        tokens = []
        for item in raw_ranks:
            if isinstance(item, str):
                tokens.extend(t.strip() for t in item.split(",") if t.strip())
            else:
                tokens.append(item)

    parsed: list[int] = []
    seen: set[int] = set()
    for tok in tokens:
        val = int(tok)
        if val <= 0:
            raise ValueError(f"子空间秩 rank_k 必须为正整数，收到：{val}")
        if val not in seen:
            seen.add(val)
            parsed.append(val)
    parsed.sort()
    return parsed if parsed else sorted(int(x) for x in default)


def parse_float_grid_values(
    raw_values: float | int | str | Iterable[float | int | str] | None,
    *,
    default: Iterable[float],
) -> list[float]:
    """解析非负浮点数网格列表（保留 0.0，升序去重）。"""
    if raw_values is None:
        tokens: list[float | int | str] = list(default)
    elif isinstance(raw_values, (int, float)):
        tokens = [float(raw_values)]
    elif isinstance(raw_values, str):
        text = raw_values.strip()
        if not text:
            tokens = list(default)
        else:
            tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
    else:
        tokens = []
        for item in raw_values:
            if isinstance(item, str):
                tokens.extend(t.strip() for t in item.split(",") if t.strip())
            else:
                tokens.append(item)

    parsed: list[float] = []
    seen: set[float] = set()
    for tok in tokens:
        val = round(float(tok), 6)
        if val < 0.0:
            raise ValueError(f"校准网格参数必须 >= 0.0，收到：{val}")
        if val not in seen:
            seen.add(val)
            parsed.append(val)
    parsed.sort()
    return parsed if parsed else sorted(round(float(x), 6) for x in default)


def subspace_coherence_loo_debiasing(
    y_clean: torch.Tensor,
    y_shifts: torch.Tensor,
    eta: float = 1.0,
    rank_k: int | None = 12,
    use_coherence_gate: bool = True,
    *,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    center_svd: bool = False,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
    precomputed_vh: torch.Tensor | None = None,
    precomputed_vh_loo: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """仅在跨历史窗口共有的低秩时间偏置子空间内执行 LOO 共模扣除，
    100% 保护正交的 (D - rank_k) 个纯神经语义维度免受 1 + eta^2/M 噪声膨胀。

    参数
    ----
    y_clean:
        [N, D]，L2 归一化的 Clean 预测特征（或 [N, V] 相似度得分矩阵）。
    y_shifts:
        [N, M, D]，M=6 个历史错位条件的 L2 归一化特征（或 [N, M, V] 得分矩阵）。
    eta:
        在低秩偏置子空间内执行的扣除强度（推荐 0.8 ~ 1.2）。
    rank_k:
        时间结构偏置的低秩子空间维度（推荐在 {8, 12, 16, 24} 中网格搜索；若为 None 则退化为全维扣除）。
    use_coherence_gate:
        是否启用跨历史窗口相干度门控（SimpleB2T Eq.7 相消对偶）。
    center_svd:
        是否对 y_shift_mean 做样本中心化后再求 SVD（默认 False：未中心化 SVD 使第 1 个奇异向量自动包含全局静态高频靶心 mu_0）。
    """
    if y_clean.ndim != 2:
        raise ValueError(f"y_clean 必须为 2 维张量 [N, D]，收到形状：{tuple(y_clean.shape)}")
    if y_shifts.ndim != 3:
        raise ValueError(f"y_shifts 必须为 3 维张量 [N, M, D]，收到形状：{tuple(y_shifts.shape)}")

    n_samples, num_shifts, dim = y_shifts.shape
    if y_clean.shape != (n_samples, dim):
        raise ValueError(
            f"y_clean 形状 {tuple(y_clean.shape)} 与 y_shifts 形状 {tuple(y_shifts.shape)} 不匹配。"
        )
    if num_shifts < 2:
        raise ValueError(
            f"留一法子空间去偏至少需要 M >= 2 个错位条件，当前 M={num_shifts}。"
        )

    clean_f = y_clean.detach().to(dtype=torch.float32)
    shifts_f = y_shifts.detach().to(dtype=torch.float32)
    if normalize_inputs:
        clean_f = F.normalize(clean_f, p=2.0, dim=-1, eps=1e-12)
        shifts_f = F.normalize(shifts_f, p=2.0, dim=-1, eps=1e-12)

    eta_val = float(eta)
    if abs(eta_val) <= 1e-12:
        return clean_f, shifts_f

    # 1. 计算全样本的错位共模均值特征 [N, D] (完全不接触 y_clean 和任何测试标签)
    y_shift_mean_all = shifts_f.mean(dim=1)  # [N, D]

    # 2. 对错位共模均值做 SVD，提取前 rank_k 个主奇异向量 U_k [D, rank_k]
    # 当 center_svd=False 时，直接对未中心化的 y_shift_mean_all 做 SVD，使第 1 个奇异向量自动包含全局高频靶心 mu_0
    u_k: torch.Tensor | None = None
    eff_k: int | None = None
    if rank_k is not None and int(rank_k) < dim:
        eff_k = max(1, min(int(rank_k), n_samples, dim))
        if precomputed_vh is not None:
            vh = precomputed_vh
        else:
            svd_in = (
                y_shift_mean_all - y_shift_mean_all.mean(dim=0, keepdim=True)
                if center_svd
                else y_shift_mean_all
            )
            _, _, vh = torch.linalg.svd(svd_in, full_matrices=False)
        u_k = vh[:eff_k, :].t().contiguous()  # [D, eff_k]

    # 3. 跨历史窗口相干度门控 (SimpleB2T Eq.7 的相消对偶):
    denom_gate = max(float(coherence_high) - float(coherence_low), 1e-6)

    def get_adaptive_weight(shift_mean_vec: torch.Tensor) -> torch.Tensor | float:
        if not use_coherence_gate:
            return 1.0
        coherence = shift_mean_vec.norm(p=2, dim=-1, keepdim=True)  # [N, 1]
        gate = ((coherence - float(coherence_low)) / denom_gate).clamp(
            min=float(gate_min), max=float(gate_max)
        )
        return gate

    def project_to_subspace(
        vec: torch.Tensor, basis_u: torch.Tensor | None
    ) -> torch.Tensor:
        if basis_u is None:
            return vec
        return torch.matmul(torch.matmul(vec, basis_u), basis_u.t())

    # 4. Clean 评估: 仅减去 y_shift_mean 在低秩子空间 U_k 上的投影
    gate_clean = get_adaptive_weight(y_shift_mean_all)
    proj_clean = project_to_subspace(y_shift_mean_all, u_k)  # [N, D]
    y_clean_debiased = clean_f - (eta_val * gate_clean) * proj_clean
    if normalize_outputs:
        y_clean_debiased = F.normalize(y_clean_debiased, p=2.0, dim=-1, eps=1e-12)

    # 5. Shifted LOO 公平评估: 第 k 个错位条件仅使用其余 M-1=5 个错位条件的低秩投影进行扣除
    sum_all_shifts = shifts_f.sum(dim=1)  # [N, D]
    loo_denom = float(num_shifts - 1)
    y_shifts_debiased_list: list[torch.Tensor] = []
    for k in range(num_shifts):
        y_shift_loo_mean = (sum_all_shifts - shifts_f[:, k, :]) / loo_denom  # [N, D]
        if eff_k is not None and precomputed_vh_loo is not None and k < len(precomputed_vh_loo):
            u_k_loo = precomputed_vh_loo[k][:eff_k, :].t().contiguous()
        elif eff_k is not None and precomputed_vh is None:
            svd_loo_in = (
                y_shift_loo_mean - y_shift_loo_mean.mean(dim=0, keepdim=True)
                if center_svd
                else y_shift_loo_mean
            )
            _, _, vh_loo_k = torch.linalg.svd(svd_loo_in, full_matrices=False)
            u_k_loo = vh_loo_k[:eff_k, :].t().contiguous()
        else:
            u_k_loo = u_k
        gate_k = get_adaptive_weight(y_shift_loo_mean)
        proj_k = project_to_subspace(y_shift_loo_mean, u_k_loo)  # [N, D]
        y_k_debiased = shifts_f[:, k, :] - (eta_val * gate_k) * proj_k
        if normalize_outputs:
            y_k_debiased = F.normalize(y_k_debiased, p=2.0, dim=-1, eps=1e-12)
        y_shifts_debiased_list.append(y_k_debiased)

    y_shifts_debiased = torch.stack(y_shifts_debiased_list, dim=1)  # [N, M, D]
    return y_clean_debiased, y_shifts_debiased


def subspace_loo_with_column_calibration(
    y_clean: torch.Tensor,
    y_shifts: torch.Tensor,
    vocab_embeds: torch.Tensor,
    rank_k: int | None = 8,
    eta: float = 0.80,
    beta: float = 1.0,
    tau: float = 0.55,
    *,
    use_coherence_gate: bool = False,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    center_svd: bool = False,
    normalize_inputs: bool = True,
    precomputed_vh: torch.Tensor | None = None,
    precomputed_vh_loo: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """低秩共模子空间 LOO 去偏 + 候选词列向错位先验校准（Column-Wise Null Standardization / Z-Norm Anti-Hubness）。

    严格遵循 LOO 因果协议：
    1. Step 1 (未中心化 Subspace LOO 特征去偏)：
       - Clean (0s) 使用全部 M=6 个历史错位窗口均值做未中心化 SVD（第 1 个奇异向量自动包含全局静态均值 mu_0）；
       - Shifted (dk) 严格仅使用其余 M-1=5 个错位窗口均值做 LOO SVD 并扣除投影。
    2. Step 2 (原始余弦相似度得分)：
       - scores_clean  = y_clean_deb  @ vocab_embeds.T   [N, V]
       - scores_shifts = y_shifts_deb @ vocab_embeds.T   [N, M, V]
    3. Step 3 (候选词列向错位先验校准，消除局部窗口高频词方差膨胀 sigma_{v<=50} >> sigma_{v>50} 诱导的 Hubness 垄断)：
       - Clean (0s)  : S_tilde_{i,v} = (S_{i,v} - beta * mu_v^{shift}) / (sigma_v^{shift})^tau
                       其中 mu_v^{shift}, sigma_v^{shift} 由全部 M=6 个错位分支统计得出；
       - Shifted (dk): S_tilde_{i,v}^{(k)} = (S_{i,v}^{(k)} - beta * mu_{v,LOO}^{shift}) / (sigma_{v,LOO}^{shift})^tau
                       其中 mu_{v,LOO}^{shift}, sigma_{v,LOO}^{shift} 严格仅使用其余 M-1=5 个错位分支统计得出！
    """
    n_samples, num_shifts, dim = y_shifts.shape
    vocab_f = vocab_embeds.detach().to(device=y_clean.device, dtype=torch.float32)
    if normalize_inputs:
        vocab_f = F.normalize(vocab_f, p=2.0, dim=-1, eps=1e-12)

    cache_key = (
        int(y_clean.data_ptr()),
        tuple(y_clean.shape),
        int(y_shifts.data_ptr()),
        tuple(y_shifts.shape),
        int(vocab_embeds.data_ptr()),
        tuple(vocab_embeds.shape),
        int(rank_k) if rank_k is not None else -1,
        round(float(eta), 6),
        bool(use_coherence_gate),
        round(float(coherence_low), 4),
        round(float(coherence_high), 4),
        round(float(gate_min), 4),
        round(float(gate_max), 4),
        bool(center_svd),
        bool(normalize_inputs),
    )

    cached = _RAW_SCORE_CALIBRATION_CACHE.get(cache_key)
    if cached is None:
        # --- Step 1: 原版 Subspace LOO 特征去偏 (默认未中心化 SVD 以同步切除全局静态均值 mu_0，且 Shifted 采用严格 LOO SVD) ---
        y_clean_deb, y_shifts_deb = subspace_coherence_loo_debiasing(
            y_clean,
            y_shifts,
            eta=eta,
            rank_k=rank_k,
            use_coherence_gate=use_coherence_gate,
            coherence_low=coherence_low,
            coherence_high=coherence_high,
            gate_min=gate_min,
            gate_max=gate_max,
            center_svd=center_svd,
            normalize_inputs=normalize_inputs,
            normalize_outputs=True,
            precomputed_vh=precomputed_vh,
            precomputed_vh_loo=precomputed_vh_loo,
        )

        # --- Step 2: 计算词库原始余弦相似度得分 ---
        scores_clean = torch.matmul(y_clean_deb, vocab_f.t())    # [N, V]
        scores_shifts = torch.matmul(y_shifts_deb, vocab_f.t())  # [N, M, V]

        # 预计算 Clean 所需的全 M=6 错位列向均值/标准差与 Shifted LOO 所需的 M-1 错位列向均值/标准差
        mu_v_all = scores_shifts.mean(dim=(0, 1), keepdim=True).squeeze(0)  # [1, V]
        std_v_all = (
            scores_shifts.std(dim=(0, 1), keepdim=True)
            .squeeze(0)
            .clamp(min=1e-4)
        )  # [1, V]

        mu_v_loo_list: list[torch.Tensor] = []
        std_v_loo_list: list[torch.Tensor] = []
        for k in range(num_shifts):
            loo_idx = [m for m in range(num_shifts) if m != k]
            s_loo = scores_shifts[:, loo_idx, :]  # [N, M-1, V]
            mu_v_loo = s_loo.mean(dim=(0, 1), keepdim=True).squeeze(0)  # [1, V]
            std_v_loo = (
                s_loo.std(dim=(0, 1), keepdim=True)
                .squeeze(0)
                .clamp(min=1e-4)
            )  # [1, V]
            mu_v_loo_list.append(mu_v_loo)
            std_v_loo_list.append(std_v_loo)

        if len(_RAW_SCORE_CALIBRATION_CACHE) >= 16:
            _RAW_SCORE_CALIBRATION_CACHE.pop(next(iter(_RAW_SCORE_CALIBRATION_CACHE)))
        cached = {
            "scores_clean": scores_clean,
            "scores_shifts": scores_shifts,
            "mu_v_all": mu_v_all,
            "std_v_all": std_v_all,
            "mu_v_loo_list": mu_v_loo_list,
            "std_v_loo_list": std_v_loo_list,
        }
        _RAW_SCORE_CALIBRATION_CACHE[cache_key] = cached

    scores_clean = cached["scores_clean"]
    scores_shifts = cached["scores_shifts"]
    beta_val = float(beta)
    tau_val = float(tau)
    if abs(beta_val) <= 1e-12 and abs(tau_val) <= 1e-12:
        return scores_clean, scores_shifts

    # --- Step 3: 候选词列向错位先验校准 (Column-Wise Null Standardization, 严格 LOO) ---
    mu_v_all = cached["mu_v_all"]
    std_v_all = cached["std_v_all"]
    denom_all = std_v_all.pow(tau_val) if abs(tau_val) > 1e-12 else 1.0
    scores_clean_cal = (scores_clean - beta_val * mu_v_all) / denom_all

    scores_shifts_cal_list: list[torch.Tensor] = []
    for k in range(num_shifts):
        mu_v_loo = cached["mu_v_loo_list"][k]
        std_v_loo = cached["std_v_loo_list"][k]
        denom_loo = std_v_loo.pow(tau_val) if abs(tau_val) > 1e-12 else 1.0
        s_k_cal = (scores_shifts[:, k, :] - beta_val * mu_v_loo) / denom_loo
        scores_shifts_cal_list.append(s_k_cal)
    scores_shifts_cal = torch.stack(scores_shifts_cal_list, dim=1)  # [N, M, V]

    return scores_clean_cal, scores_shifts_cal


def analyze_shift_subspace_and_coherence(
    y_shifts: torch.Tensor,
    *,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    center_svd: bool = False,
    normalize_inputs: bool = True,
    compute_loo_svd: bool = False,
) -> tuple[dict, torch.Tensor]:
    """对错位特征矩阵 [N, M, D] 计算 SVD 奇异值能量谱与跨历史窗口相干度门控统计量。

    返回 ``(统计信息字典, Vh 右奇异向量矩阵)``，其中 ``Vh`` 可直接复用于同一视野下的多组 (rank_k, eta) 扫描。
    当 ``compute_loo_svd=True`` 时，还会在 ``diagnostics["_vh_loo_list"]`` 中预存 M 个严格留一法 SVD 基。
    """
    shifts_f = y_shifts.detach().to(dtype=torch.float32)
    if normalize_inputs:
        shifts_f = F.normalize(shifts_f, p=2.0, dim=-1, eps=1e-12)

    n_samples, num_shifts, dim = shifts_f.shape
    y_shift_mean_all = shifts_f.mean(dim=1)  # [N, D]
    global_mean_vec = y_shift_mean_all.mean(dim=0, keepdim=True)
    global_static_mean_norm = float(global_mean_vec.norm(p=2).item())

    svd_in = y_shift_mean_all - global_mean_vec if center_svd else y_shift_mean_all
    _, singular_values, vh = torch.linalg.svd(svd_in, full_matrices=False)

    vh_loo_list: list[torch.Tensor] = []
    if compute_loo_svd and num_shifts >= 2:
        sum_all = shifts_f.sum(dim=1)
        loo_denom = float(num_shifts - 1)
        for k in range(num_shifts):
            y_loo = (sum_all - shifts_f[:, k, :]) / loo_denom
            svd_loo_in = y_loo - y_loo.mean(dim=0, keepdim=True) if center_svd else y_loo
            _, _, vh_k = torch.linalg.svd(svd_loo_in, full_matrices=False)
            vh_loo_list.append(vh_k)

    energy = singular_values.square()
    total_energy = float(energy.sum().item())
    cum_energy = torch.cumsum(energy, dim=0) / max(total_energy, 1e-12)

    energy_by_rank: dict[str, float] = {}
    for r in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128):
        if r <= len(cum_energy):
            energy_by_rank[f"top_{r}"] = float(cum_energy[r - 1].item())

    sum_sq = float(energy.sum().item())
    sum_quad = float(energy.square().sum().item())
    participation_ratio = (
        (sum_sq * sum_sq) / max(sum_quad, 1e-12) if sum_quad > 0 else 0.0
    )

    coherence = y_shift_mean_all.norm(p=2, dim=-1)  # [N]
    denom_gate = max(float(coherence_high) - float(coherence_low), 1e-6)
    gate = ((coherence - float(coherence_low)) / denom_gate).clamp(
        min=float(gate_min), max=float(gate_max)
    )
    coh_np = coherence.cpu().numpy()
    gate_np = gate.cpu().numpy()
    random_floor = float(1.0 / np.sqrt(max(1, num_shifts)))

    diagnostics: dict = {
        "num_samples": int(n_samples),
        "num_shifts_M": int(num_shifts),
        "feature_dim_D": int(dim),
        "svd_centered": bool(center_svd),
        "global_static_mean_norm": round(global_static_mean_norm, 4),
        "theoretical_random_orthogonal_norm": round(random_floor, 4),
        "effective_rank_participation_ratio": round(participation_ratio, 2),
        "cumulative_energy_explained": energy_by_rank,
        "top_16_singular_values": [
            round(float(x), 4) for x in singular_values[:16].cpu().tolist()
        ],
        "coherence_norm": {
            "mean": float(np.mean(coh_np)),
            "std": float(np.std(coh_np)),
            "median": float(np.median(coh_np)),
            "p10": float(np.percentile(coh_np, 10)),
            "p90": float(np.percentile(coh_np, 90)),
            "min": float(np.min(coh_np)),
            "max": float(np.max(coh_np)),
        },
        "adaptive_gate": {
            "mean": float(np.mean(gate_np)),
            "std": float(np.std(gate_np)),
            "median": float(np.median(gate_np)),
            "fraction_at_min_clamp": float(
                np.mean(gate_np <= float(gate_min) + 1e-6)
            ),
            "fraction_at_max_clamp": float(
                np.mean(gate_np >= float(gate_max) - 1e-6)
            ),
        },
    }
    if vh_loo_list:
        diagnostics["_vh_loo_list"] = vh_loo_list
    return diagnostics, vh


def apply_subspace_coherence_debiasing_to_encoded(
    clean_encoded: dict,
    shifted_encoded_map: dict[str, dict],
    *,
    eta: float = 1.0,
    rank_k: int | None = 12,
    use_coherence_gate: bool = True,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    center_svd: bool = False,
    normalize_inputs: bool = True,
    precomputed_vh_by_key: dict[str, torch.Tensor] | None = None,
    precomputed_vh_loo_by_key: dict[str, list[torch.Tensor]] | None = None,
) -> tuple[dict, dict[str, dict]]:
    """对 encode_loader 输出的完整字典（含 Stage 2.2 的 K0/K1/K2 修订视野预测）执行低秩子空间 + 相干度门控 LOO 去偏。"""
    shift_keys = list(shifted_encoded_map.keys())
    if len(shift_keys) < 2:
        raise ValueError(
            f"子空间 LOO 去偏至少需要 2 个错位条件，当前仅有 {len(shift_keys)} 个。"
        )

    new_clean = dict(clean_encoded)
    new_shifted: dict[str, dict] = {
        k: dict(shifted_encoded_map[k]) for k in shift_keys
    }

    has_revision = "revision_predictions" in clean_encoded and isinstance(
        clean_encoded["revision_predictions"], dict
    )
    if has_revision:
        clean_rev = dict(clean_encoded["revision_predictions"])
        new_clean_rev: dict = {}
        new_shifted_rev: dict[str, dict] = {k: {} for k in shift_keys}

        for delay, clean_delay_tensor in sorted(clean_rev.items()):
            y_shifts_delay = torch.stack(
                [
                    shifted_encoded_map[k]["revision_predictions"][delay]
                    for k in shift_keys
                ],
                dim=1,
            )
            vh_delay = (
                precomputed_vh_by_key.get(f"K{delay}")
                if precomputed_vh_by_key is not None
                else None
            )
            vh_loo_delay = (
                precomputed_vh_loo_by_key.get(f"K{delay}")
                if precomputed_vh_loo_by_key is not None
                else None
            )
            deb_clean_d, deb_shifts_d = subspace_coherence_loo_debiasing(
                clean_delay_tensor,
                y_shifts_delay,
                eta=eta,
                rank_k=rank_k,
                use_coherence_gate=use_coherence_gate,
                coherence_low=coherence_low,
                coherence_high=coherence_high,
                gate_min=gate_min,
                gate_max=gate_max,
                center_svd=center_svd,
                normalize_inputs=normalize_inputs,
                precomputed_vh=vh_delay,
                precomputed_vh_loo=vh_loo_delay,
            )
            new_clean_rev[delay] = deb_clean_d
            for idx_k, k in enumerate(shift_keys):
                new_shifted_rev[k][delay] = deb_shifts_d[:, idx_k, :]

        max_delay = max(new_clean_rev)
        new_clean["revision_predictions"] = new_clean_rev
        new_clean["predictions"] = new_clean_rev[max_delay]
        for k in shift_keys:
            new_shifted[k]["revision_predictions"] = new_shifted_rev[k]
            new_shifted[k]["predictions"] = new_shifted_rev[k][max_delay]
        return new_clean, new_shifted

    y_clean_main = clean_encoded["predictions"]
    y_shifts_main = torch.stack(
        [shifted_encoded_map[k]["predictions"] for k in shift_keys], dim=1
    )
    vh_main = (
        precomputed_vh_by_key.get("main")
        if precomputed_vh_by_key is not None
        else None
    )
    vh_loo_main = (
        precomputed_vh_loo_by_key.get("main")
        if precomputed_vh_loo_by_key is not None
        else None
    )
    deb_clean_main, deb_shifts_main = subspace_coherence_loo_debiasing(
        y_clean_main,
        y_shifts_main,
        eta=eta,
        rank_k=rank_k,
        use_coherence_gate=use_coherence_gate,
        coherence_low=coherence_low,
        coherence_high=coherence_high,
        gate_min=gate_min,
        gate_max=gate_max,
        center_svd=center_svd,
        normalize_inputs=normalize_inputs,
        precomputed_vh=vh_main,
        precomputed_vh_loo=vh_loo_main,
    )
    new_clean["predictions"] = deb_clean_main
    for idx_k, k in enumerate(shift_keys):
        new_shifted[k]["predictions"] = deb_shifts_main[:, idx_k, :]
    return new_clean, new_shifted


def apply_column_calibrated_subspace_loo_to_encoded(
    clean_encoded: dict,
    shifted_encoded_map: dict[str, dict],
    *,
    full_bank_tensor: torch.Tensor,
    vocab50_bank_tensor: torch.Tensor,
    rank_k: int | None = 8,
    eta: float = 0.80,
    beta: float = 1.0,
    tau: float = 0.55,
    use_coherence_gate: bool = False,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    center_svd: bool = False,
    normalize_inputs: bool = True,
    precomputed_vh_by_key: dict[str, torch.Tensor] | None = None,
    precomputed_vh_loo_by_key: dict[str, list[torch.Tensor]] | None = None,
    calibrate_vocab50: bool = True,
) -> tuple[dict, dict[str, dict]]:
    """对 encode_loader 输出的完整字典（含 K0/K1/K2 视野）执行 Subspace LOO + 候选词列向错位先验校准（Z-Norm Anti-Hubness），
    将校准后的全词库得分矩阵 [N, V] 与 50 词得分矩阵 [N, 50] 挂载至返回字典供快速排名评估。
    """
    shift_keys = list(shifted_encoded_map.keys())
    if len(shift_keys) < 2:
        raise ValueError(
            f"列向错位校准至少需要 2 个错位条件，当前仅有 {len(shift_keys)} 个。"
        )

    new_clean = dict(clean_encoded)
    new_shifted: dict[str, dict] = {
        k: dict(shifted_encoded_map[k]) for k in shift_keys
    }

    has_revision = "revision_predictions" in clean_encoded and isinstance(
        clean_encoded["revision_predictions"], dict
    )
    eff_beta_50 = float(beta) if calibrate_vocab50 else 0.0
    eff_tau_50 = float(tau) if calibrate_vocab50 else 0.0

    if has_revision:
        clean_rev = dict(clean_encoded["revision_predictions"])
        clean_scores_full_by_delay: dict[int, torch.Tensor] = {}
        clean_scores_50_by_delay: dict[int, torch.Tensor] = {}
        shifted_scores_full_by_key: dict[str, dict[int, torch.Tensor]] = {
            k: {} for k in shift_keys
        }
        shifted_scores_50_by_key: dict[str, dict[int, torch.Tensor]] = {
            k: {} for k in shift_keys
        }

        for delay, clean_delay_tensor in sorted(clean_rev.items()):
            y_shifts_delay = torch.stack(
                [
                    shifted_encoded_map[k]["revision_predictions"][delay]
                    for k in shift_keys
                ],
                dim=1,
            )
            vh_delay = (
                precomputed_vh_by_key.get(f"K{delay}")
                if precomputed_vh_by_key is not None
                else None
            )
            vh_loo_delay = (
                precomputed_vh_loo_by_key.get(f"K{delay}")
                if precomputed_vh_loo_by_key is not None
                else None
            )
            sc_clean_full, sc_shifts_full = subspace_loo_with_column_calibration(
                clean_delay_tensor,
                y_shifts_delay,
                full_bank_tensor,
                rank_k=rank_k,
                eta=eta,
                beta=beta,
                tau=tau,
                use_coherence_gate=use_coherence_gate,
                coherence_low=coherence_low,
                coherence_high=coherence_high,
                gate_min=gate_min,
                gate_max=gate_max,
                center_svd=center_svd,
                normalize_inputs=normalize_inputs,
                precomputed_vh=vh_delay,
                precomputed_vh_loo=vh_loo_delay,
            )
            sc_clean_50, sc_shifts_50 = subspace_loo_with_column_calibration(
                clean_delay_tensor,
                y_shifts_delay,
                vocab50_bank_tensor,
                rank_k=rank_k,
                eta=eta,
                beta=eff_beta_50,
                tau=eff_tau_50,
                use_coherence_gate=use_coherence_gate,
                coherence_low=coherence_low,
                coherence_high=coherence_high,
                gate_min=gate_min,
                gate_max=gate_max,
                center_svd=center_svd,
                normalize_inputs=normalize_inputs,
                precomputed_vh=vh_delay,
                precomputed_vh_loo=vh_loo_delay,
            )
            clean_scores_full_by_delay[delay] = sc_clean_full
            clean_scores_50_by_delay[delay] = sc_clean_50
            for idx_k, k in enumerate(shift_keys):
                shifted_scores_full_by_key[k][delay] = sc_shifts_full[:, idx_k, :]
                shifted_scores_50_by_key[k][delay] = sc_shifts_50[:, idx_k, :]

        new_clean["calibrated_scores_full"] = clean_scores_full_by_delay
        new_clean["calibrated_scores_50"] = clean_scores_50_by_delay
        for k in shift_keys:
            new_shifted[k]["calibrated_scores_full"] = shifted_scores_full_by_key[k]
            new_shifted[k]["calibrated_scores_50"] = shifted_scores_50_by_key[k]
        return new_clean, new_shifted

    y_clean_main = clean_encoded["predictions"]
    y_shifts_main = torch.stack(
        [shifted_encoded_map[k]["predictions"] for k in shift_keys], dim=1
    )
    vh_main = (
        precomputed_vh_by_key.get("main")
        if precomputed_vh_by_key is not None
        else None
    )
    vh_loo_main = (
        precomputed_vh_loo_by_key.get("main")
        if precomputed_vh_loo_by_key is not None
        else None
    )
    sc_clean_full, sc_shifts_full = subspace_loo_with_column_calibration(
        y_clean_main,
        y_shifts_main,
        full_bank_tensor,
        rank_k=rank_k,
        eta=eta,
        beta=beta,
        tau=tau,
        use_coherence_gate=use_coherence_gate,
        coherence_low=coherence_low,
        coherence_high=coherence_high,
        gate_min=gate_min,
        gate_max=gate_max,
        center_svd=center_svd,
        normalize_inputs=normalize_inputs,
        precomputed_vh=vh_main,
        precomputed_vh_loo=vh_loo_main,
    )
    sc_clean_50, sc_shifts_50 = subspace_loo_with_column_calibration(
        y_clean_main,
        y_shifts_main,
        vocab50_bank_tensor,
        rank_k=rank_k,
        eta=eta,
        beta=eff_beta_50,
        tau=eff_tau_50,
        use_coherence_gate=use_coherence_gate,
        coherence_low=coherence_low,
        coherence_high=coherence_high,
        gate_min=gate_min,
        gate_max=gate_max,
        center_svd=center_svd,
        normalize_inputs=normalize_inputs,
        precomputed_vh=vh_main,
        precomputed_vh_loo=vh_loo_main,
    )
    new_clean["calibrated_scores_full"] = {0: sc_clean_full}
    new_clean["calibrated_scores_50"] = {0: sc_clean_50}
    for idx_k, k in enumerate(shift_keys):
        new_shifted[k]["calibrated_scores_full"] = {0: sc_shifts_full[:, idx_k, :]}
        new_shifted[k]["calibrated_scores_50"] = {0: sc_shifts_50[:, idx_k, :]}
    return new_clean, new_shifted


def analyze_lexical_hubness_diagnostics(
    y_clean: torch.Tensor,
    y_shifts: torch.Tensor,
    vocab_bank_tensor: torch.Tensor,
    words: list[str],
    vocab_list: list[str],
    vocab50_list: list[str],
    *,
    rank_k: int | None = 8,
    eta: float = 0.80,
    beta: float = 1.0,
    tau_values: Iterable[float] = DEFAULT_CALIBRATION_TAUS,
    use_coherence_gate: bool = False,
    center_svd: bool = False,
    precomputed_vh: torch.Tensor | None = None,
    precomputed_vh_loo: list[torch.Tensor] | None = None,
) -> dict:
    """量化诊断齐普夫高频词中心度垄断效应（Zipf Hubness / Anisotropy）：
    1. 统计测试集查询词落入 Top-50 高频词的真实比例 P(w_i in Top-50)；
    2. 对比校准前 Top-50 高频词与其余长尾词在错位背景下的列向标准差 sigma_{v<=50} vs sigma_{v>50} 及列向均值 mu_{v<=50} vs mu_{v>50}；
    3. 统计不同 tau 下，错位分支 Top-10 预测席位被 Top-50 高频词占据的比例（理论均匀分布应为 50 / |V| = 2.48%）。
    """
    vocab_set50 = set(str(w) for w in vocab50_list)
    words_str = [str(w) for w in words]
    test_top50_share = (
        float(np.mean([w in vocab_set50 for w in words_str]))
        if words_str
        else 0.0
    )
    is_top50_col = torch.tensor(
        [str(w) in vocab_set50 for w in vocab_list],
        dtype=torch.bool,
        device=y_clean.device,
    )
    uniform_top50_col_share = float(is_top50_col.float().mean().item()) if len(vocab_list) > 0 else 0.0

    # 先获取 tau=0, beta=0 下的原始去偏余弦得分 [N, M, V]
    _, raw_scores_shifts = subspace_loo_with_column_calibration(
        y_clean,
        y_shifts,
        vocab_bank_tensor,
        rank_k=rank_k,
        eta=eta,
        beta=0.0,
        tau=0.0,
        use_coherence_gate=use_coherence_gate,
        center_svd=center_svd,
        precomputed_vh=precomputed_vh,
        precomputed_vh_loo=precomputed_vh_loo,
    )
    mu_v = raw_scores_shifts.mean(dim=(0, 1))   # [V]
    std_v = raw_scores_shifts.std(dim=(0, 1))   # [V]

    has_top50 = bool(is_top50_col.any().item())
    has_tail = bool((~is_top50_col).any().item())
    std_top50 = float(std_v[is_top50_col].mean().item()) if has_top50 else 0.0
    std_tail = float(std_v[~is_top50_col].mean().item()) if has_tail else 0.0
    mu_top50 = float(mu_v[is_top50_col].mean().item()) if has_top50 else 0.0
    mu_tail = float(mu_v[~is_top50_col].mean().item()) if has_tail else 0.0

    top_k_slots = min(10, len(vocab_list))
    tau_slot_occupancy: dict[str, dict] = {}
    for tau in tau_values:
        _, cal_shifts = subspace_loo_with_column_calibration(
            y_clean,
            y_shifts,
            vocab_bank_tensor,
            rank_k=rank_k,
            eta=eta,
            beta=beta,
            tau=float(tau),
            use_coherence_gate=use_coherence_gate,
            center_svd=center_svd,
            precomputed_vh=precomputed_vh,
            precomputed_vh_loo=precomputed_vh_loo,
        )
        # 取每个错位样本的前 10 名候选词索引 [N, M, 10]
        top10_indices = torch.topk(cal_shifts, k=top_k_slots, dim=-1).indices
        top10_in_50 = is_top50_col[top10_indices].float().mean().item()
        expected_collision = test_top50_share * top10_in_50 * (10.0 / max(1, int(is_top50_col.sum().item())))
        tau_slot_occupancy[f"{float(tau):g}"] = {
            "tau": float(tau),
            "beta": float(beta),
            "shifted_top10_slot_share_by_top50": round(float(top10_in_50), 4),
            "top50_occupancy_in_shifted_top10": round(float(top10_in_50), 4),
            "predicted_zipf_collision_floor": round(float(expected_collision), 4),
        }

    # 同样记录完全未校准 (beta=0, tau=0) 时的 Top-10 席位垄断率
    raw_top10_idx = torch.topk(raw_scores_shifts, k=top_k_slots, dim=-1).indices
    raw_top10_in_50 = float(is_top50_col[raw_top10_idx].float().mean().item())
    n_queries = len(words_str)
    n_top50_queries = int(sum(w in vocab_set50 for w in words_str))

    return {
        "anchor_rank_k": int(rank_k) if rank_k is not None else None,
        "anchor_eta": float(eta),
        "total_queries": n_queries,
        "vocab_size": len(vocab_list),
        "top50_query_count": n_top50_queries,
        "top50_query_share": round(test_top50_share, 4),
        "test_query_top50_share": round(test_top50_share, 4),
        "uniform_vocab_top50_share": round(uniform_top50_col_share, 4),
        "theoretical_uniform_top50_slot_share": round(uniform_top50_col_share, 4),
        "uncalibrated_std_top50_mean": round(std_top50, 5),
        "uncalibrated_std_tail_mean": round(std_tail, 5),
        "top50_col_std_mean": round(std_top50, 5),
        "tail_col_std_mean": round(std_tail, 5),
        "std_ratio_top50_over_tail": round(std_top50 / max(std_tail, 1e-8), 3),
        "std_ratio_top50_vs_tail": round(std_top50 / max(std_tail, 1e-8), 3),
        "uncalibrated_mu_top50_mean": round(mu_top50, 5),
        "uncalibrated_mu_tail_mean": round(mu_tail, 5),
        "top50_abs_mean_bias": round(abs(mu_top50), 5),
        "tail_abs_mean_bias": round(abs(mu_tail), 5),
        "expected_random_within_top50_hub_pool": round(test_top50_share * (10.0 / 60.0), 4),
        "uncalibrated_shifted_top10_slot_share_by_top50": round(raw_top10_in_50, 4),
        "by_tau": tau_slot_occupancy,
        "shifted_top10_hubness_by_tau": tau_slot_occupancy,
    }


def _fast_ranks_from_score_tensor(
    scores_sub: torch.Tensor, target_idx: torch.Tensor
) -> np.ndarray:
    """在 PyTorch 张量上直接按并列取中位名次计算排名（1 + greater + 0.5 * (equal - 1)），避免 118MB 得分矩阵频繁转 NumPy 与字典查找。"""
    if scores_sub.numel() == 0 or target_idx.numel() == 0:
        return np.empty(0, dtype=np.float64)
    row_idx = torch.arange(scores_sub.shape[0], device=scores_sub.device)
    target_scores = scores_sub[row_idx, target_idx].unsqueeze(1)
    greater = (scores_sub > target_scores).sum(dim=1).to(dtype=torch.float64)
    equal = (scores_sub == target_scores).sum(dim=1).to(dtype=torch.float64)
    ranks = 1.0 + greater + 0.5 * (equal - 1.0)
    return ranks.detach().cpu().numpy()


def _build_fast_summary_evaluator(
    eval_ds,
    vocabulary: list[str],
    vocab50: list[str],
    *,
    stage: str,
) -> Callable[[dict, str, str, float | None, dict | None], dict]:
    """构建零冗余的快速检索指标提取器，同时支持特征向量余弦检索与列向校准得分矩阵（calibrated_scores_full / calibrated_scores_50）直接排名。"""
    from braindecoding.evaluation.retrieval_metrics import (
        fixed_vocabulary_retrieval_metrics,
        retrieval_ranks,
        summarize_retrieval,
    )
    from braindecoding.training.workflow_runner import (
        _extract_control_summary_row,
        _score_full_vocabulary_population,
    )

    vocab_list = [str(w) for w in vocabulary]
    vocab50_list = [str(w) for w in vocab50]
    full_cand_count = max(1, len(vocab_list))
    full_bank = (
        np.stack(
            [np.asarray(eval_ds.embedding_map[w], dtype=np.float32) for w in vocab_list]
        )
        if vocab_list
        else np.zeros((0, 1), dtype=np.float32)
    )
    vocab50_bank = (
        np.stack(
            [np.asarray(eval_ds.embedding_map[w], dtype=np.float32) for w in vocab50_list]
        )
        if vocab50_list
        else np.zeros((0, 1), dtype=np.float32)
    )
    vocab_index_map = {w: idx for idx, w in enumerate(vocab_list)}
    vocab50_index_map = {w: idx for idx, w in enumerate(vocab50_list)}

    subject_map = dict(
        zip(
            eval_ds.table["event_id"].astype(str),
            eval_ds.table["subject_id"].astype(str),
        )
    )
    fallback_subjects = eval_ds.table["subject_id"].astype(str).to_numpy()

    def evaluate_encoded_row(
        encoded: dict,
        key: str,
        label: str,
        shift_seconds: float | None,
        clean_summary: dict | None = None,
    ) -> dict:
        revision = stage == "revision" and "revision_predictions" in encoded
        cal_full_map = encoded.get("calibrated_scores_full")
        cal_50_map = encoded.get("calibrated_scores_50")

        if revision:
            max_delay = max(encoded["revision_predictions"])
            primary_preds = encoded["revision_predictions"][max_delay]
            avail_mask = encoded["revision_availability_masks"][max_delay].bool()
        else:
            max_delay = 0
            primary_preds = encoded["predictions"]
            avail_mask = torch.ones(len(encoded["words"]), dtype=torch.bool)

        strict_encoded = (
            encoded
            if revision
            else {
                **encoded,
                "revision_predictions": {0: encoded["predictions"]},
                "revision_availability_masks": {
                    0: torch.ones(len(encoded["words"]), dtype=torch.bool)
                },
            }
        )
        words_arr = np.asarray([str(w) for w in strict_encoded["words"]], dtype=object)
        eligible50 = np.isin(words_arr, vocab50_list)
        covered_full = np.isin(words_arr, vocab_list)
        target_idx_50 = torch.tensor(
            [vocab50_index_map[str(w)] for w in words_arr[eligible50]],
            dtype=torch.long,
        )
        target_idx_full = torch.tensor(
            [vocab_index_map[str(w)] for w in words_arr[covered_full]],
            dtype=torch.long,
        )

        # Primary N50 block
        if isinstance(cal_50_map, dict) and max_delay in cal_50_map:
            avail_np = avail_mask.detach().cpu().numpy().astype(bool)
            el50_avail = eligible50 & avail_np
            t_idx_50_avail = torch.tensor(
                [vocab50_index_map[str(w)] for w in words_arr[el50_avail]],
                dtype=torch.long,
                device=cal_50_map[max_delay].device,
            )
            r50_primary = _fast_ranks_from_score_tensor(
                cal_50_map[max_delay][el50_avail], t_idx_50_avail
            )
            m50 = summarize_retrieval(
                r50_primary,
                words_arr[el50_avail],
                len(vocab50_list),
                top_ks=(1, 5, 10),
            )
        else:
            eval_preds = primary_preds[avail_mask]
            eval_targets = encoded["targets"][avail_mask]
            eval_words = [
                w for w, keep in zip(encoded["words"], avail_mask.tolist()) if keep
            ]
            m50 = fixed_vocabulary_retrieval_metrics(
                eval_preds,
                eval_targets,
                eval_words,
                vocab50_list,
                top_ks=(1, 5, 10),
                vocabulary_name="N50",
            )
        n50_block = {
            "macro_top1": float(m50.get("macro_recall_at_1", 0.0)),
            "macro_top5": float(m50.get("macro_recall_at_5", 0.0)),
            "macro_top10": float(m50.get("macro_recall_at_10", 0.0)),
            "micro_top1": float(m50.get("micro_recall_at_1", 0.0)),
            "micro_top5": float(m50.get("micro_recall_at_5", 0.0)),
            "micro_top10": float(m50.get("micro_recall_at_10", 0.0)),
            "mrr": float(m50.get("mean_reciprocal_rank", 0.0)),
            "median_rank": float(m50.get("median_rank", 25.0)),
            "chance_top10": float(
                min(10.0, float(len(vocab50_list))) / max(1, len(vocab50_list))
            ),
        }

        subjects_arr = np.asarray(
            [
                subject_map.get(
                    str(eid),
                    str(fallback_subjects[idx])
                    if idx < len(fallback_subjects)
                    else "unknown",
                )
                for idx, eid in enumerate(strict_encoded["event_ids"])
            ],
            dtype=object,
        )
        rev_preds = strict_encoded["revision_predictions"]
        rev_masks = strict_encoded["revision_availability_masks"]
        common_mask = np.logical_and.reduce(
            [m.detach().cpu().numpy().astype(bool) for m in rev_masks.values()]
        )

        revision_horizons: dict[str, dict] = {}
        full_block_for_primary: dict = {}
        primary_delay = max(rev_preds) if revision else 0

        for delay in sorted(rev_preds):
            pred_tensor = rev_preds[delay]
            d_avail = rev_masks[delay].detach().cpu().numpy().astype(bool)
            all_mask = d_avail if d_avail.any() else np.ones(len(words_arr), dtype=bool)
            com_mask = common_mask if common_mask.any() else all_mask

            # 1) Strict 50-word bank metrics for this horizon
            if isinstance(cal_50_map, dict) and delay in cal_50_map:
                sc50_d = cal_50_map[delay]
                ranks50 = _fast_ranks_from_score_tensor(
                    sc50_d[eligible50], target_idx_50.to(sc50_d.device)
                )
            else:
                ranks50 = retrieval_ranks(
                    pred_tensor[eligible50],
                    words_arr[eligible50],
                    vocab50_bank,
                    vocab50_list,
                )
            com_el50 = com_mask[eligible50]
            s50_common = summarize_retrieval(
                ranks50[com_el50],
                words_arr[eligible50][com_el50],
                len(vocab50_list),
                top_ks=(1, 10),
            )
            s50_all = summarize_retrieval(
                ranks50,
                words_arr[eligible50],
                len(vocab50_list),
                top_ks=(1, 10),
            )

            # 2) Full-vocabulary metrics for this horizon
            ranks_full = np.full(len(words_arr), np.inf, dtype=np.float64)
            if covered_full.any() and len(vocab_list) > 0:
                if isinstance(cal_full_map, dict) and delay in cal_full_map:
                    scf_d = cal_full_map[delay]
                    ranks_full[covered_full] = _fast_ranks_from_score_tensor(
                        scf_d[covered_full], target_idx_full.to(scf_d.device)
                    )
                else:
                    ranks_full[covered_full] = retrieval_ranks(
                        pred_tensor[covered_full],
                        words_arr[covered_full],
                        full_bank,
                        vocab_list,
                    )
            fv_all = _score_full_vocabulary_population(
                ranks_full[all_mask],
                words_arr[all_mask],
                subjects_arr[all_mask],
                covered_full[all_mask],
                full_cand_count,
            )
            fv_common = _score_full_vocabulary_population(
                ranks_full[com_mask],
                words_arr[com_mask],
                subjects_arr[com_mask],
                covered_full[com_mask],
                full_cand_count,
            )
            if int(delay) == int(primary_delay):
                full_block_for_primary = fv_common if revision else fv_all

            if revision:
                revision_horizons[f"K{delay}"] = {
                    "delay": int(delay),
                    "label": f"K{delay} (+{delay}w)",
                    "common_vocab50_macro_top10": float(
                        s50_common.get("macro_recall_at_10", 0.0)
                    ),
                    "common_vocab50_micro_top10": float(
                        s50_common.get("micro_recall_at_10", 0.0)
                    ),
                    "common_vocab50_macro_top1": float(
                        s50_common.get("macro_recall_at_1", 0.0)
                    ),
                    "all_vocab50_macro_top10": float(
                        s50_all.get("macro_recall_at_10", 0.0)
                    ),
                    "common_full_vocab_macro_top10": float(
                        fv_common.get("subject_equal_macro_recall_at_10", 0.0)
                    ),
                    "common_full_vocab_word_macro_top10": float(
                        fv_common.get("word_macro_recall_at_10", 0.0)
                    ),
                    "common_full_vocab_micro_top10": float(
                        fv_common.get("micro_recall_at_10", 0.0)
                    ),
                    "common_full_vocab_macro_top1": float(
                        fv_common.get("word_macro_recall_at_1", 0.0)
                    ),
                    "common_full_vocab_mrr": float(
                        fv_common.get("mean_reciprocal_rank", 0.0)
                    ),
                    "common_full_vocab_median_rank": float(
                        fv_common.get("median_rank", full_cand_count / 2.0)
                    ),
                }

        fv_sweep_block = {
            "macro_top1": float(
                full_block_for_primary.get("word_macro_recall_at_1", 0.0)
            ),
            "macro_top5": float(
                full_block_for_primary.get("word_macro_recall_at_5", 0.0)
            ),
            "macro_top10": float(
                full_block_for_primary.get(
                    "subject_equal_macro_recall_at_10", 0.0
                )
            ),
            "word_macro_top10": float(
                full_block_for_primary.get("word_macro_recall_at_10", 0.0)
            ),
            "micro_top1": float(
                full_block_for_primary.get("micro_recall_at_1", 0.0)
            ),
            "micro_top5": float(
                full_block_for_primary.get("micro_recall_at_5", 0.0)
            ),
            "micro_top10": float(
                full_block_for_primary.get("micro_recall_at_10", 0.0)
            ),
            "mrr": float(
                full_block_for_primary.get("mean_reciprocal_rank", 0.0)
            ),
            "median_rank": float(
                full_block_for_primary.get("median_rank", full_cand_count / 2.0)
            ),
        }
        metrics_stub = {
            "vocabulary_sweep": {
                "N50": n50_block,
                "full_vocab": fv_sweep_block,
            },
            "revision_horizons": revision_horizons,
        }
        row = _extract_control_summary_row(
            key, label, shift_seconds, metrics_stub, clean_summary=clean_summary
        )
        row["full_vocab_word_macro_top10"] = float(
            fv_sweep_block["word_macro_top10"]
        )
        return row

    return evaluate_encoded_row


def _build_method_entry(
    *,
    method_id: str,
    method_family: str,
    label: str,
    eta: float,
    rank_k: int | None,
    use_coherence_gate: bool,
    clean_summary: dict,
    condition_entries: list[dict],
    shifted_aggregate: dict,
    beta: float = 0.0,
    tau: float = 0.0,
    baseline_raw_entry: dict | None = None,
    baseline_full_loo_entry: dict | None = None,
) -> dict:
    """构建单个去偏配置（含特征层 Subspace LOO 或候选词列向先验校准 Column-Calibrated Subspace LOO）的完整指标条目。"""
    s_word_macro = [
        float(c["summary"].get("full_vocab_word_macro_top10", 0.0))
        for c in condition_entries
    ]
    if s_word_macro:
        shifted_aggregate["mean_full_vocab_word_macro_top10"] = float(
            np.mean(s_word_macro)
        )
        shifted_aggregate["clean_minus_shifted_full_vocab_word_macro_top10"] = float(
            clean_summary.get("full_vocab_word_macro_top10", 0.0)
            - float(np.mean(s_word_macro))
        )

    entry = {
        "method_id": method_id,
        "method_family": method_family,
        "label": label,
        "eta": float(eta),
        "rank_k": int(rank_k) if rank_k is not None else None,
        "use_coherence_gate": bool(use_coherence_gate),
        "beta": float(beta),
        "tau": float(tau),
        "clean_reference": clean_summary,
        "conditions": condition_entries,
        "shifted_aggregate": shifted_aggregate,
    }

    def _compute_delta(ref_entry: dict) -> dict:
        ref_clean = ref_entry["clean_reference"]
        ref_agg = ref_entry["shifted_aggregate"]
        delta = {
            "vocab50_macro_top10": float(
                clean_summary.get("vocab50_macro_top10", 0.0)
                - ref_clean.get("vocab50_macro_top10", 0.0)
            ),
            "shifted_mean_vocab50_macro_top10": float(
                shifted_aggregate.get("mean_vocab50_macro_top10", 0.0)
                - ref_agg.get("mean_vocab50_macro_top10", 0.0)
            ),
            "clean_minus_shifted_vocab50_macro_top10": float(
                shifted_aggregate.get("clean_minus_shifted_vocab50_macro_top10", 0.0)
                - ref_agg.get("clean_minus_shifted_vocab50_macro_top10", 0.0)
            ),
            "full_vocab_macro_top10": float(
                clean_summary.get("full_vocab_macro_top10", 0.0)
                - ref_clean.get("full_vocab_macro_top10", 0.0)
            ),
            "shifted_mean_full_vocab_macro_top10": float(
                shifted_aggregate.get("mean_full_vocab_macro_top10", 0.0)
                - ref_agg.get("mean_full_vocab_macro_top10", 0.0)
            ),
            "clean_minus_shifted_full_vocab_macro_top10": float(
                shifted_aggregate.get("clean_minus_shifted_full_vocab_macro_top10", 0.0)
                - ref_agg.get("clean_minus_shifted_full_vocab_macro_top10", 0.0)
            ),
            "full_vocab_mrr": float(
                clean_summary.get("full_vocab_mrr", 0.0)
                - ref_clean.get("full_vocab_mrr", 0.0)
            ),
        }
        return delta

    if baseline_raw_entry is not None:
        entry["delta_vs_raw_eta0"] = _compute_delta(baseline_raw_entry)
    if baseline_full_loo_entry is not None:
        entry["delta_vs_full_loo_eta1"] = _compute_delta(baseline_full_loo_entry)
    return entry


def run_subspace_coherence_debiasing_sweep(
    *,
    dataset: str,
    stage: str,
    split: str,
    seed: int,
    exp_tag: str,
    word_exp_tag: str,
    clean_encoded: dict,
    shifted_encoded_by_shift: list[tuple[float, str, str, dict]],
    eval_ds,
    vocabulary: list[str],
    vocab50: list[str],
    eta_values: Iterable[float] = DEFAULT_SUBSPACE_ETAS,
    rank_k_values: Iterable[int] = DEFAULT_SUBSPACE_RANKS,
    coherence_gate_modes: Iterable[bool] = (True, False),
    full_loo_etas: Iterable[float] = DEFAULT_FULL_LOO_ETAS,
    calibration_betas: Iterable[float] = DEFAULT_CALIBRATION_BETAS,
    calibration_taus: Iterable[float] = DEFAULT_CALIBRATION_TAUS,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    center_svd: bool = False,
    normalize_inputs: bool = True,
    context_reference_summary: dict | None = None,
    output_dir: str | Path | None = None,
    plot: bool = True,
    print_report: bool = True,
) -> dict:
    """在单次编码特征上运行方案一（低秩子空间投影 + 相干度门控 LOO + 候选词列向错位先验校准 Z-Norm）完整网格扫描、诊断分析、落盘与作图。"""
    from braindecoding.training.workflow_runner import (
        _summarize_shifted_conditions,
    )

    etas = parse_eta_values(eta_values, default=DEFAULT_SUBSPACE_ETAS)
    ranks = parse_rank_k_values(rank_k_values, default=DEFAULT_SUBSPACE_RANKS)
    full_etas = parse_eta_values(full_loo_etas, default=DEFAULT_FULL_LOO_ETAS)
    cal_betas = parse_float_grid_values(
        calibration_betas, default=DEFAULT_CALIBRATION_BETAS
    )
    cal_taus = parse_float_grid_values(
        calibration_taus, default=DEFAULT_CALIBRATION_TAUS
    )
    gate_modes = list(dict.fromkeys(bool(m) for m in coherence_gate_modes))
    if not gate_modes:
        gate_modes = [True]

    shift_meta = [
        (float(sec), str(key), str(label))
        for sec, key, label, _ in shifted_encoded_by_shift
    ]
    shifted_encoded_map = {
        str(key): enc for _, key, _, enc in shifted_encoded_by_shift
    }
    if len(shift_meta) < 2:
        raise ValueError(
            f"方案一（Subspace-Projected LOO）至少需要 2 个历史错位条件，当前仅有 {len(shift_meta)} 个。"
        )

    shift_keys = [key for _, key, _ in shift_meta]
    eval_row_fn = _build_fast_summary_evaluator(
        eval_ds, vocabulary, vocab50, stage=stage
    )
    vocab_list = [str(w) for w in vocabulary]
    vocab50_list = [str(w) for w in vocab50]
    full_bank_tensor = (
        torch.tensor(
            np.stack(
                [np.asarray(eval_ds.embedding_map[w], dtype=np.float32) for w in vocab_list]
            ),
            dtype=torch.float32,
        )
        if vocab_list
        else torch.zeros((0, 1), dtype=torch.float32)
    )
    vocab50_bank_tensor = (
        torch.tensor(
            np.stack(
                [np.asarray(eval_ds.embedding_map[w], dtype=np.float32) for w in vocab50_list]
            ),
            dtype=torch.float32,
        )
        if vocab50_list
        else torch.zeros((0, 1), dtype=torch.float32)
    )

    # 1. 预计算各视野的未中心化 SVD 奇异向量矩阵 Vh、严格留一法 Vh_loo 与相干度诊断信息
    subspace_diagnostics: dict[str, dict] = {}
    vh_cache: dict[str, torch.Tensor] = {}
    vh_loo_cache: dict[str, list[torch.Tensor]] = {}
    has_revision = "revision_predictions" in clean_encoded and isinstance(
        clean_encoded["revision_predictions"], dict
    )
    if has_revision:
        for delay in sorted(clean_encoded["revision_predictions"]):
            y_shifts_d = torch.stack(
                [
                    shifted_encoded_map[k]["revision_predictions"][delay]
                    for k in shift_keys
                ],
                dim=1,
            )
            diag_d, vh_d = analyze_shift_subspace_and_coherence(
                y_shifts_d,
                coherence_low=coherence_low,
                coherence_high=coherence_high,
                gate_min=gate_min,
                gate_max=gate_max,
                center_svd=center_svd,
                normalize_inputs=normalize_inputs,
                compute_loo_svd=True,
            )
            vh_loo_d = diag_d.pop("_vh_loo_list", None)
            subspace_diagnostics[f"K{delay}"] = diag_d
            vh_cache[f"K{delay}"] = vh_d
            if vh_loo_d is not None:
                vh_loo_cache[f"K{delay}"] = vh_loo_d
    else:
        y_shifts_main = torch.stack(
            [shifted_encoded_map[k]["predictions"] for k in shift_keys], dim=1
        )
        diag_main, vh_main = analyze_shift_subspace_and_coherence(
            y_shifts_main,
            coherence_low=coherence_low,
            coherence_high=coherence_high,
            gate_min=gate_min,
            gate_max=gate_max,
            center_svd=center_svd,
            normalize_inputs=normalize_inputs,
            compute_loo_svd=True,
        )
        vh_loo_main = diag_main.pop("_vh_loo_list", None)
        subspace_diagnostics["main"] = diag_main
        vh_cache["main"] = vh_main
        if vh_loo_main is not None:
            vh_loo_cache["main"] = vh_loo_main

    def _evaluate_pair(
        c_enc: dict, s_enc_map: dict[str, dict]
    ) -> tuple[dict, list[dict], dict]:
        c_row = eval_row_fn(c_enc, "clean_0s", "Clean (0s)", 0.0, None)
        s_rows: list[dict] = []
        cond_entries: list[dict] = []
        for sec, key, label in shift_meta:
            s_row = eval_row_fn(s_enc_map[key], key, label, sec, c_row)
            s_rows.append(s_row)
            cond_entries.append(
                {
                    "key": key,
                    "label": label,
                    "shift_seconds": sec,
                    "summary": s_row,
                }
            )
        s_agg = _summarize_shifted_conditions(c_row, s_rows)
        return c_row, cond_entries, s_agg

    # 2. 评估 Raw Baseline (eta = 0.0)
    raw_c_row, raw_conds, raw_agg = _evaluate_pair(
        clean_encoded, shifted_encoded_map
    )
    raw_entry = _build_method_entry(
        method_id="raw_eta0",
        method_family="raw_baseline",
        label="Raw Baseline (eta=0.0)",
        eta=0.0,
        rank_k=None,
        use_coherence_gate=False,
        clean_summary=raw_c_row,
        condition_entries=raw_conds,
        shifted_aggregate=raw_agg,
    )

    # 3. 评估标准全维 LOO 对照基线 (Full-Rank D, Gate=Off, eta in {0.5, 0.8, 1.0})
    full_loo_entries: list[dict] = []
    full_loo_eta1_entry: dict | None = None
    for f_eta in full_etas:
        deb_c, deb_s = apply_subspace_coherence_debiasing_to_encoded(
            clean_encoded,
            shifted_encoded_map,
            eta=f_eta,
            rank_k=None,
            use_coherence_gate=False,
            center_svd=center_svd,
            normalize_inputs=normalize_inputs,
            precomputed_vh_by_key=vh_cache,
            precomputed_vh_loo_by_key=vh_loo_cache,
        )
        c_row, conds, s_agg = _evaluate_pair(deb_c, deb_s)
        f_entry = _build_method_entry(
            method_id=f"full_loo_eta{f_eta:g}",
            method_family="full_rank_loo",
            label=f"Full-Rank LOO (rank=Full, eta={f_eta:g}, gate=Off)",
            eta=f_eta,
            rank_k=None,
            use_coherence_gate=False,
            clean_summary=c_row,
            condition_entries=conds,
            shifted_aggregate=s_agg,
            baseline_raw_entry=raw_entry,
        )
        full_loo_entries.append(f_entry)
        if abs(f_eta - 1.0) <= 1e-6 or full_loo_eta1_entry is None:
            full_loo_eta1_entry = f_entry

    # 4. 扫描方案一（特征层）：低秩共模子空间投影 + 相干度门控去偏 (rank_k x eta x use_coherence_gate)
    subspace_entries: list[dict] = []
    for use_gate in gate_modes:
        for rank_k in ranks:
            for eta in etas:
                deb_c, deb_s = apply_subspace_coherence_debiasing_to_encoded(
                    clean_encoded,
                    shifted_encoded_map,
                    eta=eta,
                    rank_k=rank_k,
                    use_coherence_gate=use_gate,
                    coherence_low=coherence_low,
                    coherence_high=coherence_high,
                    gate_min=gate_min,
                    gate_max=gate_max,
                    center_svd=center_svd,
                    normalize_inputs=normalize_inputs,
                    precomputed_vh_by_key=vh_cache,
                    precomputed_vh_loo_by_key=vh_loo_cache,
                )
                c_row, conds, s_agg = _evaluate_pair(deb_c, deb_s)
                gate_str = "On" if use_gate else "Off"
                s_entry = _build_method_entry(
                    method_id=f"subspace_k{rank_k}_eta{eta:g}_gate{gate_str.lower()}",
                    method_family=(
                        "subspace_coherence_gated_loo"
                        if use_gate
                        else "subspace_only_loo"
                    ),
                    label=f"Subspace LOO (K={rank_k}, eta={eta:g}, Gate={gate_str})",
                    eta=eta,
                    rank_k=rank_k,
                    use_coherence_gate=use_gate,
                    clean_summary=c_row,
                    condition_entries=conds,
                    shifted_aggregate=s_agg,
                    baseline_raw_entry=raw_entry,
                    baseline_full_loo_entry=full_loo_eta1_entry,
                )
                subspace_entries.append(s_entry)

    # 5. 选出特征层最优方案一配置
    best_by_gap50 = max(
        subspace_entries,
        key=lambda item: (
            float(
                item["shifted_aggregate"].get(
                    "clean_minus_shifted_vocab50_macro_top10", 0.0
                )
            ),
            float(item["clean_reference"].get("vocab50_macro_top10", 0.0)),
        ),
    )
    raw_shift50 = float(raw_agg.get("mean_vocab50_macro_top10", 0.0))
    effective_suppressed = [
        item
        for item in subspace_entries
        if float(item["shifted_aggregate"].get("mean_vocab50_macro_top10", 1.0))
        <= raw_shift50 - 0.03
    ]
    pool_for_clean = effective_suppressed if effective_suppressed else subspace_entries
    best_by_clean_preserving = max(
        pool_for_clean,
        key=lambda item: (
            float(item["clean_reference"].get("vocab50_macro_top10", 0.0)),
            float(
                item["shifted_aggregate"].get(
                    "clean_minus_shifted_vocab50_macro_top10", 0.0
                )
            ),
        ),
    )

    # 6. 扫描候选词列向错位先验校准 (Column-Wise Null Standardization / Z-Norm Anti-Hubness: beta, tau in [0.0, 0.3, 0.5, 0.7, 1.0])
    #    选取代表性锚点：(K=8, eta=0.80, Gate=Off) 与 (K=12, eta=1.00, Gate=Off)（若不在网格中则自动纳入首锚点 + 最优锚点）
    anchor_specs: list[tuple[int, float, bool]] = []
    preferred_anchors = [
        (ranks[0], etas[0], False),
        (
            12 if 12 in ranks else ranks[min(1, len(ranks) - 1)],
            1.0 if 1.0 in etas else etas[min(1, len(etas) - 1)],
            False,
        ),
    ]
    for ak, aeta, agate in preferred_anchors:
        key_spec = (int(ak), round(float(aeta), 4), bool(agate))
        if key_spec not in anchor_specs:
            anchor_specs.append(key_spec)

    column_calibration_entries: list[dict] = []
    for ak, aeta, agate in anchor_specs:
        for beta_val in cal_betas:
            for tau_val in cal_taus:
                cal_c, cal_s = apply_column_calibrated_subspace_loo_to_encoded(
                    clean_encoded,
                    shifted_encoded_map,
                    full_bank_tensor=full_bank_tensor,
                    vocab50_bank_tensor=vocab50_bank_tensor,
                    rank_k=ak,
                    eta=aeta,
                    beta=beta_val,
                    tau=tau_val,
                    use_coherence_gate=agate,
                    coherence_low=coherence_low,
                    coherence_high=coherence_high,
                    gate_min=gate_min,
                    gate_max=gate_max,
                    center_svd=center_svd,
                    normalize_inputs=normalize_inputs,
                    precomputed_vh_by_key=vh_cache,
                    precomputed_vh_loo_by_key=vh_loo_cache,
                    calibrate_vocab50=True,
                )
                c_row, conds, s_agg = _evaluate_pair(cal_c, cal_s)
                cal_entry = _build_method_entry(
                    method_id=f"colcal_k{ak}_eta{aeta:g}_b{beta_val:g}_tau{tau_val:g}",
                    method_family="subspace_loo_with_column_calibration",
                    label=f"ColCal Subspace (K={ak}, eta={aeta:g}, beta={beta_val:g}, tau={tau_val:g})",
                    eta=aeta,
                    rank_k=ak,
                    use_coherence_gate=agate,
                    beta=beta_val,
                    tau=tau_val,
                    clean_summary=c_row,
                    condition_entries=conds,
                    shifted_aggregate=s_agg,
                    baseline_raw_entry=raw_entry,
                    baseline_full_loo_entry=full_loo_eta1_entry,
                )
                column_calibration_entries.append(cal_entry)

    # 选出能把 FullVocab Shifted 压低至 <= 3.8%（或最低）且保持最高 FullVocab Clean 的最优校准配置
    target_fv_suppressed = [
        item
        for item in column_calibration_entries
        if float(item["shifted_aggregate"].get("mean_full_vocab_macro_top10", 1.0))
        <= 0.040
    ]
    pool_fv = target_fv_suppressed if target_fv_suppressed else column_calibration_entries
    best_by_full_vocab_calibrated = (
        max(
            pool_fv,
            key=lambda item: (
                float(
                    item["shifted_aggregate"].get(
                        "clean_minus_shifted_full_vocab_macro_top10", 0.0
                    )
                ),
                float(item["clean_reference"].get("full_vocab_macro_top10", 0.0)),
            ),
        )
        if pool_fv
        else None
    )

    # 7. 计算齐普夫高频词中心度垄断诊断 (Zipf Hubness Diagnostics)
    if has_revision:
        max_d = max(clean_encoded["revision_predictions"])
        y_c_primary = clean_encoded["revision_predictions"][max_d]
        y_s_primary = torch.stack(
            [
                shifted_encoded_map[k]["revision_predictions"][max_d]
                for k in shift_keys
            ],
            dim=1,
        )
        vh_p = vh_cache.get(f"K{max_d}")
        vh_loo_p = vh_loo_cache.get(f"K{max_d}")
    else:
        y_c_primary = clean_encoded["predictions"]
        y_s_primary = torch.stack(
            [shifted_encoded_map[k]["predictions"] for k in shift_keys], dim=1
        )
        vh_p = vh_cache.get("main")
        vh_loo_p = vh_loo_cache.get("main")

    hub_ak, hub_aeta, hub_agate = anchor_specs[0]
    lexical_hubness_diagnostics = analyze_lexical_hubness_diagnostics(
        y_c_primary,
        y_s_primary,
        full_bank_tensor,
        list(clean_encoded["words"]),
        vocab_list,
        vocab50_list,
        rank_k=hub_ak,
        eta=hub_aeta,
        beta=cal_betas[0] if cal_betas else 1.0,
        tau_values=cal_taus,
        use_coherence_gate=hub_agate,
        center_svd=center_svd,
        precomputed_vh=vh_p,
        precomputed_vh_loo=vh_loo_p,
    )

    report_payload = {
        "schema_version": 2,
        "experiment_type": "exploratory_subspace_coherence_loo_debiasing",
        "dataset": dataset,
        "stage": stage,
        "split": split,
        "seed": int(seed),
        "exp_tag": exp_tag,
        "word_exp_tag": word_exp_tag,
        "formula": {
            "subspace_svd": "_, _, Vh = svd(y_shift_mean_all); U_K = Vh[:K, :].T  (uncentered SVD captures global static mean mu_0 in 1st singular vector)",
            "coherence_gate": f"gate_i = clamp((||y_shift_mean_i||_2 - {coherence_low}) / ({coherence_high} - {coherence_low}), min={gate_min}, max={gate_max})",
            "clean_debiasing": "y_clean_debiased = normalize(y_clean - (eta * gate_clean) * (y_shift_mean_all @ U_K @ U_K.T))",
            "shifted_loo_debiasing": "y_k_debiased = normalize(y_shift_k - (eta * gate_k) * (y_shift_loo_mean_k @ U_{K,LOO} @ U_{K,LOO}.T))",
            "column_wise_null_standardization": "S_tilde_{i,v} = (<y_hat_i, e_v> - beta * mu_v^{shift}) / (sigma_v^{shift})^tau  [strict LOO on shifted conditions]",
            "num_shifted_conditions_M": len(shift_meta),
            "shift_seconds": [sec for sec, _, _ in shift_meta],
            "coherence_low": float(coherence_low),
            "coherence_high": float(coherence_high),
            "gate_min": float(gate_min),
            "gate_max": float(gate_max),
            "center_svd": bool(center_svd),
        },
        "grid": {
            "rank_k_values": ranks,
            "eta_values": etas,
            "coherence_gate_modes": gate_modes,
            "full_loo_etas": full_etas,
            "calibration_betas": cal_betas,
            "calibration_taus": cal_taus,
        },
        "subspace_and_coherence_diagnostics": subspace_diagnostics,
        "lexical_hubness_diagnostics": lexical_hubness_diagnostics,
        "context_stage_reference": context_reference_summary,
        "raw_baseline": raw_entry,
        "full_rank_loo_baselines": full_loo_entries,
        "subspace_sweep": subspace_entries,
        "column_calibration_sweep": column_calibration_entries,
        "best_by_net_gap_vocab50": best_by_gap50,
        "best_by_clean_preservation_vocab50": best_by_clean_preserving,
        "best_by_full_vocab_calibrated": best_by_full_vocab_calibrated,
        "figures": {},
    }

    if print_report:
        print_subspace_coherence_debiasing_report(report_payload)

    if output_dir is not None:
        out_path = Path(output_dir)
        exploratory_dir = out_path / "exploratory"
        exploratory_dir.mkdir(parents=True, exist_ok=True)
        json_file = (
            exploratory_dir / f"subspace_coherence_debiasing_{split}.json"
        )
        json_file.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_payload["exploratory_json"] = str(json_file)
        if plot:
            figs = plot_subspace_coherence_debiasing_figures(
                report_payload, out_path / "figures"
            )
            report_payload["figures"].update(figs)
            json_file.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return report_payload


def print_subspace_coherence_debiasing_report(report: dict) -> None:
    """在终端打印方案一（Subspace-Projected + Coherence-Gated LOO + Column-Wise Null Standardization）全维度诊断与对比报表。"""
    sep = "=" * 132
    thin = "-" * 132
    dataset = report.get("dataset", "")
    stage = report.get("stage", "")
    split = report.get("split", "test")
    seed = report.get("seed", 0)
    exp_tag = report.get("exp_tag", "") or "(default)"
    f_info = report.get("formula", {})
    m_count = f_info.get("num_shifted_conditions_M", 6)
    center_svd = bool(f_info.get("center_svd", False))

    print(
        f"\n{sep}\n"
        f" [Scheme 1 Exploratory] Subspace-Projected + Coherence-Gated LOO + Column-Wise Null Standardization\n"
        f" dataset={dataset}  |  stage={stage}  |  split={split}  |  seed={seed}  |  exp_tag={exp_tag}  |  uncentered_svd={not center_svd}\n"
        f" Clean Subspace LOO  : y_hat^(0)  = Norm(y^(0)  - (eta * gate) * (y_shift_mean @ U_K @ U_K^T))         [M={m_count} historical shifts]\n"
        f" Shifted Strict LOO  : y_hat^(dk) = Norm(y^(dk) - (eta * gate) * (y_loo_mean   @ U_{{K,LOO}} @ U_{{K,LOO}}^T)) [LOO M-1={m_count - 1} shifts]\n"
        f" Column Z-Norm Calib : S_tilde_{{i,v}} = (<y_hat_i, e_v> - beta * mu_v^shift) / (sigma_v^shift)^tau      [Strict LOO on shifted]\n"
        f"{thin}",
        flush=True,
    )

    # 1. 打印奇异值子空间能量谱与相干度门控诊断
    diags = report.get("subspace_and_coherence_diagnostics") or {}
    if diags:
        print(
            " 1. Shift Common-Mode SVD Energy Spectrum & Cross-Shift Coherence Gate Diagnostics:\n"
            f" {'Horizon':<8} | {'Dim D':>5} | {'EffRank':>7} | {'Top-8 Var%':>10} | {'Top-12 Var%':>11} | {'Top-16 Var%':>11} | {'Top-24 Var%':>11} || {'||y_shift|| Mean+-SD':>20} | {'RandFloor':>9} | {'Gate Mean':>9}\n"
            f" {'-' * 8}-+-{'-' * 5}-+-{'-' * 7}-+-{'-' * 10}-+-{'-' * 11}-+-{'-' * 11}-+-{'-' * 11}-++----------------------+-----------+----------",
            flush=True,
        )
        for h_key, d_item in diags.items():
            cum = d_item.get("cumulative_energy_explained", {})
            coh = d_item.get("coherence_norm", {})
            gt = d_item.get("adaptive_gate", {})
            e8 = float(cum.get("top_8", 0.0)) * 100.0
            e12 = float(cum.get("top_12", 0.0)) * 100.0
            e16 = float(cum.get("top_16", 0.0)) * 100.0
            e24 = float(cum.get("top_24", 0.0)) * 100.0
            c_m = float(coh.get("mean", 0.0))
            c_sd = float(coh.get("std", 0.0))
            r_fl = float(d_item.get("theoretical_random_orthogonal_norm", 0.4082))
            g_m = float(gt.get("mean", 1.0))
            print(
                f" {h_key:<8} | {int(d_item.get('feature_dim_D', 1024)):5d} | {float(d_item.get('effective_rank_participation_ratio', 0.0)):7.1f} |"
                f" {e8:9.2f}% | {e12:10.2f}% | {e16:10.2f}% | {e24:10.2f}% ||"
                f" {f'{c_m:.3f} +- {c_sd:.3f}':>20} | {r_fl:9.3f} | {g_m:9.3f}",
                flush=True,
            )
        print(thin, flush=True)

    # 若提供了 Stage 2.1 (context) 基准结果，打印参考线
    ctx_ref = report.get("context_stage_reference")
    if isinstance(ctx_ref, dict) and ctx_ref.get("vocab50_macro_top10") is not None:
        ctx_50 = float(ctx_ref["vocab50_macro_top10"]) * 100.0
        ctx_fv = float(ctx_ref.get("full_vocab_macro_top10", 0.0)) * 100.0
        print(
            f" [Reference] Stage 2.1 (Context Full-Window) Clean 50-Word Macro Top-10 = {ctx_50:.2f}%  |  Full-Vocab Top-10 = {ctx_fv:.2f}%\n"
            f"{thin}",
            flush=True,
        )

    # 2. 打印主表：Raw vs Full-Rank LOO vs Scheme 1 (Subspace + Coherence Gate, uncalibrated tau=0.0)
    print(
        " 2. Feature-Space Subspace LOO Retrieval Comparison (Before Column Z-Norm, tau=0.0):\n"
        f" {'Configuration':<44} | {'Clean Top10(50)':>15} | {'Shift Mean(50)':>14} | {'Net Gap(50)':>12} | {'vs Raw(0)':>9} | {'vs Full(1.0)':>12} || {'Clean FullTop10':>15} | {'Shift FullTop10':>15}\n"
        f" {'-' * 44}-+-{'-' * 15}-+-{'-' * 14}-+-{'-' * 12}-+-{'-' * 9}-+-{'-' * 12}-++-----------------+----------------",
        flush=True,
    )

    all_rows = [
        report["raw_baseline"],
        *report.get("full_rank_loo_baselines", ()),
        *report.get("subspace_sweep", ()),
    ]
    for item in all_rows:
        lbl = str(item["label"])
        c_ref = item["clean_reference"]
        agg = item["shifted_aggregate"]
        d_raw = item.get("delta_vs_raw_eta0")
        d_full = item.get("delta_vs_full_loo_eta1")

        c50 = float(c_ref.get("vocab50_macro_top10", 0.0)) * 100.0
        s50 = float(agg.get("mean_vocab50_macro_top10", 0.0)) * 100.0
        gap50 = float(agg.get("clean_minus_shifted_vocab50_macro_top10", 0.0)) * 100.0
        d_raw_str = (
            "baseline"
            if d_raw is None
            else f"{float(d_raw.get('vocab50_macro_top10', 0.0)) * 100.0:+7.2f}%"
        )
        d_full_str = (
            "---"
            if d_full is None
            else f"{float(d_full.get('vocab50_macro_top10', 0.0)) * 100.0:+7.2f}%"
        )
        cfv = float(c_ref.get("full_vocab_macro_top10", 0.0)) * 100.0
        sfv = float(agg.get("mean_full_vocab_macro_top10", 0.0)) * 100.0

        print(
            f" {lbl:<44} | {c50:14.2f}% | {s50:13.2f}% | {gap50:+11.2f}% | {d_raw_str:>9} | {d_full_str:>12} || {cfv:14.2f}% | {sfv:14.2f}%",
            flush=True,
        )

    # 3. 若包含 K0 / K1 / K2 视野，打印 50w 视野拆解对比表
    raw_hz = report["raw_baseline"]["clean_reference"].get("horizons") or {}
    if raw_hz:
        print(
            f"{thin}\n"
            " 3. Revision Horizons (K0 / K1 / K2) 50-Word Macro Top-10 Clean vs LOO Shifted Mean Breakdown:\n"
            f" {'Configuration':<44} | {'K0 Clean / Shift (Gap)':>24} | {'K1 Clean / Shift (Gap)':>24} | {'K2 Clean / Shift (Gap)':>24}\n"
            f" {'-' * 44}-+-{'-' * 24}-+-{'-' * 24}-+-{'-' * 24}",
            flush=True,
        )
        for item in all_rows:
            lbl = str(item["label"])
            c_hz = item["clean_reference"].get("horizons") or {}
            s_hz = item["shifted_aggregate"].get("horizons") or {}
            cols: list[str] = []
            for hk in ("K0", "K1", "K2"):
                if hk in c_hz:
                    ch_val = float(c_hz[hk].get("common_vocab50_macro_top10", 0.0)) * 100.0
                    sh_val = float(s_hz.get(hk, {}).get("mean_common_vocab50_macro_top10", 0.0)) * 100.0
                    gp_val = ch_val - sh_val
                    cols.append(f"{ch_val:5.2f}/{sh_val:5.2f} ({gp_val:+5.2f}%)")
                else:
                    cols.append("N/A")
            print(
                f" {lbl:<44} | {cols[0]:>24} | {cols[1]:>24} | {cols[2]:>24}",
                flush=True,
            )

    # 4. 打印齐普夫高频词中心度垄断 (Zipf Hubness) 与反中心度温度 tau 诊断表
    hub_diag = report.get("lexical_hubness_diagnostics") or {}
    if hub_diag:
        tot_q = int(hub_diag.get("total_queries", 0))
        v_sz = int(hub_diag.get("vocab_size", 2020))
        q50_cnt = int(hub_diag.get("top50_query_count", 0))
        q50_sh = float(hub_diag.get("top50_query_share", 0.0)) * 100.0
        q100_cnt = int(hub_diag.get("top100_query_count", 0))
        q100_sh = float(hub_diag.get("top100_query_share", 0.0)) * 100.0
        std_50 = float(hub_diag.get("top50_col_std_mean", 0.0))
        std_tail = float(hub_diag.get("tail_col_std_mean", 0.0))
        std_ratio = float(hub_diag.get("std_ratio_top50_vs_tail", 1.0))
        mu_50 = float(hub_diag.get("top50_abs_mean_bias", 0.0))
        mu_tail = float(hub_diag.get("tail_abs_mean_bias", 0.0))
        exp_hub_hit = float(hub_diag.get("expected_random_within_top50_hub_pool", 0.0)) * 100.0
        unif_top50_share = float(hub_diag.get("theoretical_uniform_top50_slot_share", 50.0 / max(1, v_sz))) * 100.0

        print(
            f"{thin}\n"
            " 4. Zipf Lexical Hubness & Column Variance Anisotropy Diagnostics (Why Uncalibrated FullVocab Shifted Stays ~8.4%):\n"
            f"    • Test Query Zipf Concentration : Top-50 words = {q50_cnt}/{tot_q} ({q50_sh:.1f}%)  |  Top-100 words = {q100_cnt}/{tot_q} ({q100_sh:.1f}%)\n"
            f"    • Shifted Column Std Anisotropy : sigma(v<=50) = {std_50:.4f} vs sigma(v>50) = {std_tail:.4f} (Ratio = {std_ratio:.2f}x)  |  |mu_v|(v<=50) = {mu_50:.4f} vs {mu_tail:.4f}\n"
            f"    • Hubness Floor Mechanism       : {q50_sh:.1f}% Top-50 queries × (10 / ~60 high-variance hub candidates) ≈ {exp_hub_hit:.2f}% FullVocab Shifted Top-10!\n"
            f"    • Top-50 Word Occupancy in Shifted Top-10 Slots Across Anti-Hubness Temperature tau (Uniform Random = {unif_top50_share:.2f}%):",
            flush=True,
        )
        hub_by_tau = hub_diag.get("shifted_top10_hubness_by_tau") or {}
        if hub_by_tau:
            tau_parts: list[str] = []
            for t_key, t_info in hub_by_tau.items():
                tau_v = float(t_info.get("tau", 0.0))
                occ_v = float(t_info.get("top50_occupancy_in_shifted_top10", 0.0)) * 100.0
                tau_parts.append(f"tau={tau_v:.2f}: {occ_v:5.2f}%")
            print(f"      [ {'  |  '.join(tau_parts)} ]", flush=True)

    # 5. 打印逐列零分布标准化 (Column-Wise Null Standardization, tau in [0.0, 0.3, 0.5, 0.7, 1.0]) 扫描主表
    cal_rows = list(report.get("column_calibration_sweep", ()))
    if cal_rows:
        print(
            f"{thin}\n"
            " 5. Column-Wise Null Standardization (subspace_loo_with_column_calibration, Strict LOO on Shifted) — Full-Vocab (2020w) & 50w Sweep:\n"
            f" {'Configuration (Subspace + Column Z-Norm)':<52} | {'FV Clean(Micro)':>15} | {'FV Shift(LOO)':>14} | {'FV Net Gap':>11} | {'FV WordMacro C/S':>18} || {'50w Clean / Shift (Gap)':>25}\n"
            f" {'-' * 52}-+-{'-' * 15}-+-{'-' * 14}-+-{'-' * 11}-+-{'-' * 18}-++--------------------------",
            flush=True,
        )
        for item in [report["raw_baseline"], *cal_rows]:
            lbl = str(item["label"])
            c_ref = item["clean_reference"]
            agg = item["shifted_aggregate"]

            cfv = float(c_ref.get("full_vocab_macro_top10", 0.0)) * 100.0
            sfv = float(agg.get("mean_full_vocab_macro_top10", 0.0)) * 100.0
            gfv = float(agg.get("clean_minus_shifted_full_vocab_macro_top10", cfv / 100.0 - sfv / 100.0)) * 100.0

            cw_fv = float(c_ref.get("full_vocab_word_macro_top10", cfv / 100.0)) * 100.0
            sw_fv = float(agg.get("mean_full_vocab_word_macro_top10", sfv / 100.0)) * 100.0

            c50 = float(c_ref.get("vocab50_macro_top10", 0.0)) * 100.0
            s50 = float(agg.get("mean_vocab50_macro_top10", 0.0)) * 100.0
            g50 = float(agg.get("clean_minus_shifted_vocab50_macro_top10", 0.0)) * 100.0

            print(
                f" {lbl:<52} | {cfv:14.2f}% | {sfv:13.2f}% | {gfv:+10.2f}% | {f'{cw_fv:5.2f}% / {sw_fv:5.2f}%':>18} || {f'{c50:5.2f}/{s50:5.2f} ({g50:+5.2f}%)':>25}",
                flush=True,
            )

        # 6. 若包含 K0 / K1 / K2 视野，打印全词表 (2020w) 下的 K0 / K1 / K2 Clean vs Shifted 拆解表
        if raw_hz:
            print(
                f"{thin}\n"
                " 6. Revision Horizons (K0 / K1 / K2) Full-Vocab (2020 Words) Clean vs Strict-LOO Shifted Breakdown Across tau:\n"
                f" {'Configuration (Subspace + Column Z-Norm)':<52} | {'K0 FV Clean/Shift (Gap)':>24} | {'K1 FV Clean/Shift (Gap)':>24} | {'K2 FV Clean/Shift (Gap)':>24}\n"
                f" {'-' * 52}-+-{'-' * 24}-+-{'-' * 24}-+-{'-' * 24}",
                flush=True,
            )
            for item in [report["raw_baseline"], *cal_rows]:
                lbl = str(item["label"])
                c_hz = item["clean_reference"].get("horizons") or {}
                s_hz = item["shifted_aggregate"].get("horizons") or {}
                cols_fv: list[str] = []
                for hk in ("K0", "K1", "K2"):
                    if hk in c_hz:
                        ch_fv = float(c_hz[hk].get("common_full_vocab_macro_top10", 0.0)) * 100.0
                        sh_fv = float(s_hz.get(hk, {}).get("mean_common_full_vocab_macro_top10", 0.0)) * 100.0
                        gp_fv = ch_fv - sh_fv
                        cols_fv.append(f"{ch_fv:5.2f}/{sh_fv:5.2f} ({gp_fv:+5.2f}%)")
                    else:
                        cols_fv.append("N/A")
                print(
                    f" {lbl:<52} | {cols_fv[0]:>24} | {cols_fv[1]:>24} | {cols_fv[2]:>24}",
                    flush=True,
                )

    # 7. 打印推荐的最优配置摘要
    best_gap = report.get("best_by_net_gap_vocab50")
    best_clean = report.get("best_by_clean_preservation_vocab50")
    best_fv_cal = report.get("best_by_full_vocab_calibrated")
    print(thin, flush=True)
    if best_clean:
        bc_c50 = float(best_clean["clean_reference"]["vocab50_macro_top10"]) * 100.0
        bc_s50 = float(best_clean["shifted_aggregate"]["mean_vocab50_macro_top10"]) * 100.0
        bc_g50 = float(best_clean["shifted_aggregate"]["clean_minus_shifted_vocab50_macro_top10"]) * 100.0
        print(
            f" ★ Best 50w Clean-Preserving    : {best_clean['label']:<50} -> 50w Clean={bc_c50:.2f}%, Shifted={bc_s50:.2f}%, Net Gap={bc_g50:+.2f}%",
            flush=True,
        )
    if best_gap:
        bg_c50 = float(best_gap["clean_reference"]["vocab50_macro_top10"]) * 100.0
        bg_s50 = float(best_gap["shifted_aggregate"]["mean_vocab50_macro_top10"]) * 100.0
        bg_g50 = float(best_gap["shifted_aggregate"]["clean_minus_shifted_vocab50_macro_top10"]) * 100.0
        print(
            f" ★ Best 50w Net-Gap             : {best_gap['label']:<50} -> 50w Clean={bg_c50:.2f}%, Shifted={bg_s50:.2f}%, Net Gap={bg_g50:+.2f}%",
            flush=True,
        )
    if best_fv_cal:
        bf_cfv = float(best_fv_cal["clean_reference"]["full_vocab_macro_top10"]) * 100.0
        bf_sfv = float(best_fv_cal["shifted_aggregate"]["mean_full_vocab_macro_top10"]) * 100.0
        bf_gfv = float(best_fv_cal["shifted_aggregate"]["clean_minus_shifted_full_vocab_macro_top10"]) * 100.0
        bf_c50 = float(best_fv_cal["clean_reference"]["vocab50_macro_top10"]) * 100.0
        bf_s50 = float(best_fv_cal["shifted_aggregate"]["mean_vocab50_macro_top10"]) * 100.0
        print(
            f" ★ Best Full-Vocab Calibrated   : {best_fv_cal['label']:<50} -> FV Clean={bf_cfv:.2f}%, FV Shifted={bf_sfv:.2f}%, FV Gap={bf_gfv:+.2f}% (50w: {bf_c50:.2f}%/{bf_s50:.2f}%)",
            flush=True,
        )
    print(f"{sep}\n", flush=True)


def plot_subspace_coherence_debiasing_figures(
    report: dict,
    figures_dir: str | Path,
) -> dict[str, str]:
    """生成方案一（低秩子空间投影 + 相干度门控 + 逐列零分布标准化去偏）专属四宫格诊断看板图，保存至 figures/ 目录。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = report.get("split", "test")
    dataset = report.get("dataset", "")
    stage = report.get("stage", "")
    seed = report.get("seed", 0)
    raw_entry = report["raw_baseline"]
    full_entries = list(report.get("full_rank_loo_baselines", ()))
    subspace_entries = list(report.get("subspace_sweep", ()))
    cal_entries = list(report.get("column_calibration_sweep", ()))
    if not subspace_entries:
        return {}

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.2), dpi=160)
    fig.suptitle(
        f"Scheme 1: Subspace-Projected LOO + Column-Wise Null Standardization  |  {dataset}  |  stage={stage}  |  split={split}  |  seed={seed}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Singular Value Cumulative Energy Explained across Rank K
    ax0 = axes[0, 0]
    diags = report.get("subspace_and_coherence_diagnostics") or {}
    rank_ticks = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    colors_h = {"K0": "#4e79a7", "K1": "#f28e2b", "K2": "#59a14f", "main": "#1f77b4"}
    for h_key, d_item in diags.items():
        cum = d_item.get("cumulative_energy_explained", {})
        xs = [r for r in rank_ticks if f"top_{r}" in cum]
        ys = [float(cum[f"top_{r}"]) * 100.0 for r in xs]
        if xs:
            coh_m = float(d_item.get("coherence_norm", {}).get("mean", 0.0))
            ax0.plot(
                xs,
                ys,
                "-o",
                linewidth=2.0,
                color=colors_h.get(h_key, "#1f77b4"),
                label=f"{h_key} (mean ||y_shift||={coh_m:.3f})",
            )
    ax0.axvline(8, color="#d62728", linestyle="--", linewidth=1.2, label="Recommended Rank K=8..12")
    ax0.set_xlabel("Subspace Rank K (Uncentered SVD Singular Vectors)")
    ax0.set_ylabel("Cumulative Shift Bias Energy Explained (%)")
    ax0.set_title("A. Shift Common-Mode Uncentered SVD Energy Spectrum", fontweight="bold")
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.legend(loc="lower right", fontsize=8.5)

    # Panel B: Clean vs Shifted Mean vs Net Gap Across Rank K (50-Word Macro Top-10)
    ax1 = axes[0, 1]
    gated_eta1 = [
        e
        for e in subspace_entries
        if e.get("use_coherence_gate") and abs(float(e["eta"]) - 1.0) <= 1e-6
    ]
    if not gated_eta1:
        target_eta = float(subspace_entries[0]["eta"])
        gated_eta1 = [
            e for e in subspace_entries if abs(float(e["eta"]) - target_eta) <= 1e-6
        ]
    else:
        target_eta = 1.0

    full_eta1 = next(
        (e for e in full_entries if abs(float(e["eta"]) - target_eta) <= 1e-6),
        full_entries[-1] if full_entries else raw_entry,
    )
    compare_items = [raw_entry, *gated_eta1, full_eta1]
    x_labels = [
        "Raw\n(η=0)"
        if item["method_family"] == "raw_baseline"
        else (
            f"Full-Rank\n(D, η={item['eta']:g})"
            if item["method_family"] == "full_rank_loo"
            else f"Rank K={item['rank_k']}\n(η={item['eta']:g}, Gate)"
        )
        for item in compare_items
    ]
    x_pos = np.arange(len(compare_items))
    c_vals = [
        float(item["clean_reference"].get("vocab50_macro_top10", 0.0)) * 100.0
        for item in compare_items
    ]
    s_vals = [
        float(item["shifted_aggregate"].get("mean_vocab50_macro_top10", 0.0)) * 100.0
        for item in compare_items
    ]
    g_vals = [
        float(
            item["shifted_aggregate"].get("clean_minus_shifted_vocab50_macro_top10", 0.0)
        )
        * 100.0
        for item in compare_items
    ]
    width = 0.26
    bars_c = ax1.bar(x_pos - width, c_vals, width=width, label="Clean (0s)", color="#1f77b4")
    bars_s = ax1.bar(x_pos, s_vals, width=width, label="Shifted Mean (LOO)", color="#ff7f0e")
    bars_g = ax1.bar(x_pos + width, g_vals, width=width, label="Net Gap (Clean - Shift)", color="#2ca02c")

    ctx_ref = report.get("context_stage_reference")
    if isinstance(ctx_ref, dict) and ctx_ref.get("vocab50_macro_top10") is not None:
        ctx_50 = float(ctx_ref["vocab50_macro_top10"]) * 100.0
        ax1.axhline(
            ctx_50,
            color="#9467bd",
            linestyle="-.",
            linewidth=1.4,
            label=f"Stage 2.1 Context ({ctx_50:.2f}%)",
        )
    chance_50 = float(raw_entry["clean_reference"].get("vocab50_chance_top10", 0.2)) * 100.0
    ax1.axhline(chance_50, color="#666666", linestyle="--", linewidth=1.0, label=f"Chance ({chance_50:.1f}%)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=8.5)
    ax1.set_ylabel("50-Word Macro Top-10 (%)")
    ax1.set_title("B. 50-Word Macro Top-10: Subspace Rank K vs Full-Rank LOO", fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="best", fontsize=8.0)
    for b_group in (bars_c, bars_s, bars_g):
        for b in b_group:
            h = b.get_height()
            ax1.annotate(
                f"{h:.1f}%",
                xy=(b.get_x() + b.get_width() / 2.0, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )

    # Panel C: Revision Horizons (K0 / K1 / K2) Clean vs Shifted Mean
    ax2 = axes[1, 0]
    best_sub = report.get("best_by_clean_preservation_vocab50") or subspace_entries[0]
    raw_hz = raw_entry["clean_reference"].get("horizons") or {}
    if raw_hz and all(k in raw_hz for k in ("K0", "K1", "K2")):
        hz_keys = ["K0", "K1", "K2"]
        hz_x = np.arange(len(hz_keys))
        series = [
            ("Raw (η=0)", raw_entry, "#333333"),
            (f"Full-Rank LOO (η={full_eta1['eta']:g})", full_eta1, "#d62728"),
            (f"Scheme 1 ({best_sub['label']})", best_sub, "#2ca02c"),
        ]
        for s_label, item, col in series:
            c_hz = item["clean_reference"].get("horizons") or {}
            s_hz = item["shifted_aggregate"].get("horizons") or {}
            c_line = [
                float(c_hz.get(k, {}).get("common_vocab50_macro_top10", 0.0)) * 100.0
                for k in hz_keys
            ]
            s_line = [
                float(s_hz.get(k, {}).get("mean_common_vocab50_macro_top10", 0.0)) * 100.0
                for k in hz_keys
            ]
            ax2.plot(hz_x, c_line, "-o", linewidth=2.0, color=col, label=f"Clean: {s_label}")
            ax2.plot(hz_x, s_line, ":s", linewidth=1.8, color=col, label=f"Shifted: {s_label}")
        ax2.axhline(chance_50, color="#888888", linestyle="--", linewidth=1.0)
        ax2.set_xticks(hz_x)
        ax2.set_xticklabels(["K0 (0w)", "K1 (+1w)", "K2 (+2w)"])
        ax2.set_ylabel("Common 50-Word Macro Top-10 (%)")
        ax2.set_title("C. Horizons K0/K1/K2: Preserving Clean While Suppressing Shift", fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="best", fontsize=7.8)
    else:
        for rank_k in report.get("grid", {}).get("rank_k_values", ()):
            sub_k = [
                e
                for e in subspace_entries
                if e.get("rank_k") == rank_k and e.get("use_coherence_gate")
            ]
            if sub_k:
                etas_k = [float(e["eta"]) for e in sub_k]
                gaps_k = [
                    float(
                        e["shifted_aggregate"].get(
                            "clean_minus_shifted_vocab50_macro_top10", 0.0
                        )
                    )
                    * 100.0
                    for e in sub_k
                ]
                ax2.plot(etas_k, gaps_k, "-o", linewidth=2.0, label=f"Rank K={rank_k} (Gated)")
        ax2.set_xlabel("Subtraction Weight η")
        ax2.set_ylabel("Net Gap: Clean - Shifted Mean (%)")
        ax2.set_title("C. Net Gap Across η and Subspace Rank K", fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="best", fontsize=8.5)

    # Panel D: Column-Wise Null Standardization Across tau on Full-Vocab (2020 words)
    ax3 = axes[1, 1]
    if cal_entries:
        first_k = cal_entries[0]["rank_k"]
        first_eta = cal_entries[0]["eta"]
        first_gate = cal_entries[0]["use_coherence_gate"]
        anchor_cals = [
            e
            for e in cal_entries
            if e["rank_k"] == first_k
            and abs(float(e["eta"]) - float(first_eta)) <= 1e-6
            and bool(e["use_coherence_gate"]) == bool(first_gate)
        ]
        taus_x = [float(e.get("calibration_tau", 0.0)) for e in anchor_cals]
        fv_clean_y = [
            float(e["clean_reference"].get("full_vocab_macro_top10", 0.0)) * 100.0
            for e in anchor_cals
        ]
        fv_shift_y = [
            float(e["shifted_aggregate"].get("mean_full_vocab_macro_top10", 0.0)) * 100.0
            for e in anchor_cals
        ]
        fv_gap_y = [
            float(
                e["shifted_aggregate"].get(
                    "clean_minus_shifted_full_vocab_macro_top10", 0.0
                )
            )
            * 100.0
            for e in anchor_cals
        ]
        ax3.plot(
            taus_x,
            fv_clean_y,
            "-o",
            linewidth=2.2,
            color="#1f77b4",
            label="FullVocab Clean (0s)",
        )
        ax3.plot(
            taus_x,
            fv_shift_y,
            "-s",
            linewidth=2.2,
            color="#d62728",
            label="FullVocab Shifted Mean (Strict LOO)",
        )
        ax3.plot(
            taus_x,
            fv_gap_y,
            "--^",
            linewidth=2.0,
            color="#2ca02c",
            label="FullVocab Net Gap (Clean - Shift)",
        )
        ax3.axhspan(2.0, 3.5, color="#2ca02c", alpha=0.12, label="Target Shifted Band (2.0%–3.5%)")
        chance_fv = float(raw_entry["clean_reference"].get("full_vocab_chance_top10", 10.0 / 2020.0)) * 100.0
        ax3.axhline(chance_fv, color="#666666", linestyle=":", linewidth=1.1, label=f"Uniform Chance ({chance_fv:.2f}%)")
        for tx, cy, sy in zip(taus_x, fv_clean_y, fv_shift_y):
            ax3.annotate(f"{cy:.1f}%", (tx, cy), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7.5, color="#1f77b4")
            ax3.annotate(f"{sy:.2f}%", (tx, sy), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=7.5, color="#d62728")
        ax3.set_xlabel("Anti-Hubness Column Std Temperature τ  (β=1.0)")
        ax3.set_ylabel("Full-Vocab (2,020 Words) Top-10 (%)")
        ax3.set_title(
            f"D. Column-Wise Null Standardization (K={first_k}, η={first_eta:g}): Eliminating 8.4% Zipf Hubness",
            fontweight="bold",
        )
        ax3.grid(True, linestyle=":", alpha=0.5)
        ax3.legend(loc="best", fontsize=8.0)
    else:
        cond_list = raw_entry.get("conditions", ())
        cond_names = ["Clean (0s)"] + [c["label"] for c in cond_list]
        cond_x = np.arange(len(cond_names))
        profile_series = [
            ("Raw Baseline (η=0)", raw_entry, "#333333", "--"),
            (f"Full-Rank LOO (η={full_eta1['eta']:g})", full_eta1, "#d62728", "-."),
            (best_sub["label"], best_sub, "#2ca02c", "-"),
        ]
        for p_label, item, col, ls in profile_series:
            vals = [float(item["clean_reference"].get("vocab50_macro_top10", 0.0)) * 100.0] + [
                float(c["summary"].get("vocab50_macro_top10", 0.0)) * 100.0
                for c in item.get("conditions", ())
            ]
            ax3.plot(cond_x, vals, marker="o", linestyle=ls, linewidth=2.0, color=col, label=p_label)
        ax3.axhline(chance_50, color="#888888", linestyle=":", linewidth=1.0, label="Chance (20.0%)")
        ax3.set_xticks(cond_x)
        ax3.set_xticklabels(cond_names, rotation=25, ha="right")
        ax3.set_ylabel("50-Word Macro Top-10 (%)")
        ax3.set_title("D. Clean (0s) vs Historical Shifted Conditions (-15s..-120s)", fontweight="bold")
        ax3.grid(True, linestyle=":", alpha=0.5)
        ax3.legend(loc="best", fontsize=8.0)

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    png_path = out_dir / f"subspace_coherence_debiasing_{split}.png"
    pdf_path = out_dir / f"subspace_coherence_debiasing_{split}.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {
        "subspace_coherence_debiasing_png": str(png_path),
        "subspace_coherence_debiasing_pdf": str(pdf_path),
    }


class _CachedEvalDatasetProxy:
    """轻量级评估集代理对象，用于命中磁盘特征缓存时无需解包 16.4GB 原始脑电/脑磁信号即可完成全部指标计算。"""

    def __init__(
        self,
        table,
        embedding_map: dict[str, np.ndarray],
        *,
        window_seconds: float = 1.0,
        window_samples: int = 50,
    ) -> None:
        self.table = table
        self.embedding_map = embedding_map
        self.window_seconds = float(window_seconds)
        self.window_samples = int(window_samples)

    def __len__(self) -> int:
        return int(len(self.table))


def _compact_encoded_dict(encoded: dict) -> dict:
    """仅保留检索排名与子空间去偏所需的张量与元数据，将缓存体积压缩至 ~45MB。"""
    compact = {
        "predictions": encoded["predictions"].detach().cpu().to(dtype=torch.float32),
        "targets": encoded["targets"].detach().cpu().to(dtype=torch.float32),
        "words": list(encoded["words"]),
        "event_ids": list(encoded["event_ids"]),
        "recording_ids": list(encoded.get("recording_ids", ())),
    }
    if "revision_predictions" in encoded and isinstance(
        encoded["revision_predictions"], dict
    ):
        compact["revision_predictions"] = {
            int(d): t.detach().cpu().to(dtype=torch.float32)
            for d, t in encoded["revision_predictions"].items()
        }
    if "revision_availability_masks" in encoded and isinstance(
        encoded["revision_availability_masks"], dict
    ):
        compact["revision_availability_masks"] = {
            int(d): m.detach().cpu().to(dtype=torch.bool)
            for d, m in encoded["revision_availability_masks"].items()
        }
    return compact


def evaluate_cached_pt_column_calibration(
    cache_pt_path: str | Path,
    *,
    eta_values: str | Iterable[float | int | str] | None = (0.8, 1.0, 1.2),
    rank_k_values: str | Iterable[int | str] | None = (8, 12, 16, 24),
    coherence_gate_modes: Iterable[bool] = (True, False),
    calibration_betas: str | Iterable[float | int | str] | None = DEFAULT_CALIBRATION_BETAS,
    calibration_taus: str | Iterable[float | int | str] | None = DEFAULT_CALIBRATION_TAUS,
    center_svd: bool = False,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    output_dir: str | Path | None = None,
    plot: bool = True,
    print_report: bool = True,
) -> dict:
    """直接加载已保存的 subspace_encoded_cache_{split}.pt 文件并在 3~5 秒内完成低秩子空间 + 逐列零分布标准化全量扫描。"""
    pt_path = Path(cache_pt_path)
    if not pt_path.is_file():
        raise FileNotFoundError(f"未找到特征缓存文件：{pt_path}")

    t0 = time.perf_counter()
    cached_bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
    elapsed = time.perf_counter() - t0
    shifts = list(cached_bundle.get("shift_seconds", DEFAULT_HISTORICAL_SHIFTS))
    dataset = str(cached_bundle.get("dataset", "pallier2025"))
    stage = str(cached_bundle.get("stage", "revision"))
    split = str(cached_bundle.get("split", "test"))
    seed = int(cached_bundle.get("seed", 0))

    print(
        f" [Direct Cache Load] Loaded Clean + {len(shifts)} Shifted features from {pt_path} in {elapsed:.2f}s",
        flush=True,
    )
    proxy_ds = _CachedEvalDatasetProxy(
        cached_bundle["eval_table"],
        cached_bundle["embedding_map"],
        window_seconds=float(cached_bundle.get("window_seconds", 1.0)),
        window_samples=int(cached_bundle.get("window_samples", 50)),
    )
    resolved_out = (
        Path(output_dir)
        if output_dir is not None
        else (pt_path.parent.parent if pt_path.parent.name == "exploratory" else pt_path.parent)
    )
    return run_subspace_coherence_debiasing_sweep(
        dataset=dataset,
        stage=stage,
        split=split,
        seed=seed,
        exp_tag=str(cached_bundle.get("exp_tag", "")),
        word_exp_tag=str(cached_bundle.get("word_exp_tag", "")),
        clean_encoded=cached_bundle["clean_encoded"],
        shifted_encoded_by_shift=cached_bundle["shifted_encoded_by_shift"],
        eval_ds=proxy_ds,
        vocabulary=cached_bundle["vocabulary"],
        vocab50=cached_bundle["vocab50"],
        eta_values=eta_values,
        rank_k_values=rank_k_values,
        coherence_gate_modes=coherence_gate_modes,
        calibration_betas=calibration_betas,
        calibration_taus=calibration_taus,
        center_svd=center_svd,
        coherence_low=coherence_low,
        coherence_high=coherence_high,
        gate_min=gate_min,
        gate_max=gate_max,
        output_dir=resolved_out,
        plot=plot,
        print_report=print_report,
    )


def run_standalone_subspace_evaluation(
    *,
    dataset: str | None = None,
    stage: str | None = "revision",
    seed: int = 0,
    split: str = "test",
    subject: str | None = "all",
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    config_path: str | Path | None = None,
    control_config_path: str | Path | None = None,
    cache_pt_path: str | Path | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    window_seconds: float | None = None,
    window_start_offset_seconds: float | None = None,
    shift_seconds: str | Iterable[float | int | str] | None = None,
    eta_values: str | Iterable[float | int | str] | None = None,
    rank_k_values: str | Iterable[int | str] | None = None,
    coherence_gate_mode: str = "both",
    calibration_betas: str | Iterable[float | int | str] | None = None,
    calibration_taus: str | Iterable[float | int | str] | None = None,
    center_svd: bool = False,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    force_reencode: bool = False,
    plot: bool = True,
    show_progress: bool = True,
) -> dict:
    """独立运行单阶段（如 revision 或 context）的方案一评估工作流，自动支持已编码特征缓存秒级复用。"""
    gate_mode_str = str(coherence_gate_mode or "both").strip().lower()
    if gate_mode_str in {"true", "on", "1", "yes"}:
        gate_modes = [True]
    elif gate_mode_str in {"false", "off", "0", "no"}:
        gate_modes = [False]
    else:
        gate_modes = [True, False]

    # 若直接指定了 cache_pt_path，则无需加载训练配置与检查点，直接 3 秒内完成评估
    if cache_pt_path is not None and str(cache_pt_path).strip():
        return evaluate_cached_pt_column_calibration(
            cache_pt_path,
            eta_values=eta_values if eta_values is not None else DEFAULT_SUBSPACE_ETAS,
            rank_k_values=rank_k_values if rank_k_values is not None else DEFAULT_SUBSPACE_RANKS,
            coherence_gate_modes=gate_modes,
            calibration_betas=calibration_betas if calibration_betas is not None else DEFAULT_CALIBRATION_BETAS,
            calibration_taus=calibration_taus if calibration_taus is not None else DEFAULT_CALIBRATION_TAUS,
            center_svd=center_svd,
            coherence_low=coherence_low,
            coherence_high=coherence_high,
            gate_min=gate_min,
            gate_max=gate_max,
            plot=plot,
            print_report=True,
        )

    from braindecoding.data import text_embeddings as text_data
    from braindecoding.experiment import file_sha256
    from braindecoding.models import build_brain_embedding_model
    from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import (
        train as chinese_task,
    )
    from braindecoding.tasks.word_decoding.pallier2025 import (
        train as pallier_task,
    )
    from braindecoding.training.training_runtime import (
        choose_device,
        load_checkpoint,
        set_seed,
    )
    from braindecoding.training.word_encoder_training import (
        encode_loader,
        make_loader,
        resolve_amp_dtype,
    )
    from braindecoding.training.workflow_runner import (
        STAGE_DISPLAY_NAMES,
        _format_shift_label,
        _parse_shift_seconds_list,
        build_workflow_config,
        group_column,
        verify_cache_ready,
        workflow_output_directory,
    )

    cfg, subset_spec = build_workflow_config(
        dataset,
        stage=stage,
        seed=seed,
        subject=subject,
        num_subjects=num_subjects,
        subjects=subjects,
        window_seconds=window_seconds,
        window_start_offset_seconds=window_start_offset_seconds,
        mode="train",
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        config_path=config_path,
    )
    dataset = subset_spec["dataset"]
    stage = cfg["workflow"]["stage"]
    resolved_exp_tag = cfg["workflow"].get("exp_tag", "")
    resolved_word_exp_tag = cfg["workflow"].get("word_exp_tag", resolved_exp_tag)
    tc = cfg["training"]
    seed = int(tc["seed"])
    out = Path(tc["output_dir"])

    # 解析 control_config_path 中的可选默认参数
    ctrl_yaml: dict = {}
    if control_config_path is not None and str(control_config_path).strip():
        resolved_ctrl = resolve_config_path(project_path(control_config_path))
        if resolved_ctrl.is_file():
            loaded = load_yaml_with_extends(resolved_ctrl)
            if isinstance(loaded, dict):
                ctrl_yaml = loaded
    sub_cfg = (
        (ctrl_yaml.get("control") or {}).get("subspace_coherence_debiasing")
        or (ctrl_yaml.get("evaluation") or {}).get("subspace_coherence_debiasing")
        or {}
    )

    if shift_seconds is not None and str(shift_seconds).strip():
        shifts = _parse_shift_seconds_list(shift_seconds)
    elif (ctrl_yaml.get("control") or {}).get("shift_seconds"):
        shifts = _parse_shift_seconds_list(ctrl_yaml["control"]["shift_seconds"])
    else:
        shifts = list(DEFAULT_HISTORICAL_SHIFTS)

    eff_etas = (
        eta_values
        if eta_values is not None
        else sub_cfg.get("eta_values", DEFAULT_SUBSPACE_ETAS)
    )
    eff_ranks = (
        rank_k_values
        if rank_k_values is not None
        else sub_cfg.get("rank_k_values", DEFAULT_SUBSPACE_RANKS)
    )
    eff_betas = (
        calibration_betas
        if calibration_betas is not None
        else sub_cfg.get("calibration_betas", DEFAULT_CALIBRATION_BETAS)
    )
    eff_taus = (
        calibration_taus
        if calibration_taus is not None
        else sub_cfg.get("calibration_taus", DEFAULT_CALIBRATION_TAUS)
    )

    # 定位检查点
    ckpt_file = out / "best.ckpt"
    if not ckpt_file.is_file():
        ckpt_file = out / "best.pt"
    if not ckpt_file.is_file():
        raise FileNotFoundError(
            f"未找到阶段 {stage} ({STAGE_DISPLAY_NAMES.get(stage, stage)}) 的训练检查点：{out / 'best.ckpt'}。"
        )
    ckpt_sha = file_sha256(ckpt_file)

    # 尝试读取同 EXP_TAG 下 Stage 2.1 (context) 的已有 Clean 评估结果作为横向对比参考线
    context_reference_summary: dict | None = None
    if stage == "revision":
        ctx_dir = workflow_output_directory(
            dataset,
            "context",
            subset_spec["subset_tag"],
            seed,
            mode="train",
            exp_tag=resolved_exp_tag,
        )
        ctx_eval_file = ctx_dir / f"evaluation_{split}.json"
        if ctx_eval_file.is_file():
            try:
                ctx_data = json.loads(ctx_eval_file.read_text(encoding="utf-8"))
                ctx_n50 = (
                    (ctx_data.get("metrics") or {})
                    .get("vocabulary_sweep", {})
                    .get("N50", {})
                )
                ctx_fv = (
                    (ctx_data.get("metrics") or {})
                    .get("vocabulary_sweep", {})
                    .get("full_vocab", {})
                )
                if ctx_n50:
                    context_reference_summary = {
                        "stage": "context",
                        "vocab50_macro_top10": float(ctx_n50.get("macro_top10", 0.0)),
                        "vocab50_micro_top10": float(ctx_n50.get("micro_top10", 0.0)),
                        "full_vocab_macro_top10": float(ctx_fv.get("macro_top10", 0.0)),
                        "source_file": str(ctx_eval_file),
                    }
            except Exception:
                pass

    exploratory_dir = out / "exploratory"
    exploratory_dir.mkdir(parents=True, exist_ok=True)
    encoded_cache_path = (
        exploratory_dir / f"subspace_encoded_cache_{split}.pt"
    )

    task = pallier_task if dataset == "pallier2025" else chinese_task
    read_fn = task.dataset_module.load_event_table
    build_fn = task.build_dataset
    selected_subjects = cfg["dataset"]["subjects"]

    # 检查是否可以直接命中已编码特征缓存（无需解包 16.4GB 原始脑信号，0秒直达矩阵去偏！）
    if not force_reencode and encoded_cache_path.is_file():
        try:
            t0 = time.perf_counter()
            cached_bundle = torch.load(
                encoded_cache_path, map_location="cpu", weights_only=False
            )
            cached_shifts = [
                round(float(s), 4) for s in cached_bundle.get("shift_seconds", ())
            ]
            target_shifts = [round(float(s), 4) for s in shifts]
            if (
                isinstance(cached_bundle, dict)
                and cached_bundle.get("checkpoint_sha256") == ckpt_sha
                and cached_shifts == target_shifts
                and "clean_encoded" in cached_bundle
                and "shifted_encoded_by_shift" in cached_bundle
                and "vocabulary" in cached_bundle
                and "vocab50" in cached_bundle
                and "eval_table" in cached_bundle
                and "embedding_map" in cached_bundle
            ):
                elapsed = time.perf_counter() - t0
                print(
                    f" [0s Feature Cache Hit] Loaded pre-encoded Clean + {len(shifts)} Shifted features "
                    f"from {encoded_cache_path} in {elapsed:.2f}s (GPU forward & signal unpack bypassed!)",
                    flush=True,
                )
                proxy_ds = _CachedEvalDatasetProxy(
                    cached_bundle["eval_table"],
                    cached_bundle["embedding_map"],
                    window_seconds=float(cached_bundle.get("window_seconds", 1.0)),
                    window_samples=int(cached_bundle.get("window_samples", 50)),
                )
                return run_subspace_coherence_debiasing_sweep(
                    dataset=dataset,
                    stage=stage,
                    split=split,
                    seed=seed,
                    exp_tag=resolved_exp_tag,
                    word_exp_tag=resolved_word_exp_tag,
                    clean_encoded=cached_bundle["clean_encoded"],
                    shifted_encoded_by_shift=cached_bundle["shifted_encoded_by_shift"],
                    eval_ds=proxy_ds,
                    vocabulary=cached_bundle["vocabulary"],
                    vocab50=cached_bundle["vocab50"],
                    eta_values=eff_etas,
                    rank_k_values=eff_ranks,
                    coherence_gate_modes=gate_modes,
                    calibration_betas=eff_betas,
                    calibration_taus=eff_taus,
                    center_svd=center_svd,
                    coherence_low=coherence_low,
                    coherence_high=coherence_high,
                    gate_min=gate_min,
                    gate_max=gate_max,
                    context_reference_summary=context_reference_summary,
                    output_dir=out,
                    plot=plot,
                    print_report=True,
                )
        except Exception as exc:
            print(
                f" [Feature Cache Notice] 缓存读取未命中或已更新 ({exc})，将执行 GPU 前向特征编码...",
                flush=True,
            )

    # 未命中缓存时：校验信号缓存并执行 1 次 Clean + 6 次 Shifted 的前向编码
    verify_cache_ready(cfg, splits=(split,))
    device = choose_device(tc.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(seed)
    checkpoint = load_checkpoint(ckpt_file, map_location="cpu")
    if not bool(cfg["workflow"].get("window_explicitly_overridden", False)):
        ckpt_cfg_ds = (
            checkpoint.get("config", {}).get("dataset", {})
            if isinstance(checkpoint.get("config"), dict)
            else {}
        )
        ckpt_win = checkpoint.get(
            "window_seconds",
            ckpt_cfg_ds.get("window_seconds")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        ckpt_off = checkpoint.get(
            "window_start_offset_seconds",
            ckpt_cfg_ds.get("window_start_offset_seconds")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        if ckpt_win is not None:
            cfg["dataset"]["window_seconds"] = float(ckpt_win)
            cfg["workflow"]["window_seconds"] = float(ckpt_win)
            sr_hz = float(cfg["workflow"].get("sampling_rate_hz", 50.0))
            cfg["workflow"]["window_samples"] = int(round(float(ckpt_win) * sr_hz))
        if ckpt_off is not None:
            cfg["dataset"]["window_start_offset_seconds"] = float(ckpt_off)
            cfg["workflow"]["window_start_offset_seconds"] = float(ckpt_off)

    train_table = read_fn(
        cfg["cache"]["event_table"],
        split="train",
        subjects=selected_subjects,
        trainable_only=True,
    )
    eval_table = read_fn(
        cfg["cache"]["event_table"],
        split=split,
        subjects=selected_subjects,
        trainable_only=True,
    )
    eval_ds = build_fn(cfg, eval_table)
    grp_col = group_column(cfg)
    eval_bs = int(tc.get("eval_batch_size", tc["batch_size"]))
    eval_loader, _ = make_loader(
        eval_ds,
        tc,
        shuffle=False,
        group_column=grp_col,
        batch_size=eval_bs,
    )

    train_word_col = (
        "normalized_word"
        if "normalized_word" in train_table.columns
        else "标准词"
    )
    vocabulary = sorted(set(train_table[train_word_col].astype(str)))
    vocab50 = list(
        pallier_task.frozen_vocabulary()
        if dataset == "pallier2025"
        else chinese_task.frozen_vocabulary(cfg["cache"]["event_table"])
    )

    model_cfg = checkpoint.get("model_config", cfg["model"])
    ckpt_subjects = list(checkpoint.get("subject_ids", selected_subjects))
    subject_count = max(len(ckpt_subjects), eval_ds.subject_count)

    model = build_brain_embedding_model(
        eval_ds.channel_count,
        eval_ds.channel_positions,
        subject_count,
        model_cfg,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    amp_enabled = bool(tc.get("amp", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(tc)

    clean_encoded = _compact_encoded_dict(
        encode_loader(
            model,
            eval_loader,
            device,
            amp=amp_enabled,
            amp_dtype=amp_dtype,
            show_progress=show_progress,
            progress_desc=f"Encode [{stage} | {split} | Clean 0s]",
        )
    )

    shifted_encoded_by_shift: list[tuple[float, str, str, dict]] = []
    for shift_sec in shifts:
        cond_key, cond_label = _format_shift_label(float(shift_sec))
        shift_ds = build_fn(
            cfg, eval_table, temporal_shift_seconds=float(shift_sec)
        )
        shift_loader, _ = make_loader(
            shift_ds,
            tc,
            shuffle=False,
            group_column=grp_col,
            batch_size=eval_bs,
        )
        shift_enc = _compact_encoded_dict(
            encode_loader(
                model,
                shift_loader,
                device,
                amp=amp_enabled,
                amp_dtype=amp_dtype,
                show_progress=show_progress,
                progress_desc=f"Encode [{stage} | {split} | {cond_label}]",
            )
        )
        shifted_encoded_by_shift.append(
            (float(shift_sec), cond_key, cond_label, shift_enc)
        )
        del shift_ds, shift_loader

    # 保存紧凑编码缓存至 exploratory/subspace_encoded_cache_{split}.pt，供后续秒级重跑与调参复用
    try:
        eval_table_minimal = eval_ds.table[["event_id", "subject_id"]].copy()
        all_needed_words = set(vocabulary) | set(vocab50) | set(clean_encoded["words"])
        minimal_emb_map = {
            w: np.asarray(eval_ds.embedding_map[w], dtype=np.float32)
            for w in all_needed_words
            if w in eval_ds.embedding_map
        }
        tmp_cache = encoded_cache_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "version": "subspace_encoded_cache_v1",
                "checkpoint_sha256": ckpt_sha,
                "dataset": dataset,
                "stage": stage,
                "split": split,
                "seed": int(seed),
                "exp_tag": resolved_exp_tag,
                "word_exp_tag": resolved_word_exp_tag,
                "window_seconds": float(getattr(eval_ds, "window_seconds", 1.0)),
                "window_samples": int(getattr(eval_ds, "window_samples", 50)),
                "shift_seconds": [float(s) for s in shifts],
                "vocabulary": vocabulary,
                "vocab50": vocab50,
                "eval_table": eval_table_minimal,
                "embedding_map": minimal_emb_map,
                "clean_encoded": clean_encoded,
                "shifted_encoded_by_shift": shifted_encoded_by_shift,
            },
            tmp_cache,
        )
        tmp_cache.replace(encoded_cache_path)
        print(
            f" [Feature Cache Saved] Cached Clean + {len(shifts)} Shifted features to {encoded_cache_path}",
            flush=True,
        )
    except Exception:
        pass

    report = run_subspace_coherence_debiasing_sweep(
        dataset=dataset,
        stage=stage,
        split=split,
        seed=seed,
        exp_tag=resolved_exp_tag,
        word_exp_tag=resolved_word_exp_tag,
        clean_encoded=clean_encoded,
        shifted_encoded_by_shift=shifted_encoded_by_shift,
        eval_ds=eval_ds,
        vocabulary=vocabulary,
        vocab50=vocab50,
        eta_values=eff_etas,
        rank_k_values=eff_ranks,
        coherence_gate_modes=gate_modes,
        calibration_betas=eff_betas,
        calibration_taus=eff_taus,
        center_svd=center_svd,
        coherence_low=coherence_low,
        coherence_high=coherence_high,
        gate_min=gate_min,
        gate_max=gate_max,
        context_reference_summary=context_reference_summary,
        output_dir=out,
        plot=plot,
        print_report=True,
    )

    del model, eval_ds, eval_loader, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """构建方案一独立命令行入口解析器。"""
    parser = argparse.ArgumentParser(
        description="Scheme 1 Exploratory Evaluation: Subspace-Projected LOO + Column-Wise Null Standardization"
    )
    parser._negative_number_matcher = re.compile(
        r"^-\d+(?:\.\d+)?(?:,\s*[+-]?\d+(?:\.\d+)?)*$"
    )
    parser.add_argument("--dataset", default=None, help="数据集名称: pallier2025 或 chineseeeg2_littleprince")
    parser.add_argument("--stage", default="revision", help="评估阶段: revision (默认) / context / word / all")
    parser.add_argument("--config", default=None, help="可选训练阶段 YAML 配置文件路径")
    parser.add_argument(
        "--control-config",
        default="configs/controls/subspace_coherence_debiasing/subspace_coherence_debiasing_control.yaml",
        help="方案一解耦负对照 YAML 配置文件路径",
    )
    parser.add_argument(
        "--cache-pt",
        default=None,
        help="可选：直接指定已保存的 subspace_encoded_cache_{split}.pt 路径，无需训练配置与信号缓存即可在 5 秒内完成全量扫描",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子 (默认 0)")
    parser.add_argument("--split", default="test", choices=("val", "test"), help="评估集划分 (默认 test)")
    parser.add_argument("--subject", default="all", help="受试者子集 (默认 all)")
    parser.add_argument("--num-subjects", type=int, default=None)
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--exp-tag", default=None, help="实验目录标签 EXP_TAG")
    parser.add_argument("--word-exp-tag", default=None, help="阶段 1 词模型实验目录标签 WORD_EXP_TAG")
    parser.add_argument("--window-seconds", type=float, default=None, help="单字时间窗秒数")
    parser.add_argument("--window-start-offset-seconds", type=float, default=None)
    parser.add_argument(
        "--shift-seconds",
        default="-15,-30,-45,-60,-90,-120",
        help="纯历史因果时间错位秒数列表 (默认 -15,-30,-45,-60,-90,-120)",
    )
    parser.add_argument(
        "--etas",
        default="0.8,1.0,1.2",
        help="低秩子空间内共模扣除权重网格 (默认 0.8,1.0,1.2)",
    )
    parser.add_argument(
        "--rank-ks",
        default="8,12,16,24",
        help="低秩时间偏置子空间维度 K 网格 (默认 8,12,16,24)",
    )
    parser.add_argument(
        "--coherence-gate",
        default="both",
        choices=("both", "true", "false", "on", "off"),
        help="是否启用跨历史窗口相干度门控: both (同时对比开启与关闭) / true / false",
    )
    parser.add_argument(
        "--betas",
        default="1.0",
        help="全词表逐列零分布静态中心度偏置扣除系数 beta 网格 (默认 1.0)",
    )
    parser.add_argument(
        "--taus",
        default="0.0,0.3,0.5,0.7,1.0",
        help="全词表逐列标准差反中心度温度指数 tau 网格 (默认 0.0,0.3,0.5,0.7,1.0)",
    )
    parser.add_argument(
        "--center-svd",
        action="store_true",
        help="若指定则在 SVD 前减去全局样本均值（默认 False，即使用不中心化 SVD 将全局静态偏置 mu_0 收入第 1 主奇异向量）",
    )
    parser.add_argument("--coherence-low", type=float, default=0.35, help="相干度门控下界阈值 (默认 0.35)")
    parser.add_argument("--coherence-high", type=float, default=0.65, help="相干度门控上界阈值 (默认 0.65)")
    parser.add_argument("--gate-min", type=float, default=0.2, help="门控最小衰减系数 (默认 0.2)")
    parser.add_argument("--gate-max", type=float, default=1.2, help="门控最大增强系数 (默认 1.2)")
    parser.add_argument("--force-reencode", action="store_true", help="强制重新运行 GPU 前向编码而不使用已有特征缓存")
    parser.add_argument("--no-plot", action="store_true", help="禁用图表生成")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.cache_pt is not None and str(args.cache_pt).strip():
        run_standalone_subspace_evaluation(
            cache_pt_path=args.cache_pt,
            eta_values=args.etas,
            rank_k_values=args.rank_ks,
            coherence_gate_mode=args.coherence_gate,
            calibration_betas=args.betas,
            calibration_taus=args.taus,
            center_svd=bool(args.center_svd),
            coherence_low=args.coherence_low,
            coherence_high=args.coherence_high,
            gate_min=args.gate_min,
            gate_max=args.gate_max,
            plot=not bool(args.no_plot),
        )
        return 0

    stage_raw = str(args.stage or "revision").strip().lower()
    if stage_raw == "all":
        stages = ["context", "revision"]
    else:
        stages = [s.strip() for s in stage_raw.split(",") if s.strip()]

    for st in stages:
        run_standalone_subspace_evaluation(
            dataset=args.dataset,
            stage=st,
            seed=args.seed,
            split=args.split,
            subject=args.subject,
            num_subjects=args.num_subjects,
            subjects=args.subjects,
            config_path=args.config if len(stages) == 1 else None,
            control_config_path=args.control_config,
            exp_tag=args.exp_tag,
            word_exp_tag=args.word_exp_tag,
            window_seconds=args.window_seconds,
            window_start_offset_seconds=args.window_start_offset_seconds,
            shift_seconds=args.shift_seconds,
            eta_values=args.etas,
            rank_k_values=args.rank_ks,
            coherence_gate_mode=args.coherence_gate,
            calibration_betas=args.betas,
            calibration_taus=args.taus,
            center_svd=bool(args.center_svd),
            coherence_low=args.coherence_low,
            coherence_high=args.coherence_high,
            gate_min=args.gate_min,
            gate_max=args.gate_max,
            force_reencode=bool(args.force_reencode),
            plot=not bool(args.no_plot),
            show_progress=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
