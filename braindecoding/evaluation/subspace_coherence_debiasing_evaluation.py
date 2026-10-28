"""方案一（零重训 · 测试期即插即用）：低秩共模子空间投影 + 相干度门控去偏（Subspace-Projected + Coherence-Gated LOO）探索性评估模块。

本模块完全独立于主干训练与主评估流水线，用于在不重新训练模型的前提下，
利用测试期 M=6 个纯历史因果时间错位窗口（-15s, -30s, -45s, -60s, -90s, -120s）提取跨窗口共有的
低秩时间结构偏置子空间 P_bias = U_K U_K^T，并结合严格留一法基底 U_{K,LOO} 与跨历史窗口相干度门控执行留一法（LOO）共模去偏：

1. 全样本错位共模均值（完全不接触 y_clean 与测试标签）：
   y_shift_mean_all = (1 / M) * sum_{m=1..M} y_shifts[:, m, :]
2. 中心化低秩时间偏置子空间 SVD 提取（秩 K in {8, 12, 16, 24}）：
   Clean (0s)   : _, _, Vh = svd(y_shift_mean_all - mean(y_shift_mean_all, dim=0)); U_K = Vh[:K, :].T
   Shifted (dk) : 严格仅使用其余 M-1=5 个错位条件的均值求中心化 SVD 基 U_{K,LOO}（完全不接触第 k 个错位条件自身）
3. 跨历史窗口相干度门控（SimpleB2T 跨观测相消对偶，可选 Gate=Off / Gate=On）：
   coherence_i = ||shift_mean_i||_2
   gate_i = clamp((coherence_i - 0.35) / (0.65 - 0.35), min=0.2, max=1.2)
4. Clean 与 Shifted 严格留一法低秩投影扣除：
   - Clean (0s)   : 仅在低秩子空间 U_K 内扣除 (eta * gate_clean) * (y_shift_mean_all @ U_K @ U_K^T)
   - Shifted (dk) : 仅使用其余 M-1 个错位条件均值在严格留一法基底 U_{K,LOO} 内的投影扣除，100% 保护正交的 (D - K) 个语义维度。
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
DEFAULT_HISTORICAL_SHIFTS: tuple[float, ...] = (
    -15.0,
    -30.0,
    -45.0,
    -60.0,
    -90.0,
    -120.0,
)


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


def subspace_coherence_loo_debiasing(
    y_clean: torch.Tensor,
    y_shifts: torch.Tensor,
    eta: float = 1.0,
    rank_k: int | None = 12,
    use_coherence_gate: bool = False,
    *,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    center_svd: bool = True,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
    precomputed_vh: torch.Tensor | None = None,
    precomputed_vh_loo: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """仅在跨历史窗口共有的低秩时间偏置子空间内执行严格 LOO 共模扣除，
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
        是否启用跨历史窗口相干度门控（SimpleB2T Eq.7 相消对偶；默认 False 即纯子空间投影 Gate=Off）。
    center_svd:
        是否对 y_shift_mean 做样本中心化后再求 SVD（默认 True：标准中心化 SVD，对应有效秩 EffRank = 1.7 ~ 2.2）。
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

    # 2. 对错位共模均值做中心化 SVD，提取前 rank_k 个主奇异向量 U_k [D, rank_k]
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

    # 5. Shifted 严格留一法 (Strict LOO) 公平评估:
    #    第 k 个错位条件严格只用其余 M-1=5 个错位条件的均值求 SVD 基底 U_{k,LOO} 并扣除投影（完全不碰第 k 个错位条件自身）
    sum_all_shifts = shifts_f.sum(dim=1)  # [N, D]
    loo_denom = float(num_shifts - 1)
    y_shifts_debiased_list: list[torch.Tensor] = []
    for k in range(num_shifts):
        y_shift_loo_mean = (sum_all_shifts - shifts_f[:, k, :]) / loo_denom  # [N, D]
        if (
            eff_k is not None
            and precomputed_vh_loo is not None
            and k < len(precomputed_vh_loo)
        ):
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


def analyze_shift_subspace_and_coherence(
    y_shifts: torch.Tensor,
    *,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    center_svd: bool = True,
    normalize_inputs: bool = True,
    compute_loo_svd: bool = True,
) -> tuple[dict, torch.Tensor]:
    """对错位特征矩阵 [N, M, D] 计算中心化 SVD 奇异值能量谱与跨历史窗口相干度门控统计量。

    返回 ``(统计信息字典, Vh 右奇异向量矩阵)``，其中 ``Vh`` 可直接复用于同一视野下的多组 (rank_k, eta) 扫描。
    当 ``compute_loo_svd=True`` 时，还会在 ``diagnostics["_vh_loo_list"]`` 中预存 M 个严格留一法 SVD 基底 U_{K,LOO}。
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
            svd_loo_in = (
                y_loo - y_loo.mean(dim=0, keepdim=True) if center_svd else y_loo
            )
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
    use_coherence_gate: bool = False,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    center_svd: bool = True,
    normalize_inputs: bool = True,
    precomputed_vh_by_key: dict[str, torch.Tensor] | None = None,
    precomputed_vh_loo_by_key: dict[str, list[torch.Tensor]] | None = None,
) -> tuple[dict, dict[str, dict]]:
    """对 encode_loader 输出的完整字典（含 Stage 2.2 的 K0/K1/K2 修订视野预测）执行低秩子空间 + 严格留一法基底 LOO 去偏。"""
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


def _build_fast_summary_evaluator(
    eval_ds,
    vocabulary: list[str],
    vocab50: list[str],
    *,
    stage: str,
) -> Callable[[dict, str, str, float | None, dict | None], dict]:
    """构建零冗余的快速特征空间余弦检索指标提取器（包含 50w Macro Top-10、50w Macro Top-1 及全词表指标）。"""
    from braindecoding.evaluation.retrieval_metrics import (
        fixed_vocabulary_retrieval_metrics,
        retrieval_ranks,
        summarize_retrieval,
    )
    from braindecoding.evaluation.word_retrieval_evaluation import (
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
        revision = stage in ("revision", "prefix_revision") and "revision_predictions" in encoded
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
                    "common_vocab50_micro_top1": float(
                        s50_common.get("micro_recall_at_1", 0.0)
                    ),
                    "common_vocab50_mrr": float(
                        s50_common.get("mean_reciprocal_rank", 0.0)
                    ),
                    "common_vocab50_median_rank": float(
                        s50_common.get("median_rank", 25.0)
                    ),
                    "all_vocab50_macro_top10": float(
                        s50_all.get("macro_recall_at_10", 0.0)
                    ),
                    "common_full_vocab_macro_top10": float(
                        fv_common.get("word_macro_recall_at_10", 0.0)
                    ),
                    "common_full_vocab_word_macro_top10": float(
                        fv_common.get("word_macro_recall_at_10", 0.0)
                    ),
                    "common_full_vocab_subject_macro_top10": float(
                        fv_common.get("subject_equal_macro_recall_at_10", 0.0)
                    ),
                    "common_full_vocab_micro_top10": float(
                        fv_common.get("micro_recall_at_10", 0.0)
                    ),
                    "common_full_vocab_macro_top1": float(
                        fv_common.get("word_macro_recall_at_1", 0.0)
                    ),
                    "common_full_vocab_micro_top1": float(
                        fv_common.get("micro_recall_at_1", 0.0)
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
                full_block_for_primary.get("word_macro_recall_at_10", 0.0)
            ),
            "word_macro_top10": float(
                full_block_for_primary.get("word_macro_recall_at_10", 0.0)
            ),
            "subject_macro_top10": float(
                full_block_for_primary.get(
                    "subject_equal_macro_recall_at_10", 0.0
                )
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
    baseline_raw_entry: dict | None = None,
    baseline_full_loo_entry: dict | None = None,
) -> dict:
    """构建单个特征空间 Subspace LOO 去偏配置的完整指标条目（含 50w / Full-Vocab Macro Top-10 与 Top-1 及 K0/K1/K2 视野拆解）。"""
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
            "vocab50_macro_top1": float(
                clean_summary.get("vocab50_macro_top1", 0.0)
                - ref_clean.get("vocab50_macro_top1", 0.0)
            ),
            "shifted_mean_vocab50_macro_top1": float(
                shifted_aggregate.get("mean_vocab50_macro_top1", 0.0)
                - ref_agg.get("mean_vocab50_macro_top1", 0.0)
            ),
            "clean_minus_shifted_vocab50_macro_top1": float(
                shifted_aggregate.get("clean_minus_shifted_vocab50_macro_top1", 0.0)
                - ref_agg.get("clean_minus_shifted_vocab50_macro_top1", 0.0)
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
            "full_vocab_word_macro_top10": float(
                clean_summary.get("full_vocab_word_macro_top10", 0.0)
                - ref_clean.get("full_vocab_word_macro_top10", 0.0)
            ),
            "full_vocab_macro_top1": float(
                clean_summary.get("full_vocab_macro_top1", 0.0)
                - ref_clean.get("full_vocab_macro_top1", 0.0)
            ),
            "shifted_mean_full_vocab_macro_top1": float(
                shifted_aggregate.get("mean_full_vocab_macro_top1", 0.0)
                - ref_agg.get("mean_full_vocab_macro_top1", 0.0)
            ),
            "clean_minus_shifted_full_vocab_macro_top1": float(
                shifted_aggregate.get("clean_minus_shifted_full_vocab_macro_top1", 0.0)
                - ref_agg.get("clean_minus_shifted_full_vocab_macro_top1", 0.0)
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
    coherence_gate_modes: Iterable[bool] = (False, True),
    full_loo_etas: Iterable[float] = DEFAULT_FULL_LOO_ETAS,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    center_svd: bool = True,
    normalize_inputs: bool = True,
    context_reference_summary: dict | None = None,
    output_dir: str | Path | None = None,
    plot: bool = True,
    print_report: bool = True,
    verbose: bool = False,
) -> dict:
    """在单次编码特征上运行方案一（中心化 SVD + 严格留一法基底 U_{K,LOO} + 相干度门控 LOO）完整网格扫描、诊断分析、落盘与作图。"""
    from braindecoding.evaluation.word_retrieval_evaluation import (
        _summarize_shifted_conditions,
    )

    etas = parse_eta_values(eta_values, default=DEFAULT_SUBSPACE_ETAS)
    ranks = parse_rank_k_values(rank_k_values, default=DEFAULT_SUBSPACE_RANKS)
    full_etas = parse_eta_values(full_loo_etas, default=DEFAULT_FULL_LOO_ETAS)
    gate_modes = list(dict.fromkeys(bool(m) for m in coherence_gate_modes))
    if not gate_modes:
        gate_modes = [False, True]

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

    # 1. 预计算各视野的中心化 SVD 奇异向量矩阵 Vh、严格留一法基底 Vh_loo 与相干度诊断信息
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

    # 4. 扫描方案一（纯特征空间 Subspace LOO）：低秩共模子空间投影 + 严格留一法基底 U_{K,LOO} (rank_k x eta x use_coherence_gate)
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

    svd_formula = (
        "_, _, Vh = svd(y_shift_mean_all - mean(y_shift_mean_all, dim=0)); U_K = Vh[:K, :].T"
        if center_svd
        else "_, _, Vh = svd(y_shift_mean_all); U_K = Vh[:K, :].T"
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
            "subspace_svd": svd_formula,
            "coherence_gate": f"gate_i = clamp((||y_shift_mean_i||_2 - {coherence_low}) / ({coherence_high} - {coherence_low}), min={gate_min}, max={gate_max})",
            "clean_debiasing": "y_clean_debiased = normalize(y_clean - (eta * gate_clean) * (y_shift_mean_all @ U_K @ U_K.T))",
            "shifted_loo_debiasing": "y_k_debiased = normalize(y_shift_k - (eta * gate_k) * (y_shift_loo_mean_k @ U_{K,LOO} @ U_{K,LOO}.T))",
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
        },
        "subspace_and_coherence_diagnostics": subspace_diagnostics,
        "context_stage_reference": context_reference_summary,
        "raw_baseline": raw_entry,
        "full_rank_loo_baselines": full_loo_entries,
        "subspace_sweep": subspace_entries,
        "best_by_net_gap_vocab50": best_by_gap50,
        "best_by_clean_preservation_vocab50": best_by_clean_preserving,
        "figures": {},
    }

    if print_report:
        print_subspace_coherence_debiasing_report(
            report_payload,
            verbose=verbose,
            output_dir=output_dir,
        )

    if output_dir is not None:
        from braindecoding.evaluation.evaluation_exports import (
            export_subspace_coherence_debiasing_tables_and_markdown,
        )

        out_path = Path(output_dir)
        exploratory_dir = out_path / "exploratory"
        exploratory_dir.mkdir(parents=True, exist_ok=True)
        report_payload["exported_files"] = (
            export_subspace_coherence_debiasing_tables_and_markdown(
                report_payload, out_path
            )
        )
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


from braindecoding.evaluation.evaluation_figures import (
    plot_subspace_coherence_debiasing_figures,
)
from braindecoding.evaluation.evaluation_reporting import (
    format_subspace_coherence_debiasing_report,
    print_subspace_coherence_debiasing_report,
)


def save_subspace_encoded_cache(
    output_dir: str | Path,
    *,
    split: str,
    checkpoint_sha256: str,
    dataset: str,
    stage: str,
    seed: int,
    exp_tag: str,
    word_exp_tag: str,
    eval_ds,
    shifts: Sequence[float] | None = None,
    shift_seconds: Sequence[float] | None = None,
    window_seconds: float | None = None,
    window_samples: int | None = None,
    vocabulary: Sequence[str],
    vocab50: Sequence[str],
    clean_encoded: dict,
    shifted_encoded_by_shift: Sequence[tuple[float, str, str, dict]],
) -> Path | None:
    """将单次编码的 Clean + M 个 Shifted 特征压缩保存至 exploratory/subspace_encoded_cache_{split}.pt (~45MB)。"""
    try:
        out_path = Path(output_dir)
        exploratory_dir = out_path / "exploratory"
        exploratory_dir.mkdir(parents=True, exist_ok=True)
        encoded_cache_path = (
            exploratory_dir / f"subspace_encoded_cache_{split}.pt"
        )
        compact_clean = _compact_encoded_dict(clean_encoded)
        compact_shifts = [
            (float(sec), str(k), str(lbl), _compact_encoded_dict(enc))
            for sec, k, lbl, enc in shifted_encoded_by_shift
        ]
        resolved_shifts = (
            shifts
            if shifts is not None
            else (
                shift_seconds
                if shift_seconds is not None
                else [sec for sec, _, _, _ in shifted_encoded_by_shift]
            )
        )
        resolved_win_sec = (
            float(window_seconds)
            if window_seconds is not None
            else float(getattr(eval_ds, "window_seconds", 1.0))
        )
        resolved_win_samp = (
            int(window_samples)
            if window_samples is not None
            else int(getattr(eval_ds, "window_samples", 50))
        )
        eval_table_minimal = eval_ds.table[["event_id", "subject_id"]].copy()
        all_needed_words = (
            set(str(w) for w in vocabulary)
            | set(str(w) for w in vocab50)
            | set(str(w) for w in compact_clean["words"])
        )
        minimal_emb_map = {
            w: np.asarray(eval_ds.embedding_map[w], dtype=np.float32)
            for w in all_needed_words
            if w in eval_ds.embedding_map
        }
        tmp_cache = encoded_cache_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "version": "subspace_encoded_cache_v1",
                "checkpoint_sha256": str(checkpoint_sha256),
                "dataset": str(dataset),
                "stage": str(stage),
                "split": str(split),
                "seed": int(seed),
                "exp_tag": str(exp_tag),
                "word_exp_tag": str(word_exp_tag),
                "window_seconds": resolved_win_sec,
                "window_samples": resolved_win_samp,
                "shift_seconds": [float(s) for s in resolved_shifts],
                "vocabulary": [str(w) for w in vocabulary],
                "vocab50": [str(w) for w in vocab50],
                "eval_table": eval_table_minimal,
                "embedding_map": minimal_emb_map,
                "clean_encoded": compact_clean,
                "shifted_encoded_by_shift": compact_shifts,
            },
            tmp_cache,
        )
        tmp_cache.replace(encoded_cache_path)
        return encoded_cache_path
    except Exception:
        return None



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


def evaluate_cached_pt_common_mode_subtraction(
    cache_pt_path: str | Path,
    *,
    eta_values: str | Iterable[float | int | str] | None = (0.5, 0.8, 1.0),
    output_dir: str | Path | None = None,
    plot: bool = False,
    print_report: bool = False,
    verbose: bool = False,
) -> dict:
    """直接基于 subspace_encoded_cache_{split}.pt 在 1 秒内重算并更新留一法共模扣除（Common-Mode Subtraction）JSON 与 CSV 表格。"""
    from braindecoding.evaluation.common_mode_subtraction_evaluation import (
        run_common_mode_subtraction_sweep,
    )
    from braindecoding.evaluation.word_retrieval_evaluation import (
        _summarize_shifted_conditions,
    )

    pt_path = Path(cache_pt_path)
    if not pt_path.is_file():
        raise FileNotFoundError(f"未找到特征缓存文件：{pt_path}")

    cached_bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
    dataset = str(cached_bundle.get("dataset", "pallier2025"))
    stage = str(cached_bundle.get("stage", "revision"))
    split = str(cached_bundle.get("split", "test"))
    seed = int(cached_bundle.get("seed", 0))
    proxy_ds = _CachedEvalDatasetProxy(
        cached_bundle["eval_table"],
        cached_bundle["embedding_map"],
        window_seconds=float(cached_bundle.get("window_seconds", 1.0)),
        window_samples=int(cached_bundle.get("window_samples", 50)),
    )
    resolved_out = (
        Path(output_dir)
        if output_dir is not None
        else (
            pt_path.parent.parent
            if pt_path.parent.name == "exploratory"
            else pt_path.parent
        )
    )
    fast_eval = _build_fast_summary_evaluator(
        proxy_ds,
        cached_bundle["vocabulary"],
        cached_bundle["vocab50"],
        stage=stage,
    )
    return run_common_mode_subtraction_sweep(
        dataset=dataset,
        stage=stage,
        split=split,
        seed=seed,
        exp_tag=str(cached_bundle.get("exp_tag", "")),
        word_exp_tag=str(cached_bundle.get("word_exp_tag", "")),
        clean_encoded=cached_bundle["clean_encoded"],
        shifted_encoded_by_shift=cached_bundle["shifted_encoded_by_shift"],
        eval_ds=proxy_ds,
        shifted_ds_by_shift={},
        vocabulary=cached_bundle["vocabulary"],
        vocab50=cached_bundle["vocab50"],
        cfg={},
        eta_values=eta_values if eta_values is not None else (0.5, 0.8, 1.0),
        compute_metrics_fn=lambda enc, *_args, **_kw: enc,
        extract_summary_row_fn=lambda key, label, sec, enc, *, clean_summary=None: fast_eval(
            enc, key, label, sec, clean_summary=clean_summary
        ),
        summarize_shifted_fn=_summarize_shifted_conditions,
        output_dir=resolved_out,
        plot=plot,
        print_report=print_report,
        verbose=verbose,
    )


def evaluate_cached_pt_subspace_loo(
    cache_pt_path: str | Path,
    *,
    eta_values: str | Iterable[float | int | str] | None = (0.8, 1.0, 1.2),
    rank_k_values: str | Iterable[int | str] | None = (8, 12, 16, 24),
    coherence_gate_modes: Iterable[bool] = (False, True),
    center_svd: bool = True,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    output_dir: str | Path | None = None,
    plot: bool = True,
    print_report: bool = True,
    verbose: bool = False,
    update_common_mode_subtraction: bool = True,
) -> dict:
    """直接加载已保存的 subspace_encoded_cache_{split}.pt 文件并在 3~5 秒内完成中心化 SVD + 严格留一法基底 Subspace LOO 全量扫描。"""
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
        else (
            pt_path.parent.parent
            if pt_path.parent.name == "exploratory"
            else pt_path.parent
        )
    )
    if update_common_mode_subtraction:
        try:
            from braindecoding.evaluation.common_mode_subtraction_evaluation import (
                run_common_mode_subtraction_sweep,
            )
            from braindecoding.evaluation.word_retrieval_evaluation import (
                _summarize_shifted_conditions,
            )

            fast_eval = _build_fast_summary_evaluator(
                proxy_ds,
                cached_bundle["vocabulary"],
                cached_bundle["vocab50"],
                stage=stage,
            )
            run_common_mode_subtraction_sweep(
                dataset=dataset,
                stage=stage,
                split=split,
                seed=seed,
                exp_tag=str(cached_bundle.get("exp_tag", "")),
                word_exp_tag=str(cached_bundle.get("word_exp_tag", "")),
                clean_encoded=cached_bundle["clean_encoded"],
                shifted_encoded_by_shift=cached_bundle["shifted_encoded_by_shift"],
                eval_ds=proxy_ds,
                shifted_ds_by_shift={},
                vocabulary=cached_bundle["vocabulary"],
                vocab50=cached_bundle["vocab50"],
                cfg={},
                eta_values=(0.5, 0.8, 1.0),
                compute_metrics_fn=lambda enc, *_args, **_kw: enc,
                extract_summary_row_fn=lambda key, label, sec, enc, *, clean_summary=None: fast_eval(
                    enc, key, label, sec, clean_summary=clean_summary
                ),
                summarize_shifted_fn=_summarize_shifted_conditions,
                output_dir=resolved_out,
                plot=plot,
                print_report=False,
                verbose=verbose,
            )
        except Exception:
            pass
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
        center_svd=center_svd,
        coherence_low=coherence_low,
        coherence_high=coherence_high,
        gate_min=gate_min,
        gate_max=gate_max,
        output_dir=resolved_out,
        plot=plot,
        print_report=print_report,
        verbose=verbose,
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
    center_svd: bool = True,
    coherence_low: float = 0.35,
    coherence_high: float = 0.65,
    gate_min: float = 0.2,
    gate_max: float = 1.2,
    force_reencode: bool = False,
    plot: bool = True,
    show_progress: bool = True,
    verbose: bool = False,
) -> dict:
    """独立运行单阶段（如 revision 或 context）的方案一评估工作流，自动支持已编码特征缓存秒级复用。"""
    gate_mode_str = str(coherence_gate_mode or "both").strip().lower()
    if gate_mode_str in {"true", "on", "1", "yes"}:
        gate_modes = [True]
    elif gate_mode_str in {"false", "off", "0", "no"}:
        gate_modes = [False]
    else:
        gate_modes = [False, True]

    # 若直接指定了 cache_pt_path，则无需加载训练配置与检查点，直接 3 秒内完成评估
    if cache_pt_path is not None and str(cache_pt_path).strip():
        return evaluate_cached_pt_subspace_loo(
            cache_pt_path,
            eta_values=(
                eta_values if eta_values is not None else DEFAULT_SUBSPACE_ETAS
            ),
            rank_k_values=(
                rank_k_values
                if rank_k_values is not None
                else DEFAULT_SUBSPACE_RANKS
            ),
            coherence_gate_modes=gate_modes,
            center_svd=center_svd,
            coherence_low=coherence_low,
            coherence_high=coherence_high,
            gate_min=gate_min,
            gate_max=gate_max,
            plot=plot,
            print_report=True,
            verbose=verbose,
        )

    from braindecoding.evaluation.word_retrieval_evaluation import (
        STAGE_DISPLAY_NAMES,
        _format_shift_label,
        _parse_shift_seconds_list,
    )
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
    if stage in ("revision", "prefix_revision"):
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
                        "vocab50_macro_top1": float(ctx_n50.get("macro_top1", 0.0)),
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
                    center_svd=center_svd,
                    coherence_low=coherence_low,
                    coherence_high=coherence_high,
                    gate_min=gate_min,
                    gate_max=gate_max,
                    context_reference_summary=context_reference_summary,
                    output_dir=out,
                    plot=plot,
                    print_report=True,
                    verbose=verbose,
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
    saved_cache = save_subspace_encoded_cache(
        out,
        split=split,
        checkpoint_sha256=ckpt_sha,
        dataset=dataset,
        stage=stage,
        seed=seed,
        exp_tag=resolved_exp_tag,
        word_exp_tag=resolved_word_exp_tag,
        eval_ds=eval_ds,
        shifts=shifts,
        vocabulary=vocabulary,
        vocab50=vocab50,
        clean_encoded=clean_encoded,
        shifted_encoded_by_shift=shifted_encoded_by_shift,
    )
    if saved_cache is not None:
        print(
            f" [Feature Cache Saved] Cached Clean + {len(shifts)} Shifted features to {saved_cache}",
            flush=True,
        )

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
        center_svd=center_svd,
        coherence_low=coherence_low,
        coherence_high=coherence_high,
        gate_min=gate_min,
        gate_max=gate_max,
        context_reference_summary=context_reference_summary,
        output_dir=out,
        plot=plot,
        print_report=True,
        verbose=verbose,
    )

    del model, eval_ds, eval_loader, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """构建方案一独立命令行入口解析器。"""
    parser = argparse.ArgumentParser(
        description="Scheme 1 Exploratory Evaluation: Subspace-Projected + Coherence-Gated LOO Debiasing"
    )
    parser._negative_number_matcher = re.compile(
        r"^-\d+(?:\.\d+)?(?:,\s*[+-]?\d+(?:\.\d+)?)*$"
    )
    parser.add_argument("--dataset", default=None, help="数据集名称: pallier2025 或 chineseeeg2_littleprince")
    parser.add_argument("--stage", default="revision", help="评估阶段: revision (默认) / prefix_revision / context / word / all")
    parser.add_argument("--config", default=None, help="可选训练阶段 YAML 配置文件路径")
    parser.add_argument(
        "--control-config",
        default="configs/controls/subspace_coherence_debiasing/subspace_coherence_debiasing_control.yaml",
        help="方案一解耦负对照 YAML 配置文件路径",
    )
    parser.add_argument(
        "--cache-pt",
        default=None,
        help="可选：直接指定已保存的 subspace_encoded_cache_{split}.pt 路径，无需训练配置与信号缓存即可在 3 秒内完成全量扫描",
    )
    parser.add_argument("--seed", "--seeds", dest="seed", default="0", help="随机种子（支持单种子 0 或逗号分隔列表 0,1,2）")
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
        help="是否启用跨历史窗口相干度门控: both (同时对比 Gate=Off 与 Gate=On) / false (仅 Gate=Off) / true (仅 Gate=On)",
    )
    parser.add_argument(
        "--uncentered-svd",
        action="store_true",
        help="若指定则不对 y_shift_mean 减去样本均值（默认 False，即使用标准中心化 SVD）",
    )
    parser.add_argument("--coherence-low", type=float, default=0.35, help="相干度门控下界阈值 (默认 0.35)")
    parser.add_argument("--coherence-high", type=float, default=0.65, help="相干度门控上界阈值 (默认 0.65)")
    parser.add_argument("--gate-min", type=float, default=0.2, help="门控最小衰减系数 (默认 0.2)")
    parser.add_argument("--gate-max", type=float, default=1.2, help="门控最大增强系数 (默认 1.2)")
    parser.add_argument("--force-reencode", action="store_true", help="强制重新运行 GPU 前向编码而不使用已有特征缓存")
    parser.add_argument("--no-plot", action="store_true", help="禁用图表生成")
    parser.add_argument("--verbose", action="store_true", help="打印完整宽表诊断拆解")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    center_svd = not bool(args.uncentered_svd)

    if args.cache_pt is not None and str(args.cache_pt).strip():
        run_standalone_subspace_evaluation(
            cache_pt_path=args.cache_pt,
            eta_values=args.etas,
            rank_k_values=args.rank_ks,
            coherence_gate_mode=args.coherence_gate,
            center_svd=center_svd,
            coherence_low=args.coherence_low,
            coherence_high=args.coherence_high,
            gate_min=args.gate_min,
            gate_max=args.gate_max,
            plot=not bool(args.no_plot),
            verbose=bool(args.verbose),
        )
        return 0

    stage_raw = str(args.stage or "revision").strip().lower()
    if stage_raw == "all":
        stages = ["context", "revision"]
    else:
        stages = [s.strip() for s in stage_raw.split(",") if s.strip()]

    seed_tokens = [
        int(tok.strip())
        for tok in str(args.seed).replace(";", ",").split(",")
        if tok.strip()
    ]
    if not seed_tokens:
        seed_tokens = [0]

    for sd in seed_tokens:
        for st in stages:
            run_standalone_subspace_evaluation(
                dataset=args.dataset,
                stage=st,
                seed=sd,
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
                center_svd=center_svd,
                coherence_low=args.coherence_low,
                coherence_high=args.coherence_high,
                gate_min=args.gate_min,
                gate_max=args.gate_max,
                force_reencode=bool(args.force_reencode),
                plot=not bool(args.no_plot),
                show_progress=True,
                verbose=bool(args.verbose),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
