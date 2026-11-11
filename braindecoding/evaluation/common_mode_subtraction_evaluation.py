"""留一法共模时序偏置扣除（Leave-One-Out Common-Mode Subtraction）探索性评估模块。

本模块完全独立于主训练与标准评估代码，用于在不重新训练模型、不增加额外 GPU 前向推理的前提下，
对每个测试词位 i 的 1 个 Clean (0s) 预测与 M 个纯历史因果时间错位 Shifted 预测（如 -15s, -30s, -45s, -60s, -90s, -120s，M=6）
执行无泄漏的留一法共模扣除（Leave-One-Out Common-Mode Subtraction）：

- Clean (0s)：使用全部 M 个错位条件的均值作为共模时序偏置扣除
    s_tilde_i^(0) = s_i^(0) - eta * (1 / M) * sum_{m=1..M} s_i^(delta_m)
- Shifted (delta_k)：为严格防止自身噪声泄漏（保持负对照公平性），仅使用其余 M-1 个错位条件的均值扣除
    s_tilde_i^(delta_k) = s_i^(delta_k) - eta * (1 / (M - 1)) * sum_{m != k} s_i^(delta_m)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_SUBTRACTION_ETAS: tuple[float, ...] = (0.5, 0.8, 1.0)
DEFAULT_CAUSAL_CMS_ETAS: tuple[float, ...] = (0.5,)
DEFAULT_CAUSAL_CMS_EMA_DECAY: float = 0.99
VALID_CMS_MODES: tuple[str, ...] = ("leave_one_out", "causal_ema")


def normalize_cms_mode(raw: str | None, *, default: str = "leave_one_out") -> str:
    """规范化共模扣除模式标识（'leave_one_out' 或 'causal_ema'）。"""
    if raw is None:
        return default
    token = str(raw).strip().lower()
    if not token or token in {"auto", "default"}:
        return default
    alias_map = {
        "leave_one_out": "leave_one_out",
        "loo": "leave_one_out",
        "historical_shifts": "leave_one_out",
        "causal_ema": "causal_ema",
        "causal": "causal_ema",
        "ema": "causal_ema",
        "online_ema": "causal_ema",
    }
    if token not in alias_map:
        raise ValueError(
            f"不支持的 cms_mode={raw!r}，可选：{VALID_CMS_MODES}"
        )
    return alias_map[token]


def parse_eta_values(
    raw_etas: str | Iterable[float | int | str] | None,
    *,
    default: Iterable[float] = DEFAULT_SUBTRACTION_ETAS,
) -> list[float]:
    """解析共模扣除权重 eta 列表（自动排除 0.0 基线并按升序去重排列，0.0 由评估器自动作为首项参考）。"""
    if raw_etas is None:
        tokens: list[float | int | str] = list(default)
    elif isinstance(raw_etas, (int, float)):
        tokens = [float(raw_etas)]
    elif isinstance(raw_etas, str):
        text = raw_etas.strip()
        if not text:
            tokens = list(default)
        else:
            tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
    else:
        tokens = []
        for item in raw_etas:
            if isinstance(item, str):
                tokens.extend(t.strip() for t in item.split(",") if t.strip())
            else:
                tokens.append(item)

    parsed: list[float] = []
    seen: set[float] = set()
    for tok in tokens:
        val = float(tok)
        if val < 0.0:
            raise ValueError(f"共模扣除权重 eta 必须为非负数，收到：{val}")
        if abs(val) <= 1e-9:
            continue
        rounded = round(val, 6)
        if rounded not in seen:
            seen.add(rounded)
            parsed.append(val)
    return parsed if parsed else [float(x) for x in default]


def _l2_normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """对二维预测张量逐行做 L2 归一化（使内积严格等于余弦相似度得分）。"""
    t = tensor.detach().clone().to(dtype=torch.float32)
    return F.normalize(t, p=2.0, dim=-1, eps=1e-12)


def apply_leave_one_out_subtraction_to_tensors(
    clean_tensor: torch.Tensor,
    shifted_tensors: dict[str, torch.Tensor],
    eta: float,
    *,
    normalize_inputs: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """对单层预测张量执行留一法共模时序偏置扣除。

    参数
    ----
    clean_tensor:
        Clean (0s) 预测张量，形状 ``(N, D)``。
    shifted_tensors:
        有序字典 ``{cond_key: tensor}``，每个张量形状 ``(N, D)``，对应 M 个错位条件（要求 M >= 2）。
    eta:
        共模扣除权重（如 0.5, 0.8, 1.0）。
    normalize_inputs:
        扣除前是否先对各条件向量做单位 L2 归一化（默认 True，保证在单位超球面/余弦得分空间线性扣除）。
    """
    shift_keys = list(shifted_tensors.keys())
    num_shifts = len(shift_keys)
    if num_shifts < 2:
        raise ValueError(
            f"留一法共模扣除（Leave-One-Out Common-Mode Subtraction）至少需要 2 个错位条件，"
            f"当前仅有 {num_shifts} 个（推荐 6 个：-15s, -30s, -45s, -60s, -90s, -120s）。"
        )

    clean_base = (
        _l2_normalize_tensor(clean_tensor)
        if normalize_inputs
        else clean_tensor.detach().clone().to(dtype=torch.float32)
    )
    shifted_base: dict[str, torch.Tensor] = {}
    for k in shift_keys:
        st = shifted_tensors[k]
        if st.shape != clean_base.shape:
            raise ValueError(
                f"错位条件 {k!r} 张量形状 {tuple(st.shape)} 与 Clean 张量形状 {tuple(clean_base.shape)} 不一致。"
            )
        shifted_base[k] = (
            _l2_normalize_tensor(st)
            if normalize_inputs
            else st.detach().clone().to(dtype=torch.float32)
        )

    eta_val = float(eta)
    if abs(eta_val) <= 1e-12:
        return clean_base, shifted_base

    # 计算全部 M 个错位条件的总和向量 S_all = sum_{m=1..M} s^(delta_m)
    stacked_shifts = torch.stack([shifted_base[k] for k in shift_keys], dim=0)
    sum_all_shifts = stacked_shifts.sum(dim=0)

    # 1. Clean (0s): 减去全部 M 个错位条件的均值 (1 / M) * S_all
    mean_all_shifts = sum_all_shifts / float(num_shifts)
    debiased_clean = clean_base - eta_val * mean_all_shifts

    # 2. Shifted (delta_k): 留一法，仅减去其余 (M - 1) 个错位条件的均值 (1 / (M - 1)) * (S_all - s^(delta_k))
    denom_loo = float(num_shifts - 1)
    debiased_shifted: dict[str, torch.Tensor] = {}
    for k in shift_keys:
        cur = shifted_base[k]
        mean_other_shifts = (sum_all_shifts - cur) / denom_loo
        debiased_shifted[k] = cur - eta_val * mean_other_shifts

    return debiased_clean, debiased_shifted


def apply_leave_one_out_subtraction_to_encoded(
    clean_encoded: dict,
    shifted_encoded_map: dict[str, dict],
    eta: float,
    *,
    normalize_inputs: bool = True,
) -> tuple[dict, dict[str, dict]]:
    """对 encode_loader 输出的完整字典（含阶段 2.2 的 K0/K1/K2 修订预测）执行留一法共模扣除。"""
    shift_keys = list(shifted_encoded_map.keys())
    clean_preds = clean_encoded["predictions"]
    shifted_preds = {k: shifted_encoded_map[k]["predictions"] for k in shift_keys}

    deb_clean_preds, deb_shifted_preds = apply_leave_one_out_subtraction_to_tensors(
        clean_preds,
        shifted_preds,
        eta,
        normalize_inputs=normalize_inputs,
    )

    new_clean = dict(clean_encoded)
    new_clean["predictions"] = deb_clean_preds

    new_shifted: dict[str, dict] = {}
    for k in shift_keys:
        copied = dict(shifted_encoded_map[k])
        copied["predictions"] = deb_shifted_preds[k]
        new_shifted[k] = copied

    if "revision_predictions" in clean_encoded and isinstance(
        clean_encoded["revision_predictions"], dict
    ):
        clean_rev = dict(clean_encoded["revision_predictions"])
        new_clean_rev: dict = {}
        new_shifted_rev: dict[str, dict] = {k: {} for k in shift_keys}

        for delay, clean_delay_tensor in sorted(clean_rev.items()):
            shifted_delay_tensors = {
                k: shifted_encoded_map[k]["revision_predictions"][delay]
                for k in shift_keys
            }
            deb_c_delay, deb_s_delay = apply_leave_one_out_subtraction_to_tensors(
                clean_delay_tensor,
                shifted_delay_tensors,
                eta,
                normalize_inputs=normalize_inputs,
            )
            new_clean_rev[delay] = deb_c_delay
            for k in shift_keys:
                new_shifted_rev[k][delay] = deb_s_delay[k]

        new_clean["revision_predictions"] = new_clean_rev
        for k in shift_keys:
            new_shifted[k]["revision_predictions"] = new_shifted_rev[k]

    return new_clean, new_shifted


def build_stream_keys_from_encoded(encoded: Mapping[str, object]) -> list[str]:
    """按 (subject_id/subject_index, recording_id) 提取在线会话流键，保证跨被试与跨 run 严格重置 EMA 状态。"""
    preds = encoded.get("predictions")
    n_rows = int(len(preds)) if isinstance(preds, torch.Tensor) else 0
    if n_rows <= 0:
        return []

    explicit_keys = encoded.get("stream_keys")
    if isinstance(explicit_keys, Sequence) and not isinstance(explicit_keys, (str, bytes)) and len(explicit_keys) == n_rows:
        return [str(k) for k in explicit_keys]

    subj_tensor = encoded.get("subject_indices")
    subj_list: list[int] | None = None
    if isinstance(subj_tensor, torch.Tensor) and len(subj_tensor) == n_rows:
        subj_list = [int(v) for v in subj_tensor.detach().cpu().tolist()]
    elif isinstance(subj_tensor, Sequence) and not isinstance(subj_tensor, (str, bytes)) and len(subj_tensor) == n_rows:
        subj_list = [int(v) for v in subj_tensor]

    rec_ids = encoded.get("recording_ids")
    has_rec = (
        isinstance(rec_ids, Sequence)
        and not isinstance(rec_ids, (str, bytes))
        and len(rec_ids) == n_rows
    )
    event_ids = encoded.get("event_ids")
    has_eid = (
        isinstance(event_ids, Sequence)
        and not isinstance(event_ids, (str, bytes))
        and len(event_ids) == n_rows
    )

    keys: list[str] = []
    for idx in range(n_rows):
        if subj_list is not None:
            subj_part = f"subj-{subj_list[idx]}"
        elif has_eid and ":" in str(event_ids[idx]):
            subj_part = str(event_ids[idx]).split(":", 1)[0]
        else:
            subj_part = "subj-0"

        if has_rec and str(rec_ids[idx]):
            rec_part = str(rec_ids[idx])
        elif has_eid and str(event_ids[idx]).count(":") >= 2:
            rec_part = ":".join(str(event_ids[idx]).split(":", 2)[:2])
        else:
            rec_part = "rec-0"

        keys.append(f"{subj_part}::{rec_part}")
    return keys


def _causal_ema_stream_prior_means(
    stream_vectors: torch.Tensor,
    init_mu: torch.Tensor,
    ema_decay: float,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """精确计算单条会话流在每个时刻 c 之前的因果指数滑动均值序列 m_c = mu_{c-1}。

    递推定义：
    - m_0 = mu_0
    - mu_c = alpha * m_c + (1 - alpha) * y_hat_c
    - m_{c+1} = mu_c
    采用严格下三角（对角线与上三角恒为 0）分块 Toeplitz 矩阵在 float64 中精确求解，
    代数上保证时刻 c 的偏置估计对当前词 y_hat_c 及所有未来词 y_hat_{>c} 零依赖。
    """
    total_steps, dim = stream_vectors.shape
    if total_steps == 0:
        return stream_vectors.new_empty((0, dim), dtype=torch.float32)

    alpha = float(ema_decay)
    y_f64 = stream_vectors.detach().to(device="cpu", dtype=torch.float64)
    mu_curr = init_mu.detach().to(device="cpu", dtype=torch.float64).reshape(dim)
    priors_f64 = torch.empty((total_steps, dim), dtype=torch.float64)

    if abs(alpha - 1.0) <= 1e-12:
        priors_f64[:] = mu_curr.unsqueeze(0)
        return priors_f64.to(device=stream_vectors.device, dtype=torch.float32)

    step = max(1, int(chunk_size))
    for start in range(0, total_steps, step):
        stop = min(start + step, total_steps)
        length = stop - start
        y_chunk = y_f64[start:stop]

        idx = torch.arange(length, dtype=torch.float64)
        alpha_pow = torch.pow(alpha, idx).unsqueeze(1)  # [L, 1]
        diff = idx.unsqueeze(1) - idx.unsqueeze(0) - 1.0  # c - j - 1
        strictly_lower = diff >= 0.0
        exponents = torch.where(strictly_lower, diff, torch.zeros_like(diff))
        weights = torch.where(
            strictly_lower,
            (1.0 - alpha) * torch.pow(alpha, exponents),
            torch.zeros_like(diff),
        )  # [L, L], strictly lower triangular (diagonal == 0)

        priors_f64[start:stop] = alpha_pow * mu_curr.unsqueeze(0) + weights @ y_chunk

        tail_exponents = (float(length - 1) - idx).clamp(min=0.0)
        tail_weights = (1.0 - alpha) * torch.pow(alpha, tail_exponents)  # [L]
        mu_curr = (alpha**length) * mu_curr + tail_weights @ y_chunk

    return priors_f64.to(device=stream_vectors.device, dtype=torch.float32)


def apply_causal_ema_subtraction_to_tensors(
    predictions: torch.Tensor,
    mu0: torch.Tensor | None = None,
    eta: float = 0.5,
    *,
    ema_decay: float = DEFAULT_CAUSAL_CMS_EMA_DECAY,
    stream_keys: Sequence[object] | None = None,
    valid_mask: torch.Tensor | None = None,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
) -> torch.Tensor:
    """对预测张量执行严格在线因果指数滑动平均共模扣除（Causal EMA-CMS）。

    对每条独立的 (subject, recording) 会话流：
    1. 会话起点初始化：mu_{-1} = mu_0（来自训练集全局输出均值，若未提供则初始化为 0 向量）；
    2. 到达词步 c 时先扣除历史滑动均值并归一化：
         y_tilde_c = normalize(y_hat_c - eta * mu_{c-1})
    3. 再用当前到达的原始归一化预测更新滑动均值（严格不窥探任何未来词 > c）：
         mu_c = alpha * mu_{c-1} + (1 - alpha) * y_hat_c
    """
    if predictions.ndim != 2:
        raise ValueError(
            f"predictions 必须为二维张量 (N, D)，收到形状：{tuple(predictions.shape)}"
        )
    eta_val = float(eta)
    if eta_val < 0.0:
        raise ValueError(f"Causal EMA-CMS 权重 eta 必须为非负数，收到：{eta_val}")
    alpha = float(ema_decay)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(
            f"Causal EMA-CMS 衰减率 ema_decay 必须在 [0.0, 1.0] 内，收到：{alpha}"
        )

    n_rows, dim = int(predictions.shape[0]), int(predictions.shape[1])
    if n_rows == 0:
        return predictions.detach().clone().to(dtype=torch.float32)

    base = (
        _l2_normalize_tensor(predictions)
        if normalize_inputs
        else predictions.detach().clone().to(dtype=torch.float32)
    )
    if mu0 is None:
        init_mu = torch.zeros(dim, dtype=torch.float32, device=base.device)
    else:
        if int(mu0.numel()) != dim:
            raise ValueError(
                f"mu0 维度 {int(mu0.numel())} 与预测向量维度 {dim} 不一致。"
            )
        init_mu = mu0.detach().to(device=base.device, dtype=torch.float32).reshape(dim)

    if valid_mask is not None:
        if int(valid_mask.numel()) != n_rows:
            raise ValueError(
                f"valid_mask 长度 {int(valid_mask.numel())} 与样本数 {n_rows} 不一致。"
            )
        valid_bool = valid_mask.detach().to(device="cpu", dtype=torch.bool).reshape(n_rows)
    else:
        valid_bool = torch.ones(n_rows, dtype=torch.bool)

    if abs(eta_val) <= 1e-12:
        if normalize_outputs:
            out = base.clone()
            if bool(valid_bool.any()):
                v_dev = valid_bool.to(device=base.device)
                out[v_dev] = F.normalize(out[v_dev], p=2.0, dim=-1, eps=1e-12)
            return out
        return base

    if stream_keys is None:
        resolved_keys: list[object] = ["__single_stream__"] * n_rows
    else:
        if len(stream_keys) != n_rows:
            raise ValueError(
                f"stream_keys 长度 {len(stream_keys)} 与样本数 {n_rows} 不一致。"
            )
        resolved_keys = list(stream_keys)

    indices_by_stream: dict[object, list[int]] = {}
    valid_list = valid_bool.tolist()
    for row_idx, key in enumerate(resolved_keys):
        if valid_list[row_idx]:
            indices_by_stream.setdefault(key, []).append(row_idx)

    out = base.clone()
    for stream_indices in indices_by_stream.values():
        idx_t = torch.as_tensor(stream_indices, device=base.device, dtype=torch.long)
        y_stream = base[idx_t]
        prior_means = _causal_ema_stream_prior_means(
            y_stream,
            init_mu,
            alpha,
        )
        debiased_stream = y_stream - eta_val * prior_means
        if normalize_outputs:
            debiased_stream = F.normalize(debiased_stream, p=2.0, dim=-1, eps=1e-12)
        out[idx_t] = debiased_stream

    return out


def apply_causal_ema_subtraction_to_encoded(
    encoded: Mapping[str, object],
    mu0_payload: Mapping[str, object] | torch.Tensor | None = None,
    eta: float = 0.5,
    *,
    ema_decay: float = DEFAULT_CAUSAL_CMS_EMA_DECAY,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
) -> dict:
    """对单条流（Clean 或单个 Shifted 负对照流）的编码字典执行在线 Causal EMA-CMS。"""
    stream_keys = build_stream_keys_from_encoded(encoded)
    mu0_pred: torch.Tensor | None = None
    mu0_rev_map: Mapping[object, object] = {}
    if isinstance(mu0_payload, torch.Tensor):
        mu0_pred = mu0_payload
    elif isinstance(mu0_payload, Mapping):
        cand_pred = mu0_payload.get("predictions")
        if isinstance(cand_pred, torch.Tensor):
            mu0_pred = cand_pred
        cand_rev = mu0_payload.get("revision_predictions")
        if isinstance(cand_rev, Mapping):
            mu0_rev_map = cand_rev

    new_encoded = dict(encoded)
    preds = encoded["predictions"]
    if not isinstance(preds, torch.Tensor):
        raise TypeError("encoded['predictions'] 必须为 torch.Tensor。")

    new_encoded["predictions"] = apply_causal_ema_subtraction_to_tensors(
        preds,
        mu0=mu0_pred,
        eta=eta,
        ema_decay=ema_decay,
        stream_keys=stream_keys,
        valid_mask=None,
        normalize_inputs=normalize_inputs,
        normalize_outputs=normalize_outputs,
    )

    rev_preds = encoded.get("revision_predictions")
    rev_masks = encoded.get("revision_availability_masks")
    if isinstance(rev_preds, Mapping):
        new_rev: dict = {}
        mask_map = rev_masks if isinstance(rev_masks, Mapping) else {}
        for delay, delay_tensor in sorted(rev_preds.items(), key=lambda kv: int(kv[0])):
            if not isinstance(delay_tensor, torch.Tensor):
                continue
            d_int = int(delay)
            d_mu0_cand = (
                mu0_rev_map.get(delay)
                if delay in mu0_rev_map
                else (
                    mu0_rev_map.get(d_int)
                    if d_int in mu0_rev_map
                    else mu0_rev_map.get(str(d_int))
                )
            )
            d_mu0 = d_mu0_cand if isinstance(d_mu0_cand, torch.Tensor) else mu0_pred
            d_mask_cand = (
                mask_map.get(delay)
                if delay in mask_map
                else (
                    mask_map.get(d_int)
                    if d_int in mask_map
                    else mask_map.get(str(d_int))
                )
            )
            d_mask = d_mask_cand if isinstance(d_mask_cand, torch.Tensor) else None
            new_rev[delay] = apply_causal_ema_subtraction_to_tensors(
                delay_tensor,
                mu0=d_mu0,
                eta=eta,
                ema_decay=ema_decay,
                stream_keys=stream_keys,
                valid_mask=d_mask,
                normalize_inputs=normalize_inputs,
                normalize_outputs=normalize_outputs,
            )
        new_encoded["revision_predictions"] = new_rev

    return new_encoded


def _build_eta_entry(
    eta: float,
    clean_summary: dict,
    condition_entries: list[dict],
    shifted_aggregate: dict,
    *,
    baseline_entry: dict | None = None,
    method_label: str | None = None,
) -> dict:
    """构建单个 eta 权重下的完整汇总字典（含相对 eta=0 的增益差值）。"""
    num_shifts = len(condition_entries)
    c_fv_word = clean_summary.get(
        "full_vocab_word_macro_top10",
        clean_summary.get("full_vocab_macro_top10"),
    )
    if c_fv_word is not None:
        clean_summary["full_vocab_word_macro_top10"] = float(c_fv_word)
        clean_summary["full_vocab_macro_top10"] = float(c_fv_word)
    s_word_macro = [
        float(
            (c.get("summary") or c).get(
                "full_vocab_word_macro_top10",
                (c.get("summary") or c).get("full_vocab_macro_top10", 0.0),
            )
        )
        for c in condition_entries
        if isinstance(c, dict)
    ]
    if s_word_macro:
        sh_w_mean = float(np.mean(s_word_macro))
        shifted_aggregate["mean_full_vocab_word_macro_top10"] = sh_w_mean
        shifted_aggregate["mean_full_vocab_macro_top10"] = sh_w_mean
        if c_fv_word is not None:
            gap_w = float(c_fv_word) - sh_w_mean
            shifted_aggregate["clean_minus_shifted_full_vocab_word_macro_top10"] = gap_w
            shifted_aggregate["clean_minus_shifted_full_vocab_macro_top10"] = gap_w

    entry = {
        "eta": float(eta),
        "label": (
            method_label
            if method_label is not None
            else (
                "Raw Baseline (eta=0.0)"
                if abs(eta) <= 1e-9
                else f"LOO Subtracted (eta={eta:g})"
            )
        ),
        "clean_divisor": int(num_shifts),
        "shifted_loo_divisor": int(max(1, num_shifts - 1)),
        "clean_reference": clean_summary,
        "conditions": condition_entries,
        "shifted_aggregate": shifted_aggregate,
    }
    if baseline_entry is not None:
        base_clean = baseline_entry["clean_reference"]
        base_agg = baseline_entry["shifted_aggregate"]
        delta_clean = {
            "vocab50_macro_top10": float(
                clean_summary.get("vocab50_macro_top10", 0.0)
                - base_clean.get("vocab50_macro_top10", 0.0)
            ),
            "vocab50_micro_top10": float(
                clean_summary.get("vocab50_micro_top10", 0.0)
                - base_clean.get("vocab50_micro_top10", 0.0)
            ),
            "vocab50_macro_top1": float(
                clean_summary.get("vocab50_macro_top1", 0.0)
                - base_clean.get("vocab50_macro_top1", 0.0)
            ),
            "vocab50_mrr": float(
                clean_summary.get("vocab50_mrr", 0.0)
                - base_clean.get("vocab50_mrr", 0.0)
            ),
            "full_vocab_macro_top10": float(
                clean_summary.get(
                    "full_vocab_word_macro_top10",
                    clean_summary.get("full_vocab_macro_top10", 0.0),
                )
                - base_clean.get(
                    "full_vocab_word_macro_top10",
                    base_clean.get("full_vocab_macro_top10", 0.0),
                )
            ),
            "full_vocab_word_macro_top10": float(
                clean_summary.get(
                    "full_vocab_word_macro_top10",
                    clean_summary.get("full_vocab_macro_top10", 0.0),
                )
                - base_clean.get(
                    "full_vocab_word_macro_top10",
                    base_clean.get("full_vocab_macro_top10", 0.0),
                )
            ),
            "full_vocab_micro_top10": float(
                clean_summary.get("full_vocab_micro_top10", 0.0)
                - base_clean.get("full_vocab_micro_top10", 0.0)
            ),
            "full_vocab_macro_top1": float(
                clean_summary.get("full_vocab_macro_top1", 0.0)
                - base_clean.get("full_vocab_macro_top1", 0.0)
            ),
            "full_vocab_mrr": float(
                clean_summary.get("full_vocab_mrr", 0.0)
                - base_clean.get("full_vocab_mrr", 0.0)
            ),
            "full_vocab_median_rank": float(
                clean_summary.get("full_vocab_median_rank", 0.0)
                - base_clean.get("full_vocab_median_rank", 0.0)
            ),
            "clean_minus_shifted_vocab50_macro_top10": float(
                shifted_aggregate.get("clean_minus_shifted_vocab50_macro_top10", 0.0)
                - base_agg.get("clean_minus_shifted_vocab50_macro_top10", 0.0)
            ),
            "shifted_mean_vocab50_macro_top1": float(
                shifted_aggregate.get("mean_vocab50_macro_top1", 0.0)
                - base_agg.get("mean_vocab50_macro_top1", 0.0)
            ),
            "clean_minus_shifted_vocab50_macro_top1": float(
                shifted_aggregate.get("clean_minus_shifted_vocab50_macro_top1", 0.0)
                - base_agg.get("clean_minus_shifted_vocab50_macro_top1", 0.0)
            ),
            "clean_minus_shifted_full_vocab_macro_top10": float(
                shifted_aggregate.get(
                    "clean_minus_shifted_full_vocab_word_macro_top10",
                    shifted_aggregate.get("clean_minus_shifted_full_vocab_macro_top10", 0.0),
                )
                - base_agg.get(
                    "clean_minus_shifted_full_vocab_word_macro_top10",
                    base_agg.get("clean_minus_shifted_full_vocab_macro_top10", 0.0),
                )
            ),
            "clean_minus_shifted_full_vocab_word_macro_top10": float(
                shifted_aggregate.get(
                    "clean_minus_shifted_full_vocab_word_macro_top10",
                    shifted_aggregate.get("clean_minus_shifted_full_vocab_macro_top10", 0.0),
                )
                - base_agg.get(
                    "clean_minus_shifted_full_vocab_word_macro_top10",
                    base_agg.get("clean_minus_shifted_full_vocab_macro_top10", 0.0),
                )
            ),
            "shifted_mean_full_vocab_macro_top1": float(
                shifted_aggregate.get("mean_full_vocab_macro_top1", 0.0)
                - base_agg.get("mean_full_vocab_macro_top1", 0.0)
            ),
            "clean_minus_shifted_full_vocab_macro_top1": float(
                shifted_aggregate.get("clean_minus_shifted_full_vocab_macro_top1", 0.0)
                - base_agg.get("clean_minus_shifted_full_vocab_macro_top1", 0.0)
            ),
        }
        entry["delta_vs_eta0"] = delta_clean
    return entry


def run_common_mode_subtraction_sweep(
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
    shifted_ds_by_shift: dict[float, object],
    vocabulary: list[str],
    vocab50: list[str],
    cfg: dict,
    eta_values: Iterable[float] = DEFAULT_SUBTRACTION_ETAS,
    normalize_inputs: bool = True,
    baseline_clean_summary: dict | None = None,
    baseline_condition_entries: list[dict] | None = None,
    baseline_shifted_aggregate: dict | None = None,
    compute_metrics_fn: Callable[..., dict],
    extract_summary_row_fn: Callable[..., dict],
    summarize_shifted_fn: Callable[[dict, list[dict]], dict],
    output_dir: str | Path | None = None,
    plot: bool = True,
    print_report: bool = True,
    verbose: bool = False,
) -> dict:
    """在单次编码缓存上执行留一法共模扣除多权重 (eta) 扫描、终端打印、独立落盘与绘图。"""
    etas = parse_eta_values(eta_values)
    shift_meta = [
        (float(sec), str(key), str(label))
        for sec, key, label, _ in shifted_encoded_by_shift
    ]
    shifted_encoded_map = {
        str(key): enc for _, key, _, enc in shifted_encoded_by_shift
    }
    shift_sec_by_key = {str(key): float(sec) for sec, key, _, _ in shifted_encoded_by_shift}
    shift_label_by_key = {str(key): str(label) for _, key, label, _ in shifted_encoded_by_shift}

    if len(shift_meta) < 2:
        raise ValueError(
            f"留一法共模扣除至少需要 2 个时间错位条件，当前只有 {len(shift_meta)} 个。"
        )

    # 1. 基线 eta = 0.0（直接复用已算好的 Clean 与 Shifted 结果，避免重复计算）
    if (
        baseline_clean_summary is not None
        and baseline_condition_entries is not None
        and baseline_shifted_aggregate is not None
    ):
        eta0_clean = baseline_clean_summary
        eta0_conditions = [
            {
                "key": c["key"],
                "label": c["label"],
                "shift_seconds": c["shift_seconds"],
                "summary": c["summary"],
            }
            for c in baseline_condition_entries
            if c.get("control_type") == "temporal_shift"
        ]
        eta0_agg = baseline_shifted_aggregate
    else:
        raw_clean_metrics = compute_metrics_fn(
            clean_encoded,
            eval_ds,
            vocabulary,
            vocab50,
            cfg,
            stage=stage,
            split=split,
        )
        eta0_clean = extract_summary_row_fn(
            "clean_0s", "Clean (0s)", 0.0, raw_clean_metrics
        )
        eta0_shifted_rows: list[dict] = []
        eta0_conditions = []
        for sec, key, label in shift_meta:
            s_ds = shifted_ds_by_shift.get(sec, eval_ds)
            s_metrics = compute_metrics_fn(
                shifted_encoded_map[key],
                s_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            s_row = extract_summary_row_fn(
                key, label, sec, s_metrics, clean_summary=eta0_clean
            )
            eta0_shifted_rows.append(s_row)
            eta0_conditions.append(
                {
                    "key": key,
                    "label": label,
                    "shift_seconds": sec,
                    "summary": s_row,
                }
            )
        eta0_agg = summarize_shifted_fn(eta0_clean, eta0_shifted_rows)

    eta0_entry = _build_eta_entry(
        0.0, eta0_clean, eta0_conditions, eta0_agg, baseline_entry=None
    )
    sweep_results: list[dict] = [eta0_entry]

    # 2. 扫描每个 eta > 0（纯内存向量运算 + 排名计算，无需重跑神经网络）
    for eta in etas:
        deb_clean_enc, deb_shifted_enc_map = apply_leave_one_out_subtraction_to_encoded(
            clean_encoded,
            shifted_encoded_map,
            eta,
            normalize_inputs=normalize_inputs,
        )
        deb_clean_metrics = compute_metrics_fn(
            deb_clean_enc,
            eval_ds,
            vocabulary,
            vocab50,
            cfg,
            stage=stage,
            split=split,
        )
        deb_clean_row = extract_summary_row_fn(
            "clean_0s", "Clean (0s)", 0.0, deb_clean_metrics
        )

        deb_shifted_rows: list[dict] = []
        deb_conditions: list[dict] = []
        for key in shifted_encoded_map:
            sec = shift_sec_by_key[key]
            label = shift_label_by_key[key]
            s_ds = shifted_ds_by_shift.get(sec, eval_ds)
            deb_s_metrics = compute_metrics_fn(
                deb_shifted_enc_map[key],
                s_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            s_row = extract_summary_row_fn(
                key, label, sec, deb_s_metrics, clean_summary=deb_clean_row
            )
            deb_shifted_rows.append(s_row)
            deb_conditions.append(
                {
                    "key": key,
                    "label": label,
                    "shift_seconds": sec,
                    "summary": s_row,
                }
            )

        deb_agg = summarize_shifted_fn(deb_clean_row, deb_shifted_rows)
        eta_entry = _build_eta_entry(
            eta,
            deb_clean_row,
            deb_conditions,
            deb_agg,
            baseline_entry=eta0_entry,
        )
        sweep_results.append(eta_entry)

    # 选出 50 词库 Clean Macro Top-10 最优与 Clean-Shifted 净差值最优的 eta
    best_by_clean50 = max(
        sweep_results,
        key=lambda item: float(item["clean_reference"].get("vocab50_macro_top10", 0.0)),
    )
    best_by_gap50 = max(
        sweep_results,
        key=lambda item: float(
            item["shifted_aggregate"].get("clean_minus_shifted_vocab50_macro_top10", 0.0)
        ),
    )

    report_payload = {
        "schema_version": 1,
        "experiment_type": "exploratory_leave_one_out_common_mode_subtraction",
        "dataset": dataset,
        "stage": stage,
        "split": split,
        "seed": int(seed),
        "exp_tag": exp_tag,
        "word_exp_tag": word_exp_tag,
        "formula": {
            "clean": "s_tilde_i^(0) = s_i^(0) - eta * (1 / M) * sum_{delta != 0} s_i^(delta)",
            "shifted_leave_one_out": "s_tilde_i^(delta_k) = s_i^(delta_k) - eta * (1 / (M - 1)) * sum_{delta not in {0, delta_k}} s_i^(delta)",
            "num_shifted_conditions_M": len(shift_meta),
            "clean_divisor": len(shift_meta),
            "shifted_loo_divisor": len(shift_meta) - 1,
            "shift_seconds": [sec for sec, _, _ in shift_meta],
            "normalize_before_subtraction": bool(normalize_inputs),
        },
        "eta_grid": [0.0, *etas],
        "best_eta_by_clean_vocab50_top10": float(best_by_clean50["eta"]),
        "best_eta_by_clean_minus_shifted_vocab50_top10": float(best_by_gap50["eta"]),
        "baseline_raw_eta0": eta0_entry,
        "sweep": sweep_results,
        "figures": {},
    }

    if print_report:
        print_common_mode_subtraction_report(
            report_payload,
            verbose=verbose,
            output_dir=output_dir,
        )

    if output_dir is not None:
        from braindecoding.evaluation.evaluation_exports import (
            export_common_mode_subtraction_tables_and_markdown,
        )

        out_path = Path(output_dir)
        exploratory_dir = out_path / "exploratory"
        exploratory_dir.mkdir(parents=True, exist_ok=True)
        report_payload["exported_files"] = (
            export_common_mode_subtraction_tables_and_markdown(
                report_payload, out_path
            )
        )
        json_file = exploratory_dir / f"common_mode_subtraction_{split}.json"
        json_file.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_payload["exploratory_json"] = str(json_file)
        if plot:
            figs = plot_common_mode_subtraction_figures(
                report_payload, out_path / "figures"
            )
            report_payload["figures"].update(figs)
            json_file.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return report_payload


def run_causal_ema_common_mode_subtraction_sweep(
    *,
    dataset: str,
    stage: str,
    split: str,
    seed: int,
    exp_tag: str,
    word_exp_tag: str,
    clean_encoded: dict,
    shifted_encoded_by_shift: Sequence[tuple[float, str, str, dict]] = (),
    eval_ds,
    shifted_ds_by_shift: Mapping[float, object] | None = None,
    vocabulary: list[str],
    vocab50: list[str],
    cfg: dict,
    mu0_payload: Mapping[str, object] | torch.Tensor | None = None,
    eta_values: Iterable[float] = DEFAULT_CAUSAL_CMS_ETAS,
    ema_decay: float = DEFAULT_CAUSAL_CMS_EMA_DECAY,
    shift_mode: str = "preroll",
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
    baseline_clean_summary: dict | None = None,
    baseline_condition_entries: list[dict] | None = None,
    baseline_shifted_aggregate: dict | None = None,
    compute_metrics_fn: Callable[..., dict],
    extract_summary_row_fn: Callable[..., dict],
    summarize_shifted_fn: Callable[[dict, list[dict]], dict],
    output_dir: str | Path | None = None,
    plot: bool = True,
    print_report: bool = True,
    verbose: bool = False,
) -> dict:
    """执行严格在线因果 Causal EMA-CMS 多权重扫描，同时支持 M_eval = 0（纯单流推理）与 M_eval >= 1（Pre-Roll 负对照审计）。"""
    etas = parse_eta_values(eta_values, default=DEFAULT_CAUSAL_CMS_ETAS)
    alpha = float(ema_decay)
    shifted_ds_lookup = dict(shifted_ds_by_shift or {})
    shift_meta = [
        (float(sec), str(key), str(label))
        for sec, key, label, _ in shifted_encoded_by_shift
    ]
    shifted_encoded_map = {
        str(key): enc for _, key, _, enc in shifted_encoded_by_shift
    }
    shift_sec_by_key = {
        str(key): float(sec) for sec, key, _, _ in shifted_encoded_by_shift
    }
    shift_label_by_key = {
        str(key): str(label) for _, key, label, _ in shifted_encoded_by_shift
    }
    has_shifts = len(shift_meta) > 0

    mu0_source = "zeros_fallback"
    if isinstance(mu0_payload, Mapping):
        mu0_source = str(mu0_payload.get("source", "checkpoint_mu0"))
    elif isinstance(mu0_payload, torch.Tensor):
        mu0_source = "tensor_mu0"

    # 1. 基线 eta = 0.0
    if (
        baseline_clean_summary is not None
        and baseline_condition_entries is not None
        and baseline_shifted_aggregate is not None
    ):
        eta0_clean = baseline_clean_summary
        eta0_conditions = [
            {
                "key": c["key"],
                "label": c["label"],
                "shift_seconds": c["shift_seconds"],
                "summary": c["summary"],
            }
            for c in baseline_condition_entries
            if c.get("control_type") == "temporal_shift"
        ]
        eta0_agg = dict(baseline_shifted_aggregate) if has_shifts else {}
    else:
        raw_clean_metrics = compute_metrics_fn(
            clean_encoded,
            eval_ds,
            vocabulary,
            vocab50,
            cfg,
            stage=stage,
            split=split,
        )
        eta0_clean = extract_summary_row_fn(
            "clean_0s", "Clean (0s)", 0.0, raw_clean_metrics
        )
        eta0_shifted_rows: list[dict] = []
        eta0_conditions = []
        for sec, key, label in shift_meta:
            s_ds = shifted_ds_lookup.get(sec, eval_ds)
            s_metrics = compute_metrics_fn(
                shifted_encoded_map[key],
                s_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            s_row = extract_summary_row_fn(
                key, label, sec, s_metrics, clean_summary=eta0_clean
            )
            eta0_shifted_rows.append(s_row)
            eta0_conditions.append(
                {
                    "key": key,
                    "label": label,
                    "shift_seconds": sec,
                    "summary": s_row,
                }
            )
        eta0_agg = (
            summarize_shifted_fn(eta0_clean, eta0_shifted_rows)
            if eta0_shifted_rows
            else {}
        )

    eta0_entry = _build_eta_entry(
        0.0,
        eta0_clean,
        eta0_conditions,
        eta0_agg,
        baseline_entry=None,
        method_label="Raw Baseline (eta=0.0)",
    )
    sweep_results: list[dict] = [eta0_entry]

    # 2. 对每个 eta > 0 独立在单流上运行在线 Causal EMA-CMS（M_eval = 0 时零移位流前向）
    for eta in etas:
        deb_clean_enc = apply_causal_ema_subtraction_to_encoded(
            clean_encoded,
            mu0_payload=mu0_payload,
            eta=eta,
            ema_decay=alpha,
            normalize_inputs=normalize_inputs,
            normalize_outputs=normalize_outputs,
        )
        deb_clean_metrics = compute_metrics_fn(
            deb_clean_enc,
            eval_ds,
            vocabulary,
            vocab50,
            cfg,
            stage=stage,
            split=split,
        )
        deb_clean_row = extract_summary_row_fn(
            "clean_0s", "Clean (0s)", 0.0, deb_clean_metrics
        )

        deb_shifted_rows: list[dict] = []
        deb_conditions: list[dict] = []
        for key in shifted_encoded_map:
            sec = shift_sec_by_key[key]
            label = shift_label_by_key[key]
            s_ds = shifted_ds_lookup.get(sec, eval_ds)
            deb_s_enc = apply_causal_ema_subtraction_to_encoded(
                shifted_encoded_map[key],
                mu0_payload=mu0_payload,
                eta=eta,
                ema_decay=alpha,
                normalize_inputs=normalize_inputs,
                normalize_outputs=normalize_outputs,
            )
            deb_s_metrics = compute_metrics_fn(
                deb_s_enc,
                s_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            s_row = extract_summary_row_fn(
                key, label, sec, deb_s_metrics, clean_summary=deb_clean_row
            )
            deb_shifted_rows.append(s_row)
            deb_conditions.append(
                {
                    "key": key,
                    "label": label,
                    "shift_seconds": sec,
                    "summary": s_row,
                }
            )

        deb_agg = (
            summarize_shifted_fn(deb_clean_row, deb_shifted_rows)
            if deb_shifted_rows
            else {}
        )
        eta_entry = _build_eta_entry(
            eta,
            deb_clean_row,
            deb_conditions,
            deb_agg,
            baseline_entry=eta0_entry,
            method_label=f"Causal EMA-CMS (eta={eta:g}, alpha={alpha:g})",
        )
        sweep_results.append(eta_entry)

    best_by_clean50 = max(
        sweep_results,
        key=lambda item: float(item["clean_reference"].get("vocab50_macro_top10", 0.0)),
    )
    best_by_gap50 = (
        max(
            sweep_results,
            key=lambda item: float(
                item["shifted_aggregate"].get(
                    "clean_minus_shifted_vocab50_macro_top10", 0.0
                )
            ),
        )
        if has_shifts
        else best_by_clean50
    )

    report_payload = {
        "schema_version": 1,
        "experiment_type": "causal_ema_common_mode_subtraction",
        "cms_mode": "causal_ema",
        "shift_mode": str(shift_mode),
        "dataset": dataset,
        "stage": stage,
        "split": split,
        "seed": int(seed),
        "exp_tag": exp_tag,
        "word_exp_tag": word_exp_tag,
        "formula": {
            "clean": "y_tilde_c = normalize(y_hat_c - eta * mu_{c-1}), mu_c = alpha * mu_{c-1} + (1 - alpha) * y_hat_c",
            "shifted_leave_one_out": "independent single-stream Causal EMA-CMS initialized from mu_0 (zero cross-stream leakage)",
            "cms_mode": "causal_ema",
            "ema_decay": alpha,
            "mu0_source": mu0_source,
            "shift_mode": str(shift_mode),
            "clean_requires_shifted_forwards": False,
            "num_shifted_conditions_M": len(shift_meta),
            "clean_divisor": 1,
            "shifted_loo_divisor": 1,
            "shift_seconds": [sec for sec, _, _ in shift_meta],
            "normalize_before_subtraction": bool(normalize_inputs),
            "normalize_after_subtraction": bool(normalize_outputs),
        },
        "eta_grid": [0.0, *etas],
        "best_eta_by_clean_vocab50_top10": float(best_by_clean50["eta"]),
        "best_eta_by_clean_minus_shifted_vocab50_top10": float(best_by_gap50["eta"]),
        "baseline_raw_eta0": eta0_entry,
        "sweep": sweep_results,
        "figures": {},
    }

    if print_report:
        print_common_mode_subtraction_report(
            report_payload,
            verbose=verbose,
            output_dir=output_dir,
        )

    if output_dir is not None:
        from braindecoding.evaluation.evaluation_exports import (
            export_common_mode_subtraction_tables_and_markdown,
        )

        out_path = Path(output_dir)
        exploratory_dir = out_path / "exploratory"
        exploratory_dir.mkdir(parents=True, exist_ok=True)
        report_payload["exported_files"] = (
            export_common_mode_subtraction_tables_and_markdown(
                report_payload, out_path
            )
        )
        json_file = exploratory_dir / f"common_mode_subtraction_{split}.json"
        json_file.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_payload["exploratory_json"] = str(json_file)
        if plot:
            figs = plot_common_mode_subtraction_figures(
                report_payload, out_path / "figures"
            )
            report_payload["figures"].update(figs)
            json_file.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return report_payload


from braindecoding.evaluation.evaluation_figures import (
    plot_common_mode_subtraction_figures,
)
from braindecoding.evaluation.evaluation_reporting import (
    format_common_mode_subtraction_report,
    print_common_mode_subtraction_report,
)

