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
DEFAULT_POS_CAUSAL_CMS_ETAS: tuple[float, ...] = (0.5, 0.8, 1.0)
DEFAULT_CAUSAL_CMS_EMA_DECAY: float = 0.99
DEFAULT_POS_CAUSAL_EMA_DECAY: float = 0.95
DEFAULT_POS_MIN_BUCKET_COUNT: int = 3
POSITION_BUCKET_LABELS: tuple[str, ...] = (
    "0",
    "1",
    "2",
    "3-4",
    "5-7",
    "8-12",
    "13+",
)
DEFAULT_PREROLL_M_EVAL_GRID: tuple[int, ...] = (1, 2, 4, 6)
VALID_CMS_MODES: tuple[str, ...] = (
    "leave_one_out",
    "causal_ema",
    "pos_causal_ema",
)


def normalize_cms_mode(raw: str | None, *, default: str = "leave_one_out") -> str:
    """规范化共模扣除模式标识（'leave_one_out'、'causal_ema' 或 'pos_causal_ema'）。"""
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
        "causal_cms": "causal_ema",
        "ema": "causal_ema",
        "online_ema": "causal_ema",
        "pos_causal_ema": "pos_causal_ema",
        "pos_causal_cms": "pos_causal_ema",
        "position_causal_ema": "pos_causal_ema",
        "position_causal_cms": "pos_causal_ema",
        "position_locked_causal_ema": "pos_causal_ema",
    }
    if token not in alias_map:
        raise ValueError(
            f"不支持的 cms_mode={raw!r}，可选：{VALID_CMS_MODES}"
        )
    return alias_map[token]


def position_bucket_index(word_pos: int) -> int:
    """将句内 0-based 词位置映射为 7 档粗粒度位置桶 b(i) in {0..6}。

    分桶规则：
    0 -> 0, 1 -> 1, 2 -> 2, [3,4] -> 3, [5,7] -> 4, [8,12] -> 5, [13+] -> 6。
    """
    pos = int(word_pos)
    if pos <= 0:
        return 0
    if pos == 1:
        return 1
    if pos == 2:
        return 2
    if pos <= 4:
        return 3
    if pos <= 7:
        return 4
    if pos <= 12:
        return 5
    return 6


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
    stream_keys: Sequence[object] | None = None,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
) -> dict:
    """对单条流（Clean 或单个 Shifted 负对照流）的编码字典执行在线 Causal EMA-CMS。"""
    resolved_stream_keys = (
        [str(k) for k in stream_keys]
        if stream_keys is not None
        else build_stream_keys_from_encoded(encoded)
    )
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
        stream_keys=resolved_stream_keys,
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
                stream_keys=resolved_stream_keys,
                valid_mask=d_mask,
                normalize_inputs=normalize_inputs,
                normalize_outputs=normalize_outputs,
            )
        new_encoded["revision_predictions"] = new_rev

    return new_encoded


def build_sentence_positions_from_encoded(
    encoded: Mapping[str, object],
    *,
    eval_table: object | None = None,
    stream_keys: Sequence[object] | None = None,
) -> tuple[list[str], list[str], list[int]]:
    """提取每个事件的 (会话流键, 句子组键, 句内 0-based 词位置 i)。

    优先从 eval_table（或 .pt 特征缓存内置的最小事件表）按真实录音顺序还原句内词序号，
    从而即使在 common_horizon_delay 截断或旧版未存 sentence_indices 的 .pt 缓存上也能精确对齐位置桶；
    若未提供 eval_table，则使用 encoded['sentence_indices'] / sentence_keys 在同流同句内递增编号。
    """
    preds = encoded.get("predictions")
    n_rows = int(len(preds)) if isinstance(preds, torch.Tensor) else 0
    if n_rows <= 0:
        return [], [], []

    resolved_streams = (
        [str(k) for k in stream_keys]
        if stream_keys is not None and len(stream_keys) == n_rows
        else build_stream_keys_from_encoded(encoded)
    )
    explicit_positions = encoded.get("word_positions")
    has_explicit_pos = (
        (isinstance(explicit_positions, torch.Tensor) and len(explicit_positions) == n_rows)
        or (
            isinstance(explicit_positions, Sequence)
            and not isinstance(explicit_positions, (str, bytes))
            and len(explicit_positions) == n_rows
        )
    )

    event_ids = encoded.get("event_ids")
    has_eids = (
        isinstance(event_ids, Sequence)
        and not isinstance(event_ids, (str, bytes))
        and len(event_ids) == n_rows
    )

    if eval_table is not None and hasattr(eval_table, "columns") and has_eids:
        cols = set(getattr(eval_table, "columns", ()))
        eid_col = "event_id" if "event_id" in cols else ("事件编号" if "事件编号" in cols else None)
        sent_col = next(
            (
                c
                for c in (
                    "runtime_context_uid",
                    "sentence_uid",
                    "context_id",
                    "上下文编号",
                )
                if c in cols
            ),
            None,
        )
        subj_col = "subject_id" if "subject_id" in cols else ("受试者" if "受试者" in cols else None)
        rec_col = "recording_id" if "recording_id" in cols else ("记录编号" if "记录编号" in cols else None)
        if eid_col is not None and sent_col is not None:
            eid_vals = [str(v) for v in eval_table[eid_col].tolist()]
            sent_vals = [str(v) for v in eval_table[sent_col].tolist()]
            subj_vals = (
                [str(v) for v in eval_table[subj_col].tolist()]
                if subj_col is not None
                else ["subj-0"] * len(eid_vals)
            )
            rec_vals = (
                [str(v) for v in eval_table[rec_col].tolist()]
                if rec_col is not None
                else ["rec-0"] * len(eid_vals)
            )
            counters: dict[tuple[str, str, str], int] = {}
            lookup: dict[str, tuple[str, str, int]] = {}
            for e_v, su_v, rc_v, sn_v in zip(eid_vals, subj_vals, rec_vals, sent_vals):
                g_key = (su_v, rc_v, sn_v)
                cur_pos = counters.get(g_key, 0)
                counters[g_key] = cur_pos + 1
                st_k = f"{su_v}::{rc_v}"
                sn_k = f"{st_k}::sent-{sn_v}"
                lookup[e_v] = (st_k, sn_k, cur_pos)
            if all(str(eid) in lookup for eid in event_ids):
                matched = [lookup[str(eid)] for eid in event_ids]
                out_streams = (
                    resolved_streams
                    if stream_keys is not None
                    else [m[0] for m in matched]
                )
                out_sents = [m[1] for m in matched]
                out_pos = (
                    [
                        int(v)
                        for v in (
                            explicit_positions.detach().cpu().tolist()
                            if isinstance(explicit_positions, torch.Tensor)
                            else explicit_positions
                        )
                    ]
                    if has_explicit_pos
                    else [m[2] for m in matched]
                )
                return out_streams, out_sents, out_pos

    raw_sent = (
        encoded.get("sentence_keys")
        if encoded.get("sentence_keys") is not None
        else (
            encoded.get("sentence_ids")
            if encoded.get("sentence_ids") is not None
            else encoded.get("sentence_indices")
        )
    )
    sent_tokens: list[str] | None = None
    if isinstance(raw_sent, torch.Tensor) and len(raw_sent) == n_rows:
        sent_tokens = [str(int(v)) for v in raw_sent.detach().cpu().tolist()]
    elif (
        isinstance(raw_sent, Sequence)
        and not isinstance(raw_sent, (str, bytes))
        and len(raw_sent) == n_rows
    ):
        sent_tokens = [str(v) for v in raw_sent]

    if sent_tokens is not None:
        out_sents = [
            f"{resolved_streams[idx]}::sent-{sent_tokens[idx]}"
            for idx in range(n_rows)
        ]
        if has_explicit_pos:
            pos_list = [
                int(v)
                for v in (
                    explicit_positions.detach().cpu().tolist()
                    if isinstance(explicit_positions, torch.Tensor)
                    else explicit_positions
                )
            ]
        else:
            counters_s: dict[str, int] = {}
            pos_list = []
            for sk in out_sents:
                p = counters_s.get(sk, 0)
                counters_s[sk] = p + 1
                pos_list.append(p)
        return resolved_streams, out_sents, pos_list

    if has_explicit_pos:
        pos_list = [
            int(v)
            for v in (
                explicit_positions.detach().cpu().tolist()
                if isinstance(explicit_positions, torch.Tensor)
                else explicit_positions
            )
        ]
        out_sents = []
        sent_counter_by_stream: dict[str, int] = {}
        prev_pos_by_stream: dict[str, int] = {}
        for idx in range(n_rows):
            sk = resolved_streams[idx]
            p = pos_list[idx]
            if sk not in sent_counter_by_stream or p <= prev_pos_by_stream[sk]:
                sent_counter_by_stream[sk] = sent_counter_by_stream.get(sk, -1) + 1
            prev_pos_by_stream[sk] = p
            out_sents.append(f"{sk}::auto-sent-{sent_counter_by_stream[sk]}")
        return resolved_streams, out_sents, pos_list

    # 最保守兜底：当既无 eval_table 又无 sentence_indices 时，每个事件视作独立单步句（回退至桶 0 / 全局因果均值）
    out_sents = [f"{resolved_streams[idx]}::step-{idx}" for idx in range(n_rows)]
    pos_list = [0] * n_rows
    return resolved_streams, out_sents, pos_list


def apply_position_causal_ema_subtraction_to_tensors(
    predictions: torch.Tensor,
    mu0: torch.Tensor | None = None,
    eta: float = 0.5,
    *,
    ema_decay: float = DEFAULT_CAUSAL_CMS_EMA_DECAY,
    pos_ema_decay: float = DEFAULT_POS_CAUSAL_EMA_DECAY,
    min_bucket_count: int = DEFAULT_POS_MIN_BUCKET_COUNT,
    stream_keys: Sequence[object] | None = None,
    sentence_keys: Sequence[object] | None = None,
    word_positions: Sequence[int] | torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
) -> torch.Tensor:
    """执行位置分桶在线因果共模中心化（Position-Stratified / Position-Locked Causal EMA-CMS, M_eval = 0）。

    物理与因果机制：
    1. 在每条独立的 (subject, recording) 会话流内，同时维护：
       - 1 个全局慢漂移因果滑动均值 mu_global（衰减率 ema_decay=0.99，冷启动兜底）；
       - 7 个粗粒度句内位置桶均值 mu_pos[b]（b in {0, 1, 2, [3,4], [5,7], [8,12], [13+]}，衰减率 pos_ema_decay=0.95）。
    2. 对按时间顺序到达的每个句子 s：
       - 解码阶段（严格仅使用此前已完成历史句子 s' < s 的状态，零未来泄漏、零句内自相减）：
         对句内位置 i 对应的桶 b = b(i)，若历史独立句观测数 count[b] < min_bucket_count（默认 3），
         自动回退扣除 mu_global；否则扣除位置专属偏置 mu_pos[b]：
             y_tilde_{s, i} = normalize(y_hat_{s, i} - eta * mu_ref(b(i)))
       - 句末状态更新阶段：
         句子 s 结束后，依次更新 mu_global，并用本句各桶的平均响应以无偏预热步长
             w_k = max(1 - pos_ema_decay, 1 / k)
         更新 mu_pos[b]，使得 k = 3 时恰好等于前 3 句同位置桶的无偏样本均值，随后平滑过渡至指数滑动平均。
    """
    if predictions.ndim != 2:
        raise ValueError(
            f"predictions 必须为二维张量 (N, D)，收到形状：{tuple(predictions.shape)}"
        )
    eta_val = float(eta)
    if eta_val < 0.0:
        raise ValueError(f"Position Causal EMA-CMS 权重 eta 必须为非负数，收到：{eta_val}")
    alpha_global = float(ema_decay)
    alpha_pos = float(pos_ema_decay)
    if not (0.0 <= alpha_global <= 1.0):
        raise ValueError(
            f"ema_decay 必须在 [0.0, 1.0] 内，收到：{alpha_global}"
        )
    if not (0.0 <= alpha_pos <= 1.0):
        raise ValueError(
            f"pos_ema_decay 必须在 [0.0, 1.0] 内，收到：{alpha_pos}"
        )
    min_cnt = max(1, int(min_bucket_count))

    n_rows, dim = int(predictions.shape[0]), int(predictions.shape[1])
    if n_rows == 0:
        return predictions.detach().clone().to(dtype=torch.float32)

    base = (
        _l2_normalize_tensor(predictions)
        if normalize_inputs
        else predictions.detach().clone().to(dtype=torch.float32)
    )
    has_explicit_mu0 = False
    if mu0 is None:
        init_mu_f64 = torch.zeros(dim, dtype=torch.float64)
    else:
        if int(mu0.numel()) != dim:
            raise ValueError(
                f"mu0 维度 {int(mu0.numel())} 与预测向量维度 {dim} 不一致。"
            )
        init_mu_f64 = mu0.detach().to(device="cpu", dtype=torch.float64).reshape(dim)
        has_explicit_mu0 = bool(torch.any(init_mu_f64.abs() > 1e-12))

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
        resolved_streams: list[str] = ["__single_stream__"] * n_rows
    else:
        if len(stream_keys) != n_rows:
            raise ValueError(
                f"stream_keys 长度 {len(stream_keys)} 与样本数 {n_rows} 不一致。"
            )
        resolved_streams = [str(k) for k in stream_keys]

    if word_positions is not None:
        if isinstance(word_positions, torch.Tensor):
            pos_list = [int(v) for v in word_positions.detach().cpu().reshape(-1).tolist()]
        else:
            pos_list = [int(v) for v in word_positions]
        if len(pos_list) != n_rows:
            raise ValueError(
                f"word_positions 长度 {len(pos_list)} 与样本数 {n_rows} 不一致。"
            )
    else:
        pos_list = None

    if sentence_keys is not None:
        if len(sentence_keys) != n_rows:
            raise ValueError(
                f"sentence_keys 长度 {len(sentence_keys)} 与样本数 {n_rows} 不一致。"
            )
        resolved_sents = [f"{resolved_streams[i]}::{sentence_keys[i]}" for i in range(n_rows)]
        if pos_list is None:
            sent_pos_cnt: dict[str, int] = {}
            pos_list = []
            for sk in resolved_sents:
                c_p = sent_pos_cnt.get(sk, 0)
                sent_pos_cnt[sk] = c_p + 1
                pos_list.append(c_p)
    elif pos_list is not None:
        resolved_sents = []
        auto_sent_id: dict[str, int] = {}
        prev_pos: dict[str, int] = {}
        for i in range(n_rows):
            st_k = resolved_streams[i]
            p_i = pos_list[i]
            if st_k not in auto_sent_id or p_i <= prev_pos[st_k]:
                auto_sent_id[st_k] = auto_sent_id.get(st_k, -1) + 1
            prev_pos[st_k] = p_i
            resolved_sents.append(f"{st_k}::auto-{auto_sent_id[st_k]}")
    else:
        resolved_sents = [f"{resolved_streams[i]}::step-{i}" for i in range(n_rows)]
        pos_list = [0] * n_rows

    bucket_ids = [position_bucket_index(p) for p in pos_list]
    valid_list = valid_bool.tolist()

    # 按流聚类，并在每条流内按时间到达顺序切分为句子块
    rows_by_stream: dict[str, list[int]] = {}
    for row_idx in range(n_rows):
        if valid_list[row_idx]:
            rows_by_stream.setdefault(resolved_streams[row_idx], []).append(row_idx)

    base_cpu_f64 = base.detach().to(device="cpu", dtype=torch.float64)
    priors_cpu_f64 = torch.zeros((n_rows, dim), dtype=torch.float64)
    num_buckets = len(POSITION_BUCKET_LABELS)

    for stream_rows in rows_by_stream.values():
        mu_global = init_mu_f64.clone()
        global_count = 0
        mu_pos = torch.zeros((num_buckets, dim), dtype=torch.float64)
        bucket_counts = [0] * num_buckets

        # 将流内有效行按连续或相同 sentence_key 划分为有序历史句子序列
        sentence_groups: list[list[int]] = []
        sent_idx_map: dict[str, int] = {}
        for r_idx in stream_rows:
            s_key = resolved_sents[r_idx]
            existing_pos = sent_idx_map.get(s_key)
            if existing_pos is None:
                sent_idx_map[s_key] = len(sentence_groups)
                sentence_groups.append([r_idx])
            else:
                sentence_groups[existing_pos].append(r_idx)

        for sent_rows in sentence_groups:
            # 1) 严格仅使用历史已完成句 (s' < s) 的状态为当前句所有词提供偏置估计
            for r_idx in sent_rows:
                b_idx = bucket_ids[r_idx]
                if bucket_counts[b_idx] >= min_cnt:
                    priors_cpu_f64[r_idx] = mu_pos[b_idx]
                else:
                    priors_cpu_f64[r_idx] = mu_global

            # 2) 当前句结束后，因果更新全局均值与本句命中的位置桶均值
            bucket_accum: dict[int, list[torch.Tensor]] = {}
            for r_idx in sent_rows:
                y_vec = base_cpu_f64[r_idx]
                global_count += 1
                if has_explicit_mu0:
                    if abs(alpha_global - 1.0) > 1e-12:
                        mu_global = (
                            alpha_global * mu_global + (1.0 - alpha_global) * y_vec
                        )
                else:
                    w_g = (
                        1.0 / float(global_count)
                        if abs(alpha_global - 1.0) <= 1e-12
                        else max(1.0 - alpha_global, 1.0 / float(global_count))
                    )
                    mu_global = (1.0 - w_g) * mu_global + w_g * y_vec
                b_idx = bucket_ids[r_idx]
                bucket_accum.setdefault(b_idx, []).append(y_vec)

            for b_idx, vec_list in bucket_accum.items():
                y_bucket = (
                    vec_list[0]
                    if len(vec_list) == 1
                    else torch.stack(vec_list, dim=0).mean(dim=0)
                )
                bucket_counts[b_idx] += 1
                k_obs = bucket_counts[b_idx]
                w_b = (
                    1.0 / float(k_obs)
                    if abs(alpha_pos - 1.0) <= 1e-12
                    else max(1.0 - alpha_pos, 1.0 / float(k_obs))
                )
                mu_pos[b_idx] = (1.0 - w_b) * mu_pos[b_idx] + w_b * y_bucket

    out = base.clone()
    if bool(valid_bool.any()):
        v_dev = valid_bool.to(device=base.device)
        priors_dev = priors_cpu_f64.to(device=base.device, dtype=torch.float32)
        deb_valid = base[v_dev] - eta_val * priors_dev[v_dev]
        if normalize_outputs:
            deb_valid = F.normalize(deb_valid, p=2.0, dim=-1, eps=1e-12)
        out[v_dev] = deb_valid
    return out


def apply_position_causal_ema_subtraction_to_encoded(
    encoded: Mapping[str, object],
    mu0_payload: Mapping[str, object] | torch.Tensor | None = None,
    eta: float = 0.5,
    *,
    ema_decay: float = DEFAULT_CAUSAL_CMS_EMA_DECAY,
    pos_ema_decay: float = DEFAULT_POS_CAUSAL_EMA_DECAY,
    min_bucket_count: int = DEFAULT_POS_MIN_BUCKET_COUNT,
    stream_keys: Sequence[object] | None = None,
    eval_table: object | None = None,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
) -> dict:
    """对单条编码流（含 Stage 2.2/2.3 的 K0/K1/K2 逐视野状态）执行位置分桶在线因果共模中心化（pos_causal_cms）。"""
    resolved_streams, resolved_sents, word_positions = (
        build_sentence_positions_from_encoded(
            encoded,
            eval_table=eval_table,
            stream_keys=stream_keys,
        )
    )
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

    new_encoded["predictions"] = apply_position_causal_ema_subtraction_to_tensors(
        preds,
        mu0=mu0_pred,
        eta=eta,
        ema_decay=ema_decay,
        pos_ema_decay=pos_ema_decay,
        min_bucket_count=min_bucket_count,
        stream_keys=resolved_streams,
        sentence_keys=resolved_sents,
        word_positions=word_positions,
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
            new_rev[delay] = apply_position_causal_ema_subtraction_to_tensors(
                delay_tensor,
                mu0=d_mu0,
                eta=eta,
                ema_decay=ema_decay,
                pos_ema_decay=pos_ema_decay,
                min_bucket_count=min_bucket_count,
                stream_keys=resolved_streams,
                sentence_keys=resolved_sents,
                word_positions=word_positions,
                valid_mask=d_mask,
                normalize_inputs=normalize_inputs,
                normalize_outputs=normalize_outputs,
            )
        new_encoded["revision_predictions"] = new_rev

    return new_encoded


def apply_preroll_m_eval_subtraction_to_tensors(
    clean_tensor: torch.Tensor,
    shifted_tensors: Mapping[str, torch.Tensor],
    *,
    m_eval: int,
    eta: float,
    eval_shift_keys: Sequence[str] | None = None,
    normalize_inputs: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int, int]:
    """执行指定前向预算 M_eval >= 1 的严格零自泄漏 Pre-Roll 错位窗口共模扣除。

    - Clean (0s) 固定扣除移位池前 M_eval 个历史错位流的均值（共需 1 + M_eval 次前向推理）；
    - 每个待评测的 Shifted 负对照流 k 从移位池中严格排除自身（s != k），取其余流中的前 min(M_eval, Pool - 1) 个求均值扣除：
      当 Pool >= M_eval + 1（例如 Pool=6 时 M_eval in {1, 2, 4}），Clean 与 Shifted 均精确扣除 M_eval 个独立错位流，
      方差注入严格同分布且即使在 M_eval = 1 时也 100% 零自相减泄漏；当 M_eval == Pool（如 M_eval=6），自动退化为标准 LOO（1/(Pool-1)）。
    """
    pool_keys = list(shifted_tensors.keys())
    pool_size = len(pool_keys)
    m_req = int(m_eval)
    if m_req < 1:
        raise ValueError(f"m_eval 必须 >= 1，收到：{m_req}")
    if pool_size < 2:
        raise ValueError(
            f"无自泄漏评测 M_eval={m_req} 至少需要移位池包含 2 条独立错位流，当前仅有 {pool_size} 条。"
        )
    if m_req > pool_size:
        raise ValueError(
            f"请求的 M_eval={m_req} 超过当前可用错位流总数 {pool_size}。"
        )

    target_keys = (
        [str(k) for k in eval_shift_keys if str(k) in shifted_tensors]
        if eval_shift_keys is not None
        else list(pool_keys)
    )
    clean_base = (
        _l2_normalize_tensor(clean_tensor)
        if normalize_inputs
        else clean_tensor.detach().clone().to(dtype=torch.float32)
    )
    shifted_base: dict[str, torch.Tensor] = {}
    for k in pool_keys:
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

    clean_div = m_req
    shifted_div = min(m_req, pool_size - 1)
    eta_val = float(eta)
    if abs(eta_val) <= 1e-12:
        return (
            clean_base,
            {k: shifted_base[k] for k in target_keys},
            clean_div,
            shifted_div,
        )

    clean_bg_keys = pool_keys[:clean_div]
    clean_bg_mean = torch.stack([shifted_base[k] for k in clean_bg_keys], dim=0).mean(dim=0)
    deb_clean = clean_base - eta_val * clean_bg_mean

    deb_shifted: dict[str, torch.Tensor] = {}
    for k in target_keys:
        disjoint_keys = [other for other in pool_keys if other != k][:shifted_div]
        sh_bg_mean = torch.stack([shifted_base[other] for other in disjoint_keys], dim=0).mean(dim=0)
        deb_shifted[k] = shifted_base[k] - eta_val * sh_bg_mean

    return deb_clean, deb_shifted, clean_div, shifted_div


def apply_preroll_m_eval_subtraction_to_encoded(
    clean_encoded: Mapping[str, object],
    shifted_encoded_map: Mapping[str, Mapping[str, object]],
    *,
    m_eval: int,
    eta: float,
    eval_shift_keys: Sequence[str] | None = None,
    normalize_inputs: bool = True,
) -> tuple[dict, dict[str, dict], int, int]:
    """对完整编码字典（含 Stage 2.2/2.3 的 K0/K1/K2 预测）执行指定 M_eval 预算的 Pre-Roll 错位扣除。"""
    pool_keys = list(shifted_encoded_map.keys())
    target_keys = (
        [str(k) for k in eval_shift_keys if str(k) in shifted_encoded_map]
        if eval_shift_keys is not None
        else list(pool_keys)
    )
    clean_preds = clean_encoded["predictions"]
    if not isinstance(clean_preds, torch.Tensor):
        raise TypeError("clean_encoded['predictions'] 必须为 torch.Tensor。")
    shifted_preds = {
        k: shifted_encoded_map[k]["predictions"]  # type: ignore[index]
        for k in pool_keys
    }
    deb_c_pred, deb_s_pred, clean_div, shifted_div = (
        apply_preroll_m_eval_subtraction_to_tensors(
            clean_preds,
            shifted_preds,
            m_eval=m_eval,
            eta=eta,
            eval_shift_keys=target_keys,
            normalize_inputs=normalize_inputs,
        )
    )
    new_clean = dict(clean_encoded)
    new_clean["predictions"] = deb_c_pred
    new_shifted: dict[str, dict] = {}
    for k in target_keys:
        copied = dict(shifted_encoded_map[k])
        copied["predictions"] = deb_s_pred[k]
        new_shifted[k] = copied

    rev_c = clean_encoded.get("revision_predictions")
    if isinstance(rev_c, Mapping):
        new_c_rev: dict = {}
        new_s_rev: dict[str, dict] = {k: {} for k in target_keys}
        for delay, c_delay_t in sorted(rev_c.items(), key=lambda kv: int(kv[0])):
            if not isinstance(c_delay_t, torch.Tensor):
                continue
            s_delay_map = {
                k: shifted_encoded_map[k]["revision_predictions"][delay]  # type: ignore[index]
                for k in pool_keys
            }
            deb_cd, deb_sd, _, _ = apply_preroll_m_eval_subtraction_to_tensors(
                c_delay_t,
                s_delay_map,
                m_eval=m_eval,
                eta=eta,
                eval_shift_keys=target_keys,
                normalize_inputs=normalize_inputs,
            )
            new_c_rev[delay] = deb_cd
            for k in target_keys:
                new_s_rev[k][delay] = deb_sd[k]
        new_clean["revision_predictions"] = new_c_rev
        for k in target_keys:
            new_shifted[k]["revision_predictions"] = new_s_rev[k]

    return new_clean, new_shifted, clean_div, shifted_div


def _build_eta_entry(
    eta: float,
    clean_summary: dict,
    condition_entries: list[dict],
    shifted_aggregate: dict,
    *,
    baseline_entry: dict | None = None,
    method_label: str | None = None,
    short_label: str | None = None,
    method_id: str | None = None,
    sweep_key: str | None = None,
    cms_variant: str | None = None,
    clean_divisor_override: int | None = None,
    shifted_divisor_override: int | None = None,
) -> dict:
    """构建单个 eta 权重下的完整汇总字典（含相对 eta=0 的增益差值与多方法标识）。"""
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
        "clean_divisor": (
            int(clean_divisor_override)
            if clean_divisor_override is not None
            else int(num_shifts)
        ),
        "shifted_loo_divisor": (
            int(shifted_divisor_override)
            if shifted_divisor_override is not None
            else int(max(1, num_shifts - 1))
        ),
        "clean_reference": clean_summary,
        "conditions": condition_entries,
        "shifted_aggregate": shifted_aggregate,
    }
    if short_label is not None:
        entry["short_label"] = str(short_label)
    if method_id is not None:
        entry["method_id"] = str(method_id)
    if sweep_key is not None:
        entry["sweep_key"] = str(sweep_key)
    if cms_variant is not None:
        entry["cms_variant"] = str(cms_variant)
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
    cms_mode: str = "causal_ema",
    eta_values: Iterable[float] | None = None,
    ema_decay: float = DEFAULT_CAUSAL_CMS_EMA_DECAY,
    pos_ema_decay: float = DEFAULT_POS_CAUSAL_EMA_DECAY,
    min_bucket_count: int = DEFAULT_POS_MIN_BUCKET_COUNT,
    include_global_causal_baseline: bool = True,
    global_baseline_eta: float = 0.5,
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
    """执行严格在线因果 Causal EMA-CMS / 句内位置分桶 Pos-Locked Causal EMA-CMS 扫描。

    同时支持 M_eval = 0（1x 前向纯单流推理）与 M_eval >= 1（Pre-Roll 负对照审计）；
    当 cms_mode == 'pos_causal_ema' 且 include_global_causal_baseline=True 时，自动在同表并列输出：
    - Raw (eta=0.0)
    - Global Causal EMA-CMS (causal_cms:eta=0.5)
    - Pos-Locked Causal EMA-CMS (pos_causal_cms:eta in eta_values)
    """
    resolved_mode = normalize_cms_mode(cms_mode, default="causal_ema")
    if resolved_mode not in {"causal_ema", "pos_causal_ema"}:
        raise ValueError(
            f"run_causal_ema_common_mode_subtraction_sweep 仅支持 'causal_ema' 或 'pos_causal_ema'，收到：{cms_mode!r}"
        )
    default_etas = (
        DEFAULT_POS_CAUSAL_CMS_ETAS
        if resolved_mode == "pos_causal_ema"
        else DEFAULT_CAUSAL_CMS_ETAS
    )
    etas = parse_eta_values(eta_values, default=default_etas)
    alpha = float(ema_decay)
    alpha_pos = float(pos_ema_decay)
    min_cnt = max(1, int(min_bucket_count))
    eval_table = getattr(eval_ds, "table", None)

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
        short_label="Raw (eta=0)",
        method_id="raw",
        sweep_key="0",
        cms_variant="raw",
        clean_divisor_override=0,
        shifted_divisor_override=0,
    )
    sweep_results: list[dict] = [eta0_entry]

    def _evaluate_single_stream_variant(
        variant_mode: str,
        eta_val: float,
        *,
        method_label: str,
        short_label: str,
        method_id: str,
        sweep_key: str,
    ) -> dict:
        if variant_mode == "pos_causal_ema":
            deb_clean_enc = apply_position_causal_ema_subtraction_to_encoded(
                clean_encoded,
                mu0_payload=mu0_payload,
                eta=eta_val,
                ema_decay=alpha,
                pos_ema_decay=alpha_pos,
                min_bucket_count=min_cnt,
                eval_table=eval_table,
                normalize_inputs=normalize_inputs,
                normalize_outputs=normalize_outputs,
            )
        else:
            deb_clean_enc = apply_causal_ema_subtraction_to_encoded(
                clean_encoded,
                mu0_payload=mu0_payload,
                eta=eta_val,
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
            s_table = getattr(s_ds, "table", eval_table)
            if variant_mode == "pos_causal_ema":
                deb_s_enc = apply_position_causal_ema_subtraction_to_encoded(
                    shifted_encoded_map[key],
                    mu0_payload=mu0_payload,
                    eta=eta_val,
                    ema_decay=alpha,
                    pos_ema_decay=alpha_pos,
                    min_bucket_count=min_cnt,
                    eval_table=s_table,
                    normalize_inputs=normalize_inputs,
                    normalize_outputs=normalize_outputs,
                )
            else:
                deb_s_enc = apply_causal_ema_subtraction_to_encoded(
                    shifted_encoded_map[key],
                    mu0_payload=mu0_payload,
                    eta=eta_val,
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
        return _build_eta_entry(
            eta_val,
            deb_clean_row,
            deb_conditions,
            deb_agg,
            baseline_entry=eta0_entry,
            method_label=method_label,
            short_label=short_label,
            method_id=method_id,
            sweep_key=sweep_key,
            cms_variant=variant_mode,
            clean_divisor_override=1,
            shifted_divisor_override=1,
        )

    # 2. 若为 pos_causal_ema，可选附带一条全局 Causal EMA 对照行（默认 eta=0.5）以直观体现分桶收益
    if (
        resolved_mode == "pos_causal_ema"
        and include_global_causal_baseline
        and float(global_baseline_eta) > 0.0
    ):
        g_eta = float(global_baseline_eta)
        sweep_results.append(
            _evaluate_single_stream_variant(
                "causal_ema",
                g_eta,
                method_label=f"Global Causal EMA-CMS (eta={g_eta:g}, alpha={alpha:g})",
                short_label=f"Global-EMA({g_eta:g})",
                method_id=f"causal_cms:eta={g_eta:g}",
                sweep_key=f"causal_cms:eta={g_eta:g}",
            )
        )

    # 3. 扫描目标模式的每个 eta > 0（M_eval = 0 单流在线推理）
    for eta in etas:
        if resolved_mode == "pos_causal_ema":
            sweep_results.append(
                _evaluate_single_stream_variant(
                    "pos_causal_ema",
                    eta,
                    method_label=(
                        f"Pos-Locked Causal EMA-CMS "
                        f"(eta={eta:g}, alpha_pos={alpha_pos:g}, B={len(POSITION_BUCKET_LABELS)})"
                    ),
                    short_label=f"Pos-EMA({eta:g})",
                    method_id=f"pos_causal_cms:eta={eta:g}",
                    sweep_key=f"pos_causal_cms:eta={eta:g}",
                )
            )
        else:
            sweep_results.append(
                _evaluate_single_stream_variant(
                    "causal_ema",
                    eta,
                    method_label=f"Causal EMA-CMS (eta={eta:g}, alpha={alpha:g})",
                    short_label=f"Causal-EMA({eta:g})",
                    method_id=f"causal_cms:eta={eta:g}",
                    sweep_key=f"{eta:g}",
                )
            )

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

    if resolved_mode == "pos_causal_ema":
        formula_block = {
            "clean": (
                "y_tilde_{c,r} = normalize(y_hat_{c,r} - eta * mu_ref(r, b(i))), "
                "mu_ref = mu_pos[r, b(i)] if count[b(i)] >= min_bucket_count else mu_global[r]"
            ),
            "shifted_leave_one_out": (
                "independent single-stream Position-Locked Causal EMA-CMS per shifted stream "
                "(sentence-boundary updates, zero cross-stream leakage)"
            ),
            "cms_mode": "pos_causal_ema",
            "ema_decay": alpha,
            "pos_ema_decay": alpha_pos,
            "min_bucket_count": min_cnt,
            "position_buckets": list(POSITION_BUCKET_LABELS),
            "num_position_buckets": len(POSITION_BUCKET_LABELS),
            "num_state_vectors_stage23": 3 * len(POSITION_BUCKET_LABELS),
            "mu0_source": mu0_source,
            "shift_mode": str(shift_mode),
            "clean_requires_shifted_forwards": False,
            "num_shifted_conditions_M": len(shift_meta),
            "clean_divisor": 1,
            "shifted_loo_divisor": 1,
            "shift_seconds": [sec for sec, _, _ in shift_meta],
            "normalize_before_subtraction": bool(normalize_inputs),
            "normalize_after_subtraction": bool(normalize_outputs),
        }
        exp_type = "position_causal_ema_common_mode_subtraction"
    else:
        formula_block = {
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
        }
        exp_type = "causal_ema_common_mode_subtraction"

    report_payload = {
        "schema_version": 1,
        "experiment_type": exp_type,
        "cms_mode": resolved_mode,
        "shift_mode": str(shift_mode),
        "dataset": dataset,
        "stage": stage,
        "split": split,
        "seed": int(seed),
        "exp_tag": exp_tag,
        "word_exp_tag": word_exp_tag,
        "formula": formula_block,
        "eta_grid": [0.0, *etas],
        "best_eta_by_clean_vocab50_top10": float(best_by_clean50["eta"]),
        "best_eta_by_clean_minus_shifted_vocab50_top10": float(best_by_gap50["eta"]),
        "best_method_id_by_clean_vocab50_top10": str(
            best_by_clean50.get("method_id", f"eta={best_by_clean50['eta']:g}")
        ),
        "best_method_id_by_clean_minus_shifted_vocab50_top10": str(
            best_by_gap50.get("method_id", f"eta={best_by_gap50['eta']:g}")
        ),
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


def run_preroll_m_eval_tradeoff_sweep(
    *,
    dataset: str,
    stage: str,
    split: str,
    seed: int,
    exp_tag: str,
    word_exp_tag: str,
    clean_encoded: dict,
    shifted_encoded_by_shift: Sequence[tuple[float, str, str, dict]],
    eval_ds,
    shifted_ds_by_shift: Mapping[float, object] | None = None,
    vocabulary: list[str],
    vocab50: list[str],
    cfg: dict,
    mu0_payload: Mapping[str, object] | torch.Tensor | None = None,
    pos_etas: Iterable[float] = DEFAULT_POS_CAUSAL_CMS_ETAS,
    global_causal_etas: Iterable[float] = DEFAULT_CAUSAL_CMS_ETAS,
    preroll_etas: Iterable[float] = (0.8, 1.0),
    m_eval_grid: Iterable[int] = DEFAULT_PREROLL_M_EVAL_GRID,
    ema_decay: float = DEFAULT_CAUSAL_CMS_EMA_DECAY,
    pos_ema_decay: float = DEFAULT_POS_CAUSAL_EMA_DECAY,
    min_bucket_count: int = DEFAULT_POS_MIN_BUCKET_COUNT,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
    compute_metrics_fn: Callable[..., dict],
    extract_summary_row_fn: Callable[..., dict],
    summarize_shifted_fn: Callable[[dict, list[dict]], dict],
    output_dir: str | Path | None = None,
) -> dict:
    """在同一组已编码 Clean + Pre-Roll Shifted 特征上统一对比：
    - M_eval = 0（1x 前向）：Raw、Global Causal EMA、Position-Locked Causal EMA
    - M_eval in {1, 2, 4, 6}（(1 + M_eval)x 前向）：零回绕 Pre-Roll 错位窗口扣除
    """
    pos_eta_list = parse_eta_values(pos_etas, default=DEFAULT_POS_CAUSAL_CMS_ETAS)
    global_eta_list = parse_eta_values(global_causal_etas, default=DEFAULT_CAUSAL_CMS_ETAS)
    preroll_eta_list = parse_eta_values(preroll_etas, default=(0.8, 1.0))

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
    pool_size = len(shift_meta)
    if pool_size < 2:
        raise ValueError(
            f"Pre-Roll M_eval 权衡扫描至少需要 2 个历史错位流以执行无自泄漏负对照评测，当前仅有 {pool_size} 个。"
        )
    valid_m_evals = sorted(
        {
            int(m)
            for m in m_eval_grid
            if 1 <= int(m) <= pool_size
        }
    )
    eval_table = getattr(eval_ds, "table", None)

    def _eval_pair(
        c_enc: dict,
        s_enc_map: Mapping[str, dict],
    ) -> tuple[dict, list[dict], dict]:
        c_m = compute_metrics_fn(
            c_enc, eval_ds, vocabulary, vocab50, cfg, stage=stage, split=split
        )
        c_row = extract_summary_row_fn("clean_0s", "Clean (0s)", 0.0, c_m)
        s_rows: list[dict] = []
        conds: list[dict] = []
        for sec, key, label in shift_meta:
            s_ds = shifted_ds_lookup.get(sec, eval_ds)
            s_m = compute_metrics_fn(
                s_enc_map[key],
                s_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            s_row = extract_summary_row_fn(
                key, label, sec, s_m, clean_summary=c_row
            )
            s_rows.append(s_row)
            conds.append(
                {
                    "key": key,
                    "label": label,
                    "shift_seconds": sec,
                    "summary": s_row,
                }
            )
        s_agg = summarize_shifted_fn(c_row, s_rows)
        return c_row, conds, s_agg

    def _pack_tradeoff_row(
        *,
        method_id: str,
        method_family: str,
        short_label: str,
        label: str,
        m_eval: int,
        eta: float,
        clean_div: int,
        shifted_div: int,
        c_row: dict,
        conds: list[dict],
        s_agg: dict,
        baseline_entry: dict | None,
    ) -> dict:
        item = _build_eta_entry(
            eta,
            c_row,
            conds,
            s_agg,
            baseline_entry=baseline_entry,
            method_label=label,
            short_label=short_label,
            method_id=method_id,
            sweep_key=method_id,
            cms_variant=method_family,
            clean_divisor_override=clean_div,
            shifted_divisor_override=shifted_div,
        )
        item["method_family"] = method_family
        item["m_eval"] = int(m_eval)
        item["forwards_per_word"] = int(1 + m_eval)
        return item

    # 1) Raw (M_eval = 0, 1x forward)
    raw_c, raw_conds, raw_agg = _eval_pair(clean_encoded, shifted_encoded_map)
    raw_item = _pack_tradeoff_row(
        method_id="raw",
        method_family="raw",
        short_label="Raw (1x)",
        label="Raw Single-Pass (M_eval=0, 1x)",
        m_eval=0,
        eta=0.0,
        clean_div=0,
        shifted_div=0,
        c_row=raw_c,
        conds=raw_conds,
        s_agg=raw_agg,
        baseline_entry=None,
    )
    rows: list[dict] = [raw_item]

    # 2) Global Online Causal EMA (M_eval = 0, 1x forward)
    for g_eta in global_eta_list:
        g_clean = apply_causal_ema_subtraction_to_encoded(
            clean_encoded,
            mu0_payload=mu0_payload,
            eta=g_eta,
            ema_decay=ema_decay,
            normalize_inputs=normalize_inputs,
            normalize_outputs=normalize_outputs,
        )
        g_shifted = {
            key: apply_causal_ema_subtraction_to_encoded(
                shifted_encoded_map[key],
                mu0_payload=mu0_payload,
                eta=g_eta,
                ema_decay=ema_decay,
                normalize_inputs=normalize_inputs,
                normalize_outputs=normalize_outputs,
            )
            for key in shifted_encoded_map
        }
        gc_row, g_conds, g_agg = _eval_pair(g_clean, g_shifted)
        rows.append(
            _pack_tradeoff_row(
                method_id=f"causal_cms:eta={g_eta:g}",
                method_family="causal_ema",
                short_label=f"Global-EMA({g_eta:g})",
                label=f"Global Causal EMA (M_eval=0, 1x, eta={g_eta:g})",
                m_eval=0,
                eta=g_eta,
                clean_div=1,
                shifted_div=1,
                c_row=gc_row,
                conds=g_conds,
                s_agg=g_agg,
                baseline_entry=raw_item,
            )
        )

    # 3) Position-Stratified Online Causal EMA (M_eval = 0, 1x forward, 21 vectors)
    for p_eta in pos_eta_list:
        p_clean = apply_position_causal_ema_subtraction_to_encoded(
            clean_encoded,
            mu0_payload=mu0_payload,
            eta=p_eta,
            ema_decay=ema_decay,
            pos_ema_decay=pos_ema_decay,
            min_bucket_count=min_bucket_count,
            eval_table=eval_table,
            normalize_inputs=normalize_inputs,
            normalize_outputs=normalize_outputs,
        )
        p_shifted = {}
        for key in shifted_encoded_map:
            sec = shift_sec_by_key[key]
            s_ds = shifted_ds_lookup.get(sec, eval_ds)
            s_table = getattr(s_ds, "table", eval_table)
            p_shifted[key] = apply_position_causal_ema_subtraction_to_encoded(
                shifted_encoded_map[key],
                mu0_payload=mu0_payload,
                eta=p_eta,
                ema_decay=ema_decay,
                pos_ema_decay=pos_ema_decay,
                min_bucket_count=min_bucket_count,
                eval_table=s_table,
                normalize_inputs=normalize_inputs,
                normalize_outputs=normalize_outputs,
            )
        pc_row, p_conds, p_agg = _eval_pair(p_clean, p_shifted)
        rows.append(
            _pack_tradeoff_row(
                method_id=f"pos_causal_cms:eta={p_eta:g}",
                method_family="pos_causal_ema",
                short_label=f"Pos-EMA({p_eta:g})",
                label=f"Pos-Locked Causal EMA (M_eval=0, 1x, eta={p_eta:g})",
                m_eval=0,
                eta=p_eta,
                clean_div=1,
                shifted_div=1,
                c_row=pc_row,
                conds=p_conds,
                s_agg=p_agg,
                baseline_entry=raw_item,
            )
        )

    # 4) Strict Pre-Roll Shifted-Window Subtraction (M_eval in {1, 2, 4, 6}, (1 + M_eval)x forwards)
    for m_val in valid_m_evals:
        for pr_eta in preroll_eta_list:
            pr_clean, pr_shifted, c_div, s_div = (
                apply_preroll_m_eval_subtraction_to_encoded(
                    clean_encoded,
                    shifted_encoded_map,
                    m_eval=m_val,
                    eta=pr_eta,
                    eval_shift_keys=[k for _, k, _ in shift_meta],
                    normalize_inputs=normalize_inputs,
                )
            )
            prc_row, pr_conds, pr_agg = _eval_pair(pr_clean, pr_shifted)
            rows.append(
                _pack_tradeoff_row(
                    method_id=f"preroll_loo:M={m_val}:eta={pr_eta:g}",
                    method_family="preroll_shifted_subtraction",
                    short_label=f"PreRoll M={m_val} ({1 + m_val}x, eta={pr_eta:g})",
                    label=(
                        f"Pre-Roll Shifted Subtraction "
                        f"(M_eval={m_val}, {1 + m_val}x, eta={pr_eta:g})"
                    ),
                    m_eval=m_val,
                    eta=pr_eta,
                    clean_div=c_div,
                    shifted_div=s_div,
                    c_row=prc_row,
                    conds=pr_conds,
                    s_agg=pr_agg,
                    baseline_entry=raw_item,
                )
            )

    payload = {
        "schema_version": 1,
        "experiment_type": "preroll_m_eval_and_pos_causal_tradeoff",
        "dataset": dataset,
        "stage": stage,
        "split": split,
        "seed": int(seed),
        "exp_tag": exp_tag,
        "word_exp_tag": word_exp_tag,
        "pool_shift_seconds": [sec for sec, _, _ in shift_meta],
        "m_eval_grid": valid_m_evals,
        "pos_etas": pos_eta_list,
        "global_causal_etas": global_eta_list,
        "preroll_etas": preroll_eta_list,
        "ema_decay": float(ema_decay),
        "pos_ema_decay": float(pos_ema_decay),
        "min_bucket_count": int(min_bucket_count),
        "rows": rows,
    }
    if output_dir is not None:
        out_p = Path(output_dir) / "exploratory"
        out_p.mkdir(parents=True, exist_ok=True)
        json_p = out_p / f"preroll_m_eval_tradeoff_{split}.json"
        json_p.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        payload["exploratory_json"] = str(json_p)
    return payload


def plot_preroll_m_eval_tradeoff_figure(
    summary_payload: Mapping[str, object],
    output_dir: str | Path,
) -> dict[str, str]:
    """绘制多种子（或单种子）Pre-Roll M_eval 算力-特异性权衡与 Pos-Locked Causal EMA 对比双联图（PNG + PDF）。"""
    import matplotlib.pyplot as plt
    from braindecoding.evaluation.evaluation_figures import evaluation_plot_context

    rows = list(summary_payload.get("methods", []))  # type: ignore[arg-type]
    if not rows:
        return {}
    split = str(summary_payload.get("split", "test"))
    fig_dir = Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / f"preroll_m_eval_tradeoff_{split}.png"
    pdf_path = fig_dir / f"preroll_m_eval_tradeoff_{split}.pdf"

    preroll_by_eta: dict[float, list[dict]] = {}
    online_rows: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        fam = str(r.get("method_family", ""))
        if fam == "preroll_shifted_subtraction":
            eta_k = round(float(r.get("eta", 1.0)), 4)
            preroll_by_eta.setdefault(eta_k, []).append(r)
        elif int(r.get("m_eval", 0)) == 0:
            online_rows.append(r)

    raw_row = next((r for r in online_rows if r.get("method_id") == "raw"), None)

    with evaluation_plot_context():
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=220)

        # Panel A: Clean & Shifted 50w Macro Top-10 (%) vs Forward Passes per Word (1 + M_eval)
        ax_left.axhline(
            20.0,
            color="#666666",
            linestyle=":",
            linewidth=1.2,
            label="50-Word Chance Floor (20.0%)",
        )
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
        for idx, (eta_val, pr_list) in enumerate(sorted(preroll_by_eta.items())):
            color = palette[idx % len(palette)]
            ordered = sorted(pr_list, key=lambda x: int(x.get("m_eval", 0)))
            xs = [int(x["forwards_per_word"]) for x in ordered]
            c_means = [100.0 * float(x["clean_vocab50_macro_top10_mean"]) for x in ordered]
            c_stds = [100.0 * float(x["clean_vocab50_macro_top10_std"]) for x in ordered]
            s_means = [100.0 * float(x["shifted_vocab50_macro_top10_mean"]) for x in ordered]
            s_stds = [100.0 * float(x["shifted_vocab50_macro_top10_std"]) for x in ordered]
            if raw_row is not None:
                xs = [1, *xs]
                c_means = [100.0 * float(raw_row["clean_vocab50_macro_top10_mean"]), *c_means]
                c_stds = [100.0 * float(raw_row["clean_vocab50_macro_top10_std"]), *c_stds]
                s_means = [100.0 * float(raw_row["shifted_vocab50_macro_top10_mean"]), *s_means]
                s_stds = [100.0 * float(raw_row["shifted_vocab50_macro_top10_std"]), *s_stds]
            ax_left.errorbar(
                xs,
                c_means,
                yerr=c_stds,
                marker="o",
                linewidth=1.9,
                capsize=3,
                color=color,
                label=f"Pre-Roll Clean (eta={eta_val:g})",
            )
            ax_left.errorbar(
                xs,
                s_means,
                yerr=s_stds,
                marker="s",
                linestyle="--",
                linewidth=1.6,
                capsize=3,
                color=color,
                alpha=0.85,
                label=f"Pre-Roll Shifted (eta={eta_val:g})",
            )

        # Plot 1x-forward Pos-Locked Causal EMA & Global Causal EMA markers at x=1
        for r in online_rows:
            mid = str(r.get("method_id", ""))
            if mid == "raw":
                continue
            c_m = 100.0 * float(r["clean_vocab50_macro_top10_mean"])
            s_m = 100.0 * float(r["shifted_vocab50_macro_top10_mean"])
            if mid.startswith("pos_causal_cms:"):
                eta_v = float(r.get("eta", 0.8))
                if abs(eta_v - 0.8) < 1e-6 or abs(eta_v - 1.0) < 1e-6:
                    mk_color = "#d62728" if abs(eta_v - 1.0) < 1e-6 else "#e377c2"
                    ax_left.scatter(
                        [1.0],
                        [c_m],
                        marker="*",
                        s=170,
                        color=mk_color,
                        zorder=6,
                        label=f"{r['short_label']} Clean (1x)",
                    )
                    ax_left.scatter(
                        [1.0],
                        [s_m],
                        marker="D",
                        s=75,
                        color=mk_color,
                        zorder=6,
                        label=f"{r['short_label']} Shifted (1x)",
                    )

        ax_left.set_xlabel("Inference Forward Passes per Word (1 + M_eval)")
        ax_left.set_ylabel("50-Word Macro Top-10 (%)")
        ax_left.set_title("A. Compute-Specificity Curve (Clean vs Shifted)")
        ax_left.grid(True, linestyle=":", alpha=0.45)
        ax_left.legend(fontsize=7.5, loc="center right")

        # Panel B: Pareto Frontier — Clean-Shifted Specificity Gap (pp) vs Clean 50w Macro Top-10 (%)
        for r in rows:
            if not isinstance(r, dict):
                continue
            fam = str(r.get("method_family", ""))
            x_gap = 100.0 * float(r["gap_vocab50_macro_top10_mean"])
            y_cln = 100.0 * float(r["clean_vocab50_macro_top10_mean"])
            if fam == "raw":
                color, marker, size = "#333333", "o", 85
            elif fam == "causal_ema":
                color, marker, size = "#7f7f7f", "^", 95
            elif fam == "pos_causal_ema":
                color, marker, size = "#d62728", "*", 175
            else:
                color, marker, size = "#1f77b4", "s", 75
            ax_right.scatter([x_gap], [y_cln], c=color, marker=marker, s=size, zorder=5)
            ax_right.annotate(
                str(r.get("short_label", r.get("method_id", ""))),
                (x_gap, y_cln),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.5,
            )

        ax_right.set_xlabel("Clean - Shifted Net Specificity Gap (pp)")
        ax_right.set_ylabel("Clean 50-Word Macro Top-10 (%)")
        ax_right.set_title("B. Clean Accuracy vs Specificity Gap Frontier")
        ax_right.grid(True, linestyle=":", alpha=0.45)

        fig.tight_layout()
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    return {
        "preroll_m_eval_tradeoff_png": str(png_path),
        "preroll_m_eval_tradeoff_pdf": str(pdf_path),
    }


def summarize_multi_seed_preroll_m_eval_tradeoff(
    per_seed_payloads: Sequence[Mapping[str, object]],
    *,
    output_dir: str | Path | None = None,
    plot: bool = True,
) -> dict:
    """跨多种子聚合 Pre-Roll M_eval 与 Pos-Locked Causal EMA 权衡结果，导出 CSV/Markdown 与 PNG+PDF 双联图。"""
    import csv

    if not per_seed_payloads:
        raise ValueError("per_seed_payloads 不能为空。")
    first = per_seed_payloads[0]
    split = str(first.get("split", "test"))
    dataset = str(first.get("dataset", "pallier2025"))
    stage = str(first.get("stage", "prefix_revision"))
    exp_tag = str(first.get("exp_tag", ""))
    seeds = [int(p.get("seed", i)) for i, p in enumerate(per_seed_payloads)]

    ordered_ids: list[str] = []
    by_method: dict[str, list[dict]] = {}
    for p in per_seed_payloads:
        for r in p.get("rows", []):  # type: ignore[union-attr]
            if not isinstance(r, dict):
                continue
            mid = str(r.get("method_id", ""))
            if not mid:
                continue
            if mid not in by_method:
                ordered_ids.append(mid)
                by_method[mid] = []
            by_method[mid].append(r)

    def _mean_std(vals: Sequence[float]) -> tuple[float, float]:
        arr = np.asarray(list(vals), dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        return mean_v, std_v

    agg_rows: list[dict] = []
    for mid in ordered_ids:
        items = by_method[mid]
        ref = items[0]
        c50_m, c50_s = _mean_std(
            [float(x["clean_reference"].get("vocab50_macro_top10", 0.0)) for x in items]
        )
        sh50_m, sh50_s = _mean_std(
            [float(x["shifted_aggregate"].get("mean_vocab50_macro_top10", 0.0)) for x in items]
        )
        gap50_m, gap50_s = _mean_std(
            [
                float(
                    x["shifted_aggregate"].get(
                        "clean_minus_shifted_vocab50_macro_top10", 0.0
                    )
                )
                for x in items
            ]
        )
        cfv_m, cfv_s = _mean_std(
            [
                float(
                    x["clean_reference"].get(
                        "full_vocab_word_macro_top10",
                        x["clean_reference"].get("full_vocab_macro_top10", 0.0),
                    )
                )
                for x in items
            ]
        )
        cfv_mic_m, cfv_mic_s = _mean_std(
            [float(x["clean_reference"].get("full_vocab_micro_top10", 0.0)) for x in items]
        )
        shfv_m, shfv_s = _mean_std(
            [
                float(
                    x["shifted_aggregate"].get(
                        "mean_full_vocab_word_macro_top10",
                        x["shifted_aggregate"].get("mean_full_vocab_macro_top10", 0.0),
                    )
                )
                for x in items
            ]
        )
        gapfv_m, gapfv_s = _mean_std(
            [
                float(
                    x["shifted_aggregate"].get(
                        "clean_minus_shifted_full_vocab_word_macro_top10",
                        x["shifted_aggregate"].get(
                            "clean_minus_shifted_full_vocab_macro_top10", 0.0
                        ),
                    )
                )
                for x in items
            ]
        )
        k0_m, k0_s = _mean_std(
            [
                float(
                    x["clean_reference"].get(
                        "k0_vocab50_macro_top10",
                        x["clean_reference"].get("vocab50_macro_top10", 0.0),
                    )
                )
                for x in items
            ]
        )
        k1_m, k1_s = _mean_std(
            [
                float(
                    x["clean_reference"].get(
                        "k1_vocab50_macro_top10",
                        x["clean_reference"].get("vocab50_macro_top10", 0.0),
                    )
                )
                for x in items
            ]
        )
        k2_m, k2_s = _mean_std(
            [
                float(
                    x["clean_reference"].get(
                        "k2_vocab50_macro_top10",
                        x["clean_reference"].get("vocab50_macro_top10", 0.0),
                    )
                )
                for x in items
            ]
        )
        agg_rows.append(
            {
                "method_id": mid,
                "method_family": str(ref.get("method_family", "")),
                "short_label": str(ref.get("short_label", mid)),
                "label": str(ref.get("label", mid)),
                "m_eval": int(ref.get("m_eval", 0)),
                "forwards_per_word": int(ref.get("forwards_per_word", 1)),
                "eta": float(ref.get("eta", 0.0)),
                "num_seeds": len(items),
                "clean_vocab50_macro_top10_mean": c50_m,
                "clean_vocab50_macro_top10_std": c50_s,
                "shifted_vocab50_macro_top10_mean": sh50_m,
                "shifted_vocab50_macro_top10_std": sh50_s,
                "gap_vocab50_macro_top10_mean": gap50_m,
                "gap_vocab50_macro_top10_std": gap50_s,
                "clean_full_vocab_word_macro_top10_mean": cfv_m,
                "clean_full_vocab_word_macro_top10_std": cfv_s,
                "clean_full_vocab_micro_top10_mean": cfv_mic_m,
                "clean_full_vocab_micro_top10_std": cfv_mic_s,
                "shifted_full_vocab_word_macro_top10_mean": shfv_m,
                "shifted_full_vocab_word_macro_top10_std": shfv_s,
                "gap_full_vocab_word_macro_top10_mean": gapfv_m,
                "gap_full_vocab_word_macro_top10_std": gapfv_s,
                "k0_clean_vocab50_macro_top10_mean": k0_m,
                "k0_clean_vocab50_macro_top10_std": k0_s,
                "k1_clean_vocab50_macro_top10_mean": k1_m,
                "k1_clean_vocab50_macro_top10_std": k1_s,
                "k2_clean_vocab50_macro_top10_mean": k2_m,
                "k2_clean_vocab50_macro_top10_std": k2_s,
            }
        )

    summary = {
        "schema_version": 1,
        "experiment_type": "multi_seed_preroll_m_eval_and_pos_causal_tradeoff",
        "dataset": dataset,
        "stage": stage,
        "split": split,
        "exp_tag": exp_tag,
        "seeds": seeds,
        "pool_shift_seconds": list(first.get("pool_shift_seconds", [])),  # type: ignore[arg-type]
        "methods": agg_rows,
        "figures": {},
        "exported_files": {},
    }

    if output_dir is not None:
        out_root = Path(output_dir)
        tables_dir = out_root / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tables_dir / f"preroll_m_eval_tradeoff_{split}.csv"
        md_path = tables_dir / f"preroll_m_eval_tradeoff_{split}.md"
        json_path = out_root / f"preroll_m_eval_tradeoff_{split}.json"

        fieldnames = list(agg_rows[0].keys()) if agg_rows else []
        with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in agg_rows:
                writer.writerow(r)

        def _fmt_pct(m_val: float, s_val: float, n_s: int) -> str:
            if n_s > 1:
                return f"{100.0 * m_val:.2f} ± {100.0 * s_val:.2f}"
            return f"{100.0 * m_val:.2f}"

        md_lines = [
            f"# Online Causal Inference & Pre-Roll $M_{{\\text{{eval}}}}$ Trade-Off ({dataset} / {stage} / {split}, n={len(seeds)} seeds)",
            "",
            f"- **Experiment Tag**: `{exp_tag}`",
            f"- **Seeds**: `{seeds}`",
            f"- **Pre-Roll Negative-Control Shifts (s)**: `{summary['pool_shift_seconds']}`",
            "",
            "| Method | $M_{\\text{eval}}$ | Forwards | 50w Clean (%) | 50w Shifted (%) | 50w Gap (pp) | Full Word-Macro (%) | Full Micro (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in agg_rows:
            ns = int(r["num_seeds"])
            md_lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{r['short_label']}`",
                        str(r["m_eval"]),
                        f"{r['forwards_per_word']}x",
                        _fmt_pct(
                            r["clean_vocab50_macro_top10_mean"],
                            r["clean_vocab50_macro_top10_std"],
                            ns,
                        ),
                        _fmt_pct(
                            r["shifted_vocab50_macro_top10_mean"],
                            r["shifted_vocab50_macro_top10_std"],
                            ns,
                        ),
                        _fmt_pct(
                            r["gap_vocab50_macro_top10_mean"],
                            r["gap_vocab50_macro_top10_std"],
                            ns,
                        ),
                        _fmt_pct(
                            r["clean_full_vocab_word_macro_top10_mean"],
                            r["clean_full_vocab_word_macro_top10_std"],
                            ns,
                        ),
                        _fmt_pct(
                            r["clean_full_vocab_micro_top10_mean"],
                            r["clean_full_vocab_micro_top10_std"],
                            ns,
                        ),
                    ]
                )
                + " |"
            )
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        summary["exported_files"] = {
            "csv": str(csv_path),
            "markdown": str(md_path),
            "json": str(json_path),
        }
        if plot:
            figs = plot_preroll_m_eval_tradeoff_figure(summary, out_root / "figures")
            summary["figures"].update(figs)
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return summary


def evaluate_cached_pts_pos_causal_and_m_eval(
    *,
    source_dir: str | Path | None = None,
    cache_pt_paths: Sequence[str | Path] | None = None,
    output_dir: str | Path,
    split: str = "test",
    pos_etas: Iterable[float] = DEFAULT_POS_CAUSAL_CMS_ETAS,
    global_causal_etas: Iterable[float] = DEFAULT_CAUSAL_CMS_ETAS,
    preroll_etas: Iterable[float] = (0.8, 1.0),
    m_eval_grid: Iterable[int] = DEFAULT_PREROLL_M_EVAL_GRID,
    ema_decay: float = DEFAULT_CAUSAL_CMS_EMA_DECAY,
    pos_ema_decay: float = DEFAULT_POS_CAUSAL_EMA_DECAY,
    min_bucket_count: int = DEFAULT_POS_MIN_BUCKET_COUNT,
    allow_in_place: bool = False,
    plot: bool = True,
    print_report: bool = True,
) -> dict:
    """零前向、零解包、零覆盖：直接从已有 subspace_encoded_cache_{split}.pt 秒级完成多种子
    Pos-Locked Causal EMA 与 Pre-Roll M_eval 权衡评测，并将全部产物写入独立隔离目录 output_dir。
    """
    from braindecoding.evaluation.subspace_coherence_debiasing_evaluation import (
        _CachedEvalDatasetProxy,
        _build_fast_summary_evaluator,
    )
    from braindecoding.evaluation.word_retrieval_evaluation import (
        _summarize_shifted_conditions,
    )

    out_root = Path(output_dir).resolve()
    if source_dir is not None:
        src_root = Path(source_dir).resolve()
        if not allow_in_place and out_root == src_root:
            raise ValueError(
                f"为保护已有审计结果零覆盖，output_dir ({out_root}) 不得与只读 source_dir ({src_root}) 相同！"
            )
    else:
        src_root = None

    resolved_pts: list[Path] = []
    if cache_pt_paths:
        for p in cache_pt_paths:
            pt_p = Path(p)
            if not pt_p.is_file():
                raise FileNotFoundError(f"未找到特征缓存文件：{pt_p}")
            resolved_pts.append(pt_p.resolve())
    elif src_root is not None:
        direct_pt = src_root / "exploratory" / f"subspace_encoded_cache_{split}.pt"
        if direct_pt.is_file():
            resolved_pts.append(direct_pt)
        else:
            for cand in sorted(
                src_root.glob(f"seed-*/exploratory/subspace_encoded_cache_{split}.pt")
            ):
                if cand.is_file():
                    resolved_pts.append(cand.resolve())
    if not resolved_pts:
        raise FileNotFoundError(
            f"未找到任何 subspace_encoded_cache_{split}.pt（source_dir={source_dir!r}）。"
        )

    per_seed_tradeoffs: list[dict] = []
    per_seed_pos_cms: list[dict] = []

    for pt_path in resolved_pts:
        if not allow_in_place and pt_path.parent.parent.resolve() == out_root:
            raise ValueError(
                f"为防止覆盖源目录，output_dir ({out_root}) 不得等于缓存所在目录 ({pt_path.parent.parent})。"
            )
        bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
        dataset = str(bundle.get("dataset", "pallier2025"))
        stage = str(bundle.get("stage", "prefix_revision"))
        b_split = str(bundle.get("split", split))
        seed = int(bundle.get("seed", 0))
        exp_tag = str(bundle.get("exp_tag", ""))
        word_exp_tag = str(bundle.get("word_exp_tag", ""))
        mu0_payload = bundle.get("mu0_payload")
        if mu0_payload is None:
            eval_json_cand = pt_path.parent.parent / f"evaluation_{b_split}.json"
            if eval_json_cand.is_file():
                try:
                    ev_meta = json.loads(eval_json_cand.read_text(encoding="utf-8"))
                    ckpt_p_str = str((ev_meta.get("checkpoint") or {}).get("path", ""))
                    if ckpt_p_str and Path(ckpt_p_str).is_file():
                        ckpt_d = torch.load(
                            ckpt_p_str, map_location="cpu", weights_only=False
                        )
                        if isinstance(ckpt_d, Mapping) and ckpt_d.get("causal_cms_mu0") is not None:
                            mu0_payload = ckpt_d.get("causal_cms_mu0")
                except Exception:
                    mu0_payload = None

        proxy_ds = _CachedEvalDatasetProxy(
            bundle["eval_table"],
            bundle["embedding_map"],
            window_seconds=float(bundle.get("window_seconds", 1.0)),
            window_samples=int(bundle.get("window_samples", 50)),
        )
        fast_eval = _build_fast_summary_evaluator(
            proxy_ds,
            bundle["vocabulary"],
            bundle["vocab50"],
            stage=stage,
        )
        seed_out = out_root / f"seed-{seed:03d}"
        seed_out.mkdir(parents=True, exist_ok=True)

        pos_cms_res = run_causal_ema_common_mode_subtraction_sweep(
            dataset=dataset,
            stage=stage,
            split=b_split,
            seed=seed,
            exp_tag=exp_tag,
            word_exp_tag=word_exp_tag,
            clean_encoded=bundle["clean_encoded"],
            shifted_encoded_by_shift=bundle["shifted_encoded_by_shift"],
            eval_ds=proxy_ds,
            shifted_ds_by_shift={},
            vocabulary=bundle["vocabulary"],
            vocab50=bundle["vocab50"],
            cfg={},
            mu0_payload=mu0_payload,
            cms_mode="pos_causal_ema",
            eta_values=pos_etas,
            ema_decay=ema_decay,
            pos_ema_decay=pos_ema_decay,
            min_bucket_count=min_bucket_count,
            include_global_causal_baseline=True,
            global_baseline_eta=(
                float(next(iter(global_causal_etas)))
                if global_causal_etas
                else 0.5
            ),
            compute_metrics_fn=lambda enc, *_args, **_kw: enc,
            extract_summary_row_fn=lambda key, label, sec, enc, *, clean_summary=None, _fe=fast_eval: _fe(
                enc, key, label, sec, clean_summary=clean_summary
            ),
            summarize_shifted_fn=_summarize_shifted_conditions,
            output_dir=seed_out,
            plot=plot,
            print_report=print_report,
        )
        per_seed_pos_cms.append(pos_cms_res)

        tradeoff_res = run_preroll_m_eval_tradeoff_sweep(
            dataset=dataset,
            stage=stage,
            split=b_split,
            seed=seed,
            exp_tag=exp_tag,
            word_exp_tag=word_exp_tag,
            clean_encoded=bundle["clean_encoded"],
            shifted_encoded_by_shift=bundle["shifted_encoded_by_shift"],
            eval_ds=proxy_ds,
            shifted_ds_by_shift={},
            vocabulary=bundle["vocabulary"],
            vocab50=bundle["vocab50"],
            cfg={},
            mu0_payload=mu0_payload,
            pos_etas=pos_etas,
            global_causal_etas=global_causal_etas,
            preroll_etas=preroll_etas,
            m_eval_grid=m_eval_grid,
            ema_decay=ema_decay,
            pos_ema_decay=pos_ema_decay,
            min_bucket_count=min_bucket_count,
            compute_metrics_fn=lambda enc, *_args, **_kw: enc,
            extract_summary_row_fn=lambda key, label, sec, enc, *, clean_summary=None, _fe=fast_eval: _fe(
                enc, key, label, sec, clean_summary=clean_summary
            ),
            summarize_shifted_fn=_summarize_shifted_conditions,
            output_dir=seed_out,
        )
        per_seed_tradeoffs.append(tradeoff_res)

    summary = summarize_multi_seed_preroll_m_eval_tradeoff(
        per_seed_tradeoffs,
        output_dir=out_root,
        plot=plot,
    )
    if print_report and summary.get("exported_files", {}).get("markdown"):
        md_text = Path(summary["exported_files"]["markdown"]).read_text(encoding="utf-8")
        print("\n" + md_text, flush=True)
    summary["per_seed_pos_causal_cms"] = per_seed_pos_cms
    return summary


from braindecoding.evaluation.evaluation_figures import (
    plot_common_mode_subtraction_figures,
)
from braindecoding.evaluation.evaluation_reporting import (
    format_common_mode_subtraction_report,
    print_common_mode_subtraction_report,
)


def _cli_main(argv: Sequence[str] | None = None) -> int:
    """命令行秒级复评入口：直接基于已有 subspace_encoded_cache_{split}.pt 运行分桶因果中心化与 M_eval 权衡曲线。"""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Zero-retraining & zero-forward evaluation of Position-Locked Causal EMA "
            "(pos_causal_cms) and Pre-Roll M_eval trade-off from cached .pt files."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Read-only stage directory containing seed-*/exploratory/subspace_encoded_cache_{split}.pt.",
    )
    parser.add_argument(
        "--cache-pts",
        type=str,
        nargs="*",
        default=None,
        help="Explicit paths to subspace_encoded_cache_{split}.pt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Isolated output directory (must differ from --source-dir unless --allow-in-place).",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--pos-etas", type=str, default="0.5,0.8,1.0")
    parser.add_argument("--global-etas", type=str, default="0.5")
    parser.add_argument("--preroll-etas", type=str, default="0.8,1.0")
    parser.add_argument("--m-eval-grid", type=str, default="1,2,4,6")
    parser.add_argument("--ema-decay", type=float, default=DEFAULT_CAUSAL_CMS_EMA_DECAY)
    parser.add_argument("--pos-ema-decay", type=float, default=DEFAULT_POS_CAUSAL_EMA_DECAY)
    parser.add_argument("--min-bucket-count", type=int, default=DEFAULT_POS_MIN_BUCKET_COUNT)
    parser.add_argument("--allow-in-place", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    m_grid = [
        int(tok.strip())
        for tok in str(args.m_eval_grid).split(",")
        if tok.strip()
    ]
    evaluate_cached_pts_pos_causal_and_m_eval(
        source_dir=args.source_dir,
        cache_pt_paths=args.cache_pts,
        output_dir=args.output_dir,
        split=args.split,
        pos_etas=parse_eta_values(args.pos_etas, default=DEFAULT_POS_CAUSAL_CMS_ETAS),
        global_causal_etas=parse_eta_values(args.global_etas, default=DEFAULT_CAUSAL_CMS_ETAS),
        preroll_etas=parse_eta_values(args.preroll_etas, default=(0.8, 1.0)),
        m_eval_grid=m_grid,
        ema_decay=float(args.ema_decay),
        pos_ema_decay=float(args.pos_ema_decay),
        min_bucket_count=int(args.min_bucket_count),
        allow_in_place=bool(args.allow_in_place),
        plot=not bool(args.no_plot),
        print_report=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())


