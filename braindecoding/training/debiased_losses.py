"""去偏训练超参解析、中心化词表矩阵及零空间 / 差分 / 蒸馏损失函数。"""

from __future__ import annotations

from collections import Counter
import math
import os
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

DEFAULT_SIAMESE_SHIFT_SECONDS: tuple[float, ...] = (
    -15.0,
    -30.0,
    -45.0,
    -60.0,
    -90.0,
    -120.0,
)
DEFAULT_CAUSAL_PREROLL_SHIFT_SECONDS: tuple[float, ...] = (
    -10.0,
    -20.0,
    -30.0,
    -40.0,
)
VALID_DEBIAS_MODES: tuple[str, ...] = (
    "null_space",
    "amortized_timing",
    "legacy_siamese",
)
VALID_SHIFT_MODES: tuple[str, ...] = (
    "deterministic_circular",
    "preroll",
)


def _normalize_debias_mode(raw: str | None) -> str | None:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if not token or token in {"auto", "none", "default"}:
        return None
    alias_map = {
        "null_space": "null_space",
        "causal_null_space": "null_space",
        "null": "null_space",
        "tier_a": "null_space",
        "amortized_timing": "amortized_timing",
        "amortized": "amortized_timing",
        "temporal_bias_head": "amortized_timing",
        "tier_b": "amortized_timing",
        "legacy_siamese": "legacy_siamese",
        "legacy": "legacy_siamese",
        "differential_cms": "legacy_siamese",
    }
    if token not in alias_map:
        raise ValueError(
            f"不支持的 debias_mode={raw!r}，可选：{VALID_DEBIAS_MODES}"
        )
    return alias_map[token]


def _normalize_shift_mode(raw: str | None) -> str | None:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if not token or token in {"auto", "none", "default"}:
        return None
    alias_map = {
        "preroll": "preroll",
        "pre_roll": "preroll",
        "causal_preroll": "preroll",
        "deterministic_circular": "deterministic_circular",
        "circular": "deterministic_circular",
    }
    if token not in alias_map:
        raise ValueError(
            f"不支持的 shift_mode={raw!r}，可选：{VALID_SHIFT_MODES}"
        )
    return alias_map[token]


def _parse_shift_list(
    raw_shifts: str | Iterable[float | int | str] | None,
) -> list[float]:
    if raw_shifts is None:
        return list(DEFAULT_SIAMESE_SHIFT_SECONDS)
    if isinstance(raw_shifts, (int, float)):
        val = float(raw_shifts)
        return [val] if abs(val) > 1e-9 else []
    if isinstance(raw_shifts, str):
        parts = [p.strip() for p in raw_shifts.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in raw_shifts if str(p).strip()]
    parsed: list[float] = []
    for item in parts:
        val = float(item)
        if abs(val) <= 1e-9:
            raise ValueError("孪生错位分支 shift_seconds 不能包含 0。")
        if val not in parsed:
            parsed.append(val)
    return parsed


def _parse_bool_env(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"环境变量 {name}={raw!r} 不是合法布尔值。")


def resolve_siamese_debiased_config(
    cfg: dict,
    *,
    siamese_debiased: bool | None = None,
    debias_mode: str | None = None,
    shift_mode: str | None = None,
    preroll_seconds: float | None = None,
    eta_train: float | None = None,
    eta_0: float | None = None,
    eta_delta: float | None = None,
    lambda_null: float | None = None,
    lambda_delta: float | None = None,
    lambda_delta_null: float | None = None,
    lambda_distill: float | None = None,
    beta_delta_cm: float | None = None,
    shifts_per_step: int | None = None,
    shift_seconds: str | Iterable[float | int | str] | None = None,
    preextract_frozen_cnn: bool | None = None,
    temporal_bias_hidden_dim: int | None = None,
    null_space_coherent_mean: bool | None = None,
    null_space_vocab_freq_alpha: float | None = None,
) -> dict:
    """合并 YAML `siamese_debiased` 配置块、环境变量与 CLI 参数。"""
    raw_block = (
        dict(cfg.get("siamese_debiased"))
        if isinstance(cfg.get("siamese_debiased"), dict)
        else {}
    )
    ds_block = (
        cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    )
    model_block = (
        cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    )
    model_tbh = (
        model_block.get("temporal_bias_head")
        if isinstance(model_block.get("temporal_bias_head"), dict)
        else {}
    )
    tc = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}

    env_enabled = _parse_bool_env("BRAINDECODING_SIAMESE_DEBIASED")
    env_preextract = _parse_bool_env("BRAINDECODING_PREEXTRACT_FROZEN_CNN")
    env_debias_mode = os.environ.get("BRAINDECODING_DEBIAS_MODE", "").strip()
    env_shift_mode = os.environ.get("BRAINDECODING_SHIFT_MODE", "").strip()
    env_preroll = os.environ.get("BRAINDECODING_PREROLL_SECONDS", "").strip()
    env_eta = (
        os.environ.get("BRAINDECODING_ETA_0", "").strip()
        or os.environ.get("BRAINDECODING_ETA_TRAIN", "").strip()
    )
    env_eta_delta = os.environ.get("BRAINDECODING_ETA_DELTA", "").strip()
    env_lambda = os.environ.get("BRAINDECODING_LAMBDA_NULL", "").strip()
    env_lambda_delta = (
        os.environ.get("BRAINDECODING_LAMBDA_DELTA", "").strip()
        or os.environ.get("BRAINDECODING_LAMBDA_DELTA_NULL", "").strip()
    )
    env_lambda_distill = os.environ.get("BRAINDECODING_LAMBDA_DISTILL", "").strip()
    env_beta_delta_cm = os.environ.get("BRAINDECODING_BETA_DELTA_CM", "").strip()
    env_m = os.environ.get("BRAINDECODING_SHIFTS_PER_STEP", "").strip()
    env_shifts = os.environ.get("BRAINDECODING_TRAIN_SHIFT_SECONDS", "").strip()
    env_coherent_mean = _parse_bool_env("BRAINDECODING_NULL_SPACE_COHERENT_MEAN")
    env_freq_alpha = os.environ.get(
        "BRAINDECODING_NULL_SPACE_VOCAB_FREQ_ALPHA", ""
    ).strip()

    explicit_debias_mode = (
        _normalize_debias_mode(debias_mode)
        or _normalize_debias_mode(env_debias_mode)
        or _normalize_debias_mode(raw_block.get("debias_mode"))
    )
    explicit_shift_mode = (
        _normalize_shift_mode(shift_mode)
        or _normalize_shift_mode(env_shift_mode)
        or _normalize_shift_mode(raw_block.get("shift_mode"))
        or _normalize_shift_mode(ds_block.get("shift_mode"))
    )

    explicit_eta = eta_0 if eta_0 is not None else eta_train
    explicit_lambda_delta = (
        lambda_delta if lambda_delta is not None else lambda_delta_null
    )
    enabled = (
        bool(siamese_debiased)
        if siamese_debiased is not None
        else (
            env_enabled
            if env_enabled is not None
            else bool(raw_block.get("enabled", False))
        )
    )
    if (
        explicit_debias_mode is not None
        or explicit_eta is not None
        or env_eta
        or eta_delta is not None
        or env_eta_delta
        or lambda_null is not None
        or env_lambda
        or explicit_lambda_delta is not None
        or env_lambda_delta
        or lambda_distill is not None
        or env_lambda_distill
    ):
        # 显式传入了去偏模式或正则超参时自动启用孪生去偏
        if siamese_debiased is not False and env_enabled is not False:
            enabled = True

    resolved_debias_mode = explicit_debias_mode or "legacy_siamese"
    is_causal_tier = resolved_debias_mode in {"null_space", "amortized_timing"}

    default_eta_by_mode = (
        0.0
        if resolved_debias_mode == "null_space"
        else (
            float(model_tbh["eta"])
            if resolved_debias_mode == "amortized_timing"
            and model_tbh.get("eta") is not None
            else (0.40 if resolved_debias_mode == "amortized_timing" else 0.8)
        )
    )
    yaml_eta_default = (
        raw_block["eta_0"]
        if "eta_0" in raw_block and raw_block["eta_0"] is not None
        else raw_block.get("eta_train", default_eta_by_mode)
    )
    resolved_eta = float(
        explicit_eta
        if explicit_eta is not None
        else (float(env_eta) if env_eta else yaml_eta_default)
    )
    yaml_eta_delta_default = (
        raw_block["eta_delta"]
        if "eta_delta" in raw_block and raw_block["eta_delta"] is not None
        else raw_block.get("eta_delta_train", 0.0)
    )
    resolved_eta_delta = (
        0.0
        if is_causal_tier and eta_delta is None and not env_eta_delta and "eta_delta" not in raw_block
        else float(
            eta_delta
            if eta_delta is not None
            else (float(env_eta_delta) if env_eta_delta else yaml_eta_delta_default)
        )
    )
    resolved_lambda = float(
        lambda_null
        if lambda_null is not None
        else (
            float(env_lambda)
            if env_lambda
            else raw_block.get("lambda_null", 10.0)
        )
    )
    default_lambda_delta = 2.0 if is_causal_tier else 0.0
    yaml_lambda_delta_default = (
        raw_block["lambda_delta"]
        if "lambda_delta" in raw_block and raw_block["lambda_delta"] is not None
        else raw_block.get("lambda_delta_null", default_lambda_delta)
    )
    resolved_lambda_delta = float(
        explicit_lambda_delta
        if explicit_lambda_delta is not None
        else (
            float(env_lambda_delta)
            if env_lambda_delta
            else yaml_lambda_delta_default
        )
    )
    resolved_lambda_distill = float(
        lambda_distill
        if lambda_distill is not None
        else (
            float(env_lambda_distill)
            if env_lambda_distill
            else raw_block.get(
                "lambda_distill",
                1.0 if resolved_debias_mode == "amortized_timing" else 0.0,
            )
        )
    )
    yaml_beta_delta_default = (
        raw_block["beta_delta_cm"]
        if "beta_delta_cm" in raw_block and raw_block["beta_delta_cm"] is not None
        else raw_block.get("beta_delta", 0.0 if is_causal_tier else 6.0)
    )
    resolved_beta_delta_cm = float(
        beta_delta_cm
        if beta_delta_cm is not None
        else (
            float(env_beta_delta_cm)
            if env_beta_delta_cm
            else yaml_beta_delta_default
        )
    )
    resolved_shift_mode = explicit_shift_mode or (
        "preroll" if is_causal_tier else "deterministic_circular"
    )
    resolved_preroll_seconds = float(
        preroll_seconds
        if preroll_seconds is not None
        else (
            float(env_preroll)
            if env_preroll
            else raw_block.get(
                "preroll_seconds", ds_block.get("preroll_seconds", 45.0)
            )
        )
    )
    default_shift_pool = (
        DEFAULT_CAUSAL_PREROLL_SHIFT_SECONDS
        if is_causal_tier or resolved_shift_mode == "preroll"
        else DEFAULT_SIAMESE_SHIFT_SECONDS
    )
    resolved_shifts = _parse_shift_list(
        shift_seconds
        if shift_seconds is not None
        else (
            env_shifts
            if env_shifts
            else raw_block.get("shift_seconds", default_shift_pool)
        )
    )
    if enabled and not resolved_shifts:
        raise ValueError("启用 siamese_debiased 时至少需要 1 个非零 shift_seconds。")
    if resolved_shift_mode == "preroll":
        for sh in resolved_shifts:
            if sh > 0.0:
                raise ValueError(
                    f"shift_mode='preroll' 要求严格过去移位 (shift_seconds < 0)，收到 {sh}。"
                )

    default_m = (
        1
        if is_causal_tier
        else (min(2, len(resolved_shifts)) if resolved_shifts else 2)
    )
    resolved_m = int(
        shifts_per_step
        if shifts_per_step is not None
        else (
            int(env_m)
            if env_m
            else int(raw_block.get("shifts_per_step", default_m))
        )
    )
    if enabled:
        resolved_m = max(1, min(resolved_m, len(resolved_shifts)))

    resolved_preextract = (
        bool(preextract_frozen_cnn)
        if preextract_frozen_cnn is not None
        else (
            env_preextract
            if env_preextract is not None
            else bool(
                raw_block.get(
                    "preextract_frozen_cnn",
                    tc.get("preextract_frozen_cnn", enabled),
                )
            )
        )
    )
    resolved_hidden_dim = int(
        temporal_bias_hidden_dim
        if temporal_bias_hidden_dim is not None
        else raw_block.get(
            "temporal_bias_hidden_dim",
            model_tbh.get("hidden_dimension", 128),
        )
    )
    sync_rng = bool(raw_block.get("synchronize_dropout_rng", True))
    cache_to_disk = bool(raw_block.get("cache_preextracted_to_disk", True))
    default_s3_cms_eta = 0.0 if resolved_debias_mode == "amortized_timing" else 0.5
    resolved_s3_cms_eta = float(raw_block.get("s3_cms_eta", default_s3_cms_eta))
    resolved_coherent_mean = (
        bool(null_space_coherent_mean)
        if null_space_coherent_mean is not None
        else (
            env_coherent_mean
            if env_coherent_mean is not None
            else bool(raw_block.get("null_space_coherent_mean", False))
        )
    )
    resolved_freq_alpha = float(
        null_space_vocab_freq_alpha
        if null_space_vocab_freq_alpha is not None
        else (
            float(env_freq_alpha)
            if env_freq_alpha
            else float(raw_block.get("null_space_vocab_freq_alpha", 0.0))
        )
    )
    if not math.isfinite(resolved_freq_alpha) or resolved_freq_alpha < 0.0:
        raise ValueError(
            f"null_space_vocab_freq_alpha 必须为非负有限实数，收到 {resolved_freq_alpha!r}。"
        )

    resolved_excess_coherent = bool(
        raw_block.get("null_space_excess_coherent", False)
    )
    resolved_excess_tolerance = float(
        raw_block.get("null_space_excess_tolerance", 0.08)
    )
    resolved_lambda_clean_stein = float(
        raw_block.get("lambda_clean_stein", 0.0)
    )
    resolved_lambda_delta_clean_stein = float(
        raw_block.get("lambda_delta_clean_stein", 0.0)
    )
    resolved_clean_stein_ema_gamma = float(
        raw_block.get("clean_stein_ema_gamma", 0.03)
    )
    resolved_lambda_shift_lag_decorr = float(
        raw_block.get("lambda_shift_lag_decorr", 0.0)
    )
    resolved_shift_lag_decorr_max_lag = max(
        1, int(raw_block.get("shift_lag_decorr_max_lag", 3))
    )
    resolved_increment_mode = bool(
        raw_block.get(
            "null_space_increment_mode",
            resolved_excess_coherent
            or resolved_lambda_clean_stein > 0.0
            or resolved_lambda_delta_clean_stein > 0.0,
        )
    )

    return {
        "enabled": bool(enabled),
        "debias_mode": resolved_debias_mode,
        "shift_mode": resolved_shift_mode,
        "preroll_seconds": resolved_preroll_seconds,
        "eta_0": resolved_eta,
        "eta_train": resolved_eta,
        "eta_delta": resolved_eta_delta,
        "lambda_null": resolved_lambda,
        "lambda_delta": resolved_lambda_delta,
        "lambda_delta_null": resolved_lambda_delta,
        "lambda_distill": resolved_lambda_distill,
        "beta_delta_cm": resolved_beta_delta_cm,
        "temporal_bias_hidden_dim": resolved_hidden_dim,
        "null_space_coherent_mean": bool(resolved_coherent_mean),
        "null_space_vocab_freq_alpha": float(resolved_freq_alpha),
        "null_space_excess_coherent": bool(resolved_excess_coherent),
        "null_space_excess_tolerance": float(resolved_excess_tolerance),
        "null_space_increment_mode": bool(resolved_increment_mode),
        "lambda_clean_stein": float(resolved_lambda_clean_stein),
        "lambda_delta_clean_stein": float(resolved_lambda_delta_clean_stein),
        "clean_stein_ema_gamma": float(resolved_clean_stein_ema_gamma),
        "lambda_shift_lag_decorr": float(resolved_lambda_shift_lag_decorr),
        "shift_lag_decorr_max_lag": int(resolved_shift_lag_decorr_max_lag),
        "shift_seconds": resolved_shifts,
        "shifts_per_step": int(resolved_m),
        "s3_cms_eta": resolved_s3_cms_eta,
        "synchronize_dropout_rng": sync_rng,
        "preextract_frozen_cnn": bool(resolved_preextract),
        "cache_preextracted_to_disk": cache_to_disk,
    }


def _extract_dataset_word_counts(dataset_obj) -> Counter[str]:
    """仅从训练 Dataset 提取词次分布，严禁包含 val/test 划分。"""
    table = getattr(dataset_obj, "table", None)
    if table is not None and hasattr(table, "columns"):
        for split_col in ("数据划分", "split"):
            if split_col in table.columns:
                splits = {str(s).strip().lower() for s in table[split_col].dropna().unique()}
                forbidden = splits & {"val", "validation", "test"}
                if forbidden:
                    raise ValueError(
                        f"词频加权零空间禁止使用非训练划分统计，检测到 split={sorted(forbidden)}。"
                    )
    raw_words = getattr(dataset_obj, "_words", None)
    if raw_words is None:
        raw_words = getattr(dataset_obj, "words", None)
    if raw_words is not None:
        return Counter(str(w) for w in raw_words)
    if table is not None and hasattr(table, "columns"):
        for word_col in ("标准词", "normalized_word", "word"):
            if word_col in table.columns:
                return Counter(str(w) for w in table[word_col].tolist())
    raise ValueError(
        "freq_alpha > 0 时 Dataset 缺少训练词序列 (_words/words/table)，无法计算词频权重。"
    )


def build_centered_vocabulary_bank(
    dataset_obj,
    vocabulary: Iterable[str],
    device: torch.device,
    *,
    freq_alpha: float = 0.0,
) -> torch.Tensor:
    """构建中心化训练词表矩阵 E_tilde_V in R^(|V| x D)。

    - freq_alpha == 0.0：均匀词型中心化 E_V - e_bar（向后兼容原公式）。
    - freq_alpha > 0.0 ：按训练集词频平滑幂次 w_v ∝ c_v^alpha 中心化并预乘 sqrt(|V| * w_v)，
      使 (D/|V|) * ||E_tilde_V * y||_2^2 严格等于 D * sum_v w_v * (y · (e_v - e_bar_w))^2，
      在保持跨词表 O(1) 尺度恒定的同时对齐高频词与低频词的零空间惩罚测度。
    """
    alpha = float(freq_alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"freq_alpha 必须为非负有限实数，收到 {freq_alpha!r}。")

    emb_map = getattr(dataset_obj, "embedding_map", None) or getattr(
        dataset_obj, "word_to_embedding", None
    )
    if not emb_map:
        raise ValueError("Dataset 缺少 embedding_map，无法构建中心化训练词表矩阵。")

    rows = []
    retained_words: list[str] = []
    for word in vocabulary:
        w_str = str(word)
        if w_str in emb_map:
            vec = np.asarray(emb_map[w_str], dtype=np.float32)
            rows.append(vec)
            retained_words.append(w_str)
    if not rows:
        raise ValueError("训练词表在 embedding_map 中无匹配词项。")

    if alpha == 0.0:
        bank = torch.from_numpy(np.stack(rows, axis=0)).to(
            device=device, dtype=torch.float32
        )
        bank = F.normalize(bank, dim=-1)
        mean_vec = bank.mean(dim=0, keepdim=True)
        centered_bank = bank - mean_vec
        return centered_bank

    word_counts = _extract_dataset_word_counts(dataset_obj)
    weights_np = np.asarray(
        [max(float(word_counts.get(w, 0)), 1.0) ** alpha for w in retained_words],
        dtype=np.float64,
    )
    weights_np = weights_np / float(weights_np.sum())
    bank_f64 = torch.from_numpy(np.stack(rows, axis=0)).to(
        device=device, dtype=torch.float64
    )
    bank_f64 = F.normalize(bank_f64, dim=-1)
    weights_t = torch.from_numpy(weights_np).to(
        device=device, dtype=torch.float64
    )
    mean_vec = (weights_t.unsqueeze(-1) * bank_f64).sum(dim=0, keepdim=True)
    centered = bank_f64 - mean_vec
    row_scale = torch.sqrt(float(len(retained_words)) * weights_t).unsqueeze(-1)
    return (centered * row_scale).to(dtype=torch.float32)


def compute_shift_null_loss(
    y_shift_mean: torch.Tensor,
    centered_vocab_bank: torch.Tensor,
) -> torch.Tensor:
    """计算错位零化正则项 L_null^(r) = (1 / B) * sum_{i=1}^B || E_tilde_V * y_bar_i^(r, shift) ||_2^2。"""
    if y_shift_mean.numel() == 0:
        return y_shift_mean.new_zeros(())
    queries = y_shift_mean.float()
    bank = centered_vocab_bank.to(device=queries.device, dtype=torch.float32)
    # projections: [B, |V|]
    projections = queries @ bank.T
    # 每个样本在中心化词表空间上的 L2 范数平方，再对 batch 求均值
    return (projections * projections).sum(dim=-1).mean()


def compute_delta_shift_null_loss(
    y_shift_r_mean: torch.Tensor,
    y_shift_k0_mean: torch.Tensor,
    centered_vocab_bank: torch.Tensor,
    *,
    beta_delta_cm: float = 6.0,
    y_shift_r_branches: torch.Tensor | None = None,
    y_shift_k0_branches: torch.Tensor | None = None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """计算跨状态错位跳变零化正则项 L_delta_null^(r)（严格仅作用于错位分支，保护 BGE-M3 语义锥心）。

    数学定义（r in {1, 2}）：
        1) 逐分支跨状态余弦跳变锁死（量纲 O(1)，不被 1/1024 稀释）：
           cos_sim_r = F.cosine_similarity(y_sh_r, sg(y_sh_k0), dim=-1)      # [M, B]
           loss_delta_cm = (1.0 - cos_sim_r).mean()
        2) 跨状态词库投影跳变零化：
           delta_y_sh_r = y_sh_r - sg(y_sh_k0)                              # [M, B, D]
           delta_logits_r = torch.matmul(delta_y_sh_r, vocab_embeds_centered.t())
           loss_delta_vocab = (delta_logits_r ** 2).mean()
    其中对初判态 K0 (r=0) 施加 stop-gradient (detach)，专门阻断共享 Transformer 在打开右边界引入
    i+1, i+2 词窗时利用新暴露的词长间隔 Delta_tau_i 产生跨状态旋转跳变。
    """
    if y_shift_r_mean.numel() == 0 or y_shift_k0_mean.numel() == 0:
        zero = y_shift_r_mean.new_zeros(())
        if return_components:
            return zero, zero
        return zero

    vocab_embeds_centered = centered_vocab_bank.to(
        device=y_shift_r_mean.device, dtype=torch.float32
    )
    if (
        y_shift_r_branches is not None
        and y_shift_k0_branches is not None
        and y_shift_r_branches.numel() > 0
        and y_shift_r_branches.shape == y_shift_k0_branches.shape
    ):
        y_sh_r = y_shift_r_branches.float()
        y_sh_k0_detach = y_shift_k0_branches.detach().float()
        num_branches = int(y_sh_r.shape[0])
        if (
            num_branches > 1
            and y_sh_r.requires_grad
            and not y_shift_r_branches.is_leaf
        ):
            # 直通均值估计（Straight-Through Mean）：前向精确等于全部 M 分支均值，反向使用随机激活分支 0 的无偏满幅梯度（消除 1/M 稀释）
            y_sh_r_det = y_sh_r.detach()
            cos_sim_det = F.cosine_similarity(
                y_sh_r_det, y_sh_k0_detach, dim=-1
            )
            loss_delta_cm_det = (1.0 - cos_sim_det).mean()
            cos_sim_0 = F.cosine_similarity(
                y_sh_r[0], y_sh_k0_detach[0], dim=-1
            )
            loss_delta_cm_0 = (1.0 - cos_sim_0).mean()
            loss_delta_cm = loss_delta_cm_det + (
                loss_delta_cm_0 - loss_delta_cm_0.detach()
            )

            delta_y_sh_det = y_sh_r_det - y_sh_k0_detach
            delta_logits_det = torch.matmul(
                delta_y_sh_det, vocab_embeds_centered.t()
            )
            loss_delta_vocab_det = (delta_logits_det**2).mean()
            delta_y_sh_0 = y_sh_r[0] - y_sh_k0_detach[0]
            delta_logits_0 = torch.matmul(
                delta_y_sh_0, vocab_embeds_centered.t()
            )
            loss_delta_vocab_0 = (delta_logits_0**2).mean()
            loss_delta_vocab = loss_delta_vocab_det + (
                loss_delta_vocab_0 - loss_delta_vocab_0.detach()
            )
        else:
            cos_sim_r = F.cosine_similarity(y_sh_r, y_sh_k0_detach, dim=-1)
            loss_delta_cm = (1.0 - cos_sim_r).mean()
            delta_y_sh_r = y_sh_r - y_sh_k0_detach
            delta_logits_r = torch.matmul(
                delta_y_sh_r, vocab_embeds_centered.t()
            )
            loss_delta_vocab = (delta_logits_r**2).mean()
    else:
        y_sh_r = y_shift_r_mean.float()
        y_sh_k0_detach = y_shift_k0_mean.detach().float()
        cos_sim_r = F.cosine_similarity(y_sh_r, y_sh_k0_detach, dim=-1)
        loss_delta_cm = (1.0 - cos_sim_r).mean()
        delta_y_sh_r = y_sh_r - y_sh_k0_detach
        delta_logits_r = torch.matmul(delta_y_sh_r, vocab_embeds_centered.t())
        loss_delta_vocab = (delta_logits_r**2).mean()

    if return_components:
        return loss_delta_vocab, loss_delta_cm
    if float(beta_delta_cm) > 0.0:
        return loss_delta_vocab + float(beta_delta_cm) * loss_delta_cm
    return loss_delta_vocab


def compute_scaled_shift_null_loss(
    y_shift_mean: torch.Tensor,
    centered_vocab_bank: torch.Tensor,
    *,
    y_shift_branches: torch.Tensor | None = None,
    coherent_mean: bool = False,
) -> torch.Tensor:
    """维度平衡词表零空间锚定惩罚：(D / |V|) * || E_bar_V * y^(r, delta) ||_2^2。

    使得跨语种不同词表大小（法语 |V|=1897、英语 |V|=18031）拥有严格相同的 O(1) 梯度尺度。
    - coherent_mean=False（默认向后兼容）：当 M > 1 时逐分支独立惩罚。
    - coherent_mean=True ：直接对 M 路错位分支相干均值 y_shift_mean 施加零空间惩罚，
      保留 100% 相干时序捷径梯度并使非相干单次真实神经方差梯度衰减 M 倍。
    """
    if y_shift_mean.numel() == 0:
        return y_shift_mean.new_zeros(())
    bank = centered_vocab_bank.to(
        device=y_shift_mean.device, dtype=torch.float32
    )
    dim_scale = float(bank.shape[1])

    if (
        not coherent_mean
        and y_shift_branches is not None
        and y_shift_branches.numel() > 0
    ):
        y_br = y_shift_branches.float()
        num_branches = int(y_br.shape[0])
        if (
            num_branches > 1
            and y_br.requires_grad
            and not y_shift_branches.is_leaf
        ):
            proj_det = torch.matmul(y_br.detach(), bank.t())
            loss_det = (proj_det * proj_det).mean(dim=-1).mean() * dim_scale
            proj_0 = torch.matmul(y_br[0], bank.t())
            loss_0 = (proj_0 * proj_0).mean(dim=-1).mean() * dim_scale
            return loss_det + (loss_0 - loss_0.detach())
        proj = torch.matmul(y_br, bank.t())
        return (proj * proj).mean(dim=-1).mean() * dim_scale

    queries = y_shift_mean.float()
    projections = queries @ bank.t()
    return (projections * projections).mean(dim=-1).mean() * dim_scale


def build_vocabulary_gram_sqrt(
    centered_vocab_bank: torch.Tensor,
) -> torch.Tensor:
    """预计算维度归一化词表 Gram 矩阵平方根 G_V^{1/2} in R^(D x D)。

    满足 ||X @ G_V^{1/2}||_F^2 == (D / |V|) * ||X @ E_tilde_V^T||_F^2，
    将大词表 (|V|=18031) 每步投影耗时缩减约 18 倍且数值严格等价。
    """
    bank_f64 = centered_vocab_bank.detach().to(dtype=torch.float64)
    num_vocab, dim = bank_f64.shape
    gram = (float(dim) / max(float(num_vocab), 1.0)) * (bank_f64.t() @ bank_f64)
    gram = 0.5 * (gram + gram.t())
    eigvals, eigvecs = torch.linalg.eigh(gram)
    sqrt_vals = torch.sqrt(eigvals.clamp(min=0.0))
    gram_sqrt = (eigvecs * sqrt_vals.unsqueeze(0)) @ eigvecs.t()
    return gram_sqrt.to(
        device=centered_vocab_bank.device, dtype=torch.float32
    ).contiguous()


def compute_excess_coherent_shift_null_loss(
    delta_y_shift_branches: torch.Tensor,
    centered_vocab_bank: torch.Tensor,
    *,
    vocab_gram_sqrt: torch.Tensor | None = None,
    excess_tolerance: float = 0.08,
) -> torch.Tensor:
    """超额相干错位零空间惩罚（消除满秩 T5-Large 空间中过往语音非相干方差导致的尾部挤压）。

    对 M 路错位跨词球面增量 u_i^(m) = (y_i^(m) - z_i^(m))_c：
      - 前向：ReLU(E_mean - (1 + tol) * sg(E_incoh)) >= 0（严格半正定，无相干时序偏置时精确归零）；
      - 反向（当超额能量 > 0 时）：采用跨分支非对角内积 2 * <v^(0), mean_{m>=1} sg(v^(m))>，
        提供 1.0x 满幅无偏相干时序偏置梯度且对独立过往语音语义保持精确零惩罚。
    """
    if delta_y_shift_branches is None or delta_y_shift_branches.numel() == 0:
        return centered_vocab_bank.new_zeros(())
    u_br = delta_y_shift_branches.float()
    if u_br.ndim == 2:
        u_br = u_br.unsqueeze(0)
    num_branches, batch_len, _ = u_br.shape
    if batch_len > 1:
        u_br = u_br - u_br.mean(dim=1, keepdim=True)

    if vocab_gram_sqrt is not None:
        gram_sqrt = vocab_gram_sqrt.to(device=u_br.device, dtype=torch.float32)
        v_br = torch.matmul(u_br, gram_sqrt)
    else:
        bank = centered_vocab_bank.to(device=u_br.device, dtype=torch.float32)
        scale = math.sqrt(float(bank.shape[1]) / max(float(bank.shape[0]), 1.0))
        v_br = torch.matmul(u_br, bank.t()) * scale

    if num_branches < 2:
        return (v_br * v_br).sum(dim=-1).mean()

    with torch.no_grad():
        v_det = v_br.detach()
        v_mean_det = v_det.mean(dim=0)
        e_mean_det = (v_mean_det * v_mean_det).sum(dim=-1).mean()
        diff_incoh = v_det - v_mean_det.unsqueeze(0)
        e_incoh_det = (diff_incoh * diff_incoh).sum(dim=-1).sum(dim=0).mean() / float(
            num_branches * (num_branches - 1)
        )
        excess_det = F.relu(
            e_mean_det - (1.0 + max(0.0, float(excess_tolerance))) * e_incoh_det
        )

    if not v_br.requires_grad:
        return excess_det + (v_br.sum() * 0.0)

    if not v_br[1:].requires_grad:
        other_mean = v_br[1:].detach().mean(dim=0)
        cross_term = 2.0 * (v_br[0] * other_mean).sum(dim=-1).mean()
    else:
        sum_sq = (v_br.sum(dim=0) ** 2).sum(dim=-1).mean()
        diag_sq = (v_br * v_br).sum(dim=-1).sum(dim=0).mean()
        cross_term = (sum_sq - diag_sq) / float(num_branches * (num_branches - 1))

    active_grad = torch.where(
        excess_det > 0.0,
        cross_term - cross_term.detach(),
        excess_det.new_zeros(()),
    )
    return excess_det + active_grad


def _count_english_vowels(word: str) -> int:
    return sum(1 for ch in word.lower() if ch in "aeiouy")


def build_clean_stein_residual_basis(
    words: Sequence[str],
    targets: torch.Tensor,
    sentence_indices: torch.Tensor,
    onset_seconds: torch.Tensor,
    device: torch.device,
    *,
    lags: tuple[int, ...] = (-3, -2, -1, 0, 1, 2),
    shrinkage_kappa: float = 5.0,
) -> dict[str, torch.Tensor | tuple[int, ...]]:
    """仅基于训练划分构建与目标词义严格正交的多滞后同词间隔残差 Stein-RBF 基矩阵。

    步骤：
      1. 句内相邻词间隔 ell_j = log(clamp(tau_{j+1} - tau_j, 0.02, 5.0))；
      2. 字形长度线性先验 + 留一法（LOO）经验贝叶斯收缩（消除单次词残差恒零问题）；
      3. 8 维每滞后 Stein-RBF 基：[r_j, r_j * RBF_1..6(mu_LOO(j)), r_j^2 - var_LOO]；
      4. 两步目标词正交化（LOO 同目标词均值扣除 + 对中心化目标词嵌入矩阵线性正交投影），
         消除停顿—句首词相关性混淆，确保对真实神经词义零惩罚。
    """
    n_rows = int(len(words))
    sent_np = sentence_indices.detach().cpu().numpy().astype(np.int64)
    onset_np = onset_seconds.detach().cpu().numpy().astype(np.float64)

    valid_pair = np.zeros(n_rows, dtype=np.bool_)
    log_dt = np.zeros(n_rows, dtype=np.float64)
    if n_rows > 1:
        same_sent = sent_np[:-1] == sent_np[1:]
        raw_dt = onset_np[1:] - onset_np[:-1]
        ok = same_sent & np.isfinite(raw_dt) & (raw_dt > 1e-4) & (raw_dt <= 5.0)
        valid_pair[:-1] = ok
        log_dt[:-1] = np.where(
            ok, np.log(np.clip(raw_dt, 0.02, 5.0)), 0.0
        )

    # 1. 字形先验 OLS + LOO 词均值收缩
    unique_words = sorted(set(str(w) for w in words))
    word_to_id = {w: i for i, w in enumerate(unique_words)}
    word_ids = np.asarray([word_to_id[str(w)] for w in words], dtype=np.int64)
    num_types = len(unique_words)

    ortho_feats = np.zeros((num_types, 4), dtype=np.float64)
    for wid, w_str in enumerate(unique_words):
        w_len = float(len(w_str))
        ortho_feats[wid, 0] = 1.0
        ortho_feats[wid, 1] = w_len
        ortho_feats[wid, 2] = math.log1p(w_len)
        ortho_feats[wid, 3] = float(_count_english_vowels(w_str))

    valid_idx = np.flatnonzero(valid_pair)
    if len(valid_idx) >= 16:
        x_ols = ortho_feats[word_ids[valid_idx]]
        y_ols = log_dt[valid_idx]
        beta, _, _, _ = np.linalg.lstsq(x_ols, y_ols, rcond=None)
        prior_mu_by_type = ortho_feats @ beta
    else:
        prior_mu_by_type = np.full(num_types, -1.2, dtype=np.float64)

    sum_by_type = np.bincount(
        word_ids[valid_idx], weights=log_dt[valid_idx], minlength=num_types
    )
    cnt_by_type = np.bincount(word_ids[valid_idx], minlength=num_types).astype(
        np.float64
    )
    kappa = float(shrinkage_kappa)
    wid_valid = word_ids[valid_idx]
    mu_loo_valid = (
        (sum_by_type[wid_valid] - log_dt[valid_idx])
        + kappa * prior_mu_by_type[wid_valid]
    ) / ((cnt_by_type[wid_valid] - 1.0).clip(min=0.0) + kappa)

    raw_res_valid = log_dt[valid_idx] - mu_loo_valid
    sigma_r = float(np.std(raw_res_valid)) if len(raw_res_valid) > 1 else 0.35
    sigma_r = max(sigma_r, 1e-3)
    std_res_valid = np.clip(raw_res_valid / sigma_r, -5.0, 5.0)

    # 2. 构造单位置 8 维 Stein-RBF 特征
    if len(mu_loo_valid) >= 16:
        rbf_centers = np.quantile(
            mu_loo_valid, np.linspace(0.10, 0.90, 6)
        ).astype(np.float64)
        rbf_sigma = max(
            float(np.std(mu_loo_valid)) * 0.45,
            float(rbf_centers[-1] - rbf_centers[0]) / 5.0,
            0.12,
        )
    else:
        rbf_centers = np.linspace(-2.0, -0.5, 6, dtype=np.float64)
        rbf_sigma = 0.25

    sq_res_valid = std_res_valid**2
    sum_sq_by_type = np.bincount(
        wid_valid, weights=sq_res_valid, minlength=num_types
    )
    global_var = float(np.mean(sq_res_valid)) if len(sq_res_valid) > 0 else 1.0
    var_loo_valid = (
        (sum_sq_by_type[wid_valid] - sq_res_valid) + kappa * global_var
    ) / ((cnt_by_type[wid_valid] - 1.0).clip(min=0.0) + kappa)

    pos_features = np.zeros((n_rows, 8), dtype=np.float32)
    if len(valid_idx) > 0:
        rbf_vals = np.exp(
            -0.5 * ((mu_loo_valid[:, None] - rbf_centers[None, :]) / rbf_sigma) ** 2
        )
        feat_v = np.empty((len(valid_idx), 8), dtype=np.float64)
        feat_v[:, 0] = std_res_valid
        feat_v[:, 1:7] = std_res_valid[:, None] * rbf_vals
        feat_v[:, 7] = np.clip(sq_res_valid - var_loo_valid, -6.0, 6.0)
        pos_features[valid_idx] = feat_v.astype(np.float32)

    # 3. 拼装同句各滞后 d in lags 特征并记录句尾边界距离以支持 Ch.3 各视界精确掩码
    lag_tuple = tuple(int(d) for d in lags)
    feat_dim = 8 * len(lag_tuple)
    raw_basis = np.zeros((n_rows, feat_dim), dtype=np.float64)

    # 计算每行在所属句子内的 (sent_start, sent_end_inclusive)
    sent_start = np.zeros(n_rows, dtype=np.int64)
    sent_end = np.zeros(n_rows, dtype=np.int64)
    s_cur = 0
    while s_cur < n_rows:
        e_cur = s_cur + 1
        s_val = sent_np[s_cur]
        while e_cur < n_rows and sent_np[e_cur] == s_val:
            e_cur += 1
        sent_start[s_cur:e_cur] = s_cur
        sent_end[s_cur:e_cur] = e_cur - 1
        s_cur = e_cur

    row_arange = np.arange(n_rows, dtype=np.int64)
    for l_idx, d in enumerate(lag_tuple):
        j_idx = row_arange + int(d)
        # 间隔 j 连接词 j 与 j+1，必须同时位于同一句内 [sent_start, sent_end]
        in_sent = (j_idx >= sent_start) & ((j_idx + 1) <= sent_end)
        valid_rows = np.flatnonzero(in_sent)
        if len(valid_rows) > 0:
            raw_basis[valid_rows, l_idx * 8 : (l_idx + 1) * 8] = pos_features[
                j_idx[valid_rows]
            ]

    # 4. 两步目标词正交化：(a) 同目标词 w_i LOO 均值扣除；(b) 对中心化目标词嵌入矩阵正交投影
    total_cnt_by_type = np.bincount(word_ids, minlength=num_types).astype(
        np.float64
    )
    basis_step1 = np.empty_like(raw_basis)
    for col in range(feat_dim):
        col_vals = raw_basis[:, col]
        col_sum_by_type = np.bincount(
            word_ids, weights=col_vals, minlength=num_types
        )
        loo_target_mean = (col_sum_by_type[word_ids] - col_vals) / (
            (total_cnt_by_type[word_ids] - 1.0).clip(min=0.0) + kappa
        )
        basis_step1[:, col] = col_vals - loo_target_mean
    basis_step1 -= basis_step1.mean(axis=0, keepdims=True)

    # 在 GPU 上执行对中心化唯一词嵌入的快速线性正交投影 (仅 O(N * D * P) 一次矩阵乘)
    basis_gpu = torch.from_numpy(basis_step1).to(
        device=device, dtype=torch.float32
    )
    t_f32 = F.normalize(targets.detach().to(device=device, dtype=torch.float32), dim=-1)
    t_centered = t_f32 - t_f32.mean(dim=0, keepdim=True)
    dim_t = int(t_centered.shape[1])
    gram_t = (t_centered.t() @ t_centered) / max(float(n_rows), 1.0)
    gram_reg = gram_t + 1e-3 * torch.eye(dim_t, device=device, dtype=torch.float32)
    cov_tb = (t_centered.t() @ basis_gpu) / max(float(n_rows), 1.0)
    proj_weights = torch.linalg.solve(gram_reg, cov_tb)
    basis_orth = basis_gpu - (t_centered @ proj_weights)
    basis_orth = basis_orth - basis_orth.mean(dim=0, keepdim=True)
    col_rms = basis_orth.pow(2).mean(dim=0, keepdim=True).sqrt().clamp(min=1e-4)
    basis_orth = (basis_orth / col_rms).clamp(-6.0, 6.0)

    return {
        "basis": basis_orth.to(dtype=torch.float16).contiguous(),
        "lags": lag_tuple,
        "features_per_lag": 8,
    }


class CleanSteinRunningLoss:
    """跨批次指数移动平均凸半正定同词间隔残差正交损失（消除 B=128 有限批方差偏置）。"""

    def __init__(
        self,
        stein_bundle: dict,
        vocab_gram_sqrt: torch.Tensor,
        *,
        ema_gamma: float = 0.03,
        active_lags: Iterable[int] | None = None,
    ) -> None:
        self.basis_fp16: torch.Tensor = stein_bundle["basis"]
        self.lags: tuple[int, ...] = tuple(stein_bundle["lags"])
        self.features_per_lag: int = int(stein_bundle["features_per_lag"])
        self.vocab_gram_sqrt: torch.Tensor = vocab_gram_sqrt.detach().float()
        self.ema_gamma = float(ema_gamma)

        if active_lags is None:
            col_indices = list(range(int(self.basis_fp16.shape[1])))
        else:
            allowed = {int(x) for x in active_lags}
            col_indices = []
            for idx, lag_val in enumerate(self.lags):
                if lag_val in allowed:
                    start = idx * self.features_per_lag
                    col_indices.extend(range(start, start + self.features_per_lag))
        self.col_idx = torch.as_tensor(
            col_indices, device=self.basis_fp16.device, dtype=torch.long
        )
        self.probe_dim = int(self.col_idx.numel())
        self.ema_cov = torch.zeros(
            (int(self.vocab_gram_sqrt.shape[0]), max(1, self.probe_dim)),
            device=self.basis_fp16.device,
            dtype=torch.float32,
        )
        self.initialized = False

    def __call__(
        self,
        delta_y_clean: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.probe_dim == 0
            or delta_y_clean.numel() == 0
            or int(delta_y_clean.shape[0]) < 2
        ):
            return delta_y_clean.sum() * 0.0
        bsz = int(delta_y_clean.shape[0])
        dy = delta_y_clean.float()
        dy_c = dy - dy.mean(dim=0, keepdim=True)
        v_clean = dy_c @ self.vocab_gram_sqrt  # [B, D]

        z_stein = (
            self.basis_fp16.index_select(0, row_indices)
            .index_select(1, self.col_idx)
            .float()
        )
        z_stein = z_stein - z_stein.mean(dim=0, keepdim=True)

        c_batch = (v_clean.t() @ z_stein) / float(bsz)  # [D, P]
        gamma = self.ema_gamma
        if not self.initialized:
            c_tilde = gamma * c_batch
        else:
            c_tilde = (1.0 - gamma) * self.ema_cov.detach() + gamma * c_batch
        with torch.no_grad():
            self.ema_cov.copy_(c_tilde.detach())
            self.initialized = True
        return (c_tilde * c_tilde).sum() / (2.0 * gamma * float(self.probe_dim))


def compute_shifted_lag_decorrelation_loss(
    y_shift_branches: torch.Tensor | None,
    sentence_indices: torch.Tensor,
    row_positions: torch.Tensor | None = None,
    *,
    max_lag: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """零 Host 同步错位分支句内 Lag-1..L_max 自相关白化惩罚 sum_{d=1}^{L_max} rho_shift(d)^2。

    仅作用于时间错位（纯重叠背景噪声）分支，引导 Stage-2 有符号邻居核主动消除相邻词窗
    重叠造成的 lag-1..3 伪相关性，同时完整保留 Clean 分支的真实跨词语义连续性。
    支持多分支直通均值估计（分支 0 满幅梯度 + 全部分支精确前向统计）。
    """
    if y_shift_branches is None or y_shift_branches.numel() == 0:
        return sentence_indices.new_zeros((), dtype=torch.float32)
    y_br = y_shift_branches.float()
    if y_br.ndim == 2:
        y_br = y_br.unsqueeze(0)
    num_branches, batch_len, _ = y_br.shape
    l_max = max(1, int(max_lag))
    if batch_len <= 1:
        return y_br.sum() * 0.0

    sent = sentence_indices.to(device=y_br.device, dtype=torch.long)
    if row_positions is None:
        pos = torch.arange(batch_len, device=y_br.device, dtype=torch.long)
    else:
        pos = row_positions.to(device=y_br.device, dtype=torch.long)

    def _single_branch_lag_loss(y_mat: torch.Tensor) -> torch.Tensor:
        u = y_mat - y_mat.mean(dim=0, keepdim=True)
        total = u.new_zeros(())
        for lag in range(1, l_max + 1):
            if batch_len <= lag:
                break
            pair_mask = (sent[:-lag] == sent[lag:]) & (
                (pos[lag:] - pos[:-lag]) == lag
            )
            m = pair_mask.to(dtype=torch.float32)
            u_a = u[:-lag]
            u_b = u[lag:]
            num = ((u_a * u_b).sum(dim=-1) * m).sum()
            den_a = ((u_a * u_a).sum(dim=-1) * m).sum()
            den_b = ((u_b * u_b).sum(dim=-1) * m).sum()
            rho = num / torch.sqrt((den_a + eps) * (den_b + eps))
            total = total + rho * rho
        return total

    if (
        num_branches > 1
        and y_br.requires_grad
        and not y_br[1:].requires_grad
    ):
        with torch.no_grad():
            det_losses = [
                _single_branch_lag_loss(y_br[m_idx].detach())
                for m_idx in range(num_branches)
            ]
            loss_det = torch.stack(det_losses).mean()
        loss_0 = _single_branch_lag_loss(y_br[0])
        return loss_det + (loss_0 - loss_0.detach())

    branch_losses = [
        _single_branch_lag_loss(y_br[m_idx]) for m_idx in range(num_branches)
    ]
    return torch.stack(branch_losses).mean()


def compute_euclidean_cross_horizon_loss(
    y_shift_r_mean: torch.Tensor,
    y_shift_k0_mean: torch.Tensor,
    *,
    y_shift_r_branches: torch.Tensor | None = None,
    y_shift_k0_branches: torch.Tensor | None = None,
    coherent_mean: bool = False,
) -> torch.Tensor:
    """极简欧氏跨视界平滑惩罚：|| y_i^(r, delta) - sg(y_i^(0, delta)) ||_2^2 (r > 0)。

    对单位球面向量天然满足 ||y^(r) - y^(0)||_2^2 = 2 * (1 - cos(y^(r), y^(0))) in [0, 4]，
    同时解析上界约束余弦旋转跳变与中心化词表投影跳变，无需双超参。
    - coherent_mean=False（默认向后兼容）：逐分支独立惩罚并经分支 0 直通均值估计保持满幅梯度。
    - coherent_mean=True ：对 M 路错位相干均值之差施加欧氏惩罚，保留 100% 跨视界相干时序跳变抑制，
      同时将未来词窗引入的非相干单次神经波动惩罚衰减 M 倍。
    """
    if y_shift_r_mean.numel() == 0 or y_shift_k0_mean.numel() == 0:
        return y_shift_r_mean.new_zeros(())
    if (
        not coherent_mean
        and y_shift_r_branches is not None
        and y_shift_k0_branches is not None
        and y_shift_r_branches.numel() > 0
        and y_shift_r_branches.shape == y_shift_k0_branches.shape
    ):
        y_sh_r = y_shift_r_branches.float()
        y_sh_k0_det = y_shift_k0_branches.detach().float()
        num_branches = int(y_sh_r.shape[0])
        if (
            num_branches > 1
            and y_sh_r.requires_grad
            and not y_shift_r_branches.is_leaf
        ):
            diff_det = y_sh_r.detach() - y_sh_k0_det
            loss_det = (diff_det * diff_det).sum(dim=-1).mean()
            diff_0 = y_sh_r[0] - y_sh_k0_det[0]
            loss_0 = (diff_0 * diff_0).sum(dim=-1).mean()
            return loss_det + (loss_0 - loss_0.detach())
        diff = y_sh_r - y_sh_k0_det
        return (diff * diff).sum(dim=-1).mean()

    diff = y_shift_r_mean.float() - y_shift_k0_mean.detach().float()
    return (diff * diff).sum(dim=-1).mean()


def compute_temporal_bias_distillation_loss(
    b_pred: torch.Tensor,
    y_shift_target: torch.Tensor,
) -> torch.Tensor:
    """档位 B 纯时序几何头 MSE 蒸馏损失：|| b_psi(Delta_tau_{0:i+r})_i - sg(y_i^(r, delta)) ||_2^2。"""
    if b_pred.numel() == 0 or y_shift_target.numel() == 0:
        return b_pred.new_zeros(())
    diff = b_pred.float() - y_shift_target.detach().float()
    return (diff * diff).sum(dim=-1).mean()


def compute_norm_damped_debiased_embedding(
    y_clean: torch.Tensor,
    y_shift_mean: torch.Tensor,
    eta_train: float,
) -> torch.Tensor:
    """雅可比模长阻尼重缩放（Norm-Damped Detach）计算孪生共模去偏表征 y_tilde。

    1. 计算扣除后的差向量（保留 y_shift_mean.detach() 阻断反向作弊）：
       u = y_clean - eta_train * y_shift_mean.detach()
    2. 计算模长并执行 L2 归一化：
       u_norm = u.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
       y_tilde = u / u_norm
    3. 直通梯度阻尼（前向数值 100% 不变，反向梯度乘回 u_norm.detach()）：
       精确抵消 L2 归一化分母产生的 1 / ||u||_2 梯度爆炸，使反向求导仅保留球面切空间投影 (I - y_tilde * y_tilde^T)。
    """
    # 1. 计算扣除后的差向量（保留 y_shift_mean.detach() 阻断反向作弊）
    u = y_clean - float(eta_train) * y_shift_mean.detach()

    # 2. 计算模长并执行 L2 归一化
    u_norm = u.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    y_tilde = u / u_norm

    # 3. 【核心修复】直通梯度阻尼：前向数值 100% 不变，反向梯度乘回 u_norm.detach()
    # 精确抵消 L2 归一化分母产生的 1 / ||u||_2 梯度爆炸，将等效学习率死死锁在设定值！
    y_tilde = y_tilde.detach() + u_norm.detach() * (y_tilde - y_tilde.detach())
    return y_tilde


def compute_norm_damped_rolling_debiased_embedding(
    y_clean: torch.Tensor,
    y_shift_mean: torch.Tensor,
    eta_0: float,
    *,
    y_shift_k0: torch.Tensor | None = None,
    eta_delta: float = 0.0,
) -> torch.Tensor:
    """前缀解耦的差分共模扣除（eta_0 + eta_delta）+ 雅可比模长阻尼重缩放（Norm-Damped Detach）。

    针对 Revision (K0 -> K1 -> K2) 共享前缀 [i-2, i-1, i] 的物理结构：
        u_i^(r) = y_i^(r, clean) - eta_0 * sg(y_bar_i^(0, shift)) - eta_delta * sg(y_bar_i^(r, shift) - y_bar_i^(0, shift))
    - 当 r = 0 (K0) 时：后一项自动为 0，等效扣除系数为 eta_0（保护初判信噪比）；
    - 当 r in {1, 2} (K1, K2) 时：静态背景按 eta_0 温和扣除，而对新暴露的跨状态时长跳变施加 eta_delta 强力抑制；
    - 随后执行 L2 归一化与 Norm-Damped Detach，保持反向梯度模长 1:1 恒定。
    """
    shift_r = y_shift_mean.detach()
    if y_shift_k0 is None or float(eta_delta) <= 0.0:
        u = y_clean - float(eta_0) * shift_r
    else:
        shift_0 = y_shift_k0.detach()
        u = (
            y_clean
            - float(eta_0) * shift_0
            - float(eta_delta) * (shift_r - shift_0)
        )
    u_norm = u.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    y_tilde = u / u_norm
    y_tilde = y_tilde.detach() + u_norm.detach() * (y_tilde - y_tilde.detach())
    return y_tilde
