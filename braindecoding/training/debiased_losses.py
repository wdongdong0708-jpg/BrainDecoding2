"""去偏训练超参解析、中心化词表矩阵及零空间 / 差分 / 蒸馏损失函数。"""

from __future__ import annotations

import os
from typing import Iterable

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
        "shift_seconds": resolved_shifts,
        "shifts_per_step": int(resolved_m),
        "s3_cms_eta": resolved_s3_cms_eta,
        "synchronize_dropout_rng": sync_rng,
        "preextract_frozen_cnn": bool(resolved_preextract),
        "cache_preextracted_to_disk": cache_to_disk,
    }


def build_centered_vocabulary_bank(
    dataset_obj,
    vocabulary: Iterable[str],
    device: torch.device,
) -> torch.Tensor:
    """构建中心化训练词表矩阵 E_tilde_V = E_V - e_bar in R^(|V| x D)。

    其中 E_V 的每一行为训练集唯一词的 L2 归一化目标向量，e_bar 为全词库均值向量。
    """
    emb_map = getattr(dataset_obj, "embedding_map", None) or getattr(
        dataset_obj, "word_to_embedding", None
    )
    if not emb_map:
        raise ValueError("Dataset 缺少 embedding_map，无法构建中心化训练词表矩阵。")

    rows = []
    for word in vocabulary:
        w_str = str(word)
        if w_str in emb_map:
            vec = np.asarray(emb_map[w_str], dtype=np.float32)
            rows.append(vec)
    if not rows:
        raise ValueError("训练词表在 embedding_map 中无匹配词项。")

    bank = torch.from_numpy(np.stack(rows, axis=0)).to(
        device=device, dtype=torch.float32
    )
    bank = F.normalize(bank, dim=-1)
    mean_vec = bank.mean(dim=0, keepdim=True)
    centered_bank = bank - mean_vec
    return centered_bank


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
) -> torch.Tensor:
    """维度平衡词表零空间锚定惩罚：(D / |V|) * || E_bar_V * y^(r, delta) ||_2^2。

    使得跨语种不同词表大小（法语 |V|=2020、中文 |V|=2180）拥有严格相同的 O(1) 梯度尺度。
    当 M > 1 时逐分支独立惩罚，并通过直通均值估计保持分支 0 的无稀释无偏梯度。
    """
    if y_shift_mean.numel() == 0:
        return y_shift_mean.new_zeros(())
    bank = centered_vocab_bank.to(
        device=y_shift_mean.device, dtype=torch.float32
    )
    dim_scale = float(bank.shape[1])

    if y_shift_branches is not None and y_shift_branches.numel() > 0:
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


def compute_euclidean_cross_horizon_loss(
    y_shift_r_mean: torch.Tensor,
    y_shift_k0_mean: torch.Tensor,
    *,
    y_shift_r_branches: torch.Tensor | None = None,
    y_shift_k0_branches: torch.Tensor | None = None,
) -> torch.Tensor:
    """极简欧氏跨视界平滑惩罚：|| y_i^(r, delta) - sg(y_i^(0, delta)) ||_2^2 (r > 0)。

    对单位球面向量天然满足 ||y^(r) - y^(0)||_2^2 = 2 * (1 - cos(y^(r), y^(0))) in [0, 4]，
    同时解析上界约束余弦旋转跳变与中心化词表投影跳变，无需双超参。
    """
    if y_shift_r_mean.numel() == 0 or y_shift_k0_mean.numel() == 0:
        return y_shift_r_mean.new_zeros(())
    if (
        y_shift_r_branches is not None
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
