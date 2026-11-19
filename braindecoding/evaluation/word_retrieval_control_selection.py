"""负对照指标注册表、方法选择与多种子对照视角投影。"""

from __future__ import annotations

from typing import Mapping, Sequence

_CONTROL_METRIC_REGISTRY: dict[str, dict[str, object]] = {
    "vocab50_macro_top10": {
        "key": "vocab50_macro_top10",
        "horizon_key": "common_vocab50_macro_top10",
        "label": "50-Word Word-Macro Top-10",
        "short_label": "50w Word-Macro Top-10",
        "is_ratio": True,
    },
    "vocab50_micro_top10": {
        "key": "vocab50_micro_top10",
        "horizon_key": "common_vocab50_micro_top10",
        "label": "50-Word Event-Micro Top-10",
        "short_label": "50w Event-Micro Top-10",
        "is_ratio": True,
    },
    "vocab50_subject_macro_top10": {
        "key": "vocab50_subject_macro_top10",
        "horizon_key": "common_vocab50_subject_macro_top10",
        "label": "50-Word Subject-Macro Top-10",
        "short_label": "50w Subj-Macro Top-10",
        "is_ratio": True,
    },
    "vocab50_macro_top5": {
        "key": "vocab50_macro_top5",
        "horizon_key": "common_vocab50_macro_top10",
        "label": "50-Word Word-Macro Top-5",
        "short_label": "50w Word-Macro Top-5",
        "is_ratio": True,
    },
    "vocab50_macro_top1": {
        "key": "vocab50_macro_top1",
        "horizon_key": "common_vocab50_macro_top1",
        "label": "50-Word Word-Macro Top-1",
        "short_label": "50w Word-Macro Top-1",
        "is_ratio": True,
    },
    "vocab50_micro_top1": {
        "key": "vocab50_micro_top1",
        "horizon_key": "common_vocab50_micro_top1",
        "label": "50-Word Event-Micro Top-1",
        "short_label": "50w Event-Micro Top-1",
        "is_ratio": True,
    },
    "vocab50_mrr": {
        "key": "vocab50_mrr",
        "horizon_key": "common_vocab50_mrr",
        "label": "50-Word MRR",
        "short_label": "50w MRR",
        "is_ratio": True,
    },
    "vocab50_median_rank": {
        "key": "vocab50_median_rank",
        "horizon_key": "common_vocab50_median_rank",
        "label": "50-Word Median Rank",
        "short_label": "50w MedRank",
        "is_ratio": False,
    },
    "full_vocab_macro_top10": {
        "key": "full_vocab_macro_top10",
        "horizon_key": "common_full_vocab_macro_top10",
        "label": "Full-Vocab Word-Macro Top-10",
        "short_label": "FullVoc Word-Macro Top-10",
        "is_ratio": True,
    },
    "full_vocab_subject_macro_top10": {
        "key": "full_vocab_subject_macro_top10",
        "horizon_key": "common_full_vocab_subject_macro_top10",
        "label": "Full-Vocab Subject-Macro Top-10",
        "short_label": "FullVoc Subj-Macro Top-10",
        "is_ratio": True,
    },
    "full_vocab_micro_top10": {
        "key": "full_vocab_micro_top10",
        "horizon_key": "common_full_vocab_micro_top10",
        "label": "Full-Vocab Event-Micro Top-10",
        "short_label": "FullVoc Event-Micro Top-10",
        "is_ratio": True,
    },
    "full_vocab_macro_top1": {
        "key": "full_vocab_macro_top1",
        "horizon_key": "common_full_vocab_macro_top1",
        "label": "Full-Vocab Word-Macro Top-1",
        "short_label": "FullVoc Word-Macro Top-1",
        "is_ratio": True,
    },
    "full_vocab_micro_top1": {
        "key": "full_vocab_micro_top1",
        "horizon_key": "common_full_vocab_micro_top1",
        "label": "Full-Vocab Event-Micro Top-1",
        "short_label": "FullVoc Event-Micro Top-1",
        "is_ratio": True,
    },
    "full_vocab_mrr": {
        "key": "full_vocab_mrr",
        "horizon_key": "common_full_vocab_mrr",
        "label": "Full-Vocab MRR",
        "short_label": "FullVoc MRR",
        "is_ratio": True,
    },
    "full_vocab_median_rank": {
        "key": "full_vocab_median_rank",
        "horizon_key": "common_full_vocab_median_rank",
        "label": "Full-Vocab Median Rank",
        "short_label": "FullVoc MedRank",
        "is_ratio": False,
    },
}

_CONTROL_METRIC_ALIASES: dict[str, str] = {
    "50w_top10": "vocab50_macro_top10",
    "n50_top10": "vocab50_macro_top10",
    "50w_macro_top10": "vocab50_macro_top10",
    "50w_word_macro_top10": "vocab50_macro_top10",
    "vocab50_word_macro_top10": "vocab50_macro_top10",
    "n50_macro_top10": "vocab50_macro_top10",
    "vocab50_top10": "vocab50_macro_top10",
    "50w_micro_top10": "vocab50_micro_top10",
    "50w_event_micro_top10": "vocab50_micro_top10",
    "vocab50_event_micro_top10": "vocab50_micro_top10",
    "n50_micro_top10": "vocab50_micro_top10",
    "50w_subject_macro_top10": "vocab50_subject_macro_top10",
    "n50_subject_macro_top10": "vocab50_subject_macro_top10",
    "50w_top5": "vocab50_macro_top5",
    "n50_top5": "vocab50_macro_top5",
    "50w_macro_top5": "vocab50_macro_top5",
    "n50_macro_top5": "vocab50_macro_top5",
    "50w_top1": "vocab50_macro_top1",
    "n50_top1": "vocab50_macro_top1",
    "50w_macro_top1": "vocab50_macro_top1",
    "50w_word_macro_top1": "vocab50_macro_top1",
    "vocab50_word_macro_top1": "vocab50_macro_top1",
    "n50_macro_top1": "vocab50_macro_top1",
    "vocab50_top1": "vocab50_macro_top1",
    "50w_micro_top1": "vocab50_micro_top1",
    "50w_event_micro_top1": "vocab50_micro_top1",
    "vocab50_event_micro_top1": "vocab50_micro_top1",
    "n50_micro_top1": "vocab50_micro_top1",
    "50w_mrr": "vocab50_mrr",
    "n50_mrr": "vocab50_mrr",
    "50w_median_rank": "vocab50_median_rank",
    "n50_median_rank": "vocab50_median_rank",
    "full_top10": "full_vocab_macro_top10",
    "fv_top10": "full_vocab_macro_top10",
    "full_macro_top10": "full_vocab_macro_top10",
    "fv_macro_top10": "full_vocab_macro_top10",
    "word_macro_top10": "full_vocab_macro_top10",
    "full_vocab_word_macro_top10": "full_vocab_macro_top10",
    "full_vocab_top10": "full_vocab_macro_top10",
    "subject_macro_top10": "full_vocab_subject_macro_top10",
    "subj_macro_top10": "full_vocab_subject_macro_top10",
    "full_subject_macro_top10": "full_vocab_subject_macro_top10",
    "full_micro_top10": "full_vocab_micro_top10",
    "full_vocab_event_micro_top10": "full_vocab_micro_top10",
    "fv_micro_top10": "full_vocab_micro_top10",
    "full_top1": "full_vocab_macro_top1",
    "fv_top1": "full_vocab_macro_top1",
    "full_macro_top1": "full_vocab_macro_top1",
    "fv_macro_top1": "full_vocab_macro_top1",
    "word_macro_top1": "full_vocab_macro_top1",
    "full_vocab_word_macro_top1": "full_vocab_macro_top1",
    "full_vocab_top1": "full_vocab_macro_top1",
    "full_micro_top1": "full_vocab_micro_top1",
    "full_vocab_event_micro_top1": "full_vocab_micro_top1",
    "fv_micro_top1": "full_vocab_micro_top1",
    "full_mrr": "full_vocab_mrr",
    "fv_mrr": "full_vocab_mrr",
    "full_median_rank": "full_vocab_median_rank",
    "fv_median_rank": "full_vocab_median_rank",
}


def _resolve_control_metric_spec(
    control_metric: str | Sequence[str] | None = None,
) -> dict[str, object]:
    """解析用户显式指定的单个负对照核心评估指标（若传入多个指标则返回首个主指标）。"""
    specs = _resolve_control_metric_specs(control_metric)
    return specs[0]


def _resolve_control_metric_specs(
    control_metric: str | Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """解析用户显式指定的一个或多个负对照核心评估指标（支持逗号/分号分隔、列表、'all' 及 k0_/k1_/k2_ 视界前缀）。"""
    if control_metric is None:
        raw_tokens = ["vocab50_macro_top10"]
    elif isinstance(control_metric, str):
        cleaned = control_metric.strip()
        if not cleaned or cleaned.lower() == "auto":
            raw_tokens = ["vocab50_macro_top10"]
        else:
            raw_tokens = [
                t.strip()
                for t in cleaned.replace(";", ",").replace("|", ",").split(",")
                if t.strip()
            ]
    elif isinstance(control_metric, Sequence):
        raw_tokens = []
        for item in control_metric:
            if isinstance(item, str):
                raw_tokens.extend(
                    t.strip()
                    for t in item.replace(";", ",").replace("|", ",").split(",")
                    if t.strip()
                )
            elif item is not None:
                raw_tokens.append(str(item).strip())
    else:
        raw_tokens = [str(control_metric).strip()]

    if not raw_tokens:
        raw_tokens = ["vocab50_macro_top10"]

    expanded_tokens: list[str] = []
    for tok in raw_tokens:
        low = tok.lower().replace("-", "_").replace(" ", "_")
        if low in ("all", "all_metrics", "*"):
            expanded_tokens.extend(
                [
                    "vocab50_macro_top10",
                    "vocab50_micro_top10",
                    "vocab50_macro_top1",
                    "vocab50_mrr",
                    "full_vocab_macro_top10",
                    "full_vocab_subject_macro_top10",
                    "full_vocab_micro_top10",
                    "full_vocab_macro_top1",
                    "full_vocab_mrr",
                    "k0_vocab50_macro_top10",
                    "k1_vocab50_macro_top10",
                    "k2_vocab50_macro_top10",
                    "k0_full_vocab_macro_top10",
                    "k1_full_vocab_macro_top10",
                    "k2_full_vocab_macro_top10",
                ]
            )
        elif low in ("auto", "default", ""):
            expanded_tokens.append("vocab50_macro_top10")
        else:
            expanded_tokens.append(low)

    specs: list[dict[str, object]] = []
    seen_keys: set[str] = set()
    for raw in expanded_tokens:
        horizon: str | None = None
        base_raw = raw
        for h_prefix, h_name in (("k0_", "K0"), ("k1_", "K1"), ("k2_", "K2")):
            if base_raw.startswith(h_prefix):
                horizon = h_name
                base_raw = base_raw[len(h_prefix) :]
                break
        if base_raw.startswith("common_"):
            base_raw = base_raw[len("common_") :]

        canonical = _CONTROL_METRIC_ALIASES.get(base_raw, base_raw)
        if canonical not in _CONTROL_METRIC_REGISTRY:
            valid_keys = ", ".join(sorted(_CONTROL_METRIC_REGISTRY.keys()))
            raise ValueError(
                f"不支持的负对照指标 control_metric={raw!r}。"
                f"可选指标（亦支持加 k0_/k1_/k2_ 前缀或传 'all'）：{valid_keys}"
            )
        spec = dict(_CONTROL_METRIC_REGISTRY[canonical])
        spec["horizon"] = horizon
        if horizon is not None:
            spec["requested_key"] = f"{horizon.lower()}_{canonical}"
            spec["label"] = f"{horizon} {spec['label']}"
            spec["short_label"] = f"{horizon} {spec['short_label']}"
        else:
            spec["requested_key"] = canonical
        req_k = str(spec["requested_key"])
        if req_k not in seen_keys:
            seen_keys.add(req_k)
            specs.append(spec)

    return specs or [_resolve_control_metric_specs("vocab50_macro_top10")[0]]


def _parse_control_methods_list(
    control_method: str | Sequence[str] | None = None,
) -> list[str]:
    """解析用户显式指定的一个或多个负对照/去偏方法（支持分号/竖线/逗号分隔与列表，且自动保留 subspace:K=8,eta=0.8,gate=off 内部逗号）。"""
    import re as _re

    if control_method is None:
        return ["auto"]
    raw_segments: list[str] = []
    if isinstance(control_method, str):
        raw_segments = [control_method]
    elif isinstance(control_method, Sequence):
        for item in control_method:
            if item is not None and str(item).strip():
                raw_segments.append(str(item).strip())
    else:
        raw_segments = [str(control_method).strip()]

    methods: list[str] = []
    for seg in raw_segments:
        for semi_part in _re.split(r"[;|]+", seg):
            semi_part = semi_part.strip()
            if not semi_part:
                continue
            comma_parts = [p.strip() for p in semi_part.split(",") if p.strip()]
            merged: list[str] = []
            for part in comma_parts:
                low_p = part.lower().replace(" ", "")
                is_subspace_Continuation = bool(
                    _re.match(
                        r"^(?:eta[=_:]|gate[=_:]|nogate|gated|k[=_:]?\d+|rank_k[=_:]?\d+)",
                        low_p,
                    )
                )
                if (
                    merged
                    and is_subspace_Continuation
                    and (
                        "subspace" in merged[-1].lower()
                        or _re.search(r"\bk[=_:]?\d+", merged[-1].lower())
                    )
                    and not low_p.startswith("subspace")
                    and not low_p.startswith("cms")
                ):
                    merged[-1] = f"{merged[-1]},{part}"
                else:
                    merged.append(part)
            for m_item in merged:
                if m_item and m_item not in methods:
                    methods.append(m_item)

    return methods or ["auto"]


def _extract_clean_and_shifted_metric(
    clean_ref: dict,
    sh_agg: dict,
    m_key: str,
) -> tuple[float, float]:
    """从单种子 clean_reference 与 shifted_aggregate 中提取统一口径的 Clean 与 Shifted 值。"""
    if m_key in ("full_vocab_macro_top10", "full_vocab_word_macro_top10"):
        c_val = float(
            clean_ref.get(
                "full_vocab_word_macro_top10",
                clean_ref.get("full_vocab_macro_top10", 0.0),
            )
        )
        s_val = float(
            sh_agg.get(
                "mean_full_vocab_word_macro_top10",
                sh_agg.get("mean_full_vocab_macro_top10", 0.0),
            )
        )
    elif m_key == "full_vocab_subject_macro_top10":
        c_val = float(
            clean_ref.get(
                "full_vocab_subject_macro_top10",
                clean_ref.get("full_vocab_micro_top10", 0.0),
            )
        )
        s_val = float(
            sh_agg.get(
                "mean_full_vocab_subject_macro_top10",
                sh_agg.get("mean_full_vocab_micro_top10", 0.0),
            )
        )
    elif m_key == "vocab50_subject_macro_top10":
        c_val = float(
            clean_ref.get(
                "vocab50_subject_macro_top10",
                clean_ref.get("vocab50_macro_top10", 0.0),
            )
        )
        s_val = float(
            sh_agg.get(
                "mean_vocab50_subject_macro_top10",
                sh_agg.get("mean_vocab50_macro_top10", 0.0),
            )
        )
    else:
        c_val = float(clean_ref.get(m_key, 0.0))
        s_val = float(sh_agg.get(f"mean_{m_key}", 0.0))
    return c_val, s_val

def _extract_metric_from_control_agg(
    ctrl_agg: dict,
    metric_spec: dict[str, object],
) -> dict[str, object]:
    """从已聚合的负对照字典中提取指定指标（含可选 K0/K1/K2 视界）的 Clean/Shifted/Gap Mean±SD、显著性 p 值与逐种子明细。"""
    m_key = str(metric_spec["key"])
    hm_key = str(metric_spec["horizon_key"])
    horizon = metric_spec.get("horizon")
    if isinstance(horizon, str) and horizon in (
        ctrl_agg.get("horizons_aggregate") or {}
    ):
        hb = ctrl_agg["horizons_aggregate"][horizon]
        per_seed_list = (hb.get("per_seed_metrics") or {}).get(hm_key, [])
        return {
            "metric_key": str(metric_spec["requested_key"]),
            "base_metric_key": m_key,
            "horizon": horizon,
            "metric_label": str(metric_spec["label"]),
            "short_label": str(metric_spec["short_label"]),
            "is_ratio": bool(metric_spec.get("is_ratio", True)),
            "mean_clean": float(hb.get(f"mean_clean_{hm_key}", 0.0)),
            "std_clean": float(hb.get(f"std_clean_{hm_key}", 0.0)),
            "mean_shifted": float(hb.get(f"mean_shifted_{hm_key}", 0.0)),
            "std_shifted": float(hb.get(f"std_shifted_{hm_key}", 0.0)),
            "mean_gap": float(hb.get(f"mean_gap_{hm_key}", 0.0)),
            "std_gap": float(hb.get(f"std_gap_{hm_key}", 0.0)),
            "chance_value": ctrl_agg.get(f"chance_{m_key}"),
            "p_shifted_vs_chance": hb.get(
                f"p_shifted_vs_chance_{hm_key}",
                ctrl_agg.get(f"p_shifted_vs_chance_{m_key}"),
            ),
            "p_str_shifted_vs_chance": str(
                hb.get(
                    f"p_str_shifted_vs_chance_{hm_key}",
                    ctrl_agg.get(f"p_str_shifted_vs_chance_{m_key}", ""),
                )
                or ""
            ),
            "p_gap": hb.get(f"p_gap_{hm_key}", ctrl_agg.get(f"p_gap_{m_key}")),
            "p_str_gap": str(
                hb.get(
                    f"p_str_gap_{hm_key}",
                    ctrl_agg.get(f"p_str_gap_{m_key}", ""),
                )
                or ""
            ),
            "per_seed": list(per_seed_list),
        }
    per_seed_list = (ctrl_agg.get("per_seed_metrics") or {}).get(m_key, [])
    return {
        "metric_key": str(metric_spec["requested_key"]),
        "base_metric_key": m_key,
        "horizon": None,
        "metric_label": str(metric_spec["label"]),
        "short_label": str(metric_spec["short_label"]),
        "is_ratio": bool(metric_spec.get("is_ratio", True)),
        "mean_clean": float(ctrl_agg.get(f"mean_clean_{m_key}", 0.0)),
        "std_clean": float(ctrl_agg.get(f"std_clean_{m_key}", 0.0)),
        "mean_shifted": float(ctrl_agg.get(f"mean_shifted_{m_key}", 0.0)),
        "std_shifted": float(ctrl_agg.get(f"std_shifted_{m_key}", 0.0)),
        "mean_gap": float(ctrl_agg.get(f"mean_gap_{m_key}", 0.0)),
        "std_gap": float(ctrl_agg.get(f"std_gap_{m_key}", 0.0)),
        "chance_value": ctrl_agg.get(f"chance_{m_key}"),
        "p_shifted_vs_chance": ctrl_agg.get(f"p_shifted_vs_chance_{m_key}"),
        "p_str_shifted_vs_chance": str(
            ctrl_agg.get(f"p_str_shifted_vs_chance_{m_key}", "") or ""
        ),
        "p_gap": ctrl_agg.get(f"p_gap_{m_key}"),
        "p_str_gap": str(ctrl_agg.get(f"p_str_gap_{m_key}", "") or ""),
        "per_seed": list(per_seed_list),
    }


def _select_active_control_method(
    methods_map: dict[str, dict],
    *,
    control_method: str | None = None,
    metric_spec: dict[str, object],
) -> dict:
    """根据用户传入的单个 control_method 选择对应的负对照/去偏聚合结果。"""
    if not methods_map:
        return {"enabled": False}

    raw_sel = str(control_method or "auto").strip()
    norm_sel = raw_sel.lower().replace(" ", "")

    cms_candidates = [
        v
        for k, v in methods_map.items()
        if k.startswith(("cms:", "causal_cms:", "pos_causal_cms:"))
        and float(v.get("eta") or 0.0) > 1e-9
    ]
    pos_cms_candidates = [
        v
        for k, v in methods_map.items()
        if k.startswith("pos_causal_cms:") and float(v.get("eta") or 0.0) > 1e-9
    ]
    sub_candidates = [
        v
        for k, v in methods_map.items()
        if k.startswith("subspace:") and v.get("rank_k") is not None
    ]

    if not norm_sel or norm_sel in ("auto", "default"):
        if sub_candidates:
            best = max(
                sub_candidates,
                key=lambda x: float(
                    _extract_metric_from_control_agg(x, metric_spec)["mean_gap"]
                ),
            )
            return dict(best)
        if pos_cms_candidates:
            best = max(
                pos_cms_candidates,
                key=lambda x: float(
                    _extract_metric_from_control_agg(x, metric_spec)["mean_gap"]
                ),
            )
            return dict(best)
        if cms_candidates:
            best = max(
                cms_candidates,
                key=lambda x: float(
                    _extract_metric_from_control_agg(x, metric_spec)["mean_gap"]
                ),
            )
            return dict(best)
        for default_id in ("raw", "zero_signal"):
            if default_id in methods_map:
                return dict(methods_map[default_id])
        return dict(next(iter(methods_map.values())))

    if norm_sel in ("raw", "temporal_shift", "eta=0", "eta=0.0", "eta_0", "raw_eta_0.0"):
        if "raw" in methods_map:
            return dict(methods_map["raw"])

    if norm_sel in ("zero_signal", "zero", "zerosignal"):
        if "zero_signal" in methods_map:
            return dict(methods_map["zero_signal"])

    if norm_sel in methods_map:
        return dict(methods_map[norm_sel])

    for k, v in methods_map.items():
        if k.lower().replace(" ", "") == norm_sel:
            return dict(v)

    if (
        norm_sel
        in ("pos_causal_cms", "pos_causal_ema", "pos_causal_cms:best_gap", "pos_causal_cms_best_gap")
        and pos_cms_candidates
    ):
        best = max(
            pos_cms_candidates,
            key=lambda x: float(
                _extract_metric_from_control_agg(x, metric_spec)["mean_gap"]
            ),
        )
        return dict(best)
    if (
        norm_sel in ("pos_causal_cms:best_clean", "pos_causal_cms_best_clean")
        and pos_cms_candidates
    ):
        best = max(
            pos_cms_candidates,
            key=lambda x: float(
                _extract_metric_from_control_agg(x, metric_spec)["mean_clean"]
            ),
        )
        return dict(best)

    if norm_sel in ("cms", "cms:best_gap", "cms_best_gap") and cms_candidates:
        best = max(
            cms_candidates,
            key=lambda x: float(
                _extract_metric_from_control_agg(x, metric_spec)["mean_gap"]
            ),
        )
        return dict(best)
    if norm_sel in ("cms:best_clean", "cms_best_clean") and cms_candidates:
        best = max(
            cms_candidates,
            key=lambda x: float(
                _extract_metric_from_control_agg(x, metric_spec)["mean_clean"]
            ),
        )
        return dict(best)

    if (
        norm_sel in ("subspace", "subspace:best_gap", "subspace_best_gap", "best_gap")
        and sub_candidates
    ):
        best = max(
            sub_candidates,
            key=lambda x: float(
                _extract_metric_from_control_agg(x, metric_spec)["mean_gap"]
            ),
        )
        return dict(best)
    if (
        norm_sel in ("subspace:best_clean", "subspace_best_clean", "best_clean")
        and sub_candidates
    ):
        raw_base = methods_map.get("raw")
        raw_gap = (
            float(_extract_metric_from_control_agg(raw_base, metric_spec)["mean_gap"])
            if isinstance(raw_base, dict) and raw_base.get("enabled")
            else -1e9
        )
        improving = [
            x
            for x in sub_candidates
            if float(_extract_metric_from_control_agg(x, metric_spec)["mean_gap"])
            > raw_gap
        ]
        pool = improving if improving else sub_candidates
        best = max(
            pool,
            key=lambda x: float(
                _extract_metric_from_control_agg(x, metric_spec)["mean_clean"]
            ),
        )
        return dict(best)

    # Parse eta=... or K=...,eta=...,gate=...
    import re as _re

    k_match = _re.search(r"k[=_:]?(\d+)", norm_sel)
    eta_match = _re.search(r"eta[=_:]?([0-9]+(?:\.[0-9]+)?)", norm_sel)
    gate_val: bool | None = None
    if any(g in norm_sel for g in ("gate=on", "gate=true", "gate_on", "gated")):
        gate_val = True
    elif any(
        g in norm_sel for g in ("gate=off", "gate=false", "gate_off", "nogate")
    ):
        gate_val = False

    if k_match is not None:
        target_k = int(k_match.group(1))
        target_eta = float(eta_match.group(1)) if eta_match else None
        for v in sub_candidates:
            if int(v.get("rank_k") or -1) != target_k:
                continue
            if (
                target_eta is not None
                and abs(float(v.get("eta") or 0.0) - target_eta) > 1e-6
            ):
                continue
            if (
                gate_val is not None
                and bool(v.get("use_coherence_gate")) != gate_val
            ):
                continue
            return dict(v)

    if eta_match is not None and k_match is None:
        target_eta = float(eta_match.group(1))
        if abs(target_eta) <= 1e-9 and "raw" in methods_map:
            return dict(methods_map["raw"])
        for v in cms_candidates:
            if abs(float(v.get("eta") or 0.0) - target_eta) <= 1e-6:
                return dict(v)

    for k, v in methods_map.items():
        if norm_sel in k.lower() or norm_sel in str(v.get("label", "")).lower().replace(" ", ""):
            return dict(v)

    avail = ", ".join(sorted(methods_map.keys()))
    raise ValueError(
        f"未找到匹配的负对照/去偏方法 control_method={control_method!r}。"
        f"当前可用方法：{avail}"
    )


def _select_control_methods_list(
    methods_map: dict[str, dict],
    *,
    control_method: str | Sequence[str] | None = None,
    metric_spec: dict[str, object],
) -> list[dict]:
    """根据用户传入的一个或多个 control_method（含 'all'）返回去重后的负对照/去偏聚合字典列表。"""
    if not methods_map:
        return []
    tokens = _parse_control_methods_list(control_method)
    selected_list: list[dict] = []
    seen_ids: set[str] = set()
    last_err: ValueError | None = None
    for tok in tokens:
        low_tok = tok.strip().lower()
        if low_tok in ("all", "all_methods", "*"):
            for m_key, m_val in methods_map.items():
                if not isinstance(m_val, dict) or not m_val.get("enabled"):
                    continue
                m_id = str(m_val.get("method_id", m_key))
                if m_id not in seen_ids:
                    seen_ids.add(m_id)
                    selected_list.append(dict(m_val))
        else:
            try:
                chosen = _select_active_control_method(
                    methods_map, control_method=tok, metric_spec=metric_spec
                )
            except ValueError as err:
                last_err = err
                continue
            if chosen.get("enabled"):
                m_id = str(chosen.get("method_id", tok))
                if m_id not in seen_ids:
                    seen_ids.add(m_id)
                    selected_list.append(dict(chosen))
    if not selected_list:
        has_debiased = any(
            k.startswith("cms:") or k.startswith("subspace:")
            for k in methods_map
        )
        if not has_debiased:
            fallback_chosen = _select_active_control_method(
                methods_map, control_method="auto", metric_spec=metric_spec
            )
            if fallback_chosen.get("enabled"):
                selected_list.append(dict(fallback_chosen))
                return selected_list
        if last_err is not None:
            raise last_err
    return selected_list


def _enrich_method_with_selected_metrics(
    method_agg: dict,
    metric_specs_list: Sequence[dict[str, object]],
    *,
    avail_methods: Sequence[str],
    requested_control_method: str,
) -> dict:
    """将用户选定的一个或多个评估指标挂载到单个负对照/去偏方法聚合字典上，并填充全部兼容别名字段。"""
    enriched = dict(method_agg)
    m_id = str(
        enriched.get(
            "method_id",
            enriched.get(
                "active_method_id",
                enriched.get("active_control_method", "raw"),
            ),
        )
    )
    m_lbl = str(
        enriched.get(
            "label",
            enriched.get(
                "active_method_label",
                enriched.get(
                    "active_control_method_label",
                    "Raw Temporal Shift (eta=0.0)",
                ),
            ),
        )
    )
    enriched["method_id"] = m_id
    enriched["active_method_id"] = m_id
    enriched["active_control_method"] = m_id
    enriched["label"] = m_lbl
    enriched["active_method_label"] = m_lbl
    enriched["active_control_method_label"] = m_lbl
    enriched["requested_control_method"] = requested_control_method
    enriched["available_control_methods"] = list(avail_methods)

    metrics_summaries: list[dict[str, object]] = [
        _extract_metric_from_control_agg(enriched, ms)
        for ms in metric_specs_list
    ]
    enriched["selected_metrics_summaries"] = metrics_summaries
    enriched["selected_control_metrics"] = [
        str(ms["metric_key"]) for ms in metrics_summaries
    ]
    enriched["selected_control_metric_labels"] = [
        str(ms["metric_label"]) for ms in metrics_summaries
    ]

    prim_sum = metrics_summaries[0]
    enriched["primary_control_metric"] = str(prim_sum["metric_key"])
    enriched["primary_control_metric_label"] = str(prim_sum["metric_label"])
    enriched["primary_control_metric_short_label"] = str(
        prim_sum["short_label"]
    )
    enriched["primary_metric_summary"] = prim_sum
    enriched["primary_metric_is_percentage"] = bool(
        prim_sum.get("is_ratio", True)
    )
    enriched["primary_metric_clean_mean"] = float(prim_sum["mean_clean"])
    enriched["primary_metric_clean_std"] = float(prim_sum["std_clean"])
    enriched["primary_metric_shifted_mean"] = float(prim_sum["mean_shifted"])
    enriched["primary_metric_shifted_std"] = float(prim_sum["std_shifted"])
    enriched["primary_metric_gap_mean"] = float(prim_sum["mean_gap"])
    enriched["primary_metric_gap_std"] = float(prim_sum["std_gap"])
    enriched["primary_metric_chance_value"] = prim_sum.get("chance_value")
    enriched["primary_metric_p_shifted_vs_chance"] = prim_sum.get(
        "p_shifted_vs_chance"
    )
    enriched["primary_metric_p_str_shifted_vs_chance"] = str(
        prim_sum.get("p_str_shifted_vs_chance", "") or ""
    )
    enriched["primary_metric_p_gap"] = prim_sum.get("p_gap")
    enriched["primary_metric_p_str_gap"] = str(
        prim_sum.get("p_str_gap", "") or ""
    )
    enriched["primary_metric_per_seed"] = list(prim_sum.get("per_seed") or [])
    if "conditions_aggregate" in enriched and "per_shift_aggregate" not in enriched:
        enriched["per_shift_aggregate"] = enriched["conditions_aggregate"]
    return enriched


def apply_multi_seed_control_selection(
    summary: dict,
    *,
    control_method: str | Sequence[str] | None = None,
    control_metric: str | Sequence[str] | None = None,
) -> dict:
    """在已聚合的多种子字典上动态切换或更新一个或多个负对照方法 (control_method) 与一个或多个核心评估指标 (control_metric)。"""
    if not isinstance(summary, dict):
        return summary

    if "stages_aggregate" in summary and isinstance(
        summary.get("stages_aggregate"), dict
    ):
        for st_key, st_agg in summary["stages_aggregate"].items():
            if isinstance(st_agg, dict):
                apply_multi_seed_control_selection(
                    st_agg,
                    control_method=control_method,
                    control_metric=control_metric,
                )
        summary["requested_control_method"] = (
            ";".join(_parse_control_methods_list(control_method))
            if control_method is not None
            else str(summary.get("requested_control_method", "auto"))
        )
        summary["requested_control_metric"] = (
            ",".join(
                str(s["requested_key"])
                for s in _resolve_control_metric_specs(control_metric)
            )
            if control_metric is not None
            else str(
                summary.get("requested_control_metric", "vocab50_macro_top10")
            )
        )
        return summary

    eff_method = (
        control_method
        if control_method is not None
        else summary.get("requested_control_method", "auto")
    )
    eff_metric = (
        control_metric
        if control_metric is not None
        else summary.get("requested_control_metric", "vocab50_macro_top10")
    )
    metric_specs_list = _resolve_control_metric_specs(eff_metric)
    primary_metric_spec = metric_specs_list[0]
    req_method_str = ";".join(_parse_control_methods_list(eff_method))
    req_metric_str = ",".join(
        str(ms["requested_key"]) for ms in metric_specs_list
    )

    methods_map = (
        summary.get("negative_controls_by_method")
        or summary.get("negative_control_methods_aggregate")
        or {}
    )
    avail_methods = (
        list(methods_map.keys()) if isinstance(methods_map, dict) else []
    )

    selected_methods_list: list[dict] = []
    if isinstance(methods_map, dict) and methods_map:
        raw_selected_list = _select_control_methods_list(
            methods_map,
            control_method=eff_method,
            metric_spec=primary_metric_spec,
        )
        selected_methods_list = [
            _enrich_method_with_selected_metrics(
                m_agg,
                metric_specs_list,
                avail_methods=avail_methods,
                requested_control_method=req_method_str,
            )
            for m_agg in raw_selected_list
        ]
    else:
        existing_nc = dict(
            summary.get("negative_controls_aggregate") or {"enabled": False}
        )
        if existing_nc.get("enabled"):
            selected_methods_list = [
                _enrich_method_with_selected_metrics(
                    existing_nc,
                    metric_specs_list,
                    avail_methods=avail_methods or [str(existing_nc.get("method_id", "raw"))],
                    requested_control_method=req_method_str,
                )
            ]

    if selected_methods_list:
        selected = dict(selected_methods_list[0])
        selected["selected_control_methods"] = [
            str(m["method_id"]) for m in selected_methods_list
        ]
        selected["selected_control_method_labels"] = [
            str(m["label"]) for m in selected_methods_list
        ]
        selected["selected_methods_aggregates"] = selected_methods_list

        multi_comparison: list[dict[str, object]] = []
        for m_agg in selected_methods_list:
            for m_sum in m_agg.get("selected_metrics_summaries") or ():
                if not isinstance(m_sum, dict):
                    continue
                multi_comparison.append(
                    {
                        "method_id": str(m_agg["method_id"]),
                        "method_label": str(m_agg["label"]),
                        "control_type": str(
                            m_agg.get("control_type", "temporal_shift")
                        ),
                        "eta": m_agg.get("eta"),
                        "rank_k": m_agg.get("rank_k"),
                        "use_coherence_gate": m_agg.get("use_coherence_gate"),
                        "metric_key": str(m_sum["metric_key"]),
                        "metric_id": str(m_sum["metric_key"]),
                        "base_metric_key": str(m_sum["base_metric_key"]),
                        "horizon": m_sum.get("horizon"),
                        "metric_label": str(m_sum["metric_label"]),
                        "short_label": str(m_sum["short_label"]),
                        "is_ratio": bool(m_sum.get("is_ratio", True)),
                        "is_percentage": bool(m_sum.get("is_ratio", True)),
                        "mean_clean": float(m_sum["mean_clean"]),
                        "std_clean": float(m_sum["std_clean"]),
                        "clean_mean": float(m_sum["mean_clean"]),
                        "clean_std": float(m_sum["std_clean"]),
                        "mean_shifted": float(m_sum["mean_shifted"]),
                        "std_shifted": float(m_sum["std_shifted"]),
                        "shifted_mean": float(m_sum["mean_shifted"]),
                        "shifted_std": float(m_sum["std_shifted"]),
                        "mean_gap": float(m_sum["mean_gap"]),
                        "std_gap": float(m_sum["std_gap"]),
                        "gap_mean": float(m_sum["mean_gap"]),
                        "gap_std": float(m_sum["std_gap"]),
                        "chance_value": m_sum.get("chance_value"),
                        "p_shifted_vs_chance": m_sum.get("p_shifted_vs_chance"),
                        "p_str_shifted_vs_chance": str(
                            m_sum.get("p_str_shifted_vs_chance", "") or ""
                        ),
                        "p_gap": m_sum.get("p_gap"),
                        "p_str_gap": str(m_sum.get("p_str_gap", "") or ""),
                        "per_seed": list(m_sum.get("per_seed") or []),
                    }
                )
        selected["multi_method_metric_comparison"] = multi_comparison

        # Update per_seed shifted/gap columns to reflect the primary selected method
        ps_metrics = selected.get("per_seed_metrics") or {}
        v50_10_by_sd = {
            int(r["seed"]): r
            for r in (ps_metrics.get("vocab50_macro_top10") or ())
        }
        v50_1_by_sd = {
            int(r["seed"]): r
            for r in (ps_metrics.get("vocab50_macro_top1") or ())
        }
        fv_10_by_sd = {
            int(r["seed"]): r
            for r in (ps_metrics.get("full_vocab_macro_top10") or ())
        }
        fv_1_by_sd = {
            int(r["seed"]): r
            for r in (ps_metrics.get("full_vocab_macro_top1") or ())
        }
        prim_by_sd = {
            int(r["seed"]): r
            for r in (
                (selected.get("primary_metric_summary") or {}).get("per_seed")
                or ()
            )
        }
        for row in summary.get("per_seed") or ():
            if not isinstance(row, dict):
                continue
            sd = int(row.get("seed", 0))
            if sd in v50_10_by_sd:
                row["clean_vocab50_macro_top10_under_control"] = float(
                    v50_10_by_sd[sd]["clean"]
                )
                row["shifted_vocab50_macro_top10"] = float(
                    v50_10_by_sd[sd]["shifted"]
                )
                row["clean_minus_shifted_vocab50_macro_top10"] = float(
                    v50_10_by_sd[sd]["gap"]
                )
            if sd in v50_1_by_sd:
                row["shifted_vocab50_macro_top1"] = float(
                    v50_1_by_sd[sd]["shifted"]
                )
                row["clean_minus_shifted_vocab50_macro_top1"] = float(
                    v50_1_by_sd[sd]["gap"]
                )
            if sd in fv_10_by_sd:
                row["shifted_full_vocab_macro_top10"] = float(
                    fv_10_by_sd[sd]["shifted"]
                )
                row["clean_minus_shifted_full_vocab_macro_top10"] = float(
                    fv_10_by_sd[sd]["gap"]
                )
            if sd in fv_1_by_sd:
                row["shifted_full_vocab_macro_top1"] = float(
                    fv_1_by_sd[sd]["shifted"]
                )
                row["clean_minus_shifted_full_vocab_macro_top1"] = float(
                    fv_1_by_sd[sd]["gap"]
                )
            if sd in prim_by_sd:
                row["primary_control_clean"] = float(prim_by_sd[sd]["clean"])
                row["primary_control_shifted"] = float(prim_by_sd[sd]["shifted"])
                row["primary_control_gap"] = float(prim_by_sd[sd]["gap"])
    else:
        selected = {"enabled": False}

    summary["requested_control_method"] = req_method_str
    summary["requested_control_metric"] = req_metric_str
    summary["selected_control_methods"] = (
        selected.get("selected_control_methods", [])
        if selected.get("enabled")
        else []
    )
    summary["selected_control_metrics"] = (
        selected.get("selected_control_metrics", [])
        if selected.get("enabled")
        else [str(ms["requested_key"]) for ms in metric_specs_list]
    )
    summary["negative_controls_aggregate"] = selected
    return summary
