from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from braindecoding.evaluation.common_mode_subtraction_evaluation import (
    apply_causal_ema_subtraction_to_encoded,
    apply_causal_ema_subtraction_to_tensors,
    run_causal_ema_common_mode_subtraction_sweep,
)
from braindecoding.evaluation.evaluation_reporting import (
    _auto_upgrade_seed_exploratory_files,
    _needs_exploratory_pt_upgrade,
    format_common_mode_subtraction_report,
)


def _reference_step_by_step_causal_ema(
    predictions: torch.Tensor,
    mu0: torch.Tensor,
    *,
    eta: float,
    ema_decay: float,
) -> torch.Tensor:
    """用最直观的逐词在线 for 循环复现严格因果 EMA-CMS，作为代数黄金基准。"""
    y_norm = F.normalize(predictions.detach().float(), p=2.0, dim=-1, eps=1e-12)
    mu = mu0.detach().float().clone()
    outputs = []
    for row in y_norm:
        debiased = F.normalize(row - float(eta) * mu, p=2.0, dim=-1, eps=1e-12)
        outputs.append(debiased)
        mu = float(ema_decay) * mu + (1.0 - float(ema_decay)) * row
    return torch.stack(outputs, dim=0)


def test_causal_ema_matches_step_by_step_loop_and_is_strictly_future_invariant():
    """反事实检验 1：跨分块 (>512) 严格匹配逐词递推循环，且篡改未来词 (>c) 对 <=c 输出差异恒为 0。"""
    torch.manual_seed(2026)
    n_steps = 620  # 跨过内部 chunk_size=512 边界
    dim = 24
    raw_preds = torch.randn(n_steps, dim)
    mu0 = F.normalize(torch.randn(dim), dim=-1) * 0.3

    vec_out = apply_causal_ema_subtraction_to_tensors(
        raw_preds,
        mu0=mu0,
        eta=0.5,
        ema_decay=0.99,
    )
    ref_out = _reference_step_by_step_causal_ema(
        raw_preds,
        mu0,
        eta=0.5,
        ema_decay=0.99,
    )
    torch.testing.assert_close(vec_out, ref_out, atol=1e-6, rtol=1e-6)

    # 极端反事实篡改：将 c > 135 的所有未来预测替换为巨幅对抗张量，验证 c <= 135 绝对零泄漏
    cutoff = 135
    mutated_preds = raw_preds.clone()
    mutated_preds[cutoff + 1 :] = torch.randn_like(mutated_preds[cutoff + 1 :]) * 500.0 + 99.0

    mutated_out = apply_causal_ema_subtraction_to_tensors(
        mutated_preds,
        mu0=mu0,
        eta=0.5,
        ema_decay=0.99,
    )
    max_diff_past = float((mutated_out[: cutoff + 1] - vec_out[: cutoff + 1]).abs().max().item())
    assert max_diff_past <= 1e-7, f"Causal EMA leaked future predictions: diff={max_diff_past}"
    assert float((mutated_out[cutoff + 5 :] - vec_out[cutoff + 5 :]).abs().max().item()) > 1e-3

    # 同时验证含 K0/K1/K2 与 revision_availability_masks 的完整 encoded 字典因果不变性
    avail_k2 = torch.ones(40, dtype=torch.bool)
    avail_k2[18:20] = False  # 句尾无 K2 样本不应污染后续句子的 EMA 状态
    encoded_orig = {
        "predictions": torch.randn(40, dim),
        "subject_indices": torch.zeros(40, dtype=torch.long),
        "recording_ids": ["pallier2025|sub-01|run-06"] * 40,
        "revision_predictions": {
            0: torch.randn(40, dim),
            1: torch.randn(40, dim),
            2: torch.randn(40, dim),
        },
        "revision_availability_masks": {
            0: torch.ones(40, dtype=torch.bool),
            1: torch.ones(40, dtype=torch.bool),
            2: avail_k2,
        },
    }
    mu0_payload = {
        "version": "causal_cms_mu0_v1",
        "source": "train",
        "predictions": mu0,
        "revision_predictions": {0: mu0, 1: mu0 * 0.9, 2: mu0 * 0.8},
    }
    deb_orig = apply_causal_ema_subtraction_to_encoded(
        encoded_orig, mu0_payload=mu0_payload, eta=0.5, ema_decay=0.98
    )

    encoded_mut = {
        **encoded_orig,
        "predictions": encoded_orig["predictions"].clone(),
        "revision_predictions": {
            d: t.clone() for d, t in encoded_orig["revision_predictions"].items()
        },
    }
    encoded_mut["predictions"][25:] = -777.0
    for d in (0, 1, 2):
        encoded_mut["revision_predictions"][d][25:] = 888.0

    deb_mut = apply_causal_ema_subtraction_to_encoded(
        encoded_mut, mu0_payload=mu0_payload, eta=0.5, ema_decay=0.98
    )
    assert float((deb_mut["predictions"][:25] - deb_orig["predictions"][:25]).abs().max().item()) <= 1e-7
    for d in (0, 1, 2):
        assert (
            float(
                (
                    deb_mut["revision_predictions"][d][:25]
                    - deb_orig["revision_predictions"][d][:25]
                )
                .abs()
                .max()
                .item()
            )
            <= 1e-7
        )


def test_causal_ema_resets_strictly_at_subject_and_recording_boundaries():
    """反事实检验 2：跨受试者或跨录音会话边界必须严格重置 mu <- mu_0，前序会话绝不串扰后续会话。"""
    torch.manual_seed(314)
    dim = 16
    mu0 = F.normalize(torch.randn(dim), dim=-1) * 0.25

    preds_a = torch.randn(30, dim)
    subj_idx = torch.tensor([0] * 10 + [0] * 10 + [1] * 10, dtype=torch.long)
    rec_ids = ["run-01"] * 10 + ["run-02"] * 10 + ["run-01"] * 10

    enc_a = {
        "predictions": preds_a,
        "subject_indices": subj_idx,
        "recording_ids": rec_ids,
    }
    deb_a = apply_causal_ema_subtraction_to_encoded(enc_a, mu0_payload=mu0, eta=0.5, ema_decay=0.95)

    # 极端篡改第一段会话 (subj=0, run-01, rows 0..9)，验证第二段 (subj=0, run-02) 与第三段 (subj=1, run-01) 0.0 变化
    preds_b = preds_a.clone()
    preds_b[:10] = torch.randn(10, dim) * 100.0
    enc_b = {
        "predictions": preds_b,
        "subject_indices": subj_idx,
        "recording_ids": rec_ids,
    }
    deb_b = apply_causal_ema_subtraction_to_encoded(enc_b, mu0_payload=mu0, eta=0.5, ema_decay=0.95)

    diff_later_sessions = float((deb_b["predictions"][10:] - deb_a["predictions"][10:]).abs().max().item())
    assert diff_later_sessions == 0.0

    # 验证每段会话的首词 (idx = 0, 10, 20) 精确等价于仅扣除先验 mu_0
    for start_row in (0, 10, 20):
        y0_norm = F.normalize(preds_b[start_row : start_row + 1].float(), p=2.0, dim=-1, eps=1e-12)[0]
        expected_first = F.normalize(y0_norm - 0.5 * mu0, p=2.0, dim=-1, eps=1e-12)
        torch.testing.assert_close(deb_b["predictions"][start_row], expected_first, atol=1e-6, rtol=1e-6)


def test_causal_ema_sweep_supports_m_eval_zero_and_matches_clean_metrics(tmp_path: Path):
    """反事实检验 3：M_eval=0（无任何错位前向）与 M_eval=4 的 Clean 去偏指标完全一致，且终端看板 <=86 列。"""
    torch.manual_seed(77)
    n_events = 24
    dim = 16
    vocab = [f"w{i}" for i in range(8)]
    vocab_bank = F.normalize(torch.randn(len(vocab), dim), dim=-1)
    word_idx = [i % len(vocab) for i in range(n_events)]
    targets = vocab_bank[word_idx]
    drift = F.normalize(torch.ones(dim), dim=-1) * 0.6
    clean_preds = F.normalize(targets + drift.unsqueeze(0) + 0.1 * torch.randn(n_events, dim), dim=-1)
    mu0 = drift.clone()

    clean_encoded = {
        "predictions": clean_preds,
        "targets": targets,
        "words": [vocab[i] for i in word_idx],
        "subject_indices": torch.zeros(n_events, dtype=torch.long),
        "recording_ids": ["pallier2025|sub-01|run-06"] * n_events,
    }
    shifted_list = []
    for sh_sec in (-10.0, -20.0, -30.0, -40.0):
        sh_preds = F.normalize(drift.unsqueeze(0) + 0.3 * torch.randn(n_events, dim), dim=-1)
        shifted_list.append(
            (
                sh_sec,
                f"shift_{int(abs(sh_sec))}s",
                f"Shifted ({int(sh_sec)}s)",
                {
                    "predictions": sh_preds,
                    "targets": targets,
                    "words": [vocab[i] for i in word_idx],
                    "subject_indices": torch.zeros(n_events, dtype=torch.long),
                    "recording_ids": ["pallier2025|sub-01|run-06"] * n_events,
                },
            )
        )

    def _dummy_compute_metrics(enc, _ds, _vocab, _vocab50, _cfg, *, stage, split):
        del stage, split
        scores = F.normalize(enc["predictions"].float(), dim=-1) @ vocab_bank.t()
        ranks = (scores > scores[torch.arange(n_events), torch.tensor(word_idx)].unsqueeze(1)).sum(dim=1) + 1
        top1 = float((ranks <= 1).float().mean().item())
        top4 = float((ranks <= 4).float().mean().item())
        return {
            "vocab50_macro_top10": top4,
            "vocab50_macro_top1": top1,
            "vocab50_micro_top10": top4,
            "full_vocab_macro_top10": top4,
            "full_vocab_macro_top1": top1,
        }

    def _dummy_extract_row(key, label, sec, metrics, clean_summary=None):
        row = {
            "key": key,
            "label": label,
            "shift_seconds": float(sec),
            **metrics,
        }
        if clean_summary is not None:
            row["delta_vocab50_macro_top10"] = (
                float(metrics["vocab50_macro_top10"]) - float(clean_summary["vocab50_macro_top10"])
            )
        return row

    def _dummy_summarize_shifted(clean_row, shifted_rows):
        if not shifted_rows:
            return {}
        mean_sh = sum(float(r["vocab50_macro_top10"]) for r in shifted_rows) / len(shifted_rows)
        return {
            "shifted_mean_vocab50_macro_top10": mean_sh,
            "clean_minus_shifted_vocab50_macro_top10": float(clean_row["vocab50_macro_top10"]) - mean_sh,
        }

    out_m0 = tmp_path / "m_eval_0"
    res_m0 = run_causal_ema_common_mode_subtraction_sweep(
        dataset="pallier2025",
        stage="prefix_revision",
        split="val",
        seed=0,
        exp_tag="unit_test",
        word_exp_tag="unit_test",
        clean_encoded=clean_encoded,
        shifted_encoded_by_shift=(),  # M_eval = 0!
        eval_ds=None,
        vocabulary=vocab,
        vocab50=vocab,
        cfg={},
        mu0_payload={"source": "train", "predictions": mu0},
        eta_values=[0.5],
        ema_decay=0.99,
        shift_mode="preroll",
        compute_metrics_fn=_dummy_compute_metrics,
        extract_summary_row_fn=_dummy_extract_row,
        summarize_shifted_fn=_dummy_summarize_shifted,
        output_dir=out_m0,
        plot=False,
        print_report=False,
    )

    out_m4 = tmp_path / "m_eval_4"
    res_m4 = run_causal_ema_common_mode_subtraction_sweep(
        dataset="pallier2025",
        stage="prefix_revision",
        split="val",
        seed=0,
        exp_tag="unit_test",
        word_exp_tag="unit_test",
        clean_encoded=clean_encoded,
        shifted_encoded_by_shift=shifted_list,  # M_eval = 4!
        eval_ds=None,
        vocabulary=vocab,
        vocab50=vocab,
        cfg={},
        mu0_payload={"source": "train", "predictions": mu0},
        eta_values=[0.5],
        ema_decay=0.99,
        shift_mode="preroll",
        compute_metrics_fn=_dummy_compute_metrics,
        extract_summary_row_fn=_dummy_extract_row,
        summarize_shifted_fn=_dummy_summarize_shifted,
        output_dir=out_m4,
        plot=False,
        print_report=False,
    )

    assert res_m0["cms_mode"] == "causal_ema"
    assert res_m0["formula"]["clean_requires_shifted_forwards"] is False
    assert res_m0["formula"]["num_shifted_conditions_M"] == 0
    assert (out_m0 / "exploratory" / "common_mode_subtraction_val.json").is_file()

    # Clean 去偏指标在 M_eval=0 与 M_eval=4 下必须严格 100% 相等（因为 Clean 在线流完全不依赖任何错位前向）
    assert res_m0["sweep"][1]["clean_reference"] == res_m4["sweep"][1]["clean_reference"]
    assert (
        res_m0["sweep"][1]["clean_reference"]["vocab50_macro_top1"]
        >= res_m0["sweep"][0]["clean_reference"]["vocab50_macro_top1"]
    )

    for payload in (res_m0, res_m4):
        assert _needs_exploratory_pt_upgrade(payload, is_cms=True) is False
        compact_text = format_common_mode_subtraction_report(payload, verbose=False)
        assert "Causal EMA Common-Mode Subtraction" in compact_text
        for line in compact_text.splitlines():
            assert len(line) <= 86, f"Report line exceeded 86 cols ({len(line)}): {line!r}"

    # 验证即使 exploratory/ 下同时存在 subspace_encoded_cache_val.pt，_auto_upgrade_seed_exploratory_files 也绝不将 causal_ema 覆盖降级为 legacy LOO
    torch.save({"dummy": True}, out_m4 / "exploratory" / "subspace_encoded_cache_val.pt")
    before_bytes = (out_m4 / "exploratory" / "common_mode_subtraction_val.json").read_text(
        encoding="utf-8"
    )
    _auto_upgrade_seed_exploratory_files(out_m4, "val")
    after_bytes = (out_m4 / "exploratory" / "common_mode_subtraction_val.json").read_text(
        encoding="utf-8"
    )
    assert after_bytes == before_bytes
    assert not (out_m4 / "exploratory" / "subspace_coherence_debiasing_val.json").exists()
