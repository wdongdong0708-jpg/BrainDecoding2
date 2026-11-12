from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from braindecoding.cli import build_parser
from braindecoding.evaluation.common_mode_subtraction_evaluation import (
    apply_causal_ema_subtraction_to_encoded,
    apply_causal_ema_subtraction_to_tensors,
    apply_position_causal_ema_subtraction_to_encoded,
    apply_position_causal_ema_subtraction_to_tensors,
    build_sentence_positions_from_encoded,
    evaluate_cached_pts_pos_causal_and_m_eval,
    position_bucket_index,
    run_causal_ema_common_mode_subtraction_sweep,
)
from braindecoding.evaluation.subspace_coherence_debiasing_evaluation import (
    save_subspace_encoded_cache,
)
from braindecoding.evaluation.evaluation_reporting import (
    _auto_upgrade_seed_exploratory_files,
    _needs_exploratory_pt_upgrade,
    format_common_mode_subtraction_report,
)
from braindecoding.evaluation.word_retrieval_evaluation import (
    _apply_eval_subdir_to_seed_dir,
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


def test_eval_subdir_isolates_zero_retrain_audit_outputs_from_existing_runs(tmp_path: Path):
    """验证 --eval-subdir 将零重训审计产物物理隔离到 <exp_tag>/<eval_subdir>/<stage>/<subset>/<seed>，原目录 0 字节写入。"""
    orig_seed_dir = (
        tmp_path
        / "runs"
        / "pallier2025"
        / "2_lr4e-4_bs256_siamese"
        / "prefix_revision"
        / "all"
        / "seed-000"
    )
    orig_seed_dir.mkdir(parents=True)

    isolated = _apply_eval_subdir_to_seed_dir(
        orig_seed_dir,
        "causal_fast_audit",
        stage="prefix_revision",
    )
    expected = (
        tmp_path
        / "runs"
        / "pallier2025"
        / "2_lr4e-4_bs256_siamese"
        / "causal_fast_audit"
        / "prefix_revision"
        / "all"
        / "seed-000"
    ).resolve()
    assert isolated == expected
    assert _apply_eval_subdir_to_seed_dir(orig_seed_dir, None, stage="prefix_revision") == orig_seed_dir

    parser = build_parser()
    parsed = parser.parse_args(
        [
            "eval-run",
            "--dataset",
            "pallier2025",
            "--stage",
            "prefix_revision",
            "--eval-subdir",
            "causal_fast_audit",
        ]
    )
    assert parsed.eval_subdir == "causal_fast_audit"


def test_position_bucket_index_exact_seven_bucket_boundaries():
    """验证 7 档句内位置桶映射：0, 1, 2, [3,4], [5,7], [8,12], [13+]。"""
    expected = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 4,
        6: 4,
        7: 4,
        8: 5,
        9: 5,
        10: 5,
        11: 5,
        12: 5,
        13: 6,
        14: 6,
        28: 6,
    }
    for pos, bucket in expected.items():
        assert position_bucket_index(pos) == bucket


def test_position_causal_ema_is_strictly_sentence_boundary_causal_and_session_isolated():
    """验证 pos_causal_cms 严格句末因果更新、零句内自相减、<3 冷启动回退、跨会话重置及 eval_table 位置恢复。"""
    import pandas as pd

    torch.manual_seed(2027)
    dim = 20
    n_sents_per_session = 6
    words_per_sent = 6  # 覆盖桶 0, 1, 2, [3,4], [5,7]
    n_per_session = n_sents_per_session * words_per_sent
    total_rows = 2 * n_per_session

    mu0 = F.normalize(torch.randn(dim), dim=-1) * 0.25
    preds = torch.randn(total_rows, dim)
    rev_k0 = torch.randn(total_rows, dim)
    rev_k1 = torch.randn(total_rows, dim)
    rev_k2 = torch.randn(total_rows, dim)
    avail_k2 = torch.ones(total_rows, dtype=torch.bool)
    avail_k2[words_per_sent - 1 :: words_per_sent] = False

    event_ids = []
    subj_ids = []
    rec_ids = []
    sent_uids = []
    for sess_idx, (sub, rec) in enumerate([("sub-01", "run-01"), ("sub-01", "run-02")]):
        for s_idx in range(n_sents_per_session):
            for w_idx in range(words_per_sent):
                row_id = sess_idx * n_per_session + s_idx * words_per_sent + w_idx
                event_ids.append(f"{sub}:{rec}:ev-{row_id:04d}")
                subj_ids.append(sub)
                rec_ids.append(rec)
                sent_uids.append(f"sent-{s_idx:02d}")

    eval_table = pd.DataFrame(
        {
            "event_id": event_ids,
            "subject_id": subj_ids,
            "recording_id": rec_ids,
            "sentence_uid": sent_uids,
        }
    )
    # 故意不传 sentence_indices，验证仅凭 eval_table + event_ids 即可精确还原 0-based 句内词位置
    encoded_orig = {
        "predictions": preds,
        "event_ids": event_ids,
        "recording_ids": rec_ids,
        "revision_predictions": {0: rev_k0, 1: rev_k1, 2: rev_k2},
        "revision_availability_masks": {
            0: torch.ones(total_rows, dtype=torch.bool),
            1: torch.ones(total_rows, dtype=torch.bool),
            2: avail_k2,
        },
    }
    streams, sents, positions = build_sentence_positions_from_encoded(
        encoded_orig, eval_table=eval_table
    )
    assert positions[:words_per_sent] == [0, 1, 2, 3, 4, 5]
    assert positions[words_per_sent : 2 * words_per_sent] == [0, 1, 2, 3, 4, 5]
    assert len(set(streams)) == 2
    assert len(set(sents)) == 2 * n_sents_per_session

    mu0_payload = {
        "predictions": mu0,
        "revision_predictions": {0: mu0, 1: mu0 * 0.9, 2: mu0 * 0.8},
    }
    deb_orig = apply_position_causal_ema_subtraction_to_encoded(
        encoded_orig,
        mu0_payload=mu0_payload,
        eta=0.8,
        ema_decay=0.99,
        pos_ema_decay=0.95,
        min_bucket_count=3,
        eval_table=eval_table,
    )

    # 1) 零句内自相减 & 零未来句泄漏：极端篡改第 1 段会话中第 4 句（s_idx=3, rows 18..23）的后半句及全部后续句（s_idx>=4）
    target_sent_start = 3 * words_per_sent  # row 18
    encoded_mut = {
        **encoded_orig,
        "predictions": preds.clone(),
        "revision_predictions": {
            0: rev_k0.clone(),
            1: rev_k1.clone(),
            2: rev_k2.clone(),
        },
    }
    # 篡改同句内词位 20..23 以及未来句 24..35（保持第 2 段会话 36..71 不变）
    encoded_mut["predictions"][target_sent_start + 2 : n_per_session] = -999.0
    for d in (0, 1, 2):
        encoded_mut["revision_predictions"][d][target_sent_start + 2 : n_per_session] = 777.0

    deb_mut = apply_position_causal_ema_subtraction_to_encoded(
        encoded_mut,
        mu0_payload=mu0_payload,
        eta=0.8,
        ema_decay=0.99,
        pos_ema_decay=0.95,
        min_bucket_count=3,
        eval_table=eval_table,
    )
    # 之前所有句 (0..17) 及当前第 4 句未篡改的前两个词 (18, 19) 输出必须 0 差异！
    max_diff_causal = float(
        (
            deb_mut["predictions"][: target_sent_start + 2]
            - deb_orig["predictions"][: target_sent_start + 2]
        )
        .abs()
        .max()
        .item()
    )
    assert max_diff_causal <= 1e-7
    for d in (0, 1, 2):
        assert (
            float(
                (
                    deb_mut["revision_predictions"][d][: target_sent_start + 2]
                    - deb_orig["revision_predictions"][d][: target_sent_start + 2]
                )
                .abs()
                .max()
                .item()
            )
            <= 1e-7
        )

    # 2) 跨会话严格重置：第 1 段会话被极端篡改后，第 2 段会话（rows 36..71）输出差异恒为 0.0
    assert (
        float(
            (deb_mut["predictions"][n_per_session:] - deb_orig["predictions"][n_per_session:])
            .abs()
            .max()
            .item()
        )
        == 0.0
    )

    # 3) 验证冷启动 (<3 句回退全局 mu_global，第 4 句起切换至前 3 句同桶均值 mu_pos[b])
    y_norm = F.normalize(preds.float(), p=2.0, dim=-1, eps=1e-12)
    # 首句任意词 (s_idx=0, rows 0..5) 此时 count[b]==0 < 3 且句内不更新，必须精确扣除 mu0
    for r_i in range(words_per_sent):
        expected_s0 = F.normalize(y_norm[r_i] - 0.8 * mu0, p=2.0, dim=-1, eps=1e-12)
        torch.testing.assert_close(deb_orig["predictions"][r_i], expected_s0, atol=1e-6, rtol=1e-6)
    # 第 4 句句首词 (s_idx=3, row=18, bucket=0) 此时 count[0]==3，必须精确扣除前 3 句句首词 (rows 0, 6, 12) 的无偏均值
    expected_mu_pos_b0 = torch.stack([y_norm[0], y_norm[6], y_norm[12]], dim=0).mean(dim=0)
    expected_s3_w0 = F.normalize(
        y_norm[target_sent_start] - 0.8 * expected_mu_pos_b0, p=2.0, dim=-1, eps=1e-12
    )
    torch.testing.assert_close(
        deb_orig["predictions"][target_sent_start], expected_s3_w0, atol=1e-6, rtol=1e-6
    )


def test_position_causal_ema_eliminates_intra_sentence_position_bias_where_global_ema_fails():
    """合成基准检验：当句内位置偏置跨桶零均值但同位置高度相关时，Global EMA 无法压低 Shifted，而 Pos-EMA 将 Shifted 压回 20% 下界。"""
    torch.manual_seed(42)
    n_vocab = 50
    dim = 32
    n_sentences = 60
    words_per_sent = 7  # 每个位置 0..6 对应部分桶并使某些词偏好句首/句尾

    vocab_bank = F.normalize(torch.randn(n_vocab, dim), dim=-1)
    # 构造 7 个零和正交位置共模偏置：在句首偏好词 0..6，句尾偏好词 42..48，跨整句求平均几乎抵消为 0
    raw_pos_biases = torch.stack(
        [vocab_bank[p * 7 : (p + 1) * 7].mean(dim=0) for p in range(words_per_sent)],
        dim=0,
    )
    raw_pos_biases = raw_pos_biases - raw_pos_biases.mean(dim=0, keepdim=True)
    pos_biases = F.normalize(raw_pos_biases, dim=-1) * 1.15

    word_indices = []
    clean_rows = []
    shifted_rows = []
    sent_indices = []
    for s_i in range(n_sentences):
        for p_i in range(words_per_sent):
            # 句内位置 p_i 的真实目标词以一定概率落在对应位置偏好词簇内（模拟自然语言句首冠词/主语分布）
            if s_i % 2 == 0:
                w_id = (p_i * 7 + (s_i % 7)) % n_vocab
            else:
                w_id = int(torch.randint(0, n_vocab, (1,)).item())
            word_indices.append(w_id)
            sent_indices.append(s_i)
            bias_vec = pos_biases[p_i]
            clean_vec = F.normalize(
                0.85 * vocab_bank[w_id] + bias_vec + 0.15 * torch.randn(dim), dim=-1
            )
            shifted_vec = F.normalize(bias_vec + 0.35 * torch.randn(dim), dim=-1)
            clean_rows.append(clean_vec)
            shifted_rows.append(shifted_vec)

    clean_preds = torch.stack(clean_rows, dim=0)
    shifted_preds = torch.stack(shifted_rows, dim=0)
    target_idx_t = torch.tensor(word_indices, dtype=torch.long)

    def _top10_acc(pred_tensor: torch.Tensor) -> float:
        scores = F.normalize(pred_tensor.float(), dim=-1) @ vocab_bank.t()
        true_scores = scores[torch.arange(len(word_indices)), target_idx_t].unsqueeze(1)
        ranks = (scores > true_scores).sum(dim=1) + 1
        # 仅统计预热后稳定段 (s_i >= 5)
        warmup_cut = 5 * words_per_sent
        return float((ranks[warmup_cut:] <= 10).float().mean().item())

    raw_clean_top10 = _top10_acc(clean_preds)
    raw_shift_top10 = _top10_acc(shifted_preds)

    glob_clean = apply_causal_ema_subtraction_to_tensors(
        clean_preds, mu0=None, eta=1.0, ema_decay=0.98
    )
    glob_shift = apply_causal_ema_subtraction_to_tensors(
        shifted_preds, mu0=None, eta=1.0, ema_decay=0.98
    )
    glob_shift_top10 = _top10_acc(glob_shift)

    pos_clean = apply_position_causal_ema_subtraction_to_tensors(
        clean_preds,
        mu0=None,
        eta=1.0,
        ema_decay=0.99,
        pos_ema_decay=0.95,
        min_bucket_count=3,
        sentence_keys=sent_indices,
    )
    pos_shift = apply_position_causal_ema_subtraction_to_tensors(
        shifted_preds,
        mu0=None,
        eta=1.0,
        ema_decay=0.99,
        pos_ema_decay=0.95,
        min_bucket_count=3,
        sentence_keys=sent_indices,
    )
    pos_clean_top10 = _top10_acc(pos_clean)
    pos_shift_top10 = _top10_acc(pos_shift)

    # Raw 与 Global EMA 的 Shifted 均显著高于 20% 随机下界 (>32%)，而 Pos-EMA 将 Shifted 压回 ~20% 下界且净间距显著扩大
    assert raw_shift_top10 > 0.32
    assert glob_shift_top10 > 0.30
    assert pos_shift_top10 < 0.25
    assert (pos_clean_top10 - pos_shift_top10) > (raw_clean_top10 - raw_shift_top10) + 0.10
    assert (pos_clean_top10 - pos_shift_top10) > (_top10_acc(glob_clean) - glob_shift_top10) + 0.10


def test_preroll_m_eval_tradeoff_and_cached_pt_zero_overwrite_evaluator(tmp_path: Path):
    """验证基于 .pt 特征缓存的 0 前向/0 覆盖 M_eval={0,1,2,4,6} + pos_causal_cms 多种子评测导出与双联图。"""
    import pandas as pd

    torch.manual_seed(99)
    n_vocab = 50
    dim = 16
    n_sents = 8
    words_per_sent = 6
    n_events = n_sents * words_per_sent

    vocab = [f"word_{i:02d}" for i in range(n_vocab)]
    vocab_bank = F.normalize(torch.randn(n_vocab, dim), dim=-1)
    emb_map = {w: vocab_bank[i].numpy() for i, w in enumerate(vocab)}

    source_stage_dir = tmp_path / "readonly_causal_fast_audit"
    shifts = [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0]

    for seed in (0, 1):
        g = torch.Generator().manual_seed(1000 + seed)
        w_ids = [int(torch.randint(0, n_vocab, (1,), generator=g).item()) for _ in range(n_events)]
        words = [vocab[i] for i in w_ids]
        targets = vocab_bank[w_ids]
        eids = [f"sub-01:run-01:ev-{i:04d}" for i in range(n_events)]
        s_uids = [f"sent-{i // words_per_sent:02d}" for i in range(n_events)]
        eval_df = pd.DataFrame(
            {
                "event_id": eids,
                "subject_id": ["sub-01"] * n_events,
                "recording_id": ["run-01"] * n_events,
                "sentence_uid": s_uids,
                "runtime_context_uid": [f"run-01:{s}" for s in s_uids],
            }
        )

        class _MockDs:
            table = eval_df
            embedding_map = emb_map
            window_seconds = 1.0
            window_samples = 50

        drift = F.normalize(torch.randn(dim, generator=g), dim=-1) * 0.5
        clean_p = F.normalize(targets + drift.unsqueeze(0) + 0.1 * torch.randn(n_events, dim, generator=g), dim=-1)
        clean_enc = {
            "predictions": clean_p,
            "targets": targets,
            "words": words,
            "event_ids": eids,
            "recording_ids": ["run-01"] * n_events,
            "subject_indices": torch.zeros(n_events, dtype=torch.long),
            "revision_predictions": {0: clean_p, 1: clean_p, 2: clean_p},
            "revision_availability_masks": {
                0: torch.ones(n_events, dtype=torch.bool),
                1: torch.ones(n_events, dtype=torch.bool),
                2: torch.ones(n_events, dtype=torch.bool),
            },
        }
        shifted_by_s = []
        for sec in shifts:
            sp = F.normalize(drift.unsqueeze(0) + 0.4 * torch.randn(n_events, dim, generator=g), dim=-1)
            s_enc = {
                **clean_enc,
                "predictions": sp,
                "revision_predictions": {0: sp, 1: sp, 2: sp},
            }
            shifted_by_s.append((sec, f"shift_{int(abs(sec))}s", f"Shifted ({int(sec)}s)", s_enc))

        seed_dir = source_stage_dir / f"seed-{seed:03d}"
        saved_pt = save_subspace_encoded_cache(
            seed_dir,
            split="test",
            checkpoint_sha256=f"sha_seed_{seed}",
            dataset="pallier2025",
            stage="prefix_revision",
            seed=seed,
            exp_tag="unit_exp",
            word_exp_tag="unit_word",
            eval_ds=_MockDs(),
            shifts=shifts,
            vocabulary=vocab,
            vocab50=vocab,
            clean_encoded=clean_enc,
            shifted_encoded_by_shift=shifted_by_s,
            mu0_payload={"source": "train", "predictions": drift},
        )
        assert saved_pt is not None and saved_pt.is_file()

    # 1) 零覆盖防呆断言：拒绝将 output_dir 指向只读 source_dir
    with pytest.raises(ValueError, match="零覆盖"):
        evaluate_cached_pts_pos_causal_and_m_eval(
            source_dir=source_stage_dir,
            output_dir=source_stage_dir,
            split="test",
            plot=False,
            print_report=False,
        )

    # 2) 输出至独立隔离目录，验证 CSV / Markdown / JSON / PNG / PDF 全套产物与单屏 86 列契约
    isolated_out = tmp_path / "pos_causal_audit_from_pt"
    before_src_files = sorted(p.relative_to(source_stage_dir) for p in source_stage_dir.rglob("*"))

    summary = evaluate_cached_pts_pos_causal_and_m_eval(
        source_dir=source_stage_dir,
        output_dir=isolated_out,
        split="test",
        pos_etas=(0.5, 0.8, 1.0),
        global_causal_etas=(0.5,),
        preroll_etas=(0.8, 1.0),
        m_eval_grid=(1, 2, 4, 6),
        plot=True,
        print_report=False,
    )

    after_src_files = sorted(p.relative_to(source_stage_dir) for p in source_stage_dir.rglob("*"))
    assert before_src_files == after_src_files

    method_map = {m["method_id"]: m for m in summary["methods"]}
    for expected_id, expected_fwds in [
        ("raw", 1),
        ("causal_cms:eta=0.5", 1),
        ("pos_causal_cms:eta=0.5", 1),
        ("pos_causal_cms:eta=0.8", 1),
        ("pos_causal_cms:eta=1", 1),
        ("preroll_loo:M=1:eta=1", 2),
        ("preroll_loo:M=2:eta=1", 3),
        ("preroll_loo:M=4:eta=1", 5),
        ("preroll_loo:M=6:eta=1", 7),
    ]:
        assert expected_id in method_map, f"Missing {expected_id} in {list(method_map)}"
        assert method_map[expected_id]["forwards_per_word"] == expected_fwds
        assert method_map[expected_id]["num_seeds"] == 2

    assert Path(summary["exported_files"]["csv"]).is_file()
    assert Path(summary["exported_files"]["markdown"]).is_file()
    assert Path(summary["exported_files"]["json"]).is_file()
    assert Path(summary["figures"]["preroll_m_eval_tradeoff_png"]).is_file()
    assert Path(summary["figures"]["preroll_m_eval_tradeoff_pdf"]).is_file()

    # 验证单种子 pos_causal_ema 终端看板严格 <= 86 列
    seed0_pos_cms = summary["per_seed_pos_causal_cms"][0]
    compact_pos_text = format_common_mode_subtraction_report(seed0_pos_cms, verbose=False)
    assert "Position-Locked Causal EMA" in compact_pos_text
    for line in compact_pos_text.splitlines():
        assert len(line) <= 86, f"Pos-EMA report line exceeded 86 cols ({len(line)}): {line!r}"

