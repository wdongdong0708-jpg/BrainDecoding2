"""隐式词长捷径 Clean 体检套件单元测试（图 2a/2b、Clean JS 与图 9 零重叠拼接）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from braindecoding.evaluation.duration_shortcut_diagnostics import (
    audit_spliced_zero_overlap,
    build_spliced_real_onset_dataset,
    build_train_duration_priors,
    compute_duration_oracle_n50,
    compute_duration_variability_correlation,
    compute_same_word_duration_substitution,
    extract_event_durations_ms,
    extract_prev_inter_onset_interval_ms,
    score_embeddings_against_bank,
)


class _DummyWordWindowDataset:
    """轻量内存桩 Dataset，复现 WordWindowDataset 的关键只读契约。"""

    def __init__(self, table: pd.DataFrame, dim: int = 8) -> None:
        self.table = table.reset_index(drop=True).copy()
        self.table["subject_id"] = self.table["受试者"].astype(str)
        self.table["recording_id"] = self.table["记录编号"].astype(str)
        self.table["normalized_word"] = self.table["标准词"].astype(str)
        self.table["onset_seconds"] = pd.to_numeric(
            self.table["开始时间"], errors="coerce"
        ).astype(np.float64)
        self.table["event_id"] = self.table["事件编号"].astype(str)

        ctx_vals = (
            self.table["recording_id"] + "|" + self.table["上下文编号"].astype(str)
        ).tolist()
        ctx_map = {v: i for i, v in enumerate(dict.fromkeys(ctx_vals))}
        self.sentence_indices = np.asarray([ctx_map[v] for v in ctx_vals], dtype=np.int64)
        self.subject_count = int(self.table["subject_id"].nunique())
        self.channel_count = 4
        self.channel_positions = np.zeros((4, 2), dtype=np.float32)

        rng = np.random.default_rng(123)
        self.embedding_map = {
            w: rng.standard_normal(dim).astype(np.float32)
            for w in sorted(set(self.table["normalized_word"]))
        }
        self._dim = dim

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict:
        row = self.table.iloc[idx]
        # 把唯一行号编码进 meg[:, 0]，方便断言拼接后完全保留真实供体窗口
        meg = torch.full((self.channel_count, 10), float(idx), dtype=torch.float32)
        return {
            "meg": meg,
            "text_embedding": torch.from_numpy(self.embedding_map[row["normalized_word"]]),
            "subject_index": torch.tensor(0, dtype=torch.long),
            "sentence_index": torch.tensor(int(self.sentence_indices[idx]), dtype=torch.long),
            "word": str(row["normalized_word"]),
            "event_id": str(row["event_id"]),
            "recording_id": str(row["recording_id"]),
            "onset_seconds": torch.tensor(float(row["onset_seconds"]), dtype=torch.float32),
        }


def test_same_word_duration_substitution_separates_oracle_neural_from_shortcut() -> None:
    rng = np.random.default_rng(42)
    words = []
    actual_ms = []
    pure_neural_pred_ms = []
    shortcut_pred_ms = []

    word_means = {f"w{i}": 150.0 + 40.0 * i for i in range(10)}
    for w, mu in word_means.items():
        # 每个词 80 次出现，带有真实发音快慢波动（sd=45ms）
        durs = rng.normal(mu, 45.0, size=80).clip(60.0, 900.0)
        words.extend([w] * 80)
        actual_ms.extend(durs.tolist())
        # 纯神经解码器：即使 100% 预测准词型，也不跟踪单次发音快慢（仅加独立小噪声）
        pure_neural_pred_ms.extend((mu + rng.normal(0.0, 5.0, size=80)).tolist())
        # 词长捷径模型：预测词长直接跟踪当次真实发音时长
        shortcut_pred_ms.extend((0.85 * durs + 0.15 * mu + rng.normal(0.0, 6.0, size=80)).tolist())

    res_neural = compute_same_word_duration_substitution(
        words, np.asarray(actual_ms), np.asarray(pure_neural_pred_ms)
    )
    res_shortcut = compute_same_word_duration_substitution(
        words, np.asarray(actual_ms), np.asarray(shortcut_pred_ms)
    )

    assert res_neural["eligible_queries"] == 800
    assert res_neural["eligible_words"] == 10
    assert abs(res_neural["error_reduction_micro_ms"]) < 1.0
    assert abs(res_neural["error_reduction_macro_ms"]) < 1.0
    assert abs(res_neural["within_word_corr_r"]) < 0.12
    assert abs(res_neural["fast_vs_slow_pred_shift_ms"]) < 3.0

    assert res_shortcut["error_reduction_micro_ms"] > 20.0
    assert res_shortcut["error_reduction_macro_ms"] > 20.0
    assert res_shortcut["within_word_corr_r"] > 0.90
    assert res_shortcut["fast_vs_slow_pred_shift_ms"] > 70.0
    assert res_shortcut["fast_vs_slow_js_bits"] > res_neural["fast_vs_slow_js_bits"]


def test_duration_oracle_variability_corr_and_chunked_scoring() -> None:
    rng = np.random.default_rng(7)
    vocab = [f"w{i:02d}" for i in range(12)]
    rows = []
    t = 0.0
    for rep in range(30):
        for idx, w in enumerate(vocab):
            dur_s = float(np.clip(rng.normal(0.12 + 0.04 * idx, 0.01 + 0.008 * idx), 0.05, 1.5))
            rows.append(
                {
                    "受试者": "sub-0",
                    "记录编号": "rec-1",
                    "上下文编号": f"sent-{rep}",
                    "标准词": w,
                    "开始时间": t,
                    "结束时间": t + dur_s,
                }
            )
            t += dur_s + 0.02
    df = pd.DataFrame(rows)
    priors = build_train_duration_priors(df, vocab)
    dur_ms, ioi_ms = extract_event_durations_ms(df)
    prev_ioi_ms = extract_prev_inter_onset_interval_ms(df)
    assert np.isfinite(dur_ms).all()
    assert np.isfinite(ioi_ms[:-1]).sum() > 0
    assert np.isnan(prev_ioi_ms[0])
    assert np.isfinite(prev_ioi_ms[1:]).sum() == np.isfinite(ioi_ms[:-1]).sum()
    assert "prev_inter_onset_interval" in priors
    # 同句内第 k 个词的 prev_ioi 严格等于第 k-1 个词的 ioi
    for k in range(1, len(df)):
        if df.iloc[k]["上下文编号"] == df.iloc[k - 1]["上下文编号"]:
            assert prev_ioi_ms[k] == pytest.approx(ioi_ms[k - 1], abs=1e-6)
        else:
            assert np.isnan(prev_ioi_ms[k])

    oracle = compute_duration_oracle_n50(
        df["标准词"].tolist(), dur_ms, vocab, priors["acoustic_duration"]
    )
    assert oracle["macro_top10"] > 0.80

    # 测试分块打分：当查询等于候选向量本身时，Top-1 排名应严格为 1.0
    bank = rng.standard_normal((len(vocab), 16)).astype(np.float32)
    q_idx = np.arange(len(vocab))
    scored = score_embeddings_against_bank(
        bank[q_idx], vocab, bank, vocab, device=torch.device("cpu"), top_k=5, chunk_size=4
    )
    assert np.allclose(scored["ranks"], 1.0)
    assert np.array_equal(scored["top_indices"][:, 0], q_idx)

    # 词长标准差相关性（窄方差词给高分，宽方差词给低分 => 强负相关）
    synthetic_ranks = np.asarray(
        [1.0 if int(w[1:]) < 6 else 12.0 for w in df["标准词"]], dtype=np.float64
    )
    fig2a = compute_duration_variability_correlation(
        synthetic_ranks,
        df["标准词"].tolist(),
        dur_ms,
        vocab,
        priors["acoustic_duration"],
        oracle_per_word_top10=oracle["per_word_top10"],
    )
    assert fig2a["corr_acc_vs_train_dur_sd_pearson_r"] < -0.7
    assert len(fig2a["dur_sd_quantile_bins"]) == 5


def test_spliced_real_onset_datasets_enforce_strict_zero_overlap() -> None:
    rows = []
    vocab = ["the", "of", "and", "to", "in", "sherlock", "holmes", "said"]
    event_counter = 0
    # 构造 4 个测试录音，每个录音 5 句 × 6 词 = 120 条真实词事件
    for rec_i in range(4):
        t = 10.0
        for sent_i in range(5):
            for pos in range(6):
                w = vocab[(rec_i + sent_i + pos) % len(vocab)]
                rows.append(
                    {
                        "事件编号": f"ev-{event_counter:04d}",
                        "受试者": "sub-0",
                        "记录编号": f"rec-{rec_i}",
                        "上下文编号": f"sent-{sent_i}",
                        "标准词": w,
                        "开始时间": t,
                        "结束时间": t + 0.25,
                    }
                )
                event_counter += 1
                t += 0.30
            t += 2.0
    base_ds = _DummyWordWindowDataset(pd.DataFrame(rows))

    # 1. shuffled 模式：严格双射、零违规、真实供体窗口与词标签严格配对
    shuf_ds, shuf_audit = build_spliced_real_onset_dataset(
        base_ds, mode="shuffled", seed=0, min_gap_seconds=10.0, neighbor_radius=2
    )
    assert len(shuf_ds) == len(base_ds)
    assert sorted(shuf_ds.donor_indices.tolist()) == list(range(len(base_ds)))
    assert shuf_audit["verified_violations_within_mask"] == 0
    assert shuf_audit["zero_overlap_token_rate"] > 0.95
    sample0 = shuf_ds[0]
    donor0 = int(shuf_ds.donor_indices[0])
    assert float(sample0["meg"][0, 0].item()) == pytest.approx(float(donor0))
    assert sample0["word"] == base_ds.table.iloc[donor0]["normalized_word"]

    # 2. same_word 模式：完整保留原句词序，同时达到极高零重叠率且掩码内零违规
    sw_ds, sw_audit = build_spliced_real_onset_dataset(
        base_ds, mode="same_word", seed=0, min_gap_seconds=10.0, neighbor_radius=2
    )
    assert len(sw_ds) == len(base_ds)
    assert sw_ds.table["normalized_word"].tolist() == base_ds.table["normalized_word"].tolist()
    assert sw_audit["verified_violations_within_mask"] == 0
    assert sw_audit["zero_overlap_token_rate"] > 0.90

    # 反向检验：对原始连续重叠表跑审计，应检出几乎全部相邻对重叠
    orig_donors = np.arange(len(base_ds), dtype=np.int64)
    _, orig_audit = audit_spliced_zero_overlap(
        base_ds.table,
        orig_donors,
        base_ds.sentence_indices,
        min_gap_seconds=10.0,
        neighbor_radius=2,
    )
    assert orig_audit["overlapping_adjacent_pairs"] == orig_audit["total_adjacent_pairs"]
    assert orig_audit["zero_overlap_tokens"] == 0
