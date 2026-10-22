"""S3 序列解码协议核心解码模块（Module 1–4）单元测试。

覆盖：
1. Module 1 (`neural_score_exporter`)：
   - 历史因果错位窗口共模扣除 `apply_historical_common_mode_subtraction`（Clean 与 Shifted -75s 对称扣除）
   - `.pt` 神经词表得分矩阵存取校验 `load_neural_vocabulary_scores`
2. Module 2 (`language_model_scoring`)：
   - 法汉表面渲染 `render_words` 与上下文前缀状态
   - 闭集词表静态预分词 `PreTokenizedVocabulary`（单 Token 与多 Token 候选区分）
   - KV-Cache 增量批量打分器 `CachedVocabularyLanguageScorer.score_candidates_with_cache`
     （单 Token 候选 0 次模型 Forward，多 Token 候选增量 Batch Forward）
3. Module 3 (`rolling_beam_search`)：
   - 闭集 Top-K (K=20) 候选检索 `compute_neural_log_probabilities` / `select_top_k_neural_candidates`
   - 方案一（单遍定稿主干 K2 + 尾部轻量草稿 K1/K0）流式束搜索 `decode_segment_streaming`
   - 单状态静态束搜索与双锚点基线（`S0` 与 `LM-Only`）`decode_segment_static` / `decode_dataset_segments`
4. Module 4 (`state_hyperparameter_calibrator`)：
   - 验证集状态自适应超参 `(T_r, λ_r)` 独立网格搜索与 `λ_{K0} >= λ_{K1} >= λ_{K2}` 单调递减约束
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from braindecoding.decoding.language_model_scoring import (
    CachedVocabularyLanguageScorer,
    prefix_context_kind,
    render_words,
    word_surface_for_context,
)
from braindecoding.decoding.neural_score_exporter import (
    CANONICAL_STATES,
    apply_historical_common_mode_subtraction,
    load_neural_vocabulary_scores,
)
from braindecoding.decoding.rolling_beam_search import (
    compute_neural_log_probabilities,
    decode_dataset_segments,
    decode_segment_static,
    decode_segment_streaming,
)
from braindecoding.decoding.state_hyperparameter_calibrator import (
    calibrate_state_hyperparameters,
)


class _ToyTokenizer:
    """确定性轻量分词器：支持将指定词映射为单 Token 或多 Token 序列。"""

    def __init__(self, surface_to_tokens: dict[str, tuple[int, ...]]) -> None:
        self.surface_to_tokens = dict(surface_to_tokens)
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.surface_to_tokens:
            return list(self.surface_to_tokens[text])
        return [2 + (ord(ch) % 28) for ch in text[:8]] or [2]


@dataclass
class _ToyCausalOutput:
    logits: torch.Tensor
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]


class _ToyCausalLanguageModel(nn.Module):
    """带 past_key_values 状态追踪与 Forward 计数器的确定性因果语言模型。"""

    def __init__(self, vocab_size: int = 64, hidden_size: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.forward_calls: list[tuple[int, int, int]] = []
        gen = torch.Generator().manual_seed(20261022)
        self.transition = torch.randn(vocab_size, vocab_size, generator=gen)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = True,
        **kwargs,
    ) -> _ToyCausalOutput:
        batch_size, seq_len = input_ids.shape
        past_len = 0 if past_key_values is None else int(past_key_values[0][0].shape[2])
        self.forward_calls.append((int(batch_size), int(seq_len), past_len))

        ids_clamped = input_ids.clamp(0, self.vocab_size - 1)
        logits = self.transition[ids_clamped].float()
        pos_offsets = torch.arange(
            past_len, past_len + seq_len, device=input_ids.device, dtype=torch.float32
        ).view(1, seq_len, 1)
        logits = logits + 0.01 * pos_offsets

        new_k = input_ids.unsqueeze(1).unsqueeze(-1).float().repeat(1, 2, 1, 4)
        new_v = new_k + 0.5
        if past_key_values is not None:
            prev_k, prev_v = past_key_values[0]
            k_cat = torch.cat([prev_k, new_k], dim=2)
            v_cat = torch.cat([prev_v, new_v], dim=2)
        else:
            k_cat, v_cat = new_k, new_v

        return _ToyCausalOutput(logits=logits, past_key_values=((k_cat, v_cat),))


def _build_toy_french_scorer() -> tuple[
    list[str], CachedVocabularyLanguageScorer, _ToyCausalLanguageModel
]:
    vocabulary = ["le", "petit", "prince", "l", "ami", "-t-il"]
    surface_map = {
        # initial
        "le": (10,),
        "petit": (11,),
        "prince": (12, 13),  # 多 Token 词
        "l": (14,),
        "ami": (15,),
        "-t-il": (16, 17, 18),  # 3 Token 词
        # continuation (with leading space)
        " le": (20,),
        " petit": (21,),
        " prince": (22, 23),  # 多 Token 词
        " l": (24,),
        " ami": (25,),
        # after_elision (with leading apostrophe)
        "'le": (30,),
        "'petit": (31,),
        "'prince": (32, 33),
        "'l": (34,),
        "'ami": (35,),
    }
    tokenizer = _ToyTokenizer(surface_map)
    model = _ToyCausalLanguageModel(vocab_size=64)
    scorer = CachedVocabularyLanguageScorer(
        model=model,
        tokenizer=tokenizer,
        language="fr",
        vocabulary=vocabulary,
    )
    return vocabulary, scorer, model


def test_historical_common_mode_subtraction_and_exporter_io(tmp_path: Path) -> None:
    """验证 Clean (0s) 与 Shifted (-75s) 对称扣除 6 个历史因果窗口均值并保持单位范数。"""
    rng = np.random.default_rng(42)
    n_events, dim, vocab_size = 6, 16, 8
    common_drift = torch.tensor(rng.standard_normal((n_events, dim)), dtype=torch.float32)
    true_signal = torch.tensor(rng.standard_normal((n_events, dim)), dtype=torch.float32)

    clean_raw = true_signal + 0.8 * common_drift
    shifted_75_raw = torch.roll(true_signal, shifts=2, dims=0) + 0.8 * common_drift
    hist_shifts = [
        common_drift + 0.05 * torch.tensor(rng.standard_normal((n_events, dim)), dtype=torch.float32)
        for _ in range(6)
    ]

    clean_cms = apply_historical_common_mode_subtraction(
        clean_raw, hist_shifts, eta=0.8
    )
    shifted_cms = apply_historical_common_mode_subtraction(
        shifted_75_raw, hist_shifts, eta=0.8
    )

    assert clean_cms.shape == (n_events, dim)
    assert shifted_cms.shape == (n_events, dim)
    assert torch.allclose(
        torch.norm(clean_cms, dim=-1), torch.ones(n_events), atol=1e-5
    )
    assert torch.allclose(
        torch.norm(shifted_cms, dim=-1), torch.ones(n_events), atol=1e-5
    )

    vocab = [f"w{i}" for i in range(vocab_size)]
    payload = {
        "schema_version": "s3_neural_vocabulary_scores_v1",
        "dataset": "pallier2025",
        "language": "fr",
        "split": "val",
        "seed": 0,
        "vocabulary": vocab,
        "vocabulary_embeddings": torch.randn(vocab_size, dim),
        "segments": [
            {
                "segment_index": 0,
                "group_id": "g0",
                "subject_id": "sub-01",
                "recording_id": "run-07",
                "scope": "run-07",
                "block_id": "run-07|block-000",
                "partition": "calibration",
                "event_indices": [0, 1, 2],
                "event_ids": ["e0", "e1", "e2"],
                "reference_words": ["w0", "w1", "w2"],
                "in_vocabulary_mask": [True, True, True],
                "length": 3,
            }
        ],
        "clean_scores": {
            s: torch.randn(n_events, vocab_size, dtype=torch.float32)
            for s in CANONICAL_STATES
        },
        "shifted_scores": {
            s: torch.randn(n_events, vocab_size, dtype=torch.float32)
            for s in CANONICAL_STATES
        },
    }
    pt_path = tmp_path / "val_scores.pt"
    torch.save(payload, pt_path)
    loaded = load_neural_vocabulary_scores(pt_path)
    assert loaded["dataset"] == "pallier2025"
    assert set(loaded["clean_scores"].keys()) == set(CANONICAL_STATES)
    assert set(loaded["shifted_scores"].keys()) == set(CANONICAL_STATES)


def test_kv_cache_incremental_scoring_zero_forward_for_single_token() -> None:
    """验证单 Token 候选词直接索引 last_log_probs（0 次模型 Forward），多 Token 候选仅对后缀增量 Forward。"""
    vocabulary, scorer, model = _build_toy_french_scorer()

    assert render_words(["l", "ami", "-t-il"], "fr") == "l'ami-t-il"
    assert render_words(["小", "王子"], "zh") == "小王子"
    assert prefix_context_kind(["l"], "fr") == "after_elision"
    assert word_surface_for_context("ami", "after_elision") == "'ami"

    prompt_state = scorer.init_prompt_state()
    calls_before = len(model.forward_calls)

    # 仅评估单 Token 候选词：["le", "petit", "l", "ami"] (indices 0, 1, 3, 4)
    single_token_scores = scorer.score_candidates_with_cache(
        [prompt_state], [[0, 1, 3, 4]]
    )
    assert len(single_token_scores) == 1
    assert single_token_scores[0].shape == (4,)
    assert len(model.forward_calls) == calls_before

    # 评估包含多 Token 候选词 ["prince" (2 tokens), "-t-il" (3 tokens)] (indices 2, 5)
    mixed_scores = scorer.score_candidates_with_cache(
        [prompt_state], [[0, 2, 5]]
    )
    assert mixed_scores[0].shape == (3,)
    assert np.isclose(mixed_scores[0][0], single_token_scores[0][0], atol=1e-6)
    new_calls = model.forward_calls[calls_before:]
    assert len(new_calls) >= 1
    for _, seq_len, past_len in new_calls:
        assert seq_len <= 2
        assert past_len == prompt_state.token_length


def test_scheme_one_streaming_committed_backbone_and_draft_tail() -> None:
    """验证方案一：K2 定稿主干单向推进永不撤回，K1/K0 草稿仅作屏幕显示与裕度记录。"""
    vocabulary, scorer, _ = _build_toy_french_scorer()
    seg_len = 5
    vocab_size = len(vocabulary)

    k0_raw = np.full((seg_len, vocab_size), -5.0, dtype=np.float32)
    k1_raw = np.full((seg_len, vocab_size), -5.0, dtype=np.float32)
    k2_raw = np.full((seg_len, vocab_size), -5.0, dtype=np.float32)

    k0_raw[0, 0] = 5.0   # pos 0, K0 -> "le"
    k1_raw[0, 1] = 6.0   # pos 0, K1 -> "petit"
    k2_raw[0, 2] = 8.0   # pos 0, K2 -> "prince"

    k0_raw[1, 3] = 5.0   # pos 1, K0 -> "l"
    k1_raw[1, 3] = 5.0   # pos 1, K1 -> "l"
    k2_raw[1, 4] = 8.0   # pos 1, K2 -> "ami"

    for pos in range(2, seg_len):
        k0_raw[pos, pos % vocab_size] = 6.0
        k1_raw[pos, (pos + 1) % vocab_size] = 6.5
        k2_raw[pos, (pos + 2) % vocab_size] = 7.5

    log_probs = {
        "K0": compute_neural_log_probabilities(k0_raw, temperature=0.1),
        "K1": compute_neural_log_probabilities(k1_raw, temperature=0.1),
        "K2": compute_neural_log_probabilities(k2_raw, temperature=0.1),
    }

    result = decode_segment_streaming(
        log_probs,
        vocabulary,
        scorer,
        language_weights={"K0": 0.2, "K1": 0.1, "K2": 0.05},
        beam_width=5,
        candidate_top_k=20,
        reference_words=["prince", "ami", "ami", "le", "petit"],
    )

    history = result["screen_history"]
    trajectories = result["position_trajectories"]
    assert len(history) == seg_len
    assert len(trajectories) == seg_len

    # Step 0 (t=0)：只有草稿 K0 ("le")，定稿主干为空
    assert history[0]["committed_prefix"] == []
    assert history[0]["screen_words"] == ["le"]

    # Step 1 (t=1)：词位 0 进入 K1 草稿 ("petit")，词位 1 进入 K0 草稿 ("l")，定稿主干仍为空
    assert history[1]["committed_prefix"] == []
    assert history[1]["screen_words"] == ["petit", "l"]

    # Step 2 (t=2)：词位 0 达到 K2 定稿 ("prince")，后续 t=3, 4 中词位 0 永久锁定为 "prince"
    assert history[2]["committed_prefix"] == ["prince"]
    assert history[3]["committed_prefix"][0] == "prince"
    assert history[4]["committed_prefix"][:2] == ["prince", "ami"]

    # 校验轨迹中记录了精确的正向改写裕度 Δs_i^{(r)} > 0
    assert trajectories[0]["k0_word"] == "le"
    assert trajectories[0]["k1_word"] == "petit"
    assert trajectories[0]["k2_word"] == "prince"
    assert trajectories[0]["k1_delta_score"] > 0.0
    assert trajectories[0]["k2_delta_score"] > 0.0


def test_static_baselines_and_state_hyperparameter_calibration() -> None:
    """验证 S0 / LM-Only 静态束搜索及验证集状态自适应超参校准（含 λ_{K0} >= λ_{K1} >= λ_{K2} 约束）。"""
    vocabulary, scorer, _ = _build_toy_french_scorer()
    vocab_size = len(vocabulary)

    # 1. 验证 LM-Only 双锚点基线（无神经输入，均匀分布先验）
    lm_only_res = decode_segment_static(
        None,
        vocabulary,
        scorer,
        language_weight=1.0,
        beam_width=3,
        candidate_top_k=6,
        segment_length=4,
        pure_language_model=True,
    )
    assert len(lm_only_res["final_words"]) == 4
    assert all(w in vocabulary for w in lm_only_res["final_words"])

    # 2. 验证状态自适应校准器在验证集上网格搜索并施加 λ_{K0} >= λ_{K1} >= λ_{K2}
    n_events = 6
    clean_scores = {}
    for s in CANONICAL_STATES:
        mat = torch.full((n_events, vocab_size), -1.0, dtype=torch.float32)
        for idx in range(n_events):
            mat[idx, idx % vocab_size] = 1.0
        clean_scores[s] = mat

    exported_val = {
        "dataset": "pallier2025",
        "language": "fr",
        "split": "val",
        "seed": 0,
        "vocabulary": vocabulary,
        "segments": [
            {
                "segment_index": 0,
                "group_id": "g0",
                "subject_id": "sub-01",
                "recording_id": "run-07",
                "scope": "run-07",
                "block_id": "run-07|block-000",
                "partition": "calibration",
                "event_indices": [0, 1, 2],
                "reference_words": [vocabulary[0], vocabulary[1], vocabulary[2]],
                "in_vocabulary_mask": [True, True, True],
            },
            {
                "segment_index": 1,
                "group_id": "g1",
                "subject_id": "sub-01",
                "recording_id": "run-07",
                "scope": "run-07",
                "block_id": "run-07|block-000",
                "partition": "calibration",
                "event_indices": [3, 4, 5],
                "reference_words": [vocabulary[3], vocabulary[4], vocabulary[5]],
                "in_vocabulary_mask": [True, True, True],
            },
        ],
        "clean_scores": clean_scores,
        "shifted_scores": clean_scores,
    }

    cal_report = calibrate_state_hyperparameters(
        exported_val,
        scorer,
        temperature_grid=(0.07, 0.10),
        language_weight_grid=(0.0, 0.10, 0.25),
        beam_width=3,
        candidate_top_k=6,
        enforce_monotonic_lambda=True,
    )

    states_map = cal_report["states"]
    assert set(states_map.keys()) == set(CANONICAL_STATES)
    assert (
        states_map["K0"]["language_weight"]
        >= states_map["K1"]["language_weight"]
        >= states_map["K2"]["language_weight"]
    )
    assert cal_report["monotonicity_satisfied"] is True

    # 3. 验证批量句段解码入口 `decode_dataset_segments`
    s3_clean_bundle = decode_dataset_segments(
        exported_val,
        scorer,
        cal_report,
        condition="S3",
        alignment="clean",
        beam_width=3,
        candidate_top_k=6,
        partition_filter="calibration",
    )
    assert s3_clean_bundle["segment_count"] == 2
    assert len(s3_clean_bundle["final_sentences"]) == 2
    assert len(s3_clean_bundle["screen_histories"]) == 2

    lm_only_bundle = decode_dataset_segments(
        exported_val,
        scorer,
        cal_report,
        condition="LM-Only",
        alignment="clean",
        beam_width=3,
        candidate_top_k=6,
        partition_filter="calibration",
    )
    assert lm_only_bundle["segment_count"] == 2
    assert len(lm_only_bundle["final_sentences"]) == 2


def test_decode_sentences_cli_pipeline_steps(tmp_path, monkeypatch) -> None:
    """验证 scripts/decode_sentences.py 的 calibrate -> decode -> evaluate 流水线闭环。"""
    import scripts.decode_sentences as decode_cli

    vocabulary, scorer, _ = _build_toy_french_scorer()
    vocab_size = len(vocabulary)
    n_events = 6
    clean_scores = {}
    for s in CANONICAL_STATES:
        mat = torch.full((n_events, vocab_size), -1.0, dtype=torch.float32)
        for idx in range(n_events):
            mat[idx, idx % vocab_size] = 1.0
        clean_scores[s] = mat

    segments = [
        {
            "segment_index": 0,
            "group_id": "g0",
            "subject_id": "sub-01",
            "recording_id": "run-06",
            "scope": "run-06",
            "block_id": "run-06|block-000",
            "partition": "calibration",
            "event_indices": [0, 1, 2],
            "reference_words": [vocabulary[0], vocabulary[1], vocabulary[2]],
            "in_vocabulary_mask": [True, True, True],
        },
        {
            "segment_index": 1,
            "group_id": "g1",
            "subject_id": "sub-01",
            "recording_id": "run-06",
            "scope": "run-06",
            "block_id": "run-06|block-000",
            "partition": "evaluation",
            "event_indices": [3, 4, 5],
            "reference_words": [vocabulary[3], vocabulary[4], vocabulary[5]],
            "in_vocabulary_mask": [True, True, True],
        },
    ]

    val_pt = tmp_path / "neural_vocabulary_scores_val.pt"
    test_pt = tmp_path / "neural_vocabulary_scores_test.pt"
    torch.save(
        {
            "dataset": "pallier2025",
            "language": "fr",
            "split": "val",
            "seed": 0,
            "vocabulary": vocabulary,
            "vocabulary_size": vocab_size,
            "vocabulary_embeddings": torch.eye(vocab_size, 16),
            "clean_scores": clean_scores,
            "shifted_scores": clean_scores,
            "segments": segments,
        },
        val_pt,
    )
    torch.save(
        {
            "dataset": "pallier2025",
            "language": "fr",
            "split": "test",
            "seed": 0,
            "vocabulary": vocabulary,
            "vocabulary_size": vocab_size,
            "vocabulary_embeddings": torch.eye(vocab_size, 16),
            "clean_scores": clean_scores,
            "shifted_scores": clean_scores,
            "segments": segments,
        },
        test_pt,
    )

    monkeypatch.setattr(
        decode_cli,
        "_load_or_init_language_scorer",
        lambda *args, **kwargs: scorer,
    )

    for step_name in ("calibrate", "decode", "evaluate"):
        args = decode_cli.build_parser().parse_args(
            [
                "--step",
                step_name,
                "--dataset",
                "pallier2025",
                "--split",
                "test",
                "--seed",
                "0",
                "--top-k",
                "4",
                "--beam-width",
                "2",
                "--output-dir",
                str(tmp_path),
                "--no-plot",
            ]
        )
        summary = decode_cli.run_sequence_decoding_pipeline(args)
        assert summary["step"] == step_name
        if step_name == "evaluate":
            assert "S3_clean" in summary["conditions"]
            assert "S3_clean_c_gated" in summary["conditions"]
            assert "S3_shifted_neg75s" in summary["conditions"]

