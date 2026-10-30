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
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            pos_offsets = position_ids.to(
                device=input_ids.device, dtype=torch.float32
            ).unsqueeze(-1)
        else:
            pos_offsets = torch.arange(
                past_len,
                past_len + seq_len,
                device=input_ids.device,
                dtype=torch.float32,
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
        >= 0.05
    )
    assert cal_report["monotonicity_satisfied"] is True

    # 2B. 验证 shared_language_weight=True 模式：K0/K1/K2 共享完全相同的 (T, λ >= 0.05)
    cal_shared = calibrate_state_hyperparameters(
        exported_val,
        scorer,
        temperature_grid=(0.07,),
        language_weight_grid=(0.0, 0.05, 0.15),
        beam_width=3,
        candidate_top_k=6,
        shared_language_weight=True,
    )
    assert (
        cal_shared["states"]["K0"]["language_weight"]
        == cal_shared["states"]["K1"]["language_weight"]
        == cal_shared["states"]["K2"]["language_weight"]
        >= 0.05
    )

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
            "checkpoint_provenance": {
                "revision_stage": "revision",
                "context": "mock_context",
                "revision": "mock_revision",
            },
            "cms_metadata": {
                "subtraction_eta": 0.0,
            },
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
            "checkpoint_provenance": {
                "revision_stage": "revision",
                "context": "mock_context",
                "revision": "mock_revision",
            },
            "cms_metadata": {
                "subtraction_eta": 0.0,
            },
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


def test_build_segment_inventory_supports_chinese_canonical_columns():
    """验证 build_segment_inventory 对磁盘落盘纯中文核心事件表列名的兼容与自动合成 runtime_context_uid。"""
    import pandas as pd
    from braindecoding.decoding.neural_score_exporter import build_segment_inventory

    table = pd.DataFrame(
        {
            "事件编号": ["ev-1", "ev-2", "ev-3", "ev-4"],
            "受试者": ["sub-01", "sub-01", "sub-01", "sub-01"],
            "记录编号": ["sub-01_run-06", "sub-01_run-06", "sub-01_run-06", "sub-01_run-06"],
            "运行编号": ["run-06", "run-06", "run-06", "run-06"],
            "BIDS事件行号": [10, 11, 20, 21],
            "标准词": ["le", "petit", "prince", "parle"],
            "上下文编号": ["sent-0", "sent-0", "sent-1", "sent-1"],
        }
    )
    segments = build_segment_inventory(
        "pallier2025",
        table,
        vocabulary=["le", "petit", "prince"],
        adjacent_components_per_block=1,
    )
    assert len(segments) == 2
    assert segments[0]["event_ids"] == ["ev-1", "ev-2"]
    assert segments[0]["reference_words"] == ["le", "petit"]
    assert segments[0]["in_vocabulary_mask"] == [True, True]
    assert segments[1]["event_ids"] == ["ev-3", "ev-4"]
    assert segments[1]["reference_words"] == ["prince", "parle"]
    assert segments[1]["in_vocabulary_mask"] == [True, False]


def test_batched_segment_decoding_and_transition_cache_equivalence() -> None:
    """验证跨句段锁步批处理解码与单句段解码输出 100% 一致，且重复解码命中转移缓存零额外 Forward。"""
    from braindecoding.decoding.rolling_beam_search import (
        decode_segments_static_batched,
        decode_segments_streaming_batched,
    )

    vocabulary, scorer_single, _ = _build_toy_french_scorer()
    _, scorer_batched, model_batched = _build_toy_french_scorer()
    vocab_size = len(vocabulary)
    rng = np.random.default_rng(1234)

    seg_dicts = []
    for seg_len in (3, 5, 4):
        seg_dicts.append(
            {
                "K0": compute_neural_log_probabilities(
                    rng.standard_normal((seg_len, vocab_size)).astype(np.float32),
                    temperature=0.1,
                ),
                "K1": compute_neural_log_probabilities(
                    rng.standard_normal((seg_len, vocab_size)).astype(np.float32),
                    temperature=0.1,
                ),
                "K2": compute_neural_log_probabilities(
                    rng.standard_normal((seg_len, vocab_size)).astype(np.float32),
                    temperature=0.1,
                ),
            }
        )

    lam_weights = {"K0": 0.2, "K1": 0.15, "K2": 0.1}
    single_stream = [
        decode_segment_streaming(
            sd, vocabulary, scorer_single, language_weights=lam_weights, beam_width=3, candidate_top_k=5
        )
        for sd in seg_dicts
    ]
    batched_stream = decode_segments_streaming_batched(
        seg_dicts, vocabulary, scorer_batched, language_weights=lam_weights, beam_width=3, candidate_top_k=5
    )
    for s_res, b_res in zip(single_stream, batched_stream):
        assert s_res["final_words"] == b_res["final_words"]
        assert s_res["committed_k2_words"] == b_res["committed_k2_words"]
        assert s_res["initial_k0_words"] == b_res["initial_k0_words"]

    calls_before_repeat = len(model_batched.forward_calls)
    batched_stream_repeat = decode_segments_streaming_batched(
        seg_dicts, vocabulary, scorer_batched, language_weights=lam_weights, beam_width=3, candidate_top_k=5
    )
    assert len(model_batched.forward_calls) == calls_before_repeat
    assert scorer_batched.stats["transition_cache_hits"] > 0
    for b_res, r_res in zip(batched_stream, batched_stream_repeat):
        assert b_res["final_words"] == r_res["final_words"]

    single_static = [
        decode_segment_static(
            sd["K2"], vocabulary, scorer_single, language_weight=0.15, beam_width=3, candidate_top_k=5
        )
        for sd in seg_dicts
    ]
    batched_static = decode_segments_static_batched(
        [sd["K2"] for sd in seg_dicts],
        vocabulary,
        scorer_batched,
        language_weight=0.15,
        beam_width=3,
        candidate_top_k=5,
    )
    for s_res, b_res in zip(single_static, batched_static):
        assert s_res["final_words"] == b_res["final_words"]
        assert np.isclose(s_res["score"], b_res["score"], atol=1e-6)


def test_top_k_local_softmax_and_multi_subject_observation_sweep() -> None:
    """验证 Top-K 局部 Softmax 归一化消除无关词稀释，以及 k ∈ {1, 2, 5, 10} 跨被试多观测聚合。"""
    from braindecoding.decoding.rolling_beam_search import (
        aggregate_multi_subject_payload,
        run_multi_subject_observation_sweep,
    )

    # 1. 验证局部 Top-K Softmax 概率和在 Top-K 候选上严格等于 1.0
    scores = np.array([[0.8, 0.7, 0.6, -0.5, -0.8, -0.9]], dtype=np.float32)
    lp_local = compute_neural_log_probabilities(scores, temperature=0.07, top_k=3)
    top3_prob_sum = float(np.exp(lp_local[0, :3]).sum())
    assert np.isclose(top3_prob_sum, 1.0, atol=1e-5)

    # 2. 验证跨被试多观测聚合 (k=1, 2) 提升信噪比并输出聚合解码结果
    vocabulary, scorer, _ = _build_toy_french_scorer()
    vocab_size = len(vocabulary)
    n_subjects = 4
    seg_len = 3
    total_events = n_subjects * seg_len

    rng = np.random.default_rng(2026)
    true_mat = np.full((seg_len, vocab_size), -1.0, dtype=np.float32)
    for pos in range(seg_len):
        true_mat[pos, pos] = 1.0

    clean_scores = {}
    for s in CANONICAL_STATES:
        rows = []
        for _ in range(n_subjects):
            noisy = true_mat + 0.35 * rng.standard_normal((seg_len, vocab_size)).astype(
                np.float32
            )
            rows.append(noisy)
        clean_scores[s] = torch.from_numpy(np.concatenate(rows, axis=0))

    segments = []
    for subj_idx in range(n_subjects):
        ev_start = subj_idx * seg_len
        segments.append(
            {
                "segment_index": subj_idx,
                "group_id": f"sub-{subj_idx+1:02d}|sent-0",
                "subject_id": f"sub-{subj_idx+1:02d}",
                "recording_id": f"sub-{subj_idx+1:02d}_run-06",
                "scope": "run-06",
                "block_id": "run-06|block-000",
                "partition": "evaluation",
                "event_indices": list(range(ev_start, ev_start + seg_len)),
                "reference_words": [vocabulary[0], vocabulary[1], vocabulary[2]],
                "in_vocabulary_mask": [True, True, True],
            }
        )

    payload = {
        "dataset": "pallier2025",
        "language": "fr",
        "split": "test",
        "seed": 0,
        "vocabulary": vocabulary,
        "segments": segments,
        "clean_scores": clean_scores,
        "shifted_scores": clean_scores,
    }
    agg_k2 = aggregate_multi_subject_payload(payload, k_observations=2)
    assert len(agg_k2["segments"]) == 1
    assert agg_k2["segments"][0]["k_observations"] == 2

    hyperparams = {
        "states": {
            s: {"temperature": 0.07, "language_weight": 0.05}
            for s in CANONICAL_STATES
        }
    }
    sweep = run_multi_subject_observation_sweep(
        payload,
        scorer,
        hyperparams,
        k_observations=(1, 2, 4),
        vocab_sizes=(2, 3),
        beam_width=3,
        candidate_top_k=5,
        show_progress=False,
    )
    assert set(sweep.keys()) == {"k=1", "k=2", "k=4"}
    assert sweep["k=4"]["final_sentences"][0]["predicted_words"] == [
        vocabulary[0],
        vocabulary[1],
        vocabulary[2],
    ]
    assert "V=2" in sweep["k=4"]["by_vocabulary_size"]
    assert "V=3" in sweep["k=4"]["by_vocabulary_size"]


def test_vocabulary_frequency_order_and_scaling_sweep() -> None:
    """验证按训练集词频降序排列词表与零训练成本多尺度候选词表扫描（|V| 截断掩码 + 全词库零开销复用）。"""
    from braindecoding.decoding.neural_score_exporter import (
        resolve_vocabulary_frequency_order,
    )
    from braindecoding.decoding.rolling_beam_search import (
        run_vocabulary_scaling_sweep,
    )

    vocabulary, scorer, _ = _build_toy_french_scorer()
    vocab_size = len(vocabulary)
    # 故意设置非词典序的训练词频：使索引 4 ("ami") 和索引 2 ("prince") 成为 Top-2 高频词
    train_counts = [5, 10, 50, 2, 100, 1]

    seg_len = 3
    clean_scores = {}
    for s in CANONICAL_STATES:
        mat = torch.full((seg_len, vocab_size), -1.0, dtype=torch.float32)
        # 在词位 0 给低频词索引 0 ("le", count=5) 最高分，给高频词索引 4 ("ami", count=100) 次高分
        mat[0, 0] = 5.0
        mat[0, 4] = 3.0
        mat[1, 2] = 4.0
        mat[2, 1] = 4.0
        clean_scores[s] = mat

    payload = {
        "dataset": "custom_toy",
        "language": "fr",
        "split": "test",
        "seed": 0,
        "vocabulary": vocabulary,
        "vocabulary_train_counts": train_counts,
        "segments": [
            {
                "segment_index": 0,
                "group_id": "sub-01|sent-0",
                "subject_id": "sub-01",
                "recording_id": "sub-01_run-06",
                "scope": "run-06",
                "block_id": "run-06|block-000",
                "partition": "evaluation",
                "event_indices": [0, 1, 2],
                # "ami" (count=100), "prince" (count=50), "le" (count=5)
                "reference_words": ["ami", "prince", "le"],
                "in_vocabulary_mask": [True, True, True],
            }
        ],
        "clean_scores": clean_scores,
        "shifted_scores": clean_scores,
    }

    ranked_indices, ranked_words, counts = resolve_vocabulary_frequency_order(payload)
    # Top-3 词频顺序应为: "ami" (100, idx=4), "prince" (50, idx=2), "petit" (10, idx=1)
    assert ranked_indices[:3] == [4, 2, 1]
    assert ranked_words[:3] == ["ami", "prince", "petit"]
    assert counts["ami"] == 100 and counts["prince"] == 50

    # 验证 allowed_vocab_indices 掩码：当仅允许 Top-2 高频词 [4, 2] 时，索引 0 被屏蔽为 -1e30
    lp_restricted = compute_neural_log_probabilities(
        clean_scores["K2"],
        temperature=0.07,
        top_k=2,
        allowed_vocab_indices=[4, 2],
    )
    assert lp_restricted[0, 0] <= -1e20
    assert np.isclose(
        float(np.exp(lp_restricted[0, [4, 2]]).sum()), 1.0, atol=1e-5
    )

    hyperparams = {
        "states": {
            s: {"temperature": 0.07, "language_weight": 0.0}
            for s in CANONICAL_STATES
        }
    }
    completed_order: list[str] = []
    scaling_bundles = run_vocabulary_scaling_sweep(
        payload,
        scorer,
        hyperparams,
        vocab_sizes=(2, 3, 6),
        core_vocab_size=3,
        beam_width=2,
        candidate_top_k=4,
        show_progress=False,
        on_scale_completed=lambda k, _b: completed_order.append(k),
    )

    # core_vocab_size=3 优先在第一位执行，随后按升序返回 V=2, V=3, V=6
    assert completed_order[0] == "V=3"
    assert list(scaling_bundles.keys()) == ["V=2", "V=3", "V=6"]

    # 在 V=2 下，候选池仅包含 {"ami", "prince"}：
    # 词位 0 原最高分 "le" 被截断排除，预测自动落入池内最高分 "ami"；且参考词 ["ami", "prince", "le"] 的掩码变为 [True, True, False]
    v2_s3 = scaling_bundles["V=2"]["conditions"]["S3_clean"]["final_sentences"][0]
    assert set(v2_s3["predicted_words"]).issubset({"ami", "prince"})
    assert v2_s3["in_vocabulary_mask"] == [True, True, False]
    assert scaling_bundles["V=2"]["test_token_coverage_pct"] == round(200.0 / 3.0, 4)
    # 验证每个尺度均对 S0 / K0 / K1 / S3_clean / K0_shifted_neg75s / S3_shifted_neg75s 执行统一静态束搜索，且未生成 F-local
    assert {
        "S0",
        "K0",
        "K1",
        "S3_clean",
        "K0_shifted_neg75s",
        "S3_shifted_neg75s",
    }.issubset(set(scaling_bundles["V=2"]["conditions"].keys()))
    assert "F" in scaling_bundles["V=2"]["conditions"]
    assert "F_local" not in scaling_bundles["V=2"]["conditions"]


def test_decoupled_stage_checkpoints_and_incremental_f_reuse(
    tmp_path, monkeypatch
) -> None:
    """验证 S3 三阶段检查点严格解耦（Stage 1 -> S0, Stage 2.1 -> F, Stage 2.3 -> K0/K1/K2）与无 F/增量补 F 缓存复用。"""
    import json
    import scripts.decode_sentences as decode_cli
    from braindecoding.decoding.neural_score_exporter import (
        resolve_stage_checkpoints,
    )

    runs_root = tmp_path / "runs"
    word_ckpt = runs_root / "pallier2025" / "tag_word" / "word" / "all-n10" / "seed-000" / "best.pt"
    ctx_ckpt = runs_root / "pallier2025" / "tag_ctx" / "context" / "all-n10" / "seed-000" / "best.pt"
    rev_ckpt = runs_root / "pallier2025" / "tag_rev" / "prefix_revision" / "all-n10" / "seed-000" / "best.pt"
    for p in (word_ckpt, rev_ckpt):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"ckpt")

    monkeypatch.setenv("BRAINDECODING_RESULTS_DIR", str(runs_root))
    resolved_no_ctx = resolve_stage_checkpoints(
        "pallier2025",
        seed=0,
        subject="all",
        exp_tag="tag_rev",
        word_exp_tag="tag_word",
        context_exp_tag="tag_ctx",
        revision_stage="prefix_revision",
    )
    assert resolved_no_ctx["word_checkpoint"] == word_ckpt
    assert resolved_no_ctx["revision_checkpoint"] == rev_ckpt
    assert resolved_no_ctx["context_checkpoint"] is None

    ctx_ckpt.parent.mkdir(parents=True, exist_ok=True)
    ctx_ckpt.write_bytes(b"ckpt")
    resolved_with_ctx = resolve_stage_checkpoints(
        "pallier2025",
        seed=0,
        subject="all",
        exp_tag="tag_rev",
        word_exp_tag="tag_word",
        context_exp_tag="tag_ctx",
        revision_stage="prefix_revision",
    )
    assert resolved_with_ctx["context_checkpoint"] == ctx_ckpt
    ctx_ckpt.unlink()

    # 验证无 Stage 2.1 (无 F) 时 calibrate -> decode -> evaluate 正常完成且不伪造 F，
    # 随后补入真实 Stage 2.1 (F) 时增量复用已有 S0/K0/K1/K2 校准与解码结果
    vocabulary, scorer, _ = _build_toy_french_scorer()
    vocab_size = len(vocabulary)
    n_events = 6
    clean_no_f = {}
    for s in ("S0", "K0", "K1", "K2"):
        mat = torch.full((n_events, vocab_size), -1.0, dtype=torch.float32)
        for idx in range(n_events):
            mat[idx, idx % vocab_size] = 1.0
        clean_no_f[s] = mat

    segments = [
        {
            "segment_index": 0,
            "group_id": "sub-01_run-06:0",
            "material_id": "run-06:0",
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
            "group_id": "sub-01_run-06:1",
            "material_id": "run-06:1",
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

    out_dir = tmp_path / "s3_inc_out"
    seed_dir = out_dir / "seed-000"
    seed_dir.mkdir(parents=True, exist_ok=True)
    val_pt = seed_dir / "neural_vocabulary_scores_val.pt"
    test_pt = seed_dir / "neural_vocabulary_scores_test.pt"
    cal_path = seed_dir / "calibration_hyperparameters_val.json"
    dec_path = seed_dir / "sequence_decoding_test.json"

    prov_no_f = {
        "word": str(word_ckpt),
        "context": "not_available",
        "revision": str(rev_ckpt),
        "revision_stage": "prefix_revision",
    }
    for pt_file, sp_name in ((val_pt, "val"), (test_pt, "test")):
        torch.save(
            {
                "dataset": "pallier2025",
                "language": "fr",
                "split": sp_name,
                "seed": 0,
                "vocabulary": vocabulary,
                "vocabulary_size": vocab_size,
                "vocabulary_embeddings": torch.eye(vocab_size, 16),
                "clean_scores": clean_no_f,
                "shifted_scores": {k: v for k, v in clean_no_f.items() if k != "S0"},
                "segments": segments,
                "checkpoint_provenance": prov_no_f,
                "cms_metadata": {"subtraction_eta": 0.0},
            },
            pt_file,
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
                "--exp-tag",
                "tag_rev",
                "--word-exp-tag",
                "tag_word",
                "--context-exp-tag",
                "tag_ctx",
                "--output-dir",
                str(seed_dir),
                "--export-path",
                str(test_pt),
                "--calibration-path",
                str(cal_path),
                "--decoding-path",
                str(dec_path),
                "--vocab-sizes",
                "3,6",
                "--core-vocab-size",
                "3",
                "--no-plot",
                "--no-progress",
            ]
        )
        res = decode_cli.run_sequence_decoding_pipeline(args)
        if step_name == "calibrate":
            assert set(res["state_hyperparameters"].keys()) == {"S0", "K0", "K1", "K2"}
        elif step_name == "decode":
            saved_dec = json.loads(dec_path.read_text(encoding="utf-8"))
            assert "F" not in saved_dec["decoded_bundles"]
            assert saved_dec["f_condition_mode"] == "not_available"
        elif step_name == "evaluate":
            assert "F" not in res["conditions"]

    # 现在补入 Stage 2.1 的 F 得分矩阵，验证 calibrate 与 decode 仅增量计算 F
    ctx_ckpt.write_bytes(b"ckpt")
    clean_with_f = dict(clean_no_f)
    clean_with_f["F"] = clean_no_f["K2"].clone()
    prov_with_f = {
        "word": str(word_ckpt),
        "context": str(ctx_ckpt),
        "revision": str(rev_ckpt),
        "revision_stage": "prefix_revision",
    }
    for pt_file, sp_name in ((val_pt, "val"), (test_pt, "test")):
        torch.save(
            {
                "dataset": "pallier2025",
                "language": "fr",
                "split": sp_name,
                "seed": 0,
                "vocabulary": vocabulary,
                "vocabulary_size": vocab_size,
                "vocabulary_embeddings": torch.eye(vocab_size, 16),
                "clean_scores": clean_with_f,
                "shifted_scores": {k: v for k, v in clean_no_f.items() if k != "S0"},
                "segments": segments,
                "checkpoint_provenance": prov_with_f,
                "cms_metadata": {"subtraction_eta": 0.0},
            },
            pt_file,
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
                "--exp-tag",
                "tag_rev",
                "--word-exp-tag",
                "tag_word",
                "--context-exp-tag",
                "tag_ctx",
                "--output-dir",
                str(seed_dir),
                "--export-path",
                str(test_pt),
                "--calibration-path",
                str(cal_path),
                "--decoding-path",
                str(dec_path),
                "--vocab-sizes",
                "3,6",
                "--core-vocab-size",
                "3",
                "--no-plot",
                "--no-progress",
            ]
        )
        res = decode_cli.run_sequence_decoding_pipeline(args)
        if step_name == "calibrate":
            assert set(res["state_hyperparameters"].keys()) == {
                "S0",
                "K0",
                "K1",
                "K2",
                "F",
            }
        elif step_name == "decode":
            saved_dec = json.loads(dec_path.read_text(encoding="utf-8"))
            assert "F" in saved_dec["decoded_bundles"]
            assert saved_dec["f_condition_mode"] == "F_full_context"
        elif step_name == "evaluate":
            assert "F" in res["conditions"]






