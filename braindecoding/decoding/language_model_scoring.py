"""闭集词表预分词与基于 KV-Cache (past_key_values) 的增量批量语言模型打分器（Module 2）。

彻底移除测试期调用 `model.generate()` 自由生成 OOV 词与动态重编码逻辑：
1. 默认升级语言模型底座为 `Qwen/Qwen3-8B-Base`（支持配置切换 `Qwen/Qwen3-4B-Base` 或本地权重路径）。
2. 初始化时将训练词表 V（法语 |V|=2,020，中文 |V|=2,180）静态预编码为各语法上下文状态下的 Token ID 序列。
3. 实现 `score_candidates_with_cache()`：
   - 单 Token 候选词：直接从前缀末位 `last_log_probs` O(1) 索引取值，0 次模型 Forward；
   - 多 Token 候选词：首 Token 从 `last_log_probs` 取值，仅对剩余 1~3 个后缀 Token 复用 `past_key_values`
     执行增量 Batch Forward。
4. 实现 `advance_prefix_states()`：仅对束搜索胜出的 Top-B (B=5) 束增量推进 1 个词的 KV-Cache。
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F_nn

from braindecoding.config import PROJECT_ROOT, model_root, project_path


DEFAULT_LANGUAGE_MODEL = "Qwen/Qwen3-8B-Base"
FALLBACK_LANGUAGE_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-4B-Base",
)

PROMPTS: dict[str, str] = {
    "fr": "Écrivez un paragraphe de récit en français. Donnez seulement le texte, sans titre ni liste.",
    "zh": "请用中文写一段连续的叙述文字。只给出正文，不要标题或列表。",
}
ELISIONS: frozenset[str] = frozenset(
    "c j l d n m s t qu jusqu lorsqu puisqu quelqu aujourd".split()
)
INTERFACE_VERSION = "cached_vocabulary_lm_s3_v1"


def render_words(words: Sequence[str], language: str) -> str:
    """将词序列渲染为法语（含省音撇号与连字符规则）或中文自然表面字符串。"""
    words_tuple = tuple(str(w) for w in words)
    if any(not w or w != w.strip() or any(c.isspace() for c in w) for w in words_tuple):
        raise ValueError("Expected nonempty individual evaluation units")
    if language == "zh":
        return "".join(words_tuple)
    if language != "fr":
        raise ValueError("Only fr and zh are supported")
    result = ""
    for i, word in enumerate(words_tuple):
        if not i:
            result = word
        elif word.startswith("-"):
            result += word
        elif words_tuple[i - 1].lower().lstrip("-") in ELISIONS:
            result += "'" + word
        else:
            result += " " + word
    return result


def prefix_context_kind(prefix_words: Sequence[str], language: str) -> str:
    """根据当前前缀末词判断下一候选词的拼接表面状态。"""
    if language == "zh":
        return "zh"
    if language != "fr":
        raise ValueError(f"Unsupported language: {language!r}")
    if not prefix_words:
        return "initial"
    last_word = str(prefix_words[-1]).lower().lstrip("-")
    if last_word in ELISIONS:
        return "after_elision"
    return "continuation"


def word_surface_for_context(word: str, context_kind: str) -> str:
    """返回词 `word` 在给定前缀表面状态 `context_kind` 下追加的精确子串。"""
    text = str(word)
    if context_kind in {"zh", "initial"}:
        return text
    if text.startswith("-"):
        return text
    if context_kind == "after_elision":
        return "'" + text
    if context_kind == "continuation":
        return " " + text
    raise ValueError(f"Unknown context_kind: {context_kind!r}")


def _to_legacy_kv_tuples(
    past_key_values: object,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """将 HuggingFace DynamicCache 或元组形式的 past_key_values 统一转为 ((k, v), ...) 元组。"""
    if past_key_values is None:
        raise ValueError("past_key_values cannot be None")
    if isinstance(past_key_values, tuple):
        return tuple((layer[0], layer[1]) for layer in past_key_values)
    if hasattr(past_key_values, "to_legacy_cache"):
        return tuple(
            (layer[0], layer[1]) for layer in past_key_values.to_legacy_cache()
        )
    if hasattr(past_key_values, "key_cache") and hasattr(
        past_key_values, "value_cache"
    ):
        return tuple(
            zip(past_key_values.key_cache, past_key_values.value_cache)
        )
    if hasattr(past_key_values, "layers"):
        return tuple(
            (layer.keys, layer.values) for layer in past_key_values.layers
        )
    raise TypeError(
        f"Unsupported past_key_values type: {type(past_key_values)!r}"
    )


def _from_legacy_kv_tuples(
    legacy_tuples: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    reference_cache: object,
) -> object:
    """将 ((k, v), ...) 元组恢复为模型期望的 Cache 对象或元组。"""
    if isinstance(reference_cache, tuple):
        return legacy_tuples
    cache_cls = type(reference_cache)
    if hasattr(cache_cls, "from_legacy_cache"):
        return cache_cls.from_legacy_cache(legacy_tuples)
    try:
        from transformers.cache_utils import DynamicCache

        if isinstance(reference_cache, DynamicCache):
            cache = DynamicCache()
            for layer_idx, (k_tensor, v_tensor) in enumerate(legacy_tuples):
                cache.update(k_tensor, v_tensor, layer_idx)
            return cache
    except ImportError:
        pass
    return legacy_tuples


def _stack_kv_caches(
    caches: Sequence[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
    reference_cache: object,
) -> object:
    """将具有相同序列长度的多个单样本 KV-Cache 沿 batch 维度 (dim=0) 拼接为批量 Cache。"""
    num_layers = len(caches[0])
    batched_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(num_layers):
        k_cat = torch.cat([c[layer_idx][0] for c in caches], dim=0)
        v_cat = torch.cat([c[layer_idx][1] for c in caches], dim=0)
        batched_layers.append((k_cat, v_cat))
    return _from_legacy_kv_tuples(tuple(batched_layers), reference_cache)


def _stack_padded_kv_caches(
    caches: Sequence[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
    reference_cache: object,
) -> tuple[object, list[int], int]:
    """将多个可能具有不同序列长度的单样本 KV-Cache 左填充对齐并沿 batch 维度 (dim=0) 拼接。"""
    past_lengths = [int(c[0][0].shape[2]) for c in caches]
    max_past_len = max(past_lengths)
    min_past_len = min(past_lengths)
    num_layers = len(caches[0])
    batched_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

    if min_past_len == max_past_len:
        for layer_idx in range(num_layers):
            k_cat = torch.cat([c[layer_idx][0] for c in caches], dim=0)
            v_cat = torch.cat([c[layer_idx][1] for c in caches], dim=0)
            batched_layers.append((k_cat, v_cat))
    else:
        for layer_idx in range(num_layers):
            k_list: list[torch.Tensor] = []
            v_list: list[torch.Tensor] = []
            for c, p_len in zip(caches, past_lengths):
                k_t, v_t = c[layer_idx]
                pad_left = max_past_len - p_len
                if pad_left > 0:
                    k_t = F_nn.pad(k_t, (0, 0, pad_left, 0), value=0.0)
                    v_t = F_nn.pad(v_t, (0, 0, pad_left, 0), value=0.0)
                k_list.append(k_t)
                v_list.append(v_t)
            batched_layers.append(
                (torch.cat(k_list, dim=0), torch.cat(v_list, dim=0))
            )

    return (
        _from_legacy_kv_tuples(tuple(batched_layers), reference_cache),
        past_lengths,
        max_past_len,
    )


def _unstack_kv_cache(
    batched_cache: object,
    batch_size: int,
) -> list[tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
    """将批量前向后的 past_key_values 拆回单样本 ((k, v), ...) 元组列表。"""
    legacy = _to_legacy_kv_tuples(batched_cache)
    per_item: list[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = []
    for item_idx in range(batch_size):
        item_layers = tuple(
            (
                layer_k[item_idx : item_idx + 1].contiguous(),
                layer_v[item_idx : item_idx + 1].contiguous(),
            )
            for layer_k, layer_v in legacy
        )
        per_item.append(item_layers)
    return per_item


def _unstack_padded_kv_cache(
    batched_cache: object,
    past_lengths: Sequence[int],
    step_lengths: Sequence[int],
    max_past_len: int,
) -> list[tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
    """将左填充历史 + 右填充当前步的批量 past_key_values 切片还原为各样本真实长度的 ((k, v), ...) 元组。"""
    legacy = _to_legacy_kv_tuples(batched_cache)
    per_item: list[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = []
    for item_idx, (p_len, s_len) in enumerate(zip(past_lengths, step_lengths)):
        start_col = max_past_len - int(p_len)
        end_col = max_past_len + int(s_len)
        item_layers = tuple(
            (
                layer_k[item_idx : item_idx + 1, :, start_col:end_col, :].contiguous(),
                layer_v[item_idx : item_idx + 1, :, start_col:end_col, :].contiguous(),
            )
            for layer_k, layer_v in legacy
        )
        per_item.append(item_layers)
    return per_item


@dataclass
class PrefixCacheState:
    """束搜索单个前缀束的语言模型增量缓存状态（支持按需惰性物化与跨步/跨搜索复用）。"""

    words: tuple[str, ...]
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None
    last_log_probs: torch.Tensor | None
    token_length: int
    context_kind: str
    parent_state: PrefixCacheState | None = None
    step_token_ids: tuple[int, ...] = ()



class PreTokenizedVocabulary:
    """训练词表 V 的静态预分词表，区分单 Token 候选与多 Token 候选。"""

    def __init__(
        self,
        vocabulary: Sequence[str],
        tokenizer,
        language: str,
    ) -> None:
        if language not in {"fr", "zh"}:
            raise ValueError(f"Only 'fr' and 'zh' are supported, got {language!r}")
        self.vocabulary: tuple[str, ...] = tuple(str(w) for w in vocabulary)
        if not self.vocabulary:
            raise ValueError("Vocabulary must be non-empty")
        self.word_to_index: dict[str, int] = {
            w: idx for idx, w in enumerate(self.vocabulary)
        }
        self.language = language
        context_kinds = (
            ("zh",)
            if language == "zh"
            else ("initial", "after_elision", "continuation")
        )
        self.tokens_by_context: dict[str, list[tuple[int, ...]]] = {}
        self.first_token_by_context: dict[str, torch.Tensor] = {}
        self.is_single_token_by_context: dict[str, np.ndarray] = {}

        for ctx in context_kinds:
            tok_list: list[tuple[int, ...]] = []
            first_tokens: list[int] = []
            single_flags: list[bool] = []
            for word in self.vocabulary:
                surface = word_surface_for_context(word, ctx)
                ids = tuple(
                    int(t)
                    for t in tokenizer.encode(surface, add_special_tokens=False)
                )
                if not ids:
                    raise ValueError(
                        f"Tokenizer produced empty token sequence for word={word!r} (surface={surface!r})"
                    )
                tok_list.append(ids)
                first_tokens.append(ids[0])
                single_flags.append(len(ids) == 1)
            self.tokens_by_context[ctx] = tok_list
            self.first_token_by_context[ctx] = torch.tensor(
                first_tokens, dtype=torch.long
            )
            self.is_single_token_by_context[ctx] = np.asarray(
                single_flags, dtype=bool
            )

    def token_ids_for(self, word_index: int, context_kind: str) -> tuple[int, ...]:
        return self.tokens_by_context[context_kind][int(word_index)]


class CachedVocabularyLanguageScorer:
    """基于预分词闭集词表与 past_key_values (KV-Cache) 的极速增量语言模型打分器。"""

    def __init__(
        self,
        model,
        tokenizer,
        language: str,
        vocabulary: Sequence[str],
        settings: dict | None = None,
        *,
        prompt_body: str | None = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.language = str(language)
        self.settings = dict(settings or {})
        self.device = next(model.parameters()).device
        if prompt_body is not None:
            if not prompt_body.strip():
                raise ValueError("prompt_body cannot be empty")
            self.prompt = prompt_body.strip() + "\n\n"
        else:
            self.prompt = PROMPTS[self.language] + "\n\n"
        self.prompt_ids: tuple[int, ...] = tuple(
            int(t) for t in tokenizer.encode(self.prompt, add_special_tokens=False)
        )
        if not self.prompt_ids:
            raise ValueError("Encoded prompt token sequence is empty")

        self.vocab_table = PreTokenizedVocabulary(
            vocabulary, tokenizer, self.language
        )
        self.vocabulary = self.vocab_table.vocabulary
        self.first_token_device_map: dict[str, torch.Tensor] = {
            ctx: tensor.to(self.device)
            for ctx, tensor in self.vocab_table.first_token_by_context.items()
        }
        self.max_batch_size = int(self.settings.get("score_batch_size", 128))
        self.max_prefix_cache_states = int(
            self.settings.get("max_prefix_cache_states", 64)
        )
        self.max_transition_cache_prefixes = int(
            self.settings.get("max_transition_cache_prefixes", 200000)
        )
        self._reference_cache_type: object = None
        self._prompt_cache_state: PrefixCacheState | None = None
        self._prefix_state_cache: OrderedDict[tuple[str, ...], PrefixCacheState] = (
            OrderedDict()
        )
        self._transition_score_cache: OrderedDict[
            tuple[str, ...], dict[int, float]
        ] = OrderedDict()
        self.stats = {
            "single_token_lookups": 0,
            "multi_token_evaluations": 0,
            "incremental_forward_batches": 0,
            "prefix_advance_batches": 0,
            "transition_cache_hits": 0,
            "prefix_cache_hits": 0,
        }

    def _cache_prefix_state(self, state: PrefixCacheState) -> None:
        if state.past_key_values is None or state.last_log_probs is None:
            return
        key = state.words
        self._prefix_state_cache[key] = state
        self._prefix_state_cache.move_to_end(key)
        while len(self._prefix_state_cache) > self.max_prefix_cache_states:
            self._prefix_state_cache.popitem(last=False)

    def _get_transition_bucket(self, words: tuple[str, ...]) -> dict[int, float]:
        bucket = self._transition_score_cache.get(words)
        if bucket is not None:
            self._transition_score_cache.move_to_end(words)
            return bucket
        bucket = {}
        self._transition_score_cache[words] = bucket
        while (
            len(self._transition_score_cache) > self.max_transition_cache_prefixes
        ):
            self._transition_score_cache.popitem(last=False)
        return bucket

    @torch.inference_mode()
    def init_prompt_state(self) -> PrefixCacheState:
        """编码初始语言提示词 Prompt，返回并缓存初始 PrefixCacheState。"""
        if self._prompt_cache_state is not None:
            return self._prompt_cache_state

        input_ids = torch.tensor(
            [self.prompt_ids], dtype=torch.long, device=self.device
        )
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
        )
        self._reference_cache_type = outputs.past_key_values
        legacy_kv = _to_legacy_kv_tuples(outputs.past_key_values)
        last_logits = outputs.logits[0, -1, :].float()
        last_log_probs = F_nn.log_softmax(last_logits, dim=-1)
        init_ctx = prefix_context_kind((), self.language)
        self._prompt_cache_state = PrefixCacheState(
            words=(),
            past_key_values=legacy_kv,
            last_log_probs=last_log_probs,
            token_length=len(self.prompt_ids),
            context_kind=init_ctx,
        )
        self._cache_prefix_state(self._prompt_cache_state)
        return self._prompt_cache_state

    @torch.inference_mode()
    def _ensure_states_materialized(
        self,
        states: Sequence[PrefixCacheState],
    ) -> None:
        """按需批量物化惰性 PrefixCacheState 的 past_key_values 与 last_log_probs。"""
        pending_by_words: dict[tuple[str, ...], list[PrefixCacheState]] = (
            defaultdict(list)
        )
        for st in states:
            if st.past_key_values is not None and st.last_log_probs is not None:
                continue
            cached = self._prefix_state_cache.get(st.words)
            if (
                cached is not None
                and cached.past_key_values is not None
                and cached.last_log_probs is not None
            ):
                self._prefix_state_cache.move_to_end(st.words)
                st.past_key_values = cached.past_key_values
                st.last_log_probs = cached.last_log_probs
                st.parent_state = None
                st.step_token_ids = ()
                self.stats["prefix_cache_hits"] += 1
                continue
            pending_by_words[st.words].append(st)

        if not pending_by_words:
            return

        unique_specs: list[
            tuple[
                tuple[str, ...],
                PrefixCacheState,
                tuple[int, ...],
                list[PrefixCacheState],
            ]
        ] = []
        for words_key, st_group in pending_by_words.items():
            rep = st_group[0]
            parent = rep.parent_state or self.init_prompt_state()
            step_toks = rep.step_token_ids
            if parent.past_key_values is None:
                self._ensure_states_materialized([parent])
            if not step_toks:
                for target_st in st_group:
                    target_st.past_key_values = parent.past_key_values
                    target_st.last_log_probs = parent.last_log_probs
                    target_st.parent_state = None
                continue
            unique_specs.append((words_key, parent, step_toks, st_group))

        for start in range(0, len(unique_specs), self.max_batch_size):
            batch = unique_specs[start : start + self.max_batch_size]
            bsz = len(batch)
            parent_kvs = [item[1].past_key_values for item in batch]
            batched_kv, past_lengths, max_past_len = _stack_padded_kv_caches(
                parent_kvs, self._reference_cache_type
            )
            step_lengths = [len(item[2]) for item in batch]
            max_step_len = max(step_lengths)
            min_step_len = min(step_lengths)
            min_past_len = min(past_lengths)

            input_ids = torch.zeros(
                (bsz, max_step_len), dtype=torch.long, device=self.device
            )
            position_ids = torch.zeros(
                (bsz, max_step_len), dtype=torch.long, device=self.device
            )
            need_mask = (min_past_len != max_past_len) or (
                min_step_len != max_step_len
            )
            attention_mask = (
                torch.zeros(
                    (bsz, max_past_len + max_step_len),
                    dtype=torch.long,
                    device=self.device,
                )
                if need_mask
                else None
            )

            for row_i, (_, _, step_toks, _) in enumerate(batch):
                p_len = past_lengths[row_i]
                s_len = len(step_toks)
                input_ids[row_i, :s_len] = torch.tensor(
                    step_toks, dtype=torch.long, device=self.device
                )
                pos_seq = torch.arange(
                    p_len, p_len + s_len, dtype=torch.long, device=self.device
                )
                position_ids[row_i, :s_len] = pos_seq
                if s_len < max_step_len:
                    position_ids[row_i, s_len:] = p_len + s_len - 1
                if attention_mask is not None:
                    pad_left = max_past_len - p_len
                    attention_mask[row_i, pad_left : max_past_len + s_len] = 1

            model_kwargs: dict = {
                "input_ids": input_ids,
                "past_key_values": batched_kv,
                "position_ids": position_ids,
                "use_cache": True,
            }
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask

            outputs = self.model(**model_kwargs)
            self.stats["prefix_advance_batches"] += 1
            unstacked_kvs = _unstack_padded_kv_cache(
                outputs.past_key_values,
                past_lengths,
                step_lengths,
                max_past_len,
            )
            last_indices = torch.tensor(
                [s_len - 1 for s_len in step_lengths],
                dtype=torch.long,
                device=self.device,
            )
            batch_rows = torch.arange(bsz, dtype=torch.long, device=self.device)
            last_logits = outputs.logits[batch_rows, last_indices, :].float()
            last_lps = F_nn.log_softmax(last_logits, dim=-1)

            for row_i, (_, _, _, st_group) in enumerate(batch):
                kv_i = unstacked_kvs[row_i]
                lp_i = last_lps[row_i]
                for target_st in st_group:
                    target_st.past_key_values = kv_i
                    target_st.last_log_probs = lp_i
                    target_st.parent_state = None
                    target_st.step_token_ids = ()
                self._cache_prefix_state(st_group[0])

    @torch.inference_mode()
    def score_candidates_with_cache(
        self,
        prefix_states: Sequence[PrefixCacheState],
        candidate_word_indices: Sequence[Sequence[int]],
    ) -> list[np.ndarray]:
        """利用前缀 KV-Cache 与转移概率缓存批量增量计算每个前缀束下各候选词的精确对数概率 log P_LM(w | u)。

        - 已缓存转移 `(u, w)`：直接从 `_transition_score_cache` O(1) 命中返回，0 次 GPU 调用；
        - 单 Token 候选词：直接从 `prefix_state.last_log_probs[t_0]` O(1) 索引取值，0 次 Forward；
        - 多 Token 候选词：首 Token 从 `last_log_probs[t_0]` 取值，剩余 1~3 个后缀 Token
          按唯一 `(u, tok_ids)` 去重后通过左填充 KV-Cache 组装为单次 Batch 增量 Forward。
        """
        if len(prefix_states) != len(candidate_word_indices):
            raise ValueError(
                f"prefix_states ({len(prefix_states)}) 与 candidate_word_indices ({len(candidate_word_indices)}) 数量不一致。"
            )
        results: list[np.ndarray] = [
            np.zeros(len(cands), dtype=np.float64)
            for cands in candidate_word_indices
        ]

        missing_by_beam: dict[int, list[tuple[int, int]]] = {}
        states_to_materialize: list[PrefixCacheState] = []

        for beam_idx, (state, cand_indices) in enumerate(
            zip(prefix_states, candidate_word_indices)
        ):
            if not cand_indices:
                continue
            cached_bucket = self._transition_score_cache.get(state.words)
            out_row = results[beam_idx]
            beam_misses: list[tuple[int, int]] = []
            if cached_bucket is not None:
                self._transition_score_cache.move_to_end(state.words)
                for pos, word_idx in enumerate(cand_indices):
                    w_id = int(word_idx)
                    if w_id in cached_bucket:
                        out_row[pos] = cached_bucket[w_id]
                        self.stats["transition_cache_hits"] += 1
                    else:
                        beam_misses.append((pos, w_id))
            else:
                beam_misses = [
                    (pos, int(word_idx))
                    for pos, word_idx in enumerate(cand_indices)
                ]

            if beam_misses:
                missing_by_beam[beam_idx] = beam_misses
                states_to_materialize.append(state)

        if not missing_by_beam:
            return results

        self._ensure_states_materialized(states_to_materialize)

        # 去重收集所有多 Token 候选词请求：
        # key = (state.words, int(word_idx)) -> (state, tok_ids, first_lp, [(beam_idx, pos), ...])
        unique_multi_requests: dict[
            tuple[tuple[str, ...], int],
            tuple[
                PrefixCacheState,
                tuple[int, ...],
                float,
                list[tuple[int, int]],
            ],
        ] = {}

        for beam_idx, beam_misses in missing_by_beam.items():
            state = prefix_states[beam_idx]
            ctx = state.context_kind
            log_probs_0 = state.last_log_probs
            tokens_list = self.vocab_table.tokens_by_context[ctx]
            out_row = results[beam_idx]
            bucket = self._get_transition_bucket(state.words)

            for pos, word_idx in beam_misses:
                if word_idx in bucket:
                    out_row[pos] = bucket[word_idx]
                    self.stats["transition_cache_hits"] += 1
                    continue
                token_ids = tokens_list[word_idx]
                first_lp = float(log_probs_0[token_ids[0]].item())
                out_row[pos] = first_lp
                if len(token_ids) == 1:
                    bucket[word_idx] = first_lp
                    self.stats["single_token_lookups"] += 1
                else:
                    req_key = (state.words, word_idx)
                    if req_key not in unique_multi_requests:
                        self.stats["multi_token_evaluations"] += 1
                        unique_multi_requests[req_key] = (
                            state,
                            token_ids,
                            first_lp,
                            [(beam_idx, pos)],
                        )
                    else:
                        unique_multi_requests[req_key][3].append((beam_idx, pos))

        if not unique_multi_requests:
            return results

        multi_items = list(unique_multi_requests.items())
        for start in range(0, len(multi_items), self.max_batch_size):
            batch_items = multi_items[start : start + self.max_batch_size]
            bsz = len(batch_items)
            batch_kvs = [item[1][0].past_key_values for item in batch_items]
            batched_kv, past_lengths, max_past_len = _stack_padded_kv_caches(
                batch_kvs, self._reference_cache_type
            )
            step_lengths = [len(item[1][1]) - 1 for item in batch_items]
            max_step_len = max(step_lengths)
            min_step_len = min(step_lengths)
            min_past_len = min(past_lengths)

            input_ids = torch.zeros(
                (bsz, max_step_len), dtype=torch.long, device=self.device
            )
            position_ids = torch.zeros(
                (bsz, max_step_len), dtype=torch.long, device=self.device
            )
            need_mask = (min_past_len != max_past_len) or (
                min_step_len != max_step_len
            )
            attention_mask = (
                torch.zeros(
                    (bsz, max_past_len + max_step_len),
                    dtype=torch.long,
                    device=self.device,
                )
                if need_mask
                else None
            )

            for row_i, (_, (_, tok_ids, _, _)) in enumerate(batch_items):
                p_len = past_lengths[row_i]
                step_tokens = tok_ids[:-1]
                s_len = len(step_tokens)
                input_ids[row_i, :s_len] = torch.tensor(
                    step_tokens, dtype=torch.long, device=self.device
                )
                pos_seq = torch.arange(
                    p_len, p_len + s_len, dtype=torch.long, device=self.device
                )
                position_ids[row_i, :s_len] = pos_seq
                if s_len < max_step_len:
                    position_ids[row_i, s_len:] = p_len + s_len - 1
                if attention_mask is not None:
                    pad_left = max_past_len - p_len
                    attention_mask[row_i, pad_left : max_past_len + s_len] = 1

            model_kwargs: dict = {
                "input_ids": input_ids,
                "past_key_values": batched_kv,
                "position_ids": position_ids,
                "use_cache": False,
            }
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask

            outputs = self.model(**model_kwargs)
            self.stats["incremental_forward_batches"] += 1
            step_log_probs = F_nn.log_softmax(outputs.logits.float(), dim=-1)

            for row_i, ((words_key, word_idx), (_, tok_ids, first_lp, targets_list)) in enumerate(
                batch_items
            ):
                suffix_targets = tok_ids[1:]
                n_suf = len(suffix_targets)
                target_tensor = torch.tensor(
                    suffix_targets, dtype=torch.long, device=self.device
                ).unsqueeze(-1)
                gathered = (
                    step_log_probs[row_i, :n_suf, :]
                    .gather(-1, target_tensor)
                    .squeeze(-1)
                )
                total_lp = first_lp + float(gathered.double().sum().item())
                bucket = self._get_transition_bucket(words_key)
                bucket[word_idx] = total_lp
                for beam_idx, pos in targets_list:
                    results[beam_idx][pos] = total_lp

        return results

    @torch.inference_mode()
    def propose_top_k_from_language_model(
        self,
        prefix_state: PrefixCacheState,
        top_k: int = 20,
    ) -> np.ndarray:
        """供 LM-Only 基线使用：利用当前前缀的 last_log_probs 从闭集词表 V 中零开销筛选首 Token Top-M 候选并精排返回 Top-K 词索引。"""
        self._ensure_states_materialized([prefix_state])
        k = min(int(top_k), len(self.vocabulary))
        shortlist_m = min(max(k * 2, 40), len(self.vocabulary))
        first_tok_ids = self.first_token_device_map[prefix_state.context_kind]
        first_scores = prefix_state.last_log_probs.index_select(0, first_tok_ids)
        _, top_m_indices = torch.topk(
            first_scores, k=shortlist_m, largest=True, sorted=True
        )
        cand_list = top_m_indices.cpu().tolist()
        exact_scores = self.score_candidates_with_cache([prefix_state], [cand_list])[0]
        order = np.argsort(-exact_scores, kind="stable")[:k]
        return np.asarray([cand_list[int(i)] for i in order], dtype=np.int64)

    @torch.inference_mode()
    def advance_prefix_states(
        self,
        advancement_requests: Sequence[tuple[PrefixCacheState, int]],
    ) -> list[PrefixCacheState]:
        """对选中的 (PrefixCacheState, word_index) 列表推进 1 个词状态（优先复用缓存或创建惰性状态，按需批量物化）。"""
        if not advancement_requests:
            return []

        advanced_states: list[PrefixCacheState] = []
        for state, word_idx in advancement_requests:
            w_id = int(word_idx)
            word_str = self.vocabulary[w_id]
            new_words = state.words + (word_str,)

            cached = self._prefix_state_cache.get(new_words)
            if (
                cached is not None
                and cached.past_key_values is not None
                and cached.last_log_probs is not None
            ):
                self._prefix_state_cache.move_to_end(new_words)
                self.stats["prefix_cache_hits"] += 1
                advanced_states.append(cached)
                continue

            tok_ids = self.vocab_table.token_ids_for(w_id, state.context_kind)
            new_ctx = prefix_context_kind(new_words, self.language)

            if state.past_key_values is not None and state.last_log_probs is not None:
                parent_ref = state
                pending_tokens = tok_ids
            else:
                cached_parent = self._prefix_state_cache.get(state.words)
                if (
                    cached_parent is not None
                    and cached_parent.past_key_values is not None
                    and cached_parent.last_log_probs is not None
                ):
                    parent_ref = cached_parent
                    pending_tokens = tok_ids
                else:
                    parent_ref = state.parent_state or self.init_prompt_state()
                    pending_tokens = state.step_token_ids + tok_ids

            advanced_states.append(
                PrefixCacheState(
                    words=new_words,
                    past_key_values=None,
                    last_log_probs=None,
                    token_length=state.token_length + len(tok_ids),
                    context_kind=new_ctx,
                    parent_state=parent_ref,
                    step_token_ids=pending_tokens,
                )
            )

        return advanced_states


def resolve_language_model_path(model_name_or_path: str | Path | None = None) -> str:
    """解析语言模型名称或本地缓存路径（优先检查本地 models 目录，回退至 HuggingFace Hub ID）。"""
    raw = str(model_name_or_path or DEFAULT_LANGUAGE_MODEL).strip()
    candidate_paths = [
        Path(raw),
        PROJECT_ROOT / raw,
        model_root() / raw,
        model_root() / Path(raw).name,
    ]
    for path in candidate_paths:
        if path.is_dir() and (path / "config.json").is_file():
            return str(path.resolve())
    return raw


def load_causal_language_model(
    model_name_or_path: str | Path | None = DEFAULT_LANGUAGE_MODEL,
    *,
    device: str | torch.device | None = "auto",
    precision: str = "bf16-mixed",
    local_files_only: bool = False,
):
    """加载 Qwen3-8B-Base（或 Qwen3-4B-Base）因果语言模型与分词器。"""
    import importlib.util
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_language_model_path(model_name_or_path)
    if isinstance(device, torch.device):
        torch_device = device
    elif device is None or str(device).lower() == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(str(device))

    prec = str(precision or "bf16-mixed").lower()
    if torch_device.type == "cuda":
        if "bf16" in prec or "bfloat16" in prec:
            dtype = torch.bfloat16
        elif "fp16" in prec or "float16" in prec:
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32

    t0 = time.perf_counter()
    print(
        f"[S3-LM] 正在加载语言模型与分词器: {resolved} (device={torch_device}, dtype={dtype})...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, local_files_only=local_files_only
    )
    model_kwargs: dict = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
    }
    if importlib.util.find_spec("accelerate") is not None:
        model_kwargs["low_cpu_mem_usage"] = True
    model = (
        AutoModelForCausalLM.from_pretrained(
            resolved,
            **model_kwargs,
        )
        .to(torch_device)
        .eval()
    )
    print(
        f"[S3-LM] 语言模型加载完成 ({resolved})，耗时 {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return model, tokenizer


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


class NaturalWordLM:
    """精确表面字符串对数似然差评分器（用于与 KV-Cache 增量打分做数值等价性核验）。"""

    def __init__(self, model, tokenizer, language: str, settings: dict | None = None):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.language, self.settings = language, dict(settings or {})
        self.device = next(model.parameters()).device
        self.prompt = PROMPTS[language] + "\n\n"
        self.prompt_ids = tuple(
            tokenizer.encode(self.prompt, add_special_tokens=False)
        )
        max_pos = int(getattr(model.config, "max_position_embeddings", 32768))
        self.context_tokens = min(
            int(self.settings.get("context_tokens", 32768)), max_pos
        )
        self.cache: OrderedDict[tuple[tuple[str, ...], str], float] = OrderedDict()

    def input_text(self, words: Sequence[str]) -> str:
        return self.prompt + render_words(words, self.language)

    def prefix_ids(self, words: Sequence[str], reserve: int = 0) -> tuple[int, ...]:
        ids = tuple(
            self.tokenizer.encode(self.input_text(words), add_special_tokens=False)
        )
        if not ids or len(ids) + reserve > self.context_tokens:
            raise ValueError(
                "Canonical input exceeds context; silent truncation would change the score"
            )
        return ids

    @torch.inference_mode()
    def _tail_scores(self, requests):
        result = {}
        pending = list(dict.fromkeys(requests))
        batch_size = int(self.settings.get("score_batch_size", 32))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id or 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            active = [(seq, cut) for seq, cut in batch if cut < len(seq)]
            result.update({(seq, cut): 0.0 for seq, cut in batch if cut == len(seq)})
            if not active:
                continue
            length = max(len(seq) for seq, _ in active)
            ids = torch.full(
                (len(active), length), pad, device=self.device, dtype=torch.long
            )
            mask = torch.zeros_like(ids)
            for i, (seq, _) in enumerate(active):
                ids[i, -len(seq) :] = torch.tensor(seq, device=self.device)
                mask[i, -len(seq) :] = 1
            logits = self.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.cumsum(1) - 1).clamp_min(0),
                use_cache=False,
            ).logits[:, :-1].float()
            for i, (seq, cut) in enumerate(active):
                n = len(seq) - cut
                tail = logits[i, -n:]
                targets = torch.tensor(seq[cut:], device=self.device)[:, None]
                terms = tail.gather(1, targets).squeeze(1) - torch.logsumexp(
                    tail, dim=-1
                )
                result[(seq, cut)] = float(terms.double().sum())
        return result

    def score_extensions(self, prefixes, words):
        if len(prefixes) != len(words):
            raise ValueError("Prefix and word counts differ")
        keys = [(tuple(p), str(w)) for p, w in zip(prefixes, words)]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        pairs, requests = {}, []
        for key in missing:
            prefix, word = key
            old = self.prefix_ids(prefix)
            new = self.prefix_ids(prefix + (word,))
            cut = common_prefix_length(old, new)
            if cut < 1:
                raise ValueError("Canonical pair must retain a shared conditioning token")
            pairs[key] = ((new, cut), (old, cut))
            requests.extend(pairs[key])
        scores = self._tail_scores(requests)
        resolved = {
            key: scores[new] - scores[old] for key, (new, old) in pairs.items()
        }
        if not all(np.isfinite(s) for s in resolved.values()):
            raise ValueError("Non-finite canonical language score")
        output = np.asarray(
            [resolved[k] if k in resolved else self.cache[k] for k in keys],
            dtype=np.float64,
        )
        self.cache.update(resolved)
        while len(self.cache) > int(self.settings.get("score_cache_size", 200000)):
            self.cache.popitem(last=False)
        return output


class PromptedNaturalWordLM(NaturalWordLM):
    """支持自定义提示词文本的表面对数似然差评分器。"""

    def __init__(
        self,
        model,
        tokenizer,
        language: str,
        settings: dict | None = None,
        *,
        prompt_body: str,
    ):
        if (
            not isinstance(prompt_body, str)
            or not prompt_body
            or prompt_body != prompt_body.strip()
        ):
            raise ValueError(
                "Prompt body must be nonempty and have no surrounding whitespace"
            )
        super().__init__(
            model,
            tokenizer,
            language,
            dict(settings or {}, prompt_body=prompt_body),
        )
        self.prompt = prompt_body + "\n\n"
        self.prompt_ids = tuple(
            tokenizer.encode(self.prompt, add_special_tokens=False)
        )
