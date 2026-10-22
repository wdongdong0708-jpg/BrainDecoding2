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


@dataclass
class PrefixCacheState:
    """束搜索单个前缀束的语言模型增量缓存状态。"""

    words: tuple[str, ...]
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    last_log_probs: torch.Tensor
    token_length: int
    context_kind: str


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
        self._reference_cache_type: object = None
        self._prompt_cache_state: PrefixCacheState | None = None
        self.stats = {
            "single_token_lookups": 0,
            "multi_token_evaluations": 0,
            "incremental_forward_batches": 0,
            "prefix_advance_batches": 0,
        }

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
        return self._prompt_cache_state

    @torch.inference_mode()
    def score_candidates_with_cache(
        self,
        prefix_states: Sequence[PrefixCacheState],
        candidate_word_indices: Sequence[Sequence[int]],
    ) -> list[np.ndarray]:
        """利用前缀 KV-Cache 批量增量计算每个前缀束下各候选词的精确对数概率 log P_LM(w | u)。

        - 单 Token 候选词：直接从 `prefix_state.last_log_probs[t_0]` O(1) 索引取值，0 次 Forward；
        - 多 Token 候选词：首 Token 从 `last_log_probs[t_0]` 取值，剩余 1~3 个后缀 Token
          复用 `prefix_state.past_key_values` 组装为 Batch 执行增量 Forward。
        """
        if len(prefix_states) != len(candidate_word_indices):
            raise ValueError(
                f"prefix_states ({len(prefix_states)}) 与 candidate_word_indices ({len(candidate_word_indices)}) 数量不一致。"
            )
        results: list[np.ndarray] = [
            np.zeros(len(cands), dtype=np.float64)
            for cands in candidate_word_indices
        ]
        # 收集所有多 Token 候选词请求：(beam_idx, cand_pos, prefix_state, token_tuple)
        multi_requests_by_past_len: dict[
            int, list[tuple[int, int, PrefixCacheState, tuple[int, ...]]]
        ] = defaultdict(list)

        for beam_idx, (state, cand_indices) in enumerate(
            zip(prefix_states, candidate_word_indices)
        ):
            if not cand_indices:
                continue
            ctx = state.context_kind
            log_probs_0 = state.last_log_probs
            tokens_list = self.vocab_table.tokens_by_context[ctx]
            out_row = results[beam_idx]

            for pos, word_idx in enumerate(cand_indices):
                token_ids = tokens_list[int(word_idx)]
                first_lp = float(log_probs_0[token_ids[0]].item())
                out_row[pos] = first_lp
                if len(token_ids) == 1:
                    self.stats["single_token_lookups"] += 1
                else:
                    self.stats["multi_token_evaluations"] += 1
                    multi_requests_by_past_len[state.token_length].append(
                        (beam_idx, pos, state, token_ids)
                    )

        # 按前缀 token_length 分组执行零 KV 填充的批量增量 Forward
        for past_len, group_reqs in multi_requests_by_past_len.items():
            for start in range(0, len(group_reqs), self.max_batch_size):
                batch_reqs = group_reqs[start : start + self.max_batch_size]
                bsz = len(batch_reqs)
                # 每个多 Token 词需输入前 m-1 个 Token 以预测第 1..m-1 个后缀 Token
                step_Lengths = [len(req[3]) - 1 for req in batch_reqs]
                max_step_len = max(step_Lengths)

                input_ids = torch.zeros(
                    (bsz, max_step_len), dtype=torch.long, device=self.device
                )
                for row_i, (_, _, _, tok_ids) in enumerate(batch_reqs):
                    step_tokens = tok_ids[:-1]
                    input_ids[row_i, : len(step_tokens)] = torch.tensor(
                        step_tokens, dtype=torch.long, device=self.device
                    )

                position_ids = (
                    torch.arange(
                        past_len,
                        past_len + max_step_len,
                        dtype=torch.long,
                        device=self.device,
                    )
                    .unsqueeze(0)
                    .expand(bsz, max_step_len)
                )
                batched_kv = _stack_kv_caches(
                    [req[2].past_key_values for req in batch_reqs],
                    self._reference_cache_type,
                )
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=batched_kv,
                    position_ids=position_ids,
                    use_cache=False,
                )
                self.stats["incremental_forward_batches"] += 1
                step_log_probs = F_nn.log_softmax(outputs.logits.float(), dim=-1)

                for row_i, (beam_idx, pos, _, tok_ids) in enumerate(batch_reqs):
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
                    results[beam_idx][pos] += float(gathered.double().sum().item())

        return results

    @torch.inference_mode()
    def propose_top_k_from_language_model(
        self,
        prefix_state: PrefixCacheState,
        top_k: int = 20,
    ) -> np.ndarray:
        """供 LM-Only 基线使用：利用当前前缀的 last_log_probs 从闭集词表 V 中零开销筛选首 Token Top-M 候选并精排返回 Top-K 词索引。"""
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
        """对选中的 (PrefixCacheState, word_index) 列表增量推进 1 个词的 KV-Cache 与末位对数概率。"""
        if not advancement_requests:
            return []

        advanced_states: list[PrefixCacheState | None] = [None] * len(
            advancement_requests
        )
        # 按 (past_len, word_token_len) 分组，同组可一次 Batch Forward 无损更新 KV-Cache
        grouped: dict[
            tuple[int, int],
            list[tuple[int, PrefixCacheState, int, tuple[int, ...]]],
        ] = defaultdict(list)

        for req_idx, (state, word_idx) in enumerate(advancement_requests):
            tok_ids = self.vocab_table.token_ids_for(
                int(word_idx), state.context_kind
            )
            grouped[(state.token_length, len(tok_ids))].append(
                (req_idx, state, int(word_idx), tok_ids)
            )

        for (past_len, step_len), items in grouped.items():
            bsz = len(items)
            input_ids = torch.tensor(
                [item[3] for item in items],
                dtype=torch.long,
                device=self.device,
            )
            position_ids = (
                torch.arange(
                    past_len,
                    past_len + step_len,
                    dtype=torch.long,
                    device=self.device,
                )
                .unsqueeze(0)
                .expand(bsz, step_len)
            )
            batched_kv = _stack_kv_caches(
                [item[1].past_key_values for item in items],
                self._reference_cache_type,
            )
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=batched_kv,
                position_ids=position_ids,
                use_cache=True,
            )
            self.stats["prefix_advance_batches"] += 1
            unstacked_kvs = _unstack_kv_cache(outputs.past_key_values, bsz)
            last_lps = F_nn.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)

            for row_i, (req_idx, state, word_idx, tok_ids) in enumerate(items):
                word_str = self.vocabulary[word_idx]
                new_words = state.words + (word_str,)
                new_ctx = prefix_context_kind(new_words, self.language)
                advanced_states[req_idx] = PrefixCacheState(
                    words=new_words,
                    past_key_values=unstacked_kvs[row_i],
                    last_log_probs=last_lps[row_i],
                    token_length=past_len + step_len,
                    context_kind=new_ctx,
                )

        return [st for st in advanced_states if st is not None]


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
